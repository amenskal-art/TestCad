# tools/camera_calibration.py
# Real-time camera lens calibration using a printed checkerboard.
#
# Design goals:
#   * Never blocks / lags the camera stream:
#     - Detection runs in its own thread, fed via a 1-slot queue
#       (newest frame wins, old frames dropped).
#     - Detection runs on a DOWNSCALED grayscale copy, then the corner
#       coordinates are scaled back up. ~5–10x faster than full-res.
#   * Calibrates on a real printed checkerboard (not on-screen):
#     - Uses findChessboardCornersSB when available (more robust on
#       printed paper under uneven lighting and motion blur).
#     - Falls back to classic findChessboardCorners + cornerSubPix.
#   * Auto-captures only DIFFERENT views of the board, not 15 copies
#     of the same pose — that's what actually makes calibration accurate.
#   * Persists results to disk so the user never has to recalibrate
#     after restarting the app.
#   * Pre-computes the undistortion remap once, so per-frame undistortion
#     is a single cv2.remap call (much faster than cv2.undistort).
#
# Usage from the GUI:
#   calib = CameraCalibrator(checkerboard_size=(9, 6))
#   calib.load()                        # restore previous profile if any
#   calib.start_session(target_frames=15)
#   # ... per video frame:
#   calib.feed_frame(bgr_frame)         # non-blocking, never stalls
#   overlay = calib.get_overlay()       # latest frame with corners drawn
#   status  = calib.get_status()        # dict for the GUI to display
#   # When status['captured'] >= target the session auto-finalises and
#   # calls the callback you supplied with the computed profile.
#
#   # Once calibrated:
#   img_undist = calib.undistort(bgr_image)

import os
import json
import time
import threading
from queue import Queue, Empty, Full

import cv2
import numpy as np


CALIB_FILE_DEFAULT = os.path.expanduser("~/.iqscanner_calib.json")


class CameraCalibrator:
    def __init__(self,
                 checkerboard_size=(9, 6),
                 square_size_mm=25.0,
                 calib_file_path=CALIB_FILE_DEFAULT):
        # Inner-corner counts. The OpenCV convention: a 10x7-square board
        # has 9x6 inner corners. Square size in mm only changes the units
        # of the extrinsics — intrinsics & distortion are unaffected — but
        # we keep it so the saved profile is self-describing.
        self.checkerboard_size = tuple(checkerboard_size)
        self.square_size_mm = float(square_size_mm)
        self.calib_file_path = calib_file_path

        # Calibration result (intrinsics + distortion).
        self.camera_matrix = None        # 3x3
        self.dist_coeffs = None          # 1x5 (k1,k2,p1,p2,k3)
        self.image_size = None           # (w, h) the calibration was done at
        self.reproj_error = None         # mean reprojection error in px

        # Pre-computed undistort remap (cv2.remap is much faster than
        # cv2.undistort per frame). Re-built whenever the input image
        # size changes.
        self._remap_x = None
        self._remap_y = None
        self._remap_size = None          # (w, h) the maps were built for
        self._new_camera_matrix = None

        # Capture session state.
        self._session_active = False
        self._target_frames = 15
        self._objpoints = []             # list of Nx3 world-coord arrays
        self._imgpoints = []             # list of Nx2 image-coord arrays
        self._captured_signatures = []   # for the anti-redundancy filter
        self._last_capture_time = 0.0
        self._capture_cooldown = 0.6     # s
        self._capture_min_distance = 0.10  # normalized; see _is_pose_novel
        self._completion_callback = None  # called with (success, info)

        # Detection thread plumbing.
        self._frame_queue = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._detector_thread = None

        # Shared state for the GUI to poll. Guarded by _state_lock so the
        # GUI thread never sees a torn frame or half-updated dict.
        self._state_lock = threading.Lock()
        self._overlay_frame = None
        self._last_status = {
            "captured": 0,
            "target": self._target_frames,
            "pattern_visible": False,
            "session_active": False,
            "message": "Idle.",
        }

        # Reusable object-points template for the board (in board-square
        # units). Constructed once because it never changes.
        cols, rows = self.checkerboard_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        self._objp_template = objp

        # Detect findChessboardCornersSB once (it's missing on very old
        # OpenCV builds; we fall back gracefully).
        self._has_sb = hasattr(cv2, "findChessboardCornersSB")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self):
        """Load a previously-saved calibration profile. Returns True on
        success. Safe to call at app startup — the user should not have
        to calibrate every time they open the app."""
        if not os.path.exists(self.calib_file_path):
            return False
        try:
            with open(self.calib_file_path, "r") as f:
                data = json.load(f)
            self.camera_matrix = np.array(data["mtx"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist"], dtype=np.float64)
            self.image_size = tuple(data.get("image_size", (0, 0)))
            self.reproj_error = data.get("reproj_error", None)
            return True
        except Exception:
            # Corrupt file — wipe state, return False, but don't crash
            # the host app.
            self.camera_matrix = None
            self.dist_coeffs = None
            return False

    def save(self):
        if self.camera_matrix is None:
            return False
        data = {
            "mtx": self.camera_matrix.tolist(),
            "dist": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size) if self.image_size else None,
            "reproj_error": self.reproj_error,
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size_mm,
            "saved_at": time.time(),
        }
        with open(self.calib_file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    def is_calibrated(self):
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def clear(self):
        """Forget the current profile (both in memory and on disk)."""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        self.reproj_error = None
        self._remap_x = None
        self._remap_y = None
        self._remap_size = None
        try:
            if os.path.exists(self.calib_file_path):
                os.remove(self.calib_file_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def start_session(self, target_frames=15, completion_callback=None):
        """Begin a calibration session. Spawns the detector thread.
        completion_callback(success: bool, info: dict) is invoked from
        the detector thread when target_frames have been captured (or
        when stop_session() is called)."""
        if self._session_active:
            return
        self._target_frames = int(target_frames)
        self._objpoints = []
        self._imgpoints = []
        self._captured_signatures = []
        self._last_capture_time = 0.0
        self._completion_callback = completion_callback
        self._session_active = True
        self._stop_event.clear()

        # Drain any stale frame so the new session starts clean.
        try:
            while True:
                self._frame_queue.get_nowait()
        except Empty:
            pass

        with self._state_lock:
            self._last_status = {
                "captured": 0,
                "target": self._target_frames,
                "pattern_visible": False,
                "session_active": True,
                "message": "Show the printed checkerboard to the camera.",
            }
            self._overlay_frame = None

        self._detector_thread = threading.Thread(
            target=self._detector_loop, daemon=True
        )
        self._detector_thread.start()

    def stop_session(self, finalize=False):
        """Halt the detector thread. If finalize=True and enough frames
        have been collected, compute the profile before stopping."""
        if not self._session_active:
            return
        self._session_active = False
        self._stop_event.set()
        if self._detector_thread is not None:
            self._detector_thread.join(timeout=2.0)
            self._detector_thread = None

        if finalize and len(self._objpoints) >= 5:
            self._finalize()
        else:
            with self._state_lock:
                self._last_status["session_active"] = False
                self._last_status["message"] = "Calibration session stopped."

    # ------------------------------------------------------------------
    # Frame feed (called from the camera stream thread; non-blocking)
    # ------------------------------------------------------------------
    def feed_frame(self, bgr_frame):
        """Hand the calibrator the latest camera frame. This NEVER
        blocks: if the detector is busy with the previous frame, the
        new one replaces it in the 1-slot queue and old ones are
        dropped. That keeps the camera stream perfectly smooth."""
        if not self._session_active or bgr_frame is None:
            return
        # Replace-or-insert semantics. Try to drop the stale frame first.
        try:
            self._frame_queue.put_nowait(bgr_frame)
        except Full:
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self._frame_queue.put_nowait(bgr_frame)
            except Full:
                pass  # Detector raced us; drop this frame, no harm done.

    # ------------------------------------------------------------------
    # GUI poll helpers
    # ------------------------------------------------------------------
    def get_status(self):
        with self._state_lock:
            # Return a shallow copy so the GUI can read it freely.
            return dict(self._last_status)

    def get_overlay(self):
        """Latest frame with detected corners drawn on top, or None.
        BGR. Safe to call from the GUI thread; the calibrator copies
        before storing so the GUI's reads can't race detector writes."""
        with self._state_lock:
            return self._overlay_frame

    # ------------------------------------------------------------------
    # Undistortion (the post-calibration runtime path)
    # ------------------------------------------------------------------
    def undistort(self, bgr_image):
        """Apply the saved lens correction to a single image. Uses the
        pre-computed remap tables (cv2.remap) for speed — significantly
        faster than cv2.undistort being called every frame."""
        if not self.is_calibrated() or bgr_image is None:
            return bgr_image
        h, w = bgr_image.shape[:2]
        if self._remap_size != (w, h) or self._remap_x is None:
            self._build_remap(w, h)
        return cv2.remap(bgr_image, self._remap_x, self._remap_y,
                         interpolation=cv2.INTER_LINEAR)

    def _build_remap(self, w, h):
        """(Re)compute the undistortion lookup tables for image size
        (w, h). Done once per size, not once per frame."""
        # alpha=0 → all pixels valid, some FOV cropped.
        # alpha=1 → keeps full FOV, may include black borders.
        # 0 is the right choice for downstream 3D reconstruction.
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_mtx,
            (w, h), cv2.CV_16SC2,  # 16SC2 is fastest for cv2.remap
        )
        self._remap_x = map_x
        self._remap_y = map_y
        self._remap_size = (w, h)
        self._new_camera_matrix = new_mtx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _detector_loop(self):
        """Worker thread. Pulls the latest frame from the queue and
        searches it for the checkerboard. Decoupled from the camera
        thread so detection cost never affects the video feed."""
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._process_one_frame(frame)
            except Exception as e:
                with self._state_lock:
                    self._last_status["message"] = f"Detector error: {e}"

    def _process_one_frame(self, frame):
        full_h, full_w = frame.shape[:2]

        # Downscale for the search; finding the board doesn't need full
        # res, and downscaling is the single biggest speed-up here.
        max_dim_for_detect = 960
        scale = 1.0
        if max(full_w, full_h) > max_dim_for_detect:
            scale = max_dim_for_detect / float(max(full_w, full_h))
            small = cv2.resize(frame, (int(full_w * scale), int(full_h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        found_small, corners_small = self._find_corners_fast(gray_small)

        overlay = frame.copy()
        pattern_visible = bool(found_small)

        if pattern_visible:
            # Scale the rough corner positions back up to full res, then
            # refine ON THE FULL-RES FRAME so accuracy isn't capped by
            # the downscale.
            corners_full_rough = (corners_small / scale).astype(np.float32) if scale != 1.0 else corners_small
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_full = self._refine_corners(gray_full, corners_full_rough)

            # Draw the polished corners on the overlay so the user sees
            # accurate feedback.
            cv2.drawChessboardCorners(overlay, self.checkerboard_size,
                                      corners_full, True)

            self._maybe_capture(corners_full, (full_w, full_h), overlay)

        with self._state_lock:
            self._overlay_frame = overlay
            self._last_status["pattern_visible"] = pattern_visible
            self._last_status["captured"] = len(self._objpoints)
            self._last_status["target"] = self._target_frames
            self._last_status["session_active"] = self._session_active
            if not pattern_visible:
                self._last_status["message"] = (
                    "Aim the printed board at the camera — hold it steady, fill the view."
                )
            else:
                self._last_status["message"] = (
                    f"Pattern locked. Move the board to a new angle. "
                    f"({len(self._objpoints)}/{self._target_frames})"
                )

    def _find_corners_fast(self, gray):
        """Robust + fast board detector. Prefer the SB ('Symmetric Box')
        detector — it handles motion blur and uneven lighting on real
        printed paper much better than the legacy detector — and fall
        back if it's not built into this OpenCV."""
        if self._has_sb:
            try:
                found, corners = cv2.findChessboardCornersSB(
                    gray, self.checkerboard_size, flags=0
                )
                if found:
                    return True, corners
            except cv2.error:
                pass  # fall through to the classic detector

        # Classic path. FAST_CHECK lets it bail almost instantly on
        # frames that obviously have no board.
        found, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH
                   | cv2.CALIB_CB_NORMALIZE_IMAGE
                   | cv2.CALIB_CB_FAST_CHECK)
        )
        return found, corners

    def _refine_corners(self, gray_full, corners_rough):
        """findChessboardCornersSB returns sub-pixel coords already, but
        the classic detector returns integer-pixel coords that need
        cornerSubPix to be useful for calibration."""
        # If they came from SB the refinement is a no-op cost-wise but
        # harmless; cornerSubPix on already-good points just confirms.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cv2.cornerSubPix(gray_full, corners_rough, (11, 11), (-1, -1), criteria)

    def _maybe_capture(self, corners_full, image_size, overlay_for_flash):
        """Decide whether this frame should be added to the calibration
        set. Two filters:
          * Time cooldown — don't capture 30 nearly-identical frames in
            a row when the user is holding the board still.
          * Pose novelty — only accept boards whose centroid/scale/skew
            signature is meaningfully different from every previous
            capture. This is what makes the resulting calibration
            actually accurate, instead of just having 15 frames of the
            same view."""
        now = time.time()
        if now - self._last_capture_time < self._capture_cooldown:
            return
        if len(self._objpoints) >= self._target_frames:
            return

        signature = self._pose_signature(corners_full, image_size)
        if not self._is_pose_novel(signature):
            return

        self._objpoints.append(self._objp_template.copy())
        self._imgpoints.append(corners_full)
        self._captured_signatures.append(signature)
        self._last_capture_time = now

        # Flash a green border on the overlay to confirm capture.
        h, w = overlay_for_flash.shape[:2]
        cv2.rectangle(overlay_for_flash, (0, 0), (w - 1, h - 1), (0, 255, 0), 18)

        if len(self._objpoints) >= self._target_frames:
            self._finalize()

    def _pose_signature(self, corners, image_size):
        """Compress a corner array into a small descriptor of where the
        board IS in the image and HOW it's oriented. Different poses
        produce different signatures."""
        w, h = image_size
        pts = corners.reshape(-1, 2)
        # Normalized centroid (where in the frame the board sits).
        cx = float(pts[:, 0].mean()) / w
        cy = float(pts[:, 1].mean()) / h
        # Bounding-box width and height (how big / how tilted).
        bw = (pts[:, 0].max() - pts[:, 0].min()) / w
        bh = (pts[:, 1].max() - pts[:, 1].min()) / h
        # Aspect — a strong proxy for board rotation/tilt.
        aspect = bw / max(bh, 1e-6)
        return np.array([cx, cy, bw, bh, np.tanh(aspect - 1.0)],
                        dtype=np.float32)

    def _is_pose_novel(self, signature):
        if not self._captured_signatures:
            return True
        prior = np.stack(self._captured_signatures, axis=0)
        diffs = np.linalg.norm(prior - signature[None, :], axis=1)
        return float(diffs.min()) >= self._capture_min_distance

    def _finalize(self):
        """Run cv2.calibrateCamera on the collected frames, store the
        result, persist to disk, and fire the user callback."""
        with self._state_lock:
            self._last_status["message"] = "Computing lens profile..."

        # All frames in this session share the same image size — the
        # camera doesn't change resolution mid-session — so just use
        # the first one to look it up.
        # NOTE: corners are in full-res coords (see _process_one_frame),
        # so we need the full-res image size. We grab it from the most
        # recent overlay.
        with self._state_lock:
            if self._overlay_frame is not None:
                h, w = self._overlay_frame.shape[:2]
            else:
                w, h = 0, 0

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self._objpoints, self._imgpoints, (w, h), None, None
            )
        except cv2.error as e:
            self._session_active = False
            self._stop_event.set()
            info = {"success": False, "error": str(e)}
            with self._state_lock:
                self._last_status["message"] = f"Calibration failed: {e}"
                self._last_status["session_active"] = False
            if self._completion_callback:
                self._completion_callback(False, info)
            return

        # Mean reprojection error — the headline number for "how good is
        # this calibration".
        total_err = 0.0
        for i in range(len(self._objpoints)):
            projected, _ = cv2.projectPoints(self._objpoints[i],
                                             rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(self._imgpoints[i], projected, cv2.NORM_L2) / len(projected)
            total_err += err
        reproj = total_err / max(len(self._objpoints), 1)

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.image_size = (w, h)
        self.reproj_error = float(reproj)
        # Force the undistortion remap to rebuild next time it's used.
        self._remap_x = None
        self._remap_y = None
        self._remap_size = None
        self.save()

        accuracy = ("Excellent" if reproj < 0.5
                    else "Good" if reproj < 1.0
                    else "Poor — please recalibrate")

        info = {
            "success": True,
            "reproj_error": reproj,
            "accuracy": accuracy,
            "frames_used": len(self._objpoints),
            "image_size": (w, h),
        }
        self._session_active = False
        self._stop_event.set()
        with self._state_lock:
            self._last_status["session_active"] = False
            self._last_status["message"] = (
                f"Done! Reprojection error {reproj:.3f}px ({accuracy})."
            )
        if self._completion_callback:
            self._completion_callback(True, info)
