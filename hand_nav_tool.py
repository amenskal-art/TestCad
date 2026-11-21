from tool_system import BaseTool
import imgui
import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
import math
from collections import deque

from utils import create_rotation_matrix, normalize


# ---------------------------------------------------------------------------
# Tiny one-pole low-pass filter (exponential moving average).
# alpha = 0 → frozen, alpha = 1 → no filtering (raw).
# A good working value is 0.15–0.30 for hand tracking.
# ---------------------------------------------------------------------------
class _EMA:
    def __init__(self, alpha=0.20):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = self.alpha * v + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class HandNavTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name     = "Gesture Commander"
        self.icon_key = "hand"
        self.hotkey   = "F10"

        # --- MediaPipe ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- Camera ---
        self.cap        = None
        self.camera_id  = 0
        self.is_active  = False
        self.texture_id = None
        self.show_overlay = True

        # --- Gesture debounce ---
        # Require this many consecutive frames of the same gesture before
        # committing to it. Prevents one-frame flickers from switching modes.
        self.GESTURE_CONFIRM_FRAMES = 5
        self._gesture_candidate  = "NEUTRAL"
        self._gesture_count      = 0
        self.mode                = "IDLE"   # committed gesture / mode

        # --- Per-axis EMA filters ---
        # Smooths the raw MediaPipe landmark coordinates before we diff them.
        self._ema_cx     = _EMA(alpha=0.25)   # hand centre X
        self._ema_cy     = _EMA(alpha=0.25)   # hand centre Y
        self._ema_angle  = _EMA(alpha=0.20)   # wrist roll
        self._ema_spread = _EMA(alpha=0.20)   # open-hand spread

        # Previous smoothed values (set on first frame of each gesture)
        self._prev_cx     = None
        self._prev_cy     = None
        self._prev_angle  = None
        self._prev_spread = None

        # --- Dead zones (in smoothed-coordinate units) ---
        # Movements smaller than these are ignored entirely so micro-jitter
        # never reaches the scene.
        self.DEAD_PAN   = 0.004   # fraction of frame width/height
        self.DEAD_ROT   = 0.003
        self.DEAD_ROLL  = 0.8     # degrees
        self.DEAD_ZOOM  = 0.003   # spread units

        # --- Sensitivity (intentionally modest) ---
        self.pan_speed  = 1.2
        self.rot_speed  = 1.8
        self.zoom_speed = 0.6
        self.roll_speed = 0.8

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    def on_activate(self):
        self._start_camera()

    def on_deactivate(self):
        self._stop_camera()

    def _start_camera(self):
        if self.cap:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            if self.texture_id:
                glDeleteTextures([self.texture_id])
            self.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)

            self.is_active = True
            self.app.status_text = (
                "Gestures: Pinch=Pan | Fist=Rotate | Open=Zoom"
            )
        except Exception as e:
            print(f"HandNavTool camera error: {e}")

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
        self.is_active = False

    # -----------------------------------------------------------------------
    # Math helpers
    # -----------------------------------------------------------------------
    def _orthonormalize(self, m):
        x = normalize(m[:3, 0])
        y = normalize(m[:3, 1] - np.dot(x, m[:3, 1]) * x)
        m[:3, 0], m[:3, 1], m[:3, 2] = x, y, np.cross(x, y)
        return m

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def _wrist_angle(lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _hand_spread(lm):
        cx, cy = lm[9].x, lm[9].y
        tips = [4, 8, 12, 16, 20]
        return sum(math.hypot(lm[i].x - cx, lm[i].y - cy) for i in tips) / 5.0

    # -----------------------------------------------------------------------
    # Gesture classifier
    # -----------------------------------------------------------------------
    def _classify_gesture(self, lm):
        wrist = lm[0]
        pinch_dist = self._dist(lm[4], lm[8])

        # Pinch: thumb + index close, pinky still extended
        if pinch_dist < 0.05 and self._dist(lm[20], lm[0]) > 0.15:
            return "PINCH"

        d_index  = self._dist(lm[8],  wrist)
        d_middle = self._dist(lm[12], wrist)
        d_ring   = self._dist(lm[16], wrist)
        d_pinky  = self._dist(lm[20], wrist)
        avg_dist = (d_index + d_middle + d_ring + d_pinky) / 4.0

        # Point: index clearly extended, others curled
        if d_index > 0.22 and d_middle < 0.15 and d_ring < 0.15:
            return "POINT"

        # Fist: all fingers compact
        if avg_dist < 0.15:
            return "FIST"

        # Open: all fingers extended, no pinch
        if avg_dist > 0.25 and pinch_dist > 0.10:
            return "OPEN"

        return "NEUTRAL"

    # -----------------------------------------------------------------------
    # Gesture debounce: only commit when N frames agree
    # -----------------------------------------------------------------------
    def _debounce_gesture(self, raw_gesture):
        if raw_gesture == self._gesture_candidate:
            self._gesture_count += 1
        else:
            self._gesture_candidate = raw_gesture
            self._gesture_count     = 1

        if self._gesture_count >= self.GESTURE_CONFIRM_FRAMES:
            return raw_gesture          # committed
        return self.mode                # still hold previous committed mode

    # -----------------------------------------------------------------------
    # Reset all smoothing filters + previous values
    # -----------------------------------------------------------------------
    def _reset_tracking(self):
        self._ema_cx.reset()
        self._ema_cy.reset()
        self._ema_angle.reset()
        self._ema_spread.reset()
        self._prev_cx     = None
        self._prev_cy     = None
        self._prev_angle  = None
        self._prev_spread = None

    # -----------------------------------------------------------------------
    # Main per-frame update
    # -----------------------------------------------------------------------
    def render_3d(self):
        if not self.is_active or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand visible → reset everything
            prev_mode  = self.mode
            self.mode  = "SEARCHING"
            self._gesture_candidate = "NEUTRAL"
            self._gesture_count     = 0
            if prev_mode != "SEARCHING":
                self._reset_tracking()
        else:
            lm            = results.multi_hand_landmarks[0].landmark
            raw_gesture   = self._classify_gesture(lm)
            new_mode      = self._debounce_gesture(raw_gesture)

            # When mode changes, throw away stale prev values so the first
            # frame of the new mode doesn't produce a huge delta.
            if new_mode != self.mode:
                self._reset_tracking()
            self.mode = new_mode

            # --- Smooth the raw hand position ---
            cx_raw = lm[9].x
            cy_raw = lm[9].y
            cx = self._ema_cx.update(cx_raw)
            cy = self._ema_cy.update(cy_raw)

            # ----------------------------------------------------------------
            # PAN  (Pinch)
            # ----------------------------------------------------------------
            if self.mode == "PINCH":
                cv2.circle(frame,
                           (int(lm[8].x * w), int(lm[8].y * h)),
                           15, (255, 80, 0), -1)

                if self._prev_cx is not None:
                    dx = cx - self._prev_cx
                    dy = cy - self._prev_cy

                    # Dead zone — ignore micro-jitter
                    if abs(dx) < self.DEAD_PAN: dx = 0.0
                    if abs(dy) < self.DEAD_PAN: dy = 0.0

                    if (dx or dy) and hasattr(self.app, "pan_x"):
                        scale = self.pan_speed * 80.0   # was 200 — tamed
                        self.app.pan_x += dx * scale
                        self.app.pan_y -= dy * scale

                self._prev_cx = cx
                self._prev_cy = cy
                # don't update angle / spread trackers
                self._prev_angle  = None
                self._prev_spread = None

            # ----------------------------------------------------------------
            # ROTATE  (Fist)
            # ----------------------------------------------------------------
            elif self.mode == "FIST":
                cv2.circle(frame, (int(cx * w), int(cy * h)), 20, (0, 220, 0), -1)

                raw_angle    = self._wrist_angle(lm)
                smooth_angle = self._ema_angle.update(raw_angle)

                if self._prev_cx is not None and self._prev_angle is not None:
                    dx = cx - self._prev_cx
                    dy = cy - self._prev_cy

                    d_angle = smooth_angle - self._prev_angle
                    # Wrap-around fix
                    if d_angle >  180: d_angle -= 360
                    if d_angle < -180: d_angle += 360

                    # Dead zones
                    if abs(dx)      < self.DEAD_ROT:  dx      = 0.0
                    if abs(dy)      < self.DEAD_ROT:  dy      = 0.0
                    if abs(d_angle) < self.DEAD_ROLL: d_angle = 0.0

                    if hasattr(self.app, "rotation_matrix") and (dx or dy or d_angle):
                        speed = self.rot_speed * 1.8   # was 3.0 * 3.0 = 9× — tamed

                        rx = create_rotation_matrix([1, 0, 0],  dy      * speed)
                        ry = create_rotation_matrix([0, 1, 0], -dx      * speed)
                        rz = create_rotation_matrix([0, 0, 1], -d_angle * self.roll_speed)

                        delta = rz @ rx @ ry
                        self.app.rotation_matrix = delta @ self.app.rotation_matrix
                        self._orthonormalize(self.app.rotation_matrix)

                self._prev_cx    = cx
                self._prev_cy    = cy
                self._prev_angle = smooth_angle
                self._prev_spread = None

            # ----------------------------------------------------------------
            # ZOOM  (Open hand)
            # ----------------------------------------------------------------
            elif self.mode == "OPEN":
                raw_spread    = self._hand_spread(lm)
                smooth_spread = self._ema_spread.update(raw_spread)

                if self._prev_spread is not None:
                    delta = smooth_spread - self._prev_spread

                    if abs(delta) >= self.DEAD_ZOOM and hasattr(self.app, "distance"):
                        zoom_amt = delta * 200.0 * self.zoom_speed  # was 500 — tamed
                        self.app.distance -= zoom_amt
                        self.app.distance  = max(10.0, min(2000.0, self.app.distance))

                self._prev_spread = smooth_spread
                self._prev_cx     = None
                self._prev_cy     = None
                self._prev_angle  = None

                # Visual ring showing spread
                radius = int(smooth_spread * w)
                cv2.circle(frame, (int(cx * w), int(cy * h)),
                           radius, (0, 230, 230), 2)

            # ----------------------------------------------------------------
            # POINT / NEUTRAL / other
            # ----------------------------------------------------------------
            else:
                self._prev_cx     = None
                self._prev_cy     = None
                self._prev_angle  = None
                self._prev_spread = None

            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS,
            )

        # Debug overlay text
        cv2.putText(frame, f"MODE: {self.mode}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Upload texture
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb)

        if self.show_overlay:
            self._draw_overlay()

    # -----------------------------------------------------------------------
    # PiP overlay
    # -----------------------------------------------------------------------
    def _draw_overlay(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, self.app.width, 0, self.app.height, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_TEXTURE_2D)

        pip_w, pip_h = 320, 240
        x = self.app.width  - pip_w - 20
        y = 20

        glColor3f(1, 1, 1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x,          y)
        glTexCoord2f(1, 1); glVertex2f(x + pip_w,  y)
        glTexCoord2f(1, 0); glVertex2f(x + pip_w,  y + pip_h)
        glTexCoord2f(0, 0); glVertex2f(x,          y + pip_h)
        glEnd()

        glPopAttrib()
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # -----------------------------------------------------------------------
    # ImGui panel
    # -----------------------------------------------------------------------
    def draw_options(self):
        imgui.text(f"Mode: {self.mode}")
        imgui.separator()

        if imgui.button("Switch Camera"):
            self.camera_id += 1
            self._start_camera()

        imgui.separator()
        imgui.text("Sensitivity")
        _, self.pan_speed  = imgui.slider_float("Pan",        self.pan_speed,  0.1, 5.0)
        _, self.rot_speed  = imgui.slider_float("Rotate",     self.rot_speed,  0.1, 5.0)
        _, self.roll_speed = imgui.slider_float("Wrist Roll", self.roll_speed, 0.1, 3.0)
        _, self.zoom_speed = imgui.slider_float("Zoom",       self.zoom_speed, 0.1, 2.0)

        imgui.separator()
        imgui.text("Smoothing (lower = smoother)")
        changed, v = imgui.slider_float("EMA Position",
                                        self._ema_cx.alpha, 0.05, 0.60)
        if changed:
            self._ema_cx.alpha = v
            self._ema_cy.alpha = v

        changed, v = imgui.slider_float("EMA Angle",
                                        self._ema_angle.alpha, 0.05, 0.60)
        if changed:
            self._ema_angle.alpha = v

        changed, v = imgui.slider_float("EMA Spread",
                                        self._ema_spread.alpha, 0.05, 0.60)
        if changed:
            self._ema_spread.alpha = v

        imgui.separator()
        _, self.show_overlay = imgui.checkbox("Show Camera Overlay",
                                              self.show_overlay)
