from tool_system import BaseTool
import imgui
import cv2
import cv2.aruco as aruco
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

class ARTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "AR Marker Pro"
        self.icon_key = 'camera'
        self.hotkey = 'F8'

        # --- Camera ---
        self.cap = None
        self.camera_id = 0
        self.is_active = False
        self.frame_width = 640
        self.frame_height = 480

        # --- ArUco ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        # --- Pose ---
        self.rvec = None
        self.tvec = None
        self.is_visible = False

        # --- Calibration ---
        self.marker_size = 50.0  # mm
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # --- Visuals ---
        self.texture_id = None
        self.video_opacity = 1.0

        # --- Alignment ---
        self.mesh_rot    = [0.0, 0.0, 0.0]
        self.mesh_offset = [0.0, 0.0, 0.0]
        self.mesh_scale  = 1.0
        self.auto_centered   = False
        self.centroid_offset = np.zeros(3, dtype=np.float32)
        self.mesh_scale_xyz  = [1.0, 1.0, 1.0]  # non-uniform scale per axis

    # ------------------------------------------------------------------
    # Marker object-space corners  (flat square, Z=0)
    # ------------------------------------------------------------------
    def _get_obj_pts(self):
        h = self.marker_size / 2.0
        return np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_activate(self):
        self._start_camera()

    def on_deactivate(self):
        self._stop_camera()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _start_camera(self):
        if self.cap:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.app.status_text = f"Error: Cam {self.camera_id} failed."
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            # Read back real resolution
            self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Approximate intrinsics
            f  = float(self.frame_width)
            cx = self.frame_width  / 2.0
            cy = self.frame_height / 2.0
            self.camera_matrix = np.array(
                [[f,  0, cx],
                 [0,  f, cy],
                 [0,  0,  1]], dtype=np.float64)

            # Allocate texture
            if self.texture_id:
                glDeleteTextures([self.texture_id])
            self.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                         self.frame_width, self.frame_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glBindTexture(GL_TEXTURE_2D, 0)

            self.rvec = None
            self.tvec = None
            self.is_active = True
            self.app.status_text = "Point camera at ArUco 4x4 marker."
            self._calc_centroid()

        except Exception as e:
            print(f"Cam Error: {e}")

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_active = False

    # ------------------------------------------------------------------
    # Centroid
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unit helpers — read app.coord_unit so we match the menu-bar setting
    # ------------------------------------------------------------------
    def _coord_unit_to_mm(self):
        table = getattr(self.app, '_UNIT_TO_MM',
                        {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'km': 1_000_000.0})
        return table.get(getattr(self.app, 'coord_unit', 'mm'), 1.0)

    def _coord_unit_label(self):
        return getattr(self.app, 'coord_unit', 'mm')

    # ------------------------------------------------------------------
    # Centroid + one-shot auto-scale
    # ------------------------------------------------------------------
    def _calc_centroid(self):
        if hasattr(self.app, 'points') and self.app.points is not None and len(self.app.points) > 0:
            pts = self.app.points
            self.centroid_offset = -np.mean(pts, axis=0).astype(np.float32)
            self.auto_centered   = True
            # Convert bbox to mm, then scale so bbox_mm == marker_size_mm.
            # This is a ONE-SHOT operation triggered by activate / button click.
            # It must NOT be called from a slider's per-frame change handler.
            bbox_diag_cu = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            bbox_diag_mm = bbox_diag_cu * self._coord_unit_to_mm()
            if bbox_diag_mm > 1e-9:
                self.mesh_scale = self.marker_size / bbox_diag_mm
            self.mesh_scale_xyz = [1.0, 1.0, 1.0]
            self._bbox_diag_cu  = bbox_diag_cu   # cache for drag-speed + readout

    # ------------------------------------------------------------------
    # Frame processing + pose
    # ------------------------------------------------------------------
    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict,
                                              parameters=self.parameters)
        self.is_visible = False

        if ids is not None and len(ids) > 0:
            img_pts = corners[0][0].astype(np.float64)  # (4,2)
            obj_pts = self._get_obj_pts()

            # IPPE_SQUARE is the best solver for flat square targets;
            # fall back to ITERATIVE on older OpenCV builds
            try:
                ret, rvec_raw, tvec_raw = cv2.solvePnP(
                    obj_pts, img_pts,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except Exception:
                ret, rvec_raw, tvec_raw = cv2.solvePnP(
                    obj_pts, img_pts,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE)

            if ret:
                self.is_visible = True

                if self.rvec is None:
                    self.rvec = rvec_raw.astype(np.float64)
                    self.tvec = tvec_raw.astype(np.float64)
                else:
                    # Antipodal-flip guard: OpenCV can negate rvec between
                    # frames (same rotation, opposite sign) → violent 180° jump
                    if np.dot(rvec_raw.flatten(), self.rvec.flatten()) < 0:
                        rvec_raw = -rvec_raw

                    alpha = 0.15   # low = smooth, high = responsive
                    self.rvec = self.rvec * (1.0 - alpha) + rvec_raw * alpha
                    self.tvec = self.tvec * (1.0 - alpha) + tvec_raw * alpha

                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  self.rvec, self.tvec, self.marker_size * 0.6)
        return frame

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render_3d(self):
        if not self.is_active or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        actual_h, actual_w = frame.shape[:2]
        frame_proc = self._process_frame(frame)
        frame_disp = cv2.flip(frame_proc, 0)
        frame_rgb  = np.ascontiguousarray(
            cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        actual_w, actual_h,
                        GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._draw_background_layer()
        glDepthMask(GL_TRUE)          # ensure depth writes are on before clearing
        glClear(GL_DEPTH_BUFFER_BIT)
        if self.is_visible:
            self._draw_ar_mesh()

    # ------------------------------------------------------------------
    # Background quad
    # ------------------------------------------------------------------
    def _draw_background_layer(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glColor4f(1, 1, 1, self.video_opacity)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1,  1)
        glTexCoord2f(0, 1); glVertex2f(-1,  1)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glPopAttrib()
        glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------
    # AR mesh overlay
    # ------------------------------------------------------------------
    def _draw_ar_mesh(self):
        if not hasattr(self.app, 'points') or self.app.points is None:
            return
        if self.rvec is None or self.tvec is None:
            return

        # --- Build OpenGL modelview matrix from OpenCV pose ---
        #
        # OpenCV convention:  X right, Y down,  Z into screen  (right-handed)
        # OpenGL convention:  X right, Y up,    Z out of screen (right-handed)
        # Fix: flip Y and Z rows of the rotation matrix and the translation.
        #
        rot_mat, _ = cv2.Rodrigues(self.rvec)
        R = rot_mat.astype(np.float32).copy()
        t = self.tvec.flatten().astype(np.float32).copy()

        R[1, :] *= -1   # flip Y row
        R[2, :] *= -1   # flip Z row
        t[1]    *= -1
        t[2]    *= -1

        # Assemble 4x4 in row-major numpy, then transpose to column-major
        # for OpenGL.  glMultMatrixf expects 16 floats, column-major.
        mv = np.eye(4, dtype=np.float32)
        mv[:3, :3] = R
        mv[:3,  3] = t
        mv_cm = np.ascontiguousarray(mv.T, dtype=np.float32)   # column-major

        # --- Projection using gluPerspective ---
        # Derived from the camera intrinsic focal length.
        # gluPerspective is correct for centred webcam lenses.
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        f     = float(self.camera_matrix[0, 0])
        fov_y = 2.0 * np.degrees(np.arctan2(self.frame_height / 2.0, f))
        gluPerspective(fov_y,
                       self.frame_width / float(self.frame_height),
                       0.5, 20000.0)

        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # Disable app fog — fog_end is calibrated to orbit camera (~12 units);
        # AR mesh sits at tvec[2] ~ 150-500 mm >> fog_end → blends to dark bg.
        glDisable(GL_FOG)

        # Directional lights (w=0) fixed in eye space BEFORE glMultMatrixf.
        # Positional (w=1) caused darkening on tilt: normals rotated away from
        # the camera-origin point light. Directional = constant regardless of pose.
        glDisable(GL_LIGHTING)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0,  1.0, 0.0])  # w=0 directional
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.0, 0.0,  0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.7, 0.7,  0.7, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2,  0.2, 1.0])
        glEnable(GL_LIGHT1)  # fill from behind, back-faces never black
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, -1.0, 0.0])  # w=0 directional
        glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.0, 0.0,  0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.3, 0.3,  0.3, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0,  0.0, 1.0])
        glEnable(GL_LIGHTING)

        # Apply marker pose — mv_cm is already a flat float32 C-array
        glMultMatrixf(mv_cm)

        # User transforms
        glTranslatef(*self.mesh_offset)
        glRotatef(self.mesh_rot[0], 1, 0, 0)
        glRotatef(self.mesh_rot[1], 0, 1, 0)
        glRotatef(self.mesh_rot[2], 0, 0, 1)
        glScalef(self.mesh_scale * self.mesh_scale_xyz[0],
                 self.mesh_scale * self.mesh_scale_xyz[1],
                 self.mesh_scale * self.mesh_scale_xyz[2])

        if self.auto_centered:
            glTranslatef(*self.centroid_offset)

        glEnable(GL_NORMALIZE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # Force solid fill — the app may have GL_POINT or GL_LINE active
        # for its own point-cloud / wireframe display mode.
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 40.0)

        # Draw — prefer indexed mesh (app.faces), fall back to point cloud
        pts = np.ascontiguousarray(self.app.points, dtype=np.float32)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pts)

        has_normals = (hasattr(self.app, 'normals') and
                       self.app.normals is not None and
                       len(self.app.normals) == len(pts))
        if has_normals:
            nrm = np.ascontiguousarray(self.app.normals, dtype=np.float32)
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, nrm)

        # Per-vertex colors — best approximation of "texture" available.
        # The OBJ loader stores per-vertex RGB in app.base_colors.
        has_colors = (hasattr(self.app, 'base_colors') and
                      self.app.base_colors is not None and
                      len(self.app.base_colors) == len(pts))
        if has_colors:
            cols_arr = np.ascontiguousarray(self.app.base_colors, dtype=np.float32)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, cols_arr)
        else:
            glColor4f(0.85, 0.85, 0.85, 1.0)   # neutral gray fallback

        has_tris = (hasattr(self.app, 'faces') and
                    self.app.faces is not None and
                    len(self.app.faces) > 0)
        if has_tris:
            idx = np.ascontiguousarray(
                self.app.faces.flatten(), dtype=np.uint32)
            glDrawElements(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, idx)
        else:
            # No face indices → point cloud; do NOT use GL_TRIANGLES on raw
            # unindexed vertices (produces garbage geometry).
            glDrawArrays(GL_POINTS, 0, len(pts))

        glDisableClientState(GL_VERTEX_ARRAY)
        if has_normals:
            glDisableClientState(GL_NORMAL_ARRAY)
        if has_colors:
            glDisableClientState(GL_COLOR_ARRAY)

        glPopAttrib()
        glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def draw_options(self):
        imgui.text("AR Marker Pro")
        imgui.separator()

        if imgui.button("Switch Cam"):
            self.camera_id = (self.camera_id + 1) % 4
            self._start_camera()
        imgui.same_line()
        if not self.is_active:
            if imgui.button("Start"): self._start_camera()
        else:
            if imgui.button("Stop"):  self._stop_camera()

        imgui.separator()
        if self.is_visible:
            imgui.text_colored("MARKER DETECTED", 0, 1, 0, 1)
        else:
            imgui.text_colored("Looking for ArUco 4x4...", 1, 0.3, 0.3, 1)

        imgui.separator()
        imgui.text_colored("3. Overlay", 0, 1, 1, 1)
        ch, v = imgui.slider_float("Video Opacity", self.video_opacity, 0.0, 1.0)
        if ch: self.video_opacity = v

        imgui.separator()
        imgui.text_colored("1. Calibration", 0, 1, 1, 1)
        ch, v = imgui.slider_float("Real Size (mm)", self.marker_size, 10.0, 500.0)
        if ch: self.marker_size = v
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Physical size of the printed ArUco marker.\n"
                "Affects pose estimation only — use Auto-Scale to resize the mesh.")

        imgui.separator()
        imgui.text_colored("2. Orientation Fix", 0, 1, 1, 1)
        if imgui.button("Auto-Center Mesh"):
            self._calc_centroid()
            self.mesh_offset = [0.0, 0.0, 0.0]

        imgui.text("Presets:")
        if imgui.button("Flat"):    self.mesh_rot = [0.0, 0.0, 0.0]
        imgui.same_line()
        if imgui.button("Upright"): self.mesh_rot = [-90.0, 0.0, 0.0]
        imgui.same_line()
        if imgui.button("Flip Z"):  self.mesh_rot[2] = (self.mesh_rot[2] + 180.0) % 360.0

        imgui.separator()
        imgui.text("Manual Tweak")

        ch, vals = imgui.drag_float3("Rot XYZ",
                                     self.mesh_rot[0], self.mesh_rot[1], self.mesh_rot[2],
                                     change_speed=1.0)
        if ch: self.mesh_rot = list(vals)

        _bbox_cu   = getattr(self, '_bbox_diag_cu', 1.0)
        _off_speed = max(1e-6, _bbox_cu * 0.01)  # 1% of bbox in coord-units
        _unit      = self._coord_unit_label()
        ch, vals = imgui.drag_float3(f"Pos XYZ ({_unit})",
                                     self.mesh_offset[0], self.mesh_offset[1], self.mesh_offset[2],
                                     change_speed=_off_speed)
        if ch: self.mesh_offset = list(vals)

        _spd = max(1e-9, self.mesh_scale * 0.02)
        ch, v = imgui.drag_float("Scale (uniform)", self.mesh_scale,
                                 change_speed=_spd,
                                 min_value=0.0, max_value=0.0,
                                 format="%.6g")
        if ch: self.mesh_scale = max(1e-9, v)

        imgui.text("Scale per axis (multiplies uniform):")
        _spd_xyz = max(1e-9, max(self.mesh_scale_xyz) * 0.02)
        ch, vals = imgui.drag_float3("XYZ Scale",
                                     self.mesh_scale_xyz[0],
                                     self.mesh_scale_xyz[1],
                                     self.mesh_scale_xyz[2],
                                     change_speed=_spd_xyz,
                                     format="%.6g")
        if ch: self.mesh_scale_xyz = [max(1e-9, x) for x in vals]
        imgui.same_line()
        if imgui.button("Reset##scale_xyz"):
            self.mesh_scale_xyz = [1.0, 1.0, 1.0]
        if imgui.button("Auto-Scale to Marker"):
            self._calc_centroid()
        if imgui.is_item_hovered():
            _u   = self._coord_unit_label()
            _umm = self._coord_unit_to_mm()
            _bd  = getattr(self, '_bbox_diag_cu', 0.0)
            imgui.set_tooltip(
                f"Fit mesh bbox ({_bd:.4g} {_u} = {_bd*_umm:.4g} mm)\n"
                f"to marker_size ({self.marker_size:.1f} mm)")
        _bd_cu  = getattr(self, '_bbox_diag_cu', 0.0)
        _eff_mm = _bd_cu * self._coord_unit_to_mm() * self.mesh_scale
        imgui.text_colored(
            f"Rendered: {_eff_mm:.4g} mm  (unit: {self._coord_unit_label()})",
            0.5, 0.9, 0.5, 1.0)
