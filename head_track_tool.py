from tool_system import BaseTool
import imgui
import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import glfw

class HeadTrackTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Holo-View (Nike Demo)"
        self.icon_key = 'eye' 
        self.hotkey = 'F11'
        
        # --- MediaPipe Setup ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # --- Camera State ---
        self.cap = None
        self.camera_id = 0
        self.is_active = False
        self.texture_id = None
        self.show_overlay = True
        
        # --- Physical Screen Setup (Meters) ---
        self.screen_width = 0.267  
        self.screen_height = 0.178 
        
        # --- Tracking Data ---
        self.head_pos = np.array([0.0, 0.0, 0.5]) 
        self.raw_head_pos = np.array([0.0, 0.0, 0.5])
        self.calib_offset = np.array([0.0, 0.0, 0.0])
        
        # --- Tuning ---
        self.sensitivity = np.array([0.1, 0.1, 0.3]) 
        self.alpha = 0.4
        self.max_range_x = 0.15 
        self.max_range_y = 0.15 
        
        # --- Storage ---
        self.original_setup_camera = None
        self.original_draw_grid = None
        self.original_setup_lighting = None
        self.original_setup_lighting_fixed = None
        self.original_distance = None 
        
        self.last_distance = 1.0

    def on_activate(self):
        self._start_camera()
        
        # 1. Patch Camera
        self.original_setup_camera = self.app.setup_camera
        self.app.setup_camera = self._holo_setup_camera
        
        # 2. Patch Grid
        self.original_draw_grid = self.app.draw_background_grid
        self.app.draw_background_grid = self._draw_3d_room
        
        # 3. Patch Lighting (Override everything to be safe)
        if hasattr(self.app, 'setup_lighting'):
            self.original_setup_lighting = self.app.setup_lighting
            self.app.setup_lighting = self._holo_lighting
            
        if hasattr(self.app, 'setup_lighting_fixed'):
            self.original_setup_lighting_fixed = self.app.setup_lighting_fixed
            self.app.setup_lighting_fixed = self._holo_lighting
            
        # 4. Save Zoom
        self.original_distance = self.app.distance
        self.app.status_text = "Holo-View: Active. Scroll to Scale. Click Calibrate."

    def on_deactivate(self):
        self._stop_camera()
        
        if self.original_setup_camera:
            self.app.setup_camera = self.original_setup_camera
            self.original_setup_camera = None
            
        if self.original_draw_grid:
            self.app.draw_background_grid = self.original_draw_grid
            self.original_draw_grid = None
            
        if self.original_setup_lighting:
            self.app.setup_lighting = self.original_setup_lighting
            self.original_setup_lighting = None

        if self.original_setup_lighting_fixed:
            self.app.setup_lighting_fixed = self.original_setup_lighting_fixed
            self.original_setup_lighting_fixed = None
            
        if self.original_distance:
            self.app.distance = self.original_distance
            
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        h = self.app.height if self.app.height > 0 else 1
        gluPerspective(60.0, self.app.width/h, 0.01, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        
        glUseProgram(0)
        glEnable(GL_CULL_FACE)

    def _holo_lighting(self):
        """
        Set up fixed-function lighting (No Shaders).
        Attached to CAMERA (Headlight).
        """
        glUseProgram(0) # Disable shaders
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glShadeModel(GL_SMOOTH)
        
        # High Ambient for safety (prevents black mesh)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1.0])

        # Material (High Specular)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 80.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # --- Headlight Setup ---
        # We must set position while ModelView is Identity to lock it to camera
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glEnable(GL_LIGHT0)
        # Directional light from Z axis (Your Eyes)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0]) 
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0]) 
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0]) 
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        glPopMatrix()

    def _holo_setup_camera(self):
        # --- 1. PROJECTION (The Holo Window) ---
        pa = self.screen_width 
        pb = self.screen_height
        
        pe = self.head_pos.copy()
        pe[0] = np.clip(pe[0], -self.max_range_x, self.max_range_x)
        pe[1] = np.clip(pe[1], -self.max_range_y, self.max_range_y)
        
        pe_mm = pe * 1000.0 
        pa_mm = pa * 1000.0
        pb_mm = pb * 1000.0
        
        near_mm = 1.0 
        if pe_mm[2] < near_mm + 1:
            pe_mm[2] = near_mm + 1
        
        ratio = near_mm / pe_mm[2]
        l = (-0.5 * pa_mm - pe_mm[0]) * ratio
        r = ( 0.5 * pa_mm - pe_mm[0]) * ratio
        b = (-0.5 * pb_mm - pe_mm[1]) * ratio
        t = ( 0.5 * pb_mm - pe_mm[1]) * ratio
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(l, r, b, t, near_mm, 50000.0) 
        
        # --- 2. MODELVIEW (The World) ---
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply Lighting NOW (Locked to Camera)
        self._holo_lighting()
        
        # --- 3. MOVEMENT & ZOOM ---
        
        # A. Move Head (Holo Effect)
        glTranslatef(-pe_mm[0], -pe_mm[1], -pe_mm[2])
        
        # B. Pivot Adjustment
        glTranslatef(0, 0, 0)
        
        # C. Pan
        px = getattr(self.app, 'pan_x', 0.0)
        py = getattr(self.app, 'pan_y', 0.0)
        glTranslatef(px, py, 0)
        
        # D. Zoom (THE FIX: Move Camera, Don't Scale Mesh)
        # We translate along Z. 
        # Note: app.distance is usually positive. We move -Z to see it.
        # We clamp min distance to 0.1 to avoid passing through it.
        zoom_dist = max(self.app.distance, 5.0) 
        glTranslatef(0, 0, -zoom_dist)
        
        # E. Rotate Object
        glMultMatrixf(self.app.rotation_matrix.T)
        
        # F. Center Object (Mesh Offset)
        glTranslatef(-self.app.center[0], -self.app.center[1], -self.app.center[2])
        
        # Final Safety Check
        glUseProgram(0) 

    def _draw_3d_room(self):
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT)
        glUseProgram(0)
        
        w = (self.screen_width * 1000.0) / 2.0
        h = (self.screen_height * 1000.0) / 2.0
        d = (self.screen_width * 1000.0) * 1.5 
        
        glDisable(GL_LIGHTING) 
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST) 
        
        glLineWidth(2.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        pe = self.head_pos.copy()
        pe[0] = np.clip(pe[0], -self.max_range_x, self.max_range_x)
        pe[1] = np.clip(pe[1], -self.max_range_y, self.max_range_y)
        pe_mm = pe * 1000.0
        glTranslatef(-pe_mm[0], -pe_mm[1], -pe_mm[2])
        
        step = 50.0 
        
        # Back Wall
        glColor3f(0.2, 0.8, 1.0) 
        glBegin(GL_LINES)
        x = -w
        while x <= w + 1:
            glVertex3f(x, -h, -d)
            glVertex3f(x, h, -d)
            x += step
        y = -h
        while y <= h + 1:
            glVertex3f(-w, y, -d)
            glVertex3f(w, y, -d)
            y += step
        glEnd()
        
        # Tunnel
        glColor3f(0.1, 0.3, 0.5)
        glLineWidth(1.0) 
        glBegin(GL_LINES)
        x = -w
        while x <= w + 1:
            glVertex3f(x, -h, 0)
            glVertex3f(x, -h, -d)
            glVertex3f(x, h, 0)
            glVertex3f(x, h, -d)
            x += step
        y = -h
        while y <= h + 1:
            glVertex3f(-w, y, 0)
            glVertex3f(-w, y, -d)
            glVertex3f(w, y, 0)
            glVertex3f(w, y, -d)
            y += step
        glEnd()
        
        # Screen Frame
        glColor3f(1.0, 1.0, 1.0) 
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-w, -h, 0)
        glVertex3f(w, -h, 0)
        glVertex3f(w, h, 0)
        glVertex3f(-w, h, 0)
        glEnd()

        glPopMatrix()
        glPopAttrib() 

    def render_3d(self):
        if not self.is_active or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self._update_head_position(frame)
            self._update_camera_texture(frame)

    def _update_head_position(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            nose = lm[6] 
            
            raw_x = (nose.x - 0.5) * -1.0 
            raw_y = (nose.y - 0.5) * -1.0 
            
            dx = lm[234].x - lm[454].x
            dy = lm[234].y - lm[454].y
            face_width = math.hypot(dx, dy)
            raw_z = (0.15 / max(face_width, 0.01)) * 0.4 
            
            current = np.array([raw_x, raw_y, raw_z])
            self.raw_head_pos = current
            
            target = (current - self.calib_offset) * self.sensitivity
            target[2] = current[2] 
            
            self.head_pos = (self.head_pos * (1.0 - self.alpha)) + (target * self.alpha)

    def _start_camera(self):
        if self.cap:
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        self.is_active = True

    def _stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_active = False

    def _update_camera_texture(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)

    def calibrate_zero(self):
        self.calib_offset[0] = self.raw_head_pos[0]
        self.calib_offset[1] = self.raw_head_pos[1]
        
    def draw_options(self):
        imgui.text("Holo-View Settings")
        if imgui.button("CALIBRATE CENTER"):
            self.calibrate_zero()
        imgui.separator()
        imgui.text(f"Pos: {self.head_pos[0]:.2f}, {self.head_pos[1]:.2f}, {self.head_pos[2]:.2f} m")
        imgui.separator()
        _, self.sensitivity[0] = imgui.slider_float("Sens X", self.sensitivity[0], 0.1, 1.0)
        _, self.sensitivity[1] = imgui.slider_float("Sens Y", self.sensitivity[1], 0.1, 1.0)
        _, self.alpha = imgui.slider_float("Smooth", self.alpha, 0.01, 0.5)
        
        imgui.separator()
        _, self.max_range_x = imgui.slider_float("Max X", self.max_range_x, 0.05, 0.5)

    def render_2d_overlay(self):
        if self.show_overlay and self.texture_id:
            win_w, win_h = glfw.get_window_size(self.app.window)
            pip_w, pip_h = 160, 120
            dl = imgui.get_background_draw_list()
            dl.add_image(self.texture_id, (10, win_h - pip_h - 10), (10 + pip_w, win_h - 10))
            dl.add_rect(10, win_h - pip_h - 10, 10 + pip_w, win_h - 10, 0x88FFFFFF)
