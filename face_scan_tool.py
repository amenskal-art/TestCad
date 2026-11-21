# tools/face_scan_tool.py
from tool_system import BaseTool
import imgui
import glfw
import cv2
import mediapipe as mp
import numpy as np
import threading
from OpenGL.GL import *
import os

class FaceScanTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Face Scanner"
        self.icon_key = 'face_scan'
        self.hotkey = '9'
        
        # Face detection settings
        self.input_image = None
        self.output_image = None
        self.face_detected = False
        self.processing = False
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
        # Display settings
        self.show_preview = True
        self.overlay_opacity = 0.7
        self.mesh_color = [0.0, 1.0, 1.0, 0.8]  # Cyan RGBA
        self.highlight_color = [1.0, 0.0, 1.0, 0.8]  # Magenta RGBA
        
        # Processing status
        self.status_message = "Ready to scan face"
        self.progress = 0.0
        
        # File browser state
        self.show_file_browser = False
        self.current_dir = os.path.expanduser("~")
        self.file_list = []
        self.selected_file = None
        self.file_filter = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # WiFi Camera state
        self.camera_enabled = False
        self.camera_stream = None
        self.camera_thread = None
        self.camera_running = False
        self.current_frame = None
        self.camera_ip = "192.168.1.100:8080"  # Default IP
        self.camera_ip_buffer = self.camera_ip
        self.show_camera_dialog = False
        self.camera_connection_status = "Disconnected"
        
        # OpenGL texture for preview
        self.input_texture = None
        self.output_texture = None
        self.camera_texture = None
        self.texture_width = 0
        self.texture_height = 0
        
    def on_activate(self):
        self.app.status_text = "Face Scanner Mode"
        self.status_message = "Browse image or connect camera"
        
    def on_deactivate(self):
        # Cleanup
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        
        # Stop camera if running
        self.disconnect_camera()
        
        # Delete OpenGL textures
        if self.input_texture is not None:
            glDeleteTextures([self.input_texture])
            self.input_texture = None
        if self.output_texture is not None:
            glDeleteTextures([self.output_texture])
            self.output_texture = None
        if self.camera_texture is not None:
            glDeleteTextures([self.camera_texture])
            self.camera_texture = None
        
    def on_mouse_button(self, button, action, mods, x, y):
        return False
        
    def on_mouse_move(self, x, y):
        return False
    
    def on_key(self, key, scancode, action, mods):
        """Handle keyboard input for camera capture"""
        if self.camera_enabled and self.camera_running and action == glfw.PRESS:
            # Press SPACE to capture frame
            if key == glfw.KEY_SPACE:
                self.capture_camera_frame()
                return True
            # Press C to toggle camera
            elif key == glfw.KEY_C:
                if self.camera_running:
                    self.disconnect_camera()
                else:
                    self.show_camera_dialog = True
                return True
        return False
        
    def update(self):
        # Update camera texture if camera is running
        if self.camera_running and self.current_frame is not None:
            self.update_camera_texture()
        
    def render_3d(self):
        pass
    
    # ==================== Camera Functions ====================
    
    def camera_stream_thread(self):
        """Thread function to continuously read from camera"""
        try:
            # Try different URL formats
            url_formats = [
                f"http://{self.camera_ip}/video",
                f"http://{self.camera_ip}/video/mjpeg",
                f"http://{self.camera_ip}/",
            ]
            
            self.camera_stream = None
            for url in url_formats:
                self.camera_stream = cv2.VideoCapture(url)
                if self.camera_stream.isOpened():
                    self.camera_connection_status = f"Connected to {url}"
                    break
                self.camera_stream.release()
                self.camera_stream = None
            
            if self.camera_stream is None or not self.camera_stream.isOpened():
                self.camera_connection_status = "Failed to connect"
                self.camera_running = False
                return
            
            while self.camera_running:
                ret, frame = self.camera_stream.read()
                if ret:
                    self.current_frame = frame
                else:
                    self.camera_connection_status = "Connection lost"
                    break
            
        except Exception as e:
            self.camera_connection_status = f"Error: {str(e)}"
        finally:
            if self.camera_stream is not None:
                self.camera_stream.release()
            self.camera_running = False
    
    def connect_camera(self):
        """Connect to WiFi camera"""
        if self.camera_running:
            return
        
        self.camera_ip = self.camera_ip_buffer
        self.camera_running = True
        self.camera_enabled = True
        self.camera_connection_status = "Connecting..."
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_stream_thread)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        self.status_message = f"Connecting to {self.camera_ip}..."
    
    def disconnect_camera(self):
        """Disconnect from camera"""
        self.camera_running = False
        self.camera_enabled = False
        
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=2.0)
            self.camera_thread = None
        
        if self.camera_stream is not None:
            self.camera_stream.release()
            self.camera_stream = None
        
        self.current_frame = None
        self.camera_connection_status = "Disconnected"
        self.status_message = "Camera disconnected"
        
        # Delete camera texture
        if self.camera_texture is not None:
            glDeleteTextures([self.camera_texture])
            self.camera_texture = None
    
    def update_camera_texture(self):
        """Update OpenGL texture with current camera frame"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # Convert BGR to RGB
        gl_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gl_image = np.flipud(gl_image)
        
        h, w = gl_image.shape[:2]
        
        # Create or update texture
        if self.camera_texture is None:
            self.camera_texture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.camera_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, 
                     GL_RGB, GL_UNSIGNED_BYTE, gl_image)
    
    def capture_camera_frame(self):
        """Capture current camera frame for processing"""
        if self.current_frame is None:
            self.status_message = "No camera frame available"
            return
        
        # Save current frame as input image
        self.input_image = self.current_frame.copy()
        
        # Create texture for preview
        self.create_texture_from_image(self.input_image, is_output=False)
        
        self.status_message = "Frame captured! Ready to process"
        self.output_image = None
        self.face_detected = False
    
    def draw_camera_dialog(self):
        """Draw camera connection dialog"""
        if not self.show_camera_dialog:
            return
        
        imgui.open_popup("WiFi Camera")
        
        if imgui.begin_popup_modal("WiFi Camera", True, 
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Connect to WiFi Camera")
            imgui.separator()
            
            imgui.text("Camera IP Address:")
            imgui.push_item_width(300)
            changed, self.camera_ip_buffer = imgui.input_text(
                "##ip", self.camera_ip_buffer, 256
            )
            imgui.pop_item_width()
            
            imgui.dummy(0, 5)
            imgui.text_disabled("Format: IP:PORT (e.g., 192.168.1.100:8080)")
            imgui.text_disabled("Common apps: IP Webcam, DroidCam, etc.")
            
            imgui.dummy(0, 10)
            imgui.separator()
            
            # Connection status
            if self.camera_running:
                imgui.text_colored("Status: ", 0.5, 0.5, 0.5)
                imgui.same_line()
                if "Connected" in self.camera_connection_status:
                    imgui.text_colored(self.camera_connection_status, 0.0, 1.0, 0.0)
                else:
                    imgui.text_colored(self.camera_connection_status, 1.0, 1.0, 0.0)
            
            imgui.dummy(0, 10)
            
            # Buttons
            if not self.camera_running:
                if imgui.button("Connect", width=120):
                    self.connect_camera()
                    if self.camera_running:
                        self.show_camera_dialog = False
                        imgui.close_current_popup()
            else:
                if imgui.button("Disconnect", width=120):
                    self.disconnect_camera()
            
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                self.show_camera_dialog = False
                imgui.close_current_popup()
            
            imgui.end_popup()
    
    # ==================== File Browser Functions ====================
    
    def refresh_file_list(self):
        """Refresh file list for current directory"""
        try:
            self.file_list = []
            
            # Add parent directory option
            if self.current_dir != os.path.dirname(self.current_dir):
                self.file_list.append(("..", True))
            
            # List directories and files
            items = os.listdir(self.current_dir)
            
            # Separate dirs and files
            dirs = []
            files = []
            
            for item in items:
                full_path = os.path.join(self.current_dir, item)
                if os.path.isdir(full_path):
                    dirs.append((item, True))
                else:
                    ext = os.path.splitext(item)[1].lower()
                    if ext in self.file_filter:
                        files.append((item, False))
            
            # Sort and combine
            dirs.sort()
            files.sort()
            self.file_list.extend(dirs)
            self.file_list.extend(files)
            
        except Exception as e:
            self.status_message = f"Error reading directory: {str(e)}"
    
    def draw_file_browser(self):
        """Draw file browser dialog using ImGui"""
        if not self.show_file_browser:
            return
        
        imgui.open_popup("File Browser")
        
        if imgui.begin_popup_modal("File Browser", True, 
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text(f"Current: {self.current_dir}")
            imgui.separator()
            
            # Refresh button
            if imgui.button("Refresh"):
                self.refresh_file_list()
            
            imgui.same_line()
            if imgui.button("Home"):
                self.current_dir = os.path.expanduser("~")
                self.refresh_file_list()
            
            imgui.separator()
            
            # File list
            imgui.begin_child("file_list", 500, 400, border=True)
            
            for item, is_dir in self.file_list:
                icon = "[DIR] " if is_dir else "[IMG] "
                
                if imgui.selectable(icon + item, self.selected_file == item)[0]:
                    self.selected_file = item
                    
                    # Double click to navigate/select
                    if imgui.is_mouse_double_clicked(0):
                        full_path = os.path.join(self.current_dir, item)
                        
                        if is_dir:
                            if item == "..":
                                self.current_dir = os.path.dirname(self.current_dir)
                            else:
                                self.current_dir = full_path
                            self.refresh_file_list()
                            self.selected_file = None
                        else:
                            # Load the selected image
                            self.load_image_from_path(full_path)
                            self.show_file_browser = False
                            imgui.close_current_popup()
            
            imgui.end_child()
            
            imgui.separator()
            
            # Buttons
            if imgui.button("Select", width=120):
                if self.selected_file:
                    full_path = os.path.join(self.current_dir, self.selected_file)
                    if os.path.isfile(full_path):
                        self.load_image_from_path(full_path)
                        self.show_file_browser = False
                        imgui.close_current_popup()
            
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                self.show_file_browser = False
                imgui.close_current_popup()
            
            imgui.end_popup()
    
    def load_image_from_path(self, file_path):
        """Load image from file path"""
        try:
            self.input_image = cv2.imread(file_path)
            if self.input_image is None:
                self.status_message = "Error: Could not read image"
                return False
            
            self.output_image = None
            self.face_detected = False
            
            # Create OpenGL texture for preview
            self.create_texture_from_image(self.input_image, is_output=False)
            
            self.status_message = f"Loaded: {os.path.basename(file_path)}"
            return True
            
        except Exception as e:
            self.status_message = f"Error loading image: {str(e)}"
            return False
    
    # ==================== Texture Functions ====================
    
    def create_texture_from_image(self, image, is_output=False):
        """Create OpenGL texture from OpenCV image"""
        # Convert image for OpenGL
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # BGRA to RGBA
                gl_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                format_type = GL_RGBA
            else:
                # BGR to RGB
                gl_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                format_type = GL_RGB
        else:
            gl_image = image
            format_type = GL_LUMINANCE
        
        # Flip vertically for OpenGL
        gl_image = np.flipud(gl_image)
        
        h, w = gl_image.shape[:2]
        
        # Delete old texture if exists
        if is_output and self.output_texture is not None:
            glDeleteTextures([self.output_texture])
        elif not is_output and self.input_texture is not None:
            glDeleteTextures([self.input_texture])
        
        # Create new texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, format_type, w, h, 0, 
                     format_type, GL_UNSIGNED_BYTE, gl_image)
        
        if is_output:
            self.output_texture = texture
        else:
            self.input_texture = texture
            self.texture_width = w
            self.texture_height = h
        
        return texture
    
    # ==================== Face Processing Functions ====================
    
    def process_face_thread(self):
        """Process face detection in separate thread"""
        try:
            self.processing = True
            self.progress = 0.1
            self.status_message = "Detecting face..."
            
            img = self.input_image.copy()
            img_h, img_w = img.shape[:2]
            
            self.progress = 0.3
            
            # Initialize face mesh if needed
            if self.face_mesh is None:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            
            # Convert and process
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_img)
            
            self.progress = 0.5
            
            if not results.multi_face_landmarks:
                self.status_message = "No face detected in image"
                self.face_detected = False
                self.processing = False
                return
            
            self.status_message = "Creating face mask..."
            face_landmarks = results.multi_face_landmarks[0]
            
            # Create face mask
            face_mask = self.create_face_mask(img, face_landmarks, img_h, img_w)
            
            self.progress = 0.7
            self.status_message = "Generating overlays..."
            
            # Create overlay
            overlay = self.create_futuristic_overlay(img, face_landmarks, img_h, img_w)
            
            self.progress = 0.9
            self.status_message = "Compositing..."
            
            # Composite final image
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = face_mask
            
            for c in range(3):
                img_rgba[:, :, c] = np.where(
                    overlay[:, :, 3] > 0,
                    cv2.addWeighted(img_rgba[:, :, c], self.overlay_opacity, 
                                   overlay[:, :, c], 1.0 - self.overlay_opacity, 0),
                    img_rgba[:, :, c]
                )
            
            self.output_image = img_rgba
            
            # Create texture for preview
            self.create_texture_from_image(img_rgba, is_output=True)
            
            self.face_detected = True
            self.progress = 1.0
            self.status_message = "Processing complete!"
            
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.face_detected = False
        finally:
            self.processing = False
    
    def process_face(self):
        """Start face processing"""
        if self.input_image is None:
            self.status_message = "Please load an image first"
            return
        
        # Run in thread to avoid blocking
        thread = threading.Thread(target=self.process_face_thread)
        thread.daemon = True
        thread.start()
    
    def save_image(self):
        """Save the processed image"""
        if self.output_image is None:
            self.status_message = "No processed image to save"
            return
        
        # Simple save to predefined location
        timestamp = int(threading.current_thread().ident * 1000) % 1000000
        output_path = f"face_scan_output_{timestamp}.png"
        
        try:
            cv2.imwrite(output_path, self.output_image)
            self.status_message = f"Saved: {output_path}"
        except Exception as e:
            self.status_message = f"Save error: {str(e)}"
    
    def create_face_mask(self, img, face_landmarks, img_h, img_w):
        """Create binary mask for face region"""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        face_oval = self.mp_face_mesh.FACEMESH_FACE_OVAL
        
        landmarks = face_landmarks.landmark
        points = []
        
        for connection in face_oval:
            for idx in connection:
                pt = (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
                points.append(pt)
        
        points = list(set(points))
        points = np.array(points)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask
    
    def create_futuristic_overlay(self, img, face_landmarks, img_h, img_w):
        """Create futuristic overlay on face"""
        overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        
        # Convert colors from normalized to 0-255
        cyan = (int(self.mesh_color[2]*255), int(self.mesh_color[1]*255), 
                int(self.mesh_color[0]*255), int(self.mesh_color[3]*255))
        magenta = (int(self.highlight_color[2]*255), int(self.highlight_color[1]*255),
                   int(self.highlight_color[0]*255), int(self.highlight_color[3]*255))
        green = (0, 255, 0, 150)
        
        landmarks = face_landmarks.landmark
        
        def get_point(idx):
            return (int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h))
        
        # Draw face mesh
        connections = self.mp_face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            pt1 = get_point(connection[0])
            pt2 = get_point(connection[1])
            cv2.line(overlay, pt1, pt2, cyan, 1, cv2.LINE_AA)
        
        # Eye contours
        left_eye = [33, 160, 158, 133, 153, 144, 163, 7]
        right_eye = [362, 385, 387, 263, 373, 380, 374, 249]
        
        for eye in [left_eye, right_eye]:
            pts = np.array([get_point(i) for i in eye])
            cv2.polylines(overlay, [pts], True, magenta, 2, cv2.LINE_AA)
        
        # Nose bridge
        nose_bridge = [6, 168, 197, 195, 5]
        pts = np.array([get_point(i) for i in nose_bridge])
        cv2.polylines(overlay, [pts], False, green, 2, cv2.LINE_AA)
        
        # Lips
        upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lower_lip = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        pts_upper = np.array([get_point(i) for i in upper_lip])
        pts_lower = np.array([get_point(i) for i in lower_lip])
        cv2.polylines(overlay, [pts_upper], False, cyan, 2, cv2.LINE_AA)
        cv2.polylines(overlay, [pts_lower], False, cyan, 2, cv2.LINE_AA)
        
        # Scan lines
        for y in range(0, img_h, 10):
            cv2.line(overlay, (0, y), (img_w, y), (100, 255, 255, 30), 1)
        
        # Key point markers
        key_points = [33, 263, 1, 61, 291, 199]
        for idx in key_points:
            pt = get_point(idx)
            cv2.circle(overlay, pt, 5, magenta, -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 8, cyan, 1, cv2.LINE_AA)
        
        return overlay
    
    # ==================== ImGui Interface ====================
        
    def draw_options(self):
        """Draw ImGui options panel"""
        imgui.text("Face Scanner Tool")
        imgui.separator()
        
        # Input Source Section
        imgui.text("Input Source:")
        
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.8, 0.8)
        if imgui.button("Browse Image", width=-1):
            self.show_file_browser = True
            self.refresh_file_list()
        imgui.pop_style_color()
        
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.4, 0.0)
        if imgui.button("WiFi Camera", width=-1):
            self.show_camera_dialog = True
        imgui.pop_style_color()
        
        # Camera status indicator
        if self.camera_running:
            imgui.text_colored("Camera: ", 0.5, 0.5, 0.5)
            imgui.same_line()
            imgui.text_colored("LIVE", 0.0, 1.0, 0.0)
            imgui.same_line()
            if imgui.small_button("Disconnect"):
                self.disconnect_camera()
        
        imgui.dummy(0, 5)
        imgui.separator()
        
        # Camera Live View
        if self.camera_running and self.camera_texture is not None:
            imgui.text("Live Camera Feed:")
            imgui.image(self.camera_texture, 250, 250)
            imgui.text_colored("Press SPACE to capture", 1.0, 1.0, 0.0)
            imgui.dummy(0, 5)
            imgui.separator()
        
        # Processing Section
        imgui.text("Processing:")
        
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.0, 0.8)
        disabled = self.processing or self.input_image is None
        if disabled:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
        
        if imgui.button("Process Face", width=-1) and not disabled:
            self.process_face()
        
        if disabled:
            imgui.pop_style_var()
        imgui.pop_style_color()
        
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.8, 0.0)
        if imgui.button("Save Result", width=-1):
            self.save_image()
        imgui.pop_style_color()
        
        imgui.dummy(0, 5)
        imgui.separator()
        
        # Status
        imgui.text("Status:")
        imgui.text_colored(self.status_message, 1.0, 1.0, 0.0)
        
        if self.processing:
            imgui.progress_bar(self.progress, (-1, 0))
        
        imgui.dummy(0, 5)
        imgui.separator()
        
        # Preview images
        if self.input_texture is not None:
            imgui.text("Input Preview:")
            imgui.image(self.input_texture, 200, 200)
        
        if self.output_texture is not None:
            imgui.text("Output Preview:")
            imgui.image(self.output_texture, 200, 200)
        
        imgui.dummy(0, 5)
        imgui.separator()
        
        # Settings
        imgui.text("Overlay Settings")
        imgui.push_item_width(150)
        
        _, self.overlay_opacity = imgui.slider_float(
            "Opacity", self.overlay_opacity, 0.0, 1.0
        )
        
        _, self.mesh_color = imgui.color_edit4(
            "Mesh Color", *self.mesh_color
        )
        
        _, self.highlight_color = imgui.color_edit4(
            "Highlight Color", *self.highlight_color
        )
        
        imgui.pop_item_width()
        
        imgui.dummy(0, 5)
        imgui.separator()
        
        # Info
        imgui.text("Image Info:")
        if self.input_image is not None:
            h, w = self.input_image.shape[:2]
            imgui.text_disabled(f"Size: {w}x{h}")
            imgui.text_disabled(f"Face: {'Detected' if self.face_detected else 'Not detected'}")
        else:
            imgui.text_disabled("No image loaded")
        
        imgui.dummy(0, 5)
        
        if imgui.button("Reset Settings", width=-1):
            self.overlay_opacity = 0.7
            self.mesh_color = [0.0, 1.0, 1.0, 0.8]
            self.highlight_color = [1.0, 0.0, 1.0, 0.8]
        
        imgui.dummy(0, 5)
        imgui.separator()
        imgui.text("Instructions:")
        imgui.text_disabled("Browse: Load from file")
        imgui.text_disabled("Camera: Connect WiFi cam")
        imgui.text_disabled("SPACE: Capture frame")
        imgui.text_disabled("C: Toggle camera")
        imgui.text_disabled("Process: Detect face")
        imgui.text_disabled("Save: Export PNG")
        
        # Draw dialogs
        self.draw_file_browser()
        self.draw_camera_dialog()
