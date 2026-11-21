# core.py - UPDATED WITH AUTOMATIC UI SCALING AND CRISP TEXT
import sys
import numpy as np
import glfw
from OpenGL.GL import *
import imgui
from imgui.integrations.glfw import GlfwRenderer
import os
import json

from utils import enable_high_dpi, load_texture
from gui_layout import draw_gui
from gui_style import setup_style
from app_draw import AppDrawMixin
from app_input import AppInputMixin
from app_ops import AppOpsMixin, _make_default_rotation
from file_save import FileSaveMixin
from sculpt_core import SculptSession
from gizmo_nav import NavGizmo

# Import tool system
from tool_system import ToolManager
from tools.ai_scanner_tool import AIScannerTool
from tools.navigation_tool import NavigationTool
from tools.selection_tool import SmartSelectionTool
from tools.measure_tool import MeasureTool
from tools.scale_tool import ScaleTool, ScaleGizmo  
from tools.brush_tool import BrushTool
from tools.area_tool import AreaMeasureTool
from tools.angle_measure_tool import AngleMeasureTool
from tools.volume_measure_tool import VolumeMeasureTool
from tools.possion_logic_tool import PoissonReconTool
from tools.engineering_tool import SmoothTool
from tools.geo_map_tool import GeoMapTool
from tools.export_geo_tool import GeoExportTool
from tools.sculpt_tool import SculptTool
from tools.thickness_tool import ThicknessTool


def get_resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If running normally, use the standard project folder
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    return os.path.join(base_path, relative_path)


class DentalEditorApp(AppDrawMixin, AppInputMixin, AppOpsMixin, FileSaveMixin):
    def __init__(self, points, colors=None, normals=None, faces=None):
        enable_high_dpi() 
        
        if not glfw.init(): sys.exit(1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.SAMPLES, 8)

        # App identity
        self.app_version = "1.0.0"
        
        # --- UI SCALING & PERSISTENT MEMORY ---
        self.settings_file = "app_settings.json"
        saved_scale = self._load_ui_scale()

        if saved_scale is not None:
            self.ui_scale = saved_scale
            print(f"UI Scale loaded from settings: {self.ui_scale:.2f}x")
        else:
            # Get primary monitor's info for automatic UI scaling
            primary_monitor = glfw.get_primary_monitor()
            if primary_monitor:
                # Get the video mode (current resolution)
                vidmode = glfw.get_video_mode(primary_monitor)
                monitor_width = vidmode.size.width
                monitor_height = vidmode.size.height
                
                # Get the physical size in mm
                width_mm, height_mm = glfw.get_monitor_physical_size(primary_monitor)
                
                # Calculate DPI (dots per inch) - 25.4 mm per inch
                if width_mm > 0:
                    dpi = (monitor_width / width_mm) * 25.4
                    
                    # Base scale on DPI with 96 DPI as reference
                    dpi_scale = dpi / 96.0
                    
                    # Also consider resolution: larger screens need slightly larger UI
                    # Reference resolution: 1920x1080
                    resolution_scale = (monitor_width * monitor_height) / (1920 * 1080)
                    resolution_scale = max(0.8, min(1.5, resolution_scale ** 0.5))
                    
                    # Combine scales
                    self.ui_scale = dpi_scale * resolution_scale
                    
                    # Clamp to reasonable range
                    self.ui_scale = max(0.8, min(3.0, self.ui_scale))
                else:
                    self.ui_scale = 1.5  # Fallback
            else:
                self.ui_scale = 1.5  # Fallback
                
            print(f"UI Scale automatically set to: {self.ui_scale:.2f}x")
            self._save_ui_scale()  # Save the auto-detected scale for next time

        # Create window with appropriate initial size
        window_width = int(1280 * min(1.5, self.ui_scale))
        window_height = int(800 * min(1.5, self.ui_scale))
        
        self.window = glfw.create_window(window_width, window_height, 
                                        "GridCad", None, None)
        if not self.window: glfw.terminate(); sys.exit(1)
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(0) 
        
        # Load Icons using the safe PyInstaller resource path
        self.icons = {}
        
        self.icons['nav']       = load_texture(get_resource_path(os.path.join("icons", "navigation.png")))
        self.icons['scan']      = load_texture(get_resource_path(os.path.join("icons", "scan.png")))
        self.icons['selection'] = load_texture(get_resource_path(os.path.join("icons", "selection.png")))
        self.icons['measure']   = load_texture(get_resource_path(os.path.join("icons", "measure.png")))
        self.icons['scale']     = load_texture(get_resource_path(os.path.join("icons", "scale.png")))
        self.icons['brush']     = load_texture(get_resource_path(os.path.join("icons", "brush.png")))
        self.icons['camera']    = load_texture(get_resource_path(os.path.join("icons", "ar.png")))
        self.icons['area']      = load_texture(get_resource_path(os.path.join("icons", "area.png")))
        self.icons['angle']     = load_texture(get_resource_path(os.path.join("icons", "angle.png")))
        self.icons['volume']    = load_texture(get_resource_path(os.path.join("icons", "volume.png")))
        self.icons['mesh']      = load_texture(get_resource_path(os.path.join("icons", "pos.png")))
        self.icons['smooth']    = load_texture(get_resource_path(os.path.join("icons", "smooth.png")))
        self.icons['gps']       = load_texture(get_resource_path(os.path.join("icons", "gps.png")))
        self.icons['sculpt']    = load_texture(get_resource_path(os.path.join("icons", "sculpt.png")))
        self.icons['thickness']    = load_texture(get_resource_path(os.path.join("icons", "thickness.png")))

        # Setup Mesh Data
        self.points = points.astype(np.float32)
        n = len(self.points)
        self.default_mesh_color = [0.75, 0.75, 0.75]
        
        if colors is not None and len(colors) == n:
            self.base_colors = colors.astype(np.float32)
        else:
            self.base_colors = np.tile(self.default_mesh_color, (n, 1)).astype(np.float32)

        # Handle normals and faces
        self.faces = faces
        self.has_faces = (faces is not None and len(faces) > 0)
        
        if normals is not None and len(normals) == n:
            self.normals = normals.astype(np.float32)
            self.has_normals = True
            self.render_mode_mesh = True
        else:
            self.normals = np.zeros((n, 3), dtype=np.float32)
            self.has_normals = False
            self.render_mode_mesh = False

        self.points_colors = np.hstack([self.points, self.base_colors])
        self.points_ids = None 
        self.rebuild_ids(n)
        
        # State
        self.selected = np.zeros(n, dtype=bool)
        self.rotation_matrix = _make_default_rotation()
        self.distance = 2.5

        if n > 0:
            self.center = np.mean(self.points, axis=0)
        else:
            self.center = np.zeros(3, dtype=np.float32)
        
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 20
        
        # Append mode flag (True = load multiple objects, False = replace)
        self.append_mode = True

        # Store pending file and operation mode (replace/append)
        self.pending_file = None
        self.pending_file_mode = None

        # Background file loader state (set by AppOpsMixin)
        self._load_thread_running = False
        self._loaded_result       = None
        
        # Tool system
        self.tool_manager = ToolManager(self)
        self._register_tools()
        self.tool_manager.initialize_tools()
        
        # Keep original active_tool for compatibility
        self.active_tool = 0  
        
        # Keep scale_gizmo for backward compatibility
        self.scale_gizmo = ScaleGizmo()  
        
        # Rest of original initialization
        self.selection_submode = 0 
        self.lasso = []
        self.is_lassoing = False
        self.lasso_mode = 0 
        self._dragging_rotate = False 
        self._dragging_pan = False    
        self._last_dxdy = None
        self.key_constraints = {glfw.KEY_X: False, glfw.KEY_Y: False, glfw.KEY_Z: False}
        self.rot_speed = 0.5   
        self.pan_speed = 0.001   
        
        self.measurements = [] 
        self.measure_start_pt = None 
        self.measure_drag_pt = None
        self.is_measuring_drag = False
        
        self.sculptor = SculptSession(self.points, self.normals)
        self.sculpt_radius = 5.0 
        self.sculpt_strength = 0.3
        self.sculpt_mode = 'add' 
        self.is_sculpting = False
        self.cursor_world_pos = None 
        self.vbo_dirty = False 
        
        self.knife_poly = [] 
        self.is_poly_active = False
        
        self.nav_gizmo = NavGizmo()
        
        self.point_size = 3.0
        self.gap_fill_size = 6.0 
        self.show_grid = True 
        self.show_axes = True
        self.is_orthographic = True
        
        self.bg_color_top = [0.12, 0.12, 0.14] 
        self.bg_color_bot = [0.05, 0.05, 0.05]
        
        self.light_intensity = 0.8
        self.light_ambient = 0.4
        self.mat_specular = 0.5
        self.mat_shininess = 64.0
        
        self.status_text = "Ready."
        self.save_filename = "cleaned_cloud.ply"
        self.grid_unit_str = "1 mm"

        # Flag for New Scene dialog
        self.new_dialog_requested = False
        self.current_project_path = None

        # Unit System
        # coord_unit  : what 1 raw coordinate unit represents ("mm","cm","m","km")
        # display_unit: what unit to show in measurements and grid labels
        # Both default to mm; auto-detection below may override them.
        self.coord_unit   = "mm"
        self.display_unit = "mm"
        self._auto_detect_unit()   # sets coord_unit & display_unit from bbox

        # GLFW Callbacks (mouse/key/scroll — registered before imgui init is fine)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_framebuffer_size_callback(self.window, self.on_resize)
        self.width, self.height = glfw.get_framebuffer_size(self.window)

        # ImGui Init
        imgui.create_context()
        self.impl = GlfwRenderer(self.window, attach_callbacks=False)

        # FIX: register the char callback AFTER self.impl is created.
        # Without this, imgui never receives typed characters, so input_text
        # widgets show a blinking cursor but accept no keyboard input.
        glfw.set_char_callback(self.window, self.impl.char_callback)

        io = imgui.get_io()

        # Do NOT set font_global_scale; we'll generate fonts at the correct size
        io.font_global_scale = 1.0

        # Load fonts with the calculated UI scale (size in pixels)
        self.load_gui_font(size_pixels=16)  # base size 16, will be multiplied by ui_scale

        setup_style() 

        # OpenGL Buffers
        self.vbo = glGenBuffers(1)        
        self.vbo_id = glGenBuffers(1)     
        self.vbo_normals = glGenBuffers(1)
        
        # Element Buffer Object for indexed triangle rendering
        self.ebo = glGenBuffers(1) if self.has_faces else None

        # LOD cache — built lazily by AppDrawMixin._ensure_lod_ebo()
        self._lod_ebo      = None
        self._lod_count    = 0
        self._lod_n_points = -1   # -1 = "cache empty"

        # SVD throttle counter (reset in AppDrawMixin.setup_camera every 120 frames)
        self._svd_frame_ctr = 0
        
        self.update_vbo_full()

    # --- DYNAMIC UI SCALING METHODS ---
    def _load_ui_scale(self):
        """Loads the saved UI scale from disk, defaults to None if not found."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                    return float(data.get("ui_scale", 1.0))
            except Exception:
                pass
        return None

    def _save_ui_scale(self):
        """Saves the current UI scale to disk."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump({"ui_scale": float(self.ui_scale)}, f)
        except Exception as e:
            print(f"Failed to save UI settings: {e}")

    def apply_ui_scale(self, scale_value):
        """Applies the scale dynamically to fonts and styling, then saves."""
        self.ui_scale = scale_value
        try:
            # Scale all text dynamically
            imgui.get_io().font_global_scale = self.ui_scale
            # Reset base style, then multiply all padding/sizes by the scale
            setup_style() 
            imgui.get_style().scale_all_sizes(self.ui_scale)
        except Exception:
            pass # Suppress if imgui isn't fully initialized yet
        
        self._save_ui_scale()
    # ----------------------------------

    def _register_tools(self):
        """Register all tools with the tool manager."""
        self.tool_manager.register_tool(0, AIScannerTool)
        self.tool_manager.register_tool(1, NavigationTool)
        self.tool_manager.register_tool(2, SmartSelectionTool)
        self.tool_manager.register_tool(20, MeasureTool)
        self.tool_manager.register_tool(4, SculptTool)
        self.tool_manager.register_tool(5, ScaleTool)
        self.tool_manager.register_tool(6, BrushTool)
        self.tool_manager.register_tool(9, AreaMeasureTool)
        self.tool_manager.register_tool(10, AngleMeasureTool)
        self.tool_manager.register_tool(11, VolumeMeasureTool)
        self.tool_manager.register_tool(12, PoissonReconTool)
        self.tool_manager.register_tool(13, SmoothTool)
        self.tool_manager.register_tool(14, GeoMapTool)
        self._geo_exporter = GeoExportTool(self)
        self.tool_manager.register_tool(19, ThicknessTool)

    def sync_sculptor(self):
        self.sculptor = SculptSession(self.points, self.normals)
        
    def _orthonormalize_matrix(self):
        m = self.rotation_matrix
        x = m[:3, 0]; y = m[:3, 1]
        x = x / np.linalg.norm(x)
        y = y - np.dot(x, y) * x
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        m[:3, 0] = x; m[:3, 1] = y; m[:3, 2] = z
        self.rotation_matrix = m

    def update_vbo_full(self):
        """Update all VBOs + EBO; handles zero-sized data gracefully."""
        # Vertex position + colour
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if self.points_colors is not None and len(self.points_colors) > 0:
            # Upload new data (orphan first for better performance)
            glBufferData(GL_ARRAY_BUFFER, self.points_colors.nbytes, None, GL_DYNAMIC_DRAW)
            glBufferData(GL_ARRAY_BUFFER, self.points_colors.nbytes, self.points_colors, GL_DYNAMIC_DRAW)
        else:
            # No data: delete the old buffer by creating a zero-sized one
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # ID buffer
        if self.points_ids is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
            if len(self.points_ids) > 0:
                glBufferData(GL_ARRAY_BUFFER, self.points_ids.nbytes, self.points_ids, GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Normals
        if self.has_normals:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            if len(self.normals) > 0:
                glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_DYNAMIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Face index buffer (triangle mesh)
        if self.has_faces and self.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            if self.faces is not None and len(self.faces) > 0:
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.faces.nbytes, self.faces, GL_STATIC_DRAW)
            else:
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self.vbo_dirty = False

        # Invalidate LOD point-cloud index cache whenever geometry changes
        if hasattr(self, '_lod_n_points'):
            self._lod_n_points = -1  # force _ensure_lod_ebo() to rebuild

    # ═══════════════════════════════════════════════════════ Unit System ═════

    # Conversion table: 1 unit of each type == this many mm
    _UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1_000.0, "km": 1_000_000.0}
    _ALL_UNITS   = ["mm", "cm", "m", "km"]

    def _auto_detect_unit(self):
        """
        Estimate whether the loaded geometry is in mm, cm, m, or km by looking
        at the bounding-box diagonal, then set coord_unit and display_unit.
        The 3-D geometry is never modified.
        """
        if not hasattr(self, 'points') or len(self.points) == 0:
            return
        bbox_diag = float(np.linalg.norm(
            np.max(self.points, axis=0) - np.min(self.points, axis=0)
        ))
        # Heuristic thresholds (raw-unit diagonal value)
        if   bbox_diag < 0.5:      detected = "m"
        elif bbox_diag < 5.0:      detected = "cm"
        elif bbox_diag < 50_000.0: detected = "mm"
        else:                      detected = "km"

        self.coord_unit   = detected
        self.display_unit = detected
        self._refresh_grid_unit_str()
        print(f"[UnitSystem] bbox diagonal={bbox_diag:.5g} → auto-detected '{detected}'")

    def _refresh_grid_unit_str(self):
        """Rebuild the human-readable grid-cell label shown in the viewport."""
        # We'll compute the actual step label in draw_background_grid;
        # here we just store the current unit pair for the status bar.
        self.grid_unit_str = f"coords: {self.coord_unit}  |  display: {self.display_unit}"

    def format_distance(self, raw_dist: float) -> str:
        """
        Convert *raw_dist* (in coord_unit) to display_unit and return a
        nicely formatted string like '12.34 mm' or '0.0123 m'.
        """
        factor = (self._UNIT_TO_MM[self.coord_unit] /
                  self._UNIT_TO_MM[self.display_unit])
        val = raw_dist * factor
        # Adaptive precision
        if abs(val) >= 1_000:  prec = 0
        elif abs(val) >= 10:   prec = 2
        elif abs(val) >= 0.1:  prec = 3
        else:                  prec = 5
        return f"{val:.{prec}f} {self.display_unit}"

    def format_grid_step(self, raw_step: float) -> str:
        """Like format_distance but used for grid-cell labels."""
        return self.format_distance(raw_step)

    def set_coord_unit(self, unit: str):
        """Called from UI when the user changes the coordinate unit."""
        if unit in self._UNIT_TO_MM:
            self.coord_unit = unit
            self._refresh_grid_unit_str()

    def set_display_unit(self, unit: str):
        """Called from UI when the user changes the display unit."""
        if unit in self._UNIT_TO_MM:
            self.display_unit = unit
            self._refresh_grid_unit_str()

    # ═════════════════════════════════════════════════════════════════════════

    def run(self):
        while not glfw.window_should_close(self.window):
            self.impl.process_inputs()
            imgui.new_frame()
            
            self.check_file_loader() 
            
            # Update active tool for compatibility
            if hasattr(self, 'tool_manager'):
                active_tool = self.tool_manager.get_active_tool()
                if active_tool:
                    active_tool.update()
                    
                    # Sync active_tool for compatibility
                    self.active_tool = active_tool.tool_id
                    
                    # Update status text from tool (if not already set)
                    if active_tool.name != "Navigation" and not self.status_text.startswith(active_tool.name):
                        self.status_text = f"{active_tool.name} Mode"
            
            # Original measurement drag logic (for backward compatibility)
            if self.active_tool == 2 and self.is_measuring_drag:
                mx, my = glfw.get_cursor_pos(self.window); my = self.height - my
                pt = self.pick_single_point(mx, my)
                if pt is not None: self.measure_drag_pt = pt
                
            self.draw_scene()
            draw_gui(self) 
            
            # Render tool-specific 2D overlays
            if hasattr(self, 'tool_manager'):
                active_tool = self.tool_manager.get_active_tool()
                if active_tool:
                    active_tool.render_2d_overlay()
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
        self.impl.shutdown()
        glfw.terminate()
