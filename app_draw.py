# app_draw.py  — PERFORMANCE OPTIMISED
#
# Summary of changes vs original:
#   1. update_cursor_depth()   – only runs when sculpt tool (3) is active;
#                                glReadPixels was stalling the GPU every frame.
#   2. setup_camera() SVD      – orthonormalise at most once every 120 frames
#                                (not every frame).
#   3. draw_scene()            – removed the duplicate check_file_loader() call
#                                that was already being called in run().
#   4. render_mesh_vbo() LOD   – point-cloud mode skips vertices by a computed
#                                stride so 3 M points → ~300 K draw calls at
#                                normal zoom. A cached EBO (GL_POINTS) is rebuilt
#                                only when the point count changes.
#   5. render_mesh_vbo() EBO   – mesh (triangle) mode uses the pre-built EBO
#                                from core.py instead of raw glDrawArrays.
#   6. draw_brush_cursor()     – highlighted points uploaded to a small temp VBO
#                                once per frame instead of a Python glVertex loop.
#   7. _invalidate_lod_cache() – helper called from update_vbo_full() whenever
#                                the point array is replaced so stale LOD indices
#                                are discarded.

import numpy as np
import math
import ctypes
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from utils import normalize
import glfw

# --- Scipy for surface measurement -------------------------------------------
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: Scipy not found. Surface measurement will fall back to linear.")
# -----------------------------------------------------------------------------

# LOD strides: render every Nth point to cap draw calls at ~300 K
_LOD_TABLE = [
    (300_000,   1),
    (750_000,   3),
    (1_500_000, 6),
    (2_500_000, 10),
    (float('inf'), 16),
]

def _lod_stride(n):
    for threshold, stride in _LOD_TABLE:
        if n <= threshold:
            return stride
    return 16


class AppDrawMixin:

    # ------------------------------------------------------------------ fonts
    def load_gui_font(self, size_pixels=24):
        import os
        io = imgui.get_io()
        io.fonts.clear()

        font_paths = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
        loaded = False

        for fpath in font_paths:
            if os.path.exists(fpath):
                try:
                    io.fonts.add_font_from_file_ttf(fpath, size_pixels)
                    self.impl.refresh_font_texture()
                    loaded = True
                    break
                except:
                    continue

        if not loaded:
            io.fonts.add_font_default()
            self.impl.refresh_font_texture()

    # ------------------------------------------------------------------ camera
    def setup_camera(self):
        """
        Sets up the camera.  SVD orthonormalisation is throttled to once every
        120 frames (not every frame) to avoid numpy linalg overhead at 60+ FPS.
        """
        if hasattr(self, 'window') and self.window:
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            if fb_h == 0:
                fb_h = 1
            aspect = max(0.1, fb_w / float(fb_h))
            glViewport(0, 0, fb_w, fb_h)
        else:
            safe_height = getattr(self, 'height', 1) if getattr(self, 'height', 1) > 0 else 1
            aspect = max(0.1, getattr(self, 'width', 1) / float(safe_height))

        # --- SVD throttle: only orthonormalise every 120 frames --------------
        self._svd_frame_ctr = getattr(self, '_svd_frame_ctr', 0) + 1
        if self._svd_frame_ctr >= 120:
            self._svd_frame_ctr = 0
            if hasattr(self, 'rotation_matrix') and isinstance(self.rotation_matrix, np.ndarray):
                try:
                    if self.rotation_matrix.shape == (4, 4):
                        R3 = self.rotation_matrix[:3, :3]
                        U, S, Vt = np.linalg.svd(R3)
                        R_ortho = np.dot(U, Vt)
                        if np.linalg.det(R_ortho) < 0:
                            U[:, 2] *= -1
                            R_ortho = np.dot(U, Vt)
                        self.rotation_matrix[:3, :3] = R_ortho
                        self.rotation_matrix[3, :] = [0.0, 0.0, 0.0, 1.0]
                        self.rotation_matrix[:, 3] = [0.0, 0.0, 0.0, 1.0]
                except Exception:
                    pass
        # ---------------------------------------------------------------------

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if not hasattr(self, 'distance') or self.distance < 1e-4:
            self.distance = 1e-4
        elif self.distance > 1e7:
            self.distance = 1e7

        near_plane = max(0.001, self.distance * 0.005)
        far_plane  = max(100.0, self.distance * 100.0)

        if getattr(self, 'is_orthographic', False):
            fov_rad     = math.radians(60.0)
            half_height = self.distance * math.tan(fov_rad / 2.0)
            half_width  = half_height * aspect
            glOrtho(-half_width, half_width, -half_height, half_height, -far_plane, far_plane)
        else:
            gluPerspective(60.0, aspect, near_plane, far_plane)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        px = getattr(self, 'pan_x', 0.0)
        py = getattr(self, 'pan_y', 0.0)
        glTranslatef(px, py, -self.distance)
        glMultMatrixf(self.rotation_matrix.T)

        c = getattr(self, 'center', [0.0, 0.0, 0.0])
        glTranslatef(-c[0], -c[1], -c[2])

    # ---------------------------------------------------------------- lighting
    def setup_lighting_fixed(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        a = getattr(self, 'light_ambient', 0.2)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [a, a, a, 1.0])

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [100.0, 100.0, 100.0, 0.0])

        d = getattr(self, 'light_intensity', 0.8)
        s = getattr(self, 'mat_specular', 0.5)
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [d,       d,       d,       1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [s,       s,       s,       1.0])

        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-100.0, -50.0, -100.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [d*0.4,  d*0.4,  d*0.4,  1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0,    0.0,    0.0,    1.0])

        glPopMatrix()

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  [s, s, s, 1.0])
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, getattr(self, 'mat_shininess', 32.0))

    # -------------------------------------------------------------------- fog
    def setup_fog(self):
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        avg_bg = [(t + b) * 0.5 for t, b in zip(self.bg_color_top, self.bg_color_bot)] + [1.0]
        glFogfv(GL_FOG_COLOR, avg_bg)

        dist = max(0.001, getattr(self, 'distance', 10.0))
        glFogf(GL_FOG_START, dist * 1.5)
        glFogf(GL_FOG_END,   dist * 5.0)

    # --------------------------------------------------------- background
    def draw_gradient_background(self):
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)

        glBegin(GL_QUADS)
        glColor3f(*self.bg_color_top); glVertex2f(-1,  1); glVertex2f(1,  1)
        glColor3f(*self.bg_color_bot); glVertex2f( 1, -1); glVertex2f(-1, -1)
        glEnd()

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glPopAttrib()

    # -------------------------------------------------------------------- grid
    def draw_background_grid(self):
        if not getattr(self, 'show_grid', True):
            return

        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glEnable(GL_LINE_SMOOTH)

        dist = max(0.001, getattr(self, 'distance', 10.0))

        if hasattr(self, 'window') and self.window:
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            if fb_h == 0:
                fb_h = 1
            aspect = max(0.1, fb_w / float(fb_h))
        else:
            safe_height = getattr(self, 'height', 1) if getattr(self, 'height', 1) > 0 else 1
            aspect = max(0.1, getattr(self, 'width', 1) / float(safe_height))

        near_plane = max(0.001, dist * 0.005)
        far_plane  = max(100.0, dist * 100.0)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(60.0, aspect, near_plane, far_plane)
        glMatrixMode(GL_MODELVIEW)

        glPushMatrix()

        step = 10.0 ** math.floor(math.log10(dist / 2.0))
        if step <= 0:
            step = 0.001
        if (dist / step) < 3.0:
            step /= 2.0

        num_steps = 30
        size = step * num_steps

        c  = getattr(self, 'center', [0.0, 0.0, 0.0])
        cx = math.floor(c[0] / step) * step
        cz = math.floor(c[2] / step) * step

        glLineWidth(1.0)
        glBegin(GL_LINES)

        for i in range(-num_steps, num_steps + 1):
            offset = i * step
            x = cx + offset
            z = cz + offset

            fade  = max(0.0, 1.0 - (abs(i) / float(num_steps)))
            alpha = fade * 0.35

            is_z_axis = abs(x) < (step * 0.02)
            is_x_axis = abs(z) < (step * 0.02)

            if is_z_axis:
                glColor4f(0.3, 0.4, 0.9, fade * 0.8)
            elif i % 10 == 0:
                glColor4f(0.6, 0.6, 0.6, alpha * 1.5)
            else:
                glColor4f(0.4, 0.4, 0.4, alpha)
            glVertex3f(x, 0.0, cz - size)
            glVertex3f(x, 0.0, cz + size)

            if is_x_axis:
                glColor4f(0.9, 0.3, 0.3, fade * 0.8)
            elif i % 10 == 0:
                glColor4f(0.6, 0.6, 0.6, alpha * 1.5)
            else:
                glColor4f(0.4, 0.4, 0.4, alpha)
            glVertex3f(cx - size, 0.0, z)
            glVertex3f(cx + size, 0.0, z)

        glEnd()
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glPopAttrib()

        # ── Grid step label (bottom-left corner) ─────────────────────────────
        # Show what one grid cell represents in the current display unit.
        step_label = self.format_grid_step(step)
        self.grid_unit_str = f"Grid: {step_label} / cell"

        draw_list = imgui.get_background_draw_list()
        if hasattr(self, 'window') and self.window:
            win_w, win_h = glfw.get_window_size(self.window)
        else:
            win_w, win_h = getattr(self, 'width', 800), getattr(self, 'height', 600)

        label_text = f"Grid cell = {step_label}"
        tw = imgui.calc_text_size(label_text).x
        margin = 12
        left_panel_w = int(72 * getattr(self, 'ui_scale', 1.0))
        status_bar_h = int(32 * getattr(self, 'ui_scale', 1.0))
        lx = left_panel_w + margin          # clear the left icon panel
        ly = win_h - status_bar_h - 36      # clear the status bar
        draw_list.add_rect_filled(lx - 4, ly - 2, lx + tw + 8, ly + 18,
                                  imgui.get_color_u32_rgba(0, 0, 0, 0.55))
        draw_list.add_text(lx, ly, imgui.get_color_u32_rgba(0.75, 0.85, 1.0, 0.9), label_text)

    # ------------------------------------------------------- VBO management
    def _invalidate_lod_cache(self):
        """Discard cached LOD EBO so it is rebuilt on the next draw call."""
        if hasattr(self, '_lod_ebo') and self._lod_ebo is not None:
            try:
                glDeleteBuffers(1, [self._lod_ebo])
            except Exception:
                pass
        self._lod_ebo       = None
        self._lod_count     = 0
        self._lod_n_points  = -1   # sentinel: "no cache"

    def update_vbo_full(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # Buffer orphaning: hint to driver it can discard the old store
        glBufferData(GL_ARRAY_BUFFER, self.points_colors.nbytes, None, GL_DYNAMIC_DRAW)
        glBufferData(GL_ARRAY_BUFFER, self.points_colors.nbytes, self.points_colors, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, self.points_ids.nbytes, self.points_ids, GL_STATIC_DRAW)

        if getattr(self, 'has_normals', False):
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        if hasattr(self, 'surface_graph'):
            self.surface_graph = None

        # Invalidate LOD cache whenever the point array changes size
        self._invalidate_lod_cache()

    def update_colors(self):
        mask = self.selected[:, None]
        self.points_colors[:, 3:6] = np.where(mask, [1.0, 0.2, 0.2], self.base_colors)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points_colors.nbytes, self.points_colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # --------------------------------------------------------------- draw_scene
    def draw_scene(self):
        if not hasattr(self, 'points') or self.points is None:
            return

        # 1. Clear & Background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_gradient_background()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # NOTE: check_file_loader() is intentionally NOT called here.
        #       It is already called once per frame in run().  Calling it
        #       a second time inside draw_scene() doubles the overhead.

        # 2. Camera
        self.setup_camera()

        # 3. Grid
        self.draw_background_grid()

        # 4. Lighting
        if getattr(self, 'render_mode_mesh', False) and getattr(self, 'has_normals', False):
            self.setup_lighting_fixed()
        else:
            glDisable(GL_LIGHTING)

        self.setup_fog()

        # 5. Axes gizmo
        if getattr(self, 'show_axes', False):
            glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT)
            glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            s = max(0.01, getattr(self, 'distance', 10.0) * 0.1)
            c = getattr(self, 'center', [0, 0, 0])
            glColor3f(0.8, 0.2, 0.2); glVertex3f(c[0], c[1], c[2]); glVertex3f(c[0]+s, c[1],   c[2]  )
            glColor3f(0.2, 0.8, 0.2); glVertex3f(c[0], c[1], c[2]); glVertex3f(c[0],   c[1]+s, c[2]  )
            glColor3f(0.2, 0.2, 0.8); glVertex3f(c[0], c[1], c[2]); glVertex3f(c[0],   c[1],   c[2]+s)
            glEnd()
            glPopAttrib()

        # 6. Main mesh / point cloud
        self.render_mesh_vbo()

        # 7. Tool 3-D overlays
        if hasattr(self, 'tool_manager'):
            active_tool = self.tool_manager.get_active_tool()
            if active_tool and hasattr(active_tool, 'render_3d'):
                glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT |
                             GL_CURRENT_BIT | GL_POLYGON_BIT | GL_DEPTH_BUFFER_BIT)
                active_tool.render_3d()
                glPopAttrib()

        # 7b. GPS plane — always rendered regardless of active tool or render mode.
        #     In mesh/EBO mode the GPS plane points are in the VBO but never indexed
        #     by the face EBO, so they would be invisible without this dedicated pass.
        self._render_gps_plane()

        # 8. Post-processing
        glDisable(GL_LIGHTING); glDisable(GL_FOG)
        glDisable(GL_POINT_SMOOTH)

        # update_cursor_depth performs glReadPixels which stalls the GPU.
        # Only pay that cost when the sculpt tool actually needs cursor position.
        active_tool_idx = getattr(self, 'active_tool', -1)
        if active_tool_idx in (3, 5):
            self.update_cursor_depth()

        if getattr(self, 'is_sculpting', False) and getattr(self, 'cursor_world_pos', None) is not None:
            self.process_sculpting_frame()

        if getattr(self, 'vbo_dirty', False):
            if hasattr(self, 'update_vbo_subset'):
                self.update_vbo_subset()
            self.vbo_dirty = False

        self.draw_measurements()

        if active_tool_idx == 3:
            self.draw_brush_cursor()

        

        # 9. 2-D screen overlays
        if hasattr(self, 'window') and self.window:
            win_w, win_h = glfw.get_window_size(self.window)
        else:
            win_w, win_h = getattr(self, 'width', 800), getattr(self, 'height', 600)

        if getattr(self, 'is_poly_active', False) and len(getattr(self, 'knife_poly', [])) > 0:
            self._draw_overlay_poly_open(self.knife_poly, (1.0, 0.8, 0.0), win_w, win_h)
            mx, my = imgui.get_mouse_pos()
            self._draw_overlay_line(self.knife_poly[-1], (mx, win_h - my), (1.0, 1.0, 1.0), win_w, win_h)

        if getattr(self, 'is_lassoing', False) and len(getattr(self, 'lasso', [])) > 1:
            col = (0.2, 1.0, 0.2) if getattr(self, 'lasso_mode', 0) == 1 else (1.0, 0.2, 0.2)
            self._draw_overlay_poly_open(self.lasso, col, win_w, win_h)

        draw_list = imgui.get_background_draw_list()
        # grid label is drawn inside draw_background_grid() with correct positioning

    # ------------------------------------------------------ GPS plane overlay
    def _render_gps_plane(self):
        """Dedicated render pass for the GPS reference plane.

        Called every frame unconditionally so the plane is always visible
        regardless of which tool is active or whether the scene is in
        mesh-triangle (EBO) or point-cloud mode.
        """
        if not getattr(self, 'gps_plane_active', False):
            return
        if getattr(self, 'gps_plane_hidden', False):
            return
        gps_id = getattr(self, 'gps_plane_object_id', None)
        if gps_id is None or not hasattr(self, 'point_object_ids'):
            return

        mask = self.point_object_ids == gps_id
        if not np.any(mask):
            return

        pts  = self.points[mask]
        cols = self.base_colors[mask]
        if len(pts) == 0:
            return

        # Pack [x,y,z, r,g,b] interleaved — same layout as the main VBO
        data = np.hstack([pts, cols]).astype(np.float32)

        tmp_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, tmp_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)

        glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT | GL_CURRENT_BIT |
                     GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
        glColorPointer (3, GL_FLOAT, 24, ctypes.c_void_p(12))

        # Slightly larger points, drawn on top of mesh (no depth test)
        glPointSize(max(5.0, getattr(self, 'point_size', 3.0) + 1.0))
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)     # always on top — never occluded by mesh
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

        glDrawArrays(GL_POINTS, 0, len(pts))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(1, [tmp_vbo])
        glPopAttrib()

    # --------------------------------------------------------- render_mesh_vbo
    def _ensure_lod_ebo(self):
        """
        Build (or reuse) a GPU index buffer for LOD rendering of point clouds.

        The LOD stride is determined by the total point count so that the GPU
        never has to process more than ~300 K vertices per frame regardless of
        how large the loaded cloud is.  The EBO is rebuilt only when the number
        of points changes (e.g. after a load or delete operation).
        """
        n = len(self.points_colors)
        if not hasattr(self, '_lod_n_points'):
            self._lod_n_points = -1
            self._lod_ebo      = None
            self._lod_count    = 0

        if self._lod_n_points == n and self._lod_ebo is not None:
            return  # cache still valid

        # Discard old buffer
        if self._lod_ebo is not None:
            try:
                glDeleteBuffers(1, [self._lod_ebo])
            except Exception:
                pass

        stride = _lod_stride(n)
        indices = np.arange(0, n, stride, dtype=np.uint32)

        self._lod_ebo     = glGenBuffers(1)
        self._lod_count   = len(indices)
        self._lod_n_points = n

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._lod_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        if stride > 1:
            print(f"[LOD] {n:,} points → stride {stride} → {self._lod_count:,} rendered vertices")

    def render_mesh_vbo(self):
        point_size = getattr(self, 'point_size', 1.0)
        glPointSize(max(1.0, point_size))

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, getattr(self, 'vbo', 0))
        glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
        glColorPointer (3, GL_FLOAT, 24, ctypes.c_void_p(12))

        if getattr(self, 'render_mode_mesh', False) and getattr(self, 'has_normals', False):
            # ---- Mesh / triangle mode ----------------------------------------
            glEnableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, getattr(self, 'vbo_normals', 0))
            glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))

            if getattr(self, 'has_faces', False) and getattr(self, 'ebo', None) is not None:
                # Use indexed rendering — only unique vertices are processed
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                n_indices = len(self.faces) * 3 if self.faces is not None else 0
                glDrawElements(GL_TRIANGLES, n_indices, GL_UNSIGNED_INT, ctypes.c_void_p(0))
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            else:
                glDrawArrays(GL_TRIANGLES, 0, len(self.points_colors))

            glDisableClientState(GL_NORMAL_ARRAY)
        else:
            # ---- Point cloud mode with LOD ------------------------------------
            self._ensure_lod_ebo()

            if self._lod_count > 0 and self._lod_ebo is not None:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._lod_ebo)
                glDrawElements(GL_POINTS, self._lod_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            else:
                # Fallback: no LOD buffer yet — raw draw
                glDrawArrays(GL_POINTS, 0, len(self.points_colors))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    # --------------------------------------------------------- sculpt helpers
    def process_sculpting_frame(self):
        mode = getattr(self, 'sculpt_mode', 'add')
        wnd  = self.window
        if wnd and glfw.get_key(wnd, glfw.KEY_LEFT_SHIFT)   == glfw.PRESS: mode = 'smooth'
        elif wnd and glfw.get_key(wnd, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            if   mode == 'add':    mode = 'remove'
            elif mode == 'remove': mode = 'add'

        if hasattr(self, 'sculptor') and hasattr(self, 'cursor_world_pos'):
            self.sculptor.sculpt(
                self.cursor_world_pos,
                getattr(self, 'sculpt_radius',   1.0),
                getattr(self, 'sculpt_strength',  1.0) * 0.5,
                mode
            )
            self.vbo_dirty = True

    def draw_brush_cursor(self):
        """
        Highlight points inside the sculpt radius.

        Instead of a Python-level glVertex loop (one CPython call per point),
        the subset is uploaded to a transient VBO so the GPU receives the data
        in a single DMA transfer.
        """
        if not hasattr(self, 'sculpt_radius') or getattr(self, 'cursor_world_pos', None) is None:
            return

        center = self.cursor_world_pos
        radius = self.sculpt_radius

        if hasattr(self, 'sculptor'):
            idxs = self.sculptor.query_radius(center, radius)

            if len(idxs) > 0:
                glPushAttrib(GL_ENABLE_BIT)
                glDisable(GL_LIGHTING)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_POINT_SMOOTH)
                glPointSize(getattr(self, 'point_size', 1.0) + 2.0)
                glColor3f(1.0, 0.5, 0.0)

                pts_subset = self.points[idxs].astype(np.float32)

                # --- transient VBO so we avoid a Python loop -----------------
                tmp_vbo = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, tmp_vbo)
                glBufferData(GL_ARRAY_BUFFER, pts_subset.nbytes, pts_subset, GL_STREAM_DRAW)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
                glDrawArrays(GL_POINTS, 0, len(pts_subset))
                glDisableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDeleteBuffers(1, [tmp_vbo])
                # -------------------------------------------------------------

                glDisable(GL_POINT_SMOOTH)
                glPopAttrib()

            normal = self.sculptor.get_nearest_normal(center)
        else:
            normal = None

        if normal is None:
            normal = np.array([0, 1, 0], dtype=np.float32)
        n = normalize(normal)

        if abs(n[0]) < 0.9: temp = np.array([1, 0, 0], dtype=np.float32)
        else:               temp = np.array([0, 1, 0], dtype=np.float32)

        b = normalize(np.cross(n, temp))
        t = normalize(np.cross(b, n))

        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        glColor4f(1.0, 0.7, 0.2, 0.9)
        segments  = 36
        offset_n  = n * (radius * 0.05)
        for i in range(segments):
            theta = 2.0 * math.pi * float(i) / float(segments)
            p = center + (t * math.cos(theta) + b * math.sin(theta)) * radius + offset_n
            glVertex3f(p[0], p[1], p[2])
        glEnd()
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glVertex3f(*(center + offset_n))
        glEnd()
        glPopAttrib()

    # ---------------------------------------------------------------- helpers
    def get_camera_pos(self):
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        try:
            inv = np.linalg.inv(mv.T)
            return inv[:3, 3]
        except:
            return np.array([0, 0, 10])

    def update_cursor_depth(self):
        """
        Reads ONE depth sample under the mouse cursor and unprojects it to
        world space.  This issues a glReadPixels call which STALLS the GPU
        pipeline — only call when the sculpt tool is active (see draw_scene).
        """
        if not hasattr(self, 'window') or not self.window:
            self.cursor_world_pos = None
            return

        win_w, win_h = glfw.get_window_size(self.window)
        fb_w,  fb_h  = glfw.get_framebuffer_size(self.window)
        px_ratio     = fb_w / win_w if win_w > 0 else 1.0
        mx, my       = imgui.get_mouse_pos()
        my_logical   = win_h - my

        if mx < 0 or mx > win_w or my_logical < 0 or my_logical > win_h:
            self.cursor_world_pos = None
            return

        x = int(mx * px_ratio)
        y = int(my_logical * px_ratio)
        w, h = 10, 10

        x = max(0, min(x - 5, fb_w - w))
        y = max(0, min(y - 5, fb_h - h))

        try:
            pixels       = glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, GL_FLOAT)
            depths        = np.frombuffer(pixels, dtype=np.float32)
            valid_depths  = depths[depths < 1.0]
            if len(valid_depths) > 0:
                min_z       = np.min(valid_depths)
                modelview   = glGetDoublev(GL_MODELVIEW_MATRIX)
                projection  = glGetDoublev(GL_PROJECTION_MATRIX)
                viewport    = glGetIntegerv(GL_VIEWPORT)
                center_x_fb = mx * px_ratio
                center_y_fb = my_logical * px_ratio
                wx, wy, wz  = gluUnProject(center_x_fb, center_y_fb, min_z,
                                           modelview, projection, viewport)
                self.cursor_world_pos = np.array([wx, wy, wz], dtype=np.float32)
            else:
                self.cursor_world_pos = None
        except Exception:
            self.cursor_world_pos = None

    # --------------------------------------------------- 2-D overlay helpers
    def _draw_overlay_line(self, p1, p2, col, w_log, h_log):
        glPushAttrib(GL_ENABLE_BIT)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, w_log, 0, h_log, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(*col); glVertex2f(*p1); glVertex2f(*p2)
        glEnd()
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glPopAttrib()

    def _draw_overlay_poly_open(self, points, col, w_log, h_log):
        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, w_log, 0, h_log, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_LINE_SMOOTH)
        glColor3f(*col)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for x, y in points: glVertex2f(x, y)
        glEnd()
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for x, y in points: glVertex2f(x, y)
        glEnd()
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glPopAttrib()

    # pick_single_point lives in AppInputMixin / AppOpsMixin (fast KD-tree version).
    # Do NOT add a slow offscreen-render version here.

    def project_point_to_screen(self, pt):
        modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport   = glGetIntegerv(GL_VIEWPORT)
        x, y, z    = gluProject(pt[0], pt[1], pt[2], modelview, projection, viewport)
        if hasattr(self, 'window') and self.window:
            win_w, win_h = glfw.get_window_size(self.window)
            px_ratio = glfw.get_framebuffer_size(self.window)[0] / win_w if win_w > 0 else 1.0
        else:
            win_h, px_ratio = getattr(self, 'height', 600), 1.0
        return x / px_ratio, (win_h - y / px_ratio), z

    # -------------------------------------------------- surface graph / paths
    def build_surface_graph(self):
        if not SCIPY_AVAILABLE:
            return None
        if getattr(self, 'surface_graph', None) is not None:
            return self.surface_graph
        if not hasattr(self, 'faces') or self.faces is None or len(self.faces) == 0:
            return None

        edges   = np.vstack([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        p1      = self.points[edges[:, 0]]
        p2      = self.points[edges[:, 1]]
        weights = np.linalg.norm(p1 - p2, axis=1)
        ij      = np.vstack([edges, edges[:, [1, 0]]])
        w       = np.hstack([weights, weights])
        n       = len(self.points)
        self.surface_graph = csr_matrix((w, (ij[:, 0], ij[:, 1])), shape=(n, n))
        return self.surface_graph

    def get_surface_path(self, idx_start, idx_end):
        graph = self.build_surface_graph()
        if graph is None:
            return [idx_start, idx_end]

        dist_matrix, predecessors = shortest_path(
            csgraph=graph, method='D', directed=False,
            indices=idx_start, return_predecessors=True
        )

        if np.isinf(dist_matrix[idx_end]):
            return [idx_start, idx_end]

        path, curr = [], idx_end
        while curr != -9999:
            path.append(curr)
            if curr == idx_start:
                break
            curr = predecessors[curr]
        return path[::-1]

    # ---------------------------------------------------------- measurements
    def draw_measurements(self):
        measurements  = getattr(self, 'measurements',     [])
        is_meas_drag  = getattr(self, 'is_measuring_drag', False)
        if not measurements and not is_meas_drag:
            return

        if not hasattr(self, 'measure_on_surface'):
            self.measure_on_surface = True

        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)

        def get_coord(pt):
            return self.points[pt] if isinstance(pt, (int, np.integer)) else pt

        def draw_path(p1_raw, p2_raw, is_active=False):
            is_idx = isinstance(p1_raw, (int, np.integer)) and isinstance(p2_raw, (int, np.integer))
            if getattr(self, 'measure_on_surface', False) and SCIPY_AVAILABLE and is_idx:
                path_idxs = self.get_surface_path(p1_raw, p2_raw)
                pts       = self.points[path_idxs]
                glColor3f(1.0, 0.0, 0.0) if is_active else glColor3f(1.0, 0.8, 0.0)
                glBegin(GL_LINE_STRIP)
                for p in pts: glVertex3f(*p)
                glEnd()
                dist = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)) if len(pts) > 1 else 0.0
                return dist, pts
            else:
                c1, c2 = get_coord(p1_raw), get_coord(p2_raw)
                glColor3f(1.0, 0.0, 0.0) if is_active else glColor3f(1.0, 0.8, 0.0)
                glBegin(GL_LINES)
                glVertex3f(*c1); glVertex3f(*c2)
                glEnd()
                return np.linalg.norm(c2 - c1), [c1, c2]

        draw_list = imgui.get_background_draw_list()
        self.setup_camera()

        for p1, p2 in measurements:
            dist, pts = draw_path(p1, p2, is_active=False)
            mid_pt    = pts[len(pts) // 2] if len(pts) > 0 else get_coord(p1)
            sx, sy, sz = self.project_point_to_screen(mid_pt)
            if 0.0 <= sz <= 1.0:
                text = self.format_distance(dist)
                tw   = imgui.calc_text_size(text).x
                draw_list.add_rect_filled(sx, sy, sx+tw+8, sy+20, imgui.get_color_u32_rgba(0, 0, 0, 0.7))
                draw_list.add_text(sx+4, sy+2, imgui.get_color_u32_rgba(1, 1, 1, 1), text)

        if (is_meas_drag
                and getattr(self, 'measure_start_pt', None) is not None
                and getattr(self, 'measure_drag_pt',  None) is not None):
            dist, pts  = draw_path(self.measure_start_pt, self.measure_drag_pt, is_active=True)
            mid_pt     = pts[len(pts) // 2]
            sx, sy, sz = self.project_point_to_screen(mid_pt)
            if 0.0 <= sz <= 1.0:
                text = self.format_distance(dist)
                tw   = imgui.calc_text_size(text).x
                draw_list.add_rect_filled(sx, sy, sx+tw+8, sy+20, imgui.get_color_u32_rgba(0, 0, 0, 0.7))
                draw_list.add_text(sx+4, sy+2, imgui.get_color_u32_rgba(1, 1, 1, 1), text)

        glPopAttrib()

