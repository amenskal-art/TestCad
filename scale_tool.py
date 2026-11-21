# tools/scale_tool.py
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
import glfw
from tool_system import BaseTool

# Import measurement tools to transform their shapes after scaling
from tools.area_tool import AreaMeasureTool
from tools.volume_measure_tool import VolumeMeasureTool

# --- 1. PERFORMANCE KERNELS ---
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        return lambda func: func
    def prange(n): return range(n)

if HAS_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _scale_oriented_numba(src_points, out_points, sx, sy, sz, 
                              cx, cy, cz, 
                              r00, r01, r02, 
                              r10, r11, r12, 
                              r20, r21, r22,
                              p_ox, p_oy, p_oz):
        n = src_points.shape[0]
        for i in prange(n):
            # 1. Translate to Center
            px = src_points[i, 0] - cx
            py = src_points[i, 1] - cy
            pz = src_points[i, 2] - cz
            
            # 2. Rotate to Local Space
            lx = px * r00 + py * r10 + pz * r20
            ly = px * r01 + py * r11 + pz * r21
            lz = px * r02 + py * r12 + pz * r22
            
            # 3. Apply Scale relative to the local pivot
            lx = (lx - p_ox) * sx + p_ox
            ly = (ly - p_oy) * sy + p_oy
            lz = (lz - p_oz) * sz + p_oz
            
            # 4. Rotate back to World Space
            wx = lx * r00 + ly * r01 + lz * r02
            wy = lx * r10 + ly * r11 + lz * r12
            wz = lx * r20 + ly * r21 + lz * r22
            
            # 5. Translate back
            out_points[i, 0] = wx + cx
            out_points[i, 1] = wy + cy
            out_points[i, 2] = wz + cz

    # --- WARM UP THE NUMBA COMPILER ---
    _dummy_pts = np.zeros((1, 3), dtype=np.float32)
    _scale_oriented_numba(_dummy_pts, _dummy_pts.copy(), 
                          1.0, 1.0, 1.0, 
                          0.0, 0.0, 0.0, 
                          1.0, 0.0, 0.0, 
                          0.0, 1.0, 0.0, 
                          0.0, 0.0, 1.0,
                          0.0, 0.0, 0.0)

# -------------------------------------------

class ScaleGizmo:
    def __init__(self):
        self.active = False
        self.use_local = True 
        
        self.bounds_local_min = np.zeros(3) 
        self.bounds_local_max = np.zeros(3)
        self.center = np.zeros(3)           
        self.rotation = np.eye(3)           
        
        self.original_points_cache = None 
        self.indices_cache = None         
        self.temp_buffer = None
        
        self.hovered_handle = -1
        self.active_handle = -1
        self.handle_positions = [] 
        self.drag_start_screen = None 
        self.drag_screen_axis = None  
        self.current_scale = np.ones(3)
        self.active_pivot_local = np.zeros(3) 

    def calculate_bounds(self, points, selection_mask=None):
        if selection_mask is not None and np.any(selection_mask):
            pts = points[selection_mask]
        else:
            pts = points
            
        if len(pts) == 0: 
            self.active = False
            return
            
        self.active = True
        self.center = np.mean(pts, axis=0)
        
        if self.use_local and len(pts) > 3:
            centered = pts - self.center
            step = max(1, len(pts) // 10000)
            subset = centered[::step]
            cov = np.cov(subset, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            
            x_axis = evecs[:, 2] 
            y_axis = evecs[:, 1]
            z_axis = np.cross(x_axis, y_axis)
            
            self.rotation = np.column_stack((x_axis, y_axis, z_axis))
            local_pts = np.dot(centered, self.rotation)
        else:
            self.rotation = np.eye(3)
            local_pts = pts - self.center

        self.bounds_local_min = np.min(local_pts, axis=0)
        self.bounds_local_max = np.max(local_pts, axis=0)
        
        self._update_handles()

    def _update_handles(self, current_scale=None, active_pivot_local=None):
        l_min = self.bounds_local_min.copy()
        l_max = self.bounds_local_max.copy()
        
        if current_scale is not None and active_pivot_local is not None:
            l_min = (l_min - active_pivot_local) * current_scale + active_pivot_local
            l_max = (l_max - active_pivot_local) * current_scale + active_pivot_local

        lc = (l_min + l_max) * 0.5 

        local_pos = [
            [l_max[0], lc[1], lc[2]],   # 0: +X
            [l_min[0], lc[1], lc[2]],   # 1: -X
            [lc[0], l_max[1], lc[2]],   # 2: +Y
            [lc[0], l_min[1], lc[2]],   # 3: -Y
            [lc[0], lc[1], l_max[2]],   # 4: +Z
            [lc[0], lc[1], l_min[2]],   # 5: -Z
            [lc[0], lc[1], lc[2]],      # 6: Uniform
            [l_min[0], l_min[1], l_min[2]], # 7
            [l_max[0], l_min[1], l_min[2]], # 8
            [l_min[0], l_max[1], l_min[2]], # 9
            [l_max[0], l_max[1], l_min[2]], # 10
            [l_min[0], l_min[1], l_max[2]], # 11
            [l_max[0], l_min[1], l_max[2]], # 12
            [l_min[0], l_max[1], l_max[2]], # 13
            [l_max[0], l_max[1], l_max[2]], # 14
        ]
        
        self.handle_positions = []
        for p in local_pos:
            world_pos = self.center + np.dot(self.rotation, np.array(p))
            self.handle_positions.append(world_pos)

    def _project_point(self, pt):
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        try:
            x, y, z = gluProject(pt[0], pt[1], pt[2], modelview, projection, viewport)
            return np.array([x, y])
        except: return np.array([0, 0])

    def start_drag(self, points, selection_mask, mouse_x, mouse_y):
        if self.hovered_handle == -1: return False
        
        self.active_handle = self.hovered_handle
        self.drag_start_screen = np.array([mouse_x, mouse_y])
        self.current_scale = np.ones(3)
        
        idx = self.active_handle
        l_min = self.bounds_local_min
        l_max = self.bounds_local_max
        lc = (l_min + l_max) * 0.5 

        if idx <= 5: 
            if idx == 0:   self.active_pivot_local = np.array([l_min[0], lc[1], lc[2]])
            elif idx == 1: self.active_pivot_local = np.array([l_max[0], lc[1], lc[2]])
            elif idx == 2: self.active_pivot_local = np.array([lc[0], l_min[1], lc[2]])
            elif idx == 3: self.active_pivot_local = np.array([lc[0], l_max[1], lc[2]])
            elif idx == 4: self.active_pivot_local = np.array([lc[0], lc[1], l_min[2]])
            elif idx == 5: self.active_pivot_local = np.array([lc[0], lc[1], l_max[2]])
        elif idx == 6:
            self.active_pivot_local = lc
        else:
            i = idx - 7
            px = l_max[0] if not (i & 1) else l_min[0]
            py = l_max[1] if not (i & 2) else l_min[1]
            pz = l_max[2] if not (i & 4) else l_min[2]
            self.active_pivot_local = np.array([px, py, pz])

        screen_center = self._project_point(self.center)
        screen_handle = self._project_point(self.handle_positions[idx])
        axis_vec = screen_handle - screen_center
        norm = np.linalg.norm(axis_vec)
        self.drag_screen_axis = axis_vec / norm if norm > 0.1 else np.array([1.0, 1.0]) / 1.414

        if selection_mask is not None and np.any(selection_mask):
            self.indices_cache = np.nonzero(selection_mask)[0]
            self.original_points_cache = points[self.indices_cache].copy()
        else:
            self.indices_cache = slice(None)
            self.original_points_cache = points.copy()
            
        self.temp_buffer = np.empty_like(self.original_points_cache)
        return True

    def update_drag(self, mouse_x, mouse_y):
        if self.active_handle == -1: return None
        
        current_screen = np.array([mouse_x, mouse_y])
        drag_vec = current_screen - self.drag_start_screen
        projection = np.dot(drag_vec, self.drag_screen_axis)
        
        scale_val = 1.0 + (projection * 0.005)
        if scale_val < 0.01: scale_val = 0.01
        
        self.current_scale = np.ones(3, dtype=np.float32)
        idx = self.active_handle
        
        if idx <= 5: 
            axis = idx // 2
            self.current_scale[axis] = scale_val
        else: 
            self.current_scale[:] = scale_val
            
        self._update_handles(self.current_scale, self.active_pivot_local)
        return self.current_scale

    def apply_to_points(self, points, points_colors=None):
        if self.original_points_cache is None: return

        if not points.flags.writeable:
            raise RuntimeError("ScaleGizmo.apply_to_points: points array is read-only")

        src = self.original_points_cache
        dst = self.temp_buffer
        
        sx, sy, sz = self.current_scale
        cx, cy, cz = self.center
        
        r = self.rotation
        r00, r01, r02 = r[0,0], r[0,1], r[0,2]
        r10, r11, r12 = r[1,0], r[1,1], r[1,2]
        r20, r21, r22 = r[2,0], r[2,1], r[2,2]
        
        pox, poy, poz = self.active_pivot_local

        if HAS_NUMBA:
            _scale_oriented_numba(src, dst, 
                                  float(sx), float(sy), float(sz), 
                                  float(cx), float(cy), float(cz),
                                  r00, r01, r02, r10, r11, r12, r20, r21, r22,
                                  float(pox), float(poy), float(poz))
        else:
            centered = src - self.center
            local = np.dot(centered, self.rotation)
            local = (local - self.active_pivot_local) * self.current_scale + self.active_pivot_local
            world = np.dot(local, self.rotation.T) + self.center
            np.copyto(dst, world)
        
        points[self.indices_cache] = dst
        
        if points_colors is not None:
            points_colors[self.indices_cache, 0:3] = dst

    def end_drag(self, points, selection_mask):
        self.active_handle = -1
        self.original_points_cache = None
        self.indices_cache = None
        self.temp_buffer = None
        self.current_scale = np.ones(3)
        
        self.calculate_bounds(points, selection_mask)
        return True

    def check_hover(self, ray_origin, ray_dir, handle_radius_world):
        if not self.active: return
        min_dist = float('inf'); hovered = -1
        for i, pos in enumerate(self.handle_positions):
            ac = pos - ray_origin
            proj = np.dot(ac, ray_dir)
            if proj < 0: continue 
            closest = ray_origin + ray_dir * proj
            dist = np.linalg.norm(pos - closest)
            if dist < handle_radius_world:
                if dist < min_dist: min_dist = dist; hovered = i
        self.hovered_handle = hovered

    def render(self):
        if not self.active: return
        
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT | GL_POINT_BIT)
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glLineWidth(1.5)

        corners = self.handle_positions[7:15]
        
        glColor3f(1.0, 0.6, 0.0)
        glBegin(GL_LINES)
        
        indices = [
            (0,1), (1,3), (3,2), (2,0), # Bottom
            (4,5), (5,7), (7,6), (6,4), # Top
            (0,4), (1,5), (2,6), (3,7)  # Sides
        ]
        for a, b in indices:
            glVertex3f(*corners[a]); glVertex3f(*corners[b])
        glEnd()

        axis_len = np.linalg.norm(corners[1] - corners[0]) * 0.5
        cx, cy, cz = self.center
        
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(cx,cy,cz)
        glVertex3f(*(self.center + self.rotation[:,0] * axis_len))
        glColor3f(0,1,0); glVertex3f(cx,cy,cz)
        glVertex3f(*(self.center + self.rotation[:,1] * axis_len))
        glColor3f(0,0,1); glVertex3f(cx,cy,cz)
        glVertex3f(*(self.center + self.rotation[:,2] * axis_len))
        glEnd()

        glPointSize(10.0); glEnable(GL_POINT_SMOOTH)
        glBegin(GL_POINTS)
        for i, pos in enumerate(self.handle_positions):
            if i == self.hovered_handle or i == self.active_handle:
                glColor3f(1,1,1)
            elif i == 6: glColor3f(1,1,0.8) # Center
            elif i >= 7: glColor3f(1,0,1)   # Corners
            else: # Faces
                ax = i // 2
                if ax==0: glColor3f(1,0.2,0.2)
                elif ax==1: glColor3f(0.2,1,0.2)
                else: glColor3f(0.2,0.4,1)
            glVertex3f(*pos)
        glEnd()
        glPopAttrib()

# -----------------------------------------------------------------------------
# 3. TOOL WRAPPER
# -----------------------------------------------------------------------------
class ScaleTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "3D Scale"
        self.icon_key = 'scale'
        self.gizmo = ScaleGizmo()
        self.pending_scale = False 
        
        # New Target Scale Mode Variables
        self.mode = 0 # 0: Gizmo, 1: Target Distance Match
        self.picked_indices = []
        # Store string persistently in the class so typing actually works without resetting!
        self.target_distance_str = "10.0" 
        
    def on_activate(self):
        if hasattr(self.app, 'points') and self.app.points is not None:
            self.gizmo.calculate_bounds(self.app.points, self.app.selected)

    def _ensure_points_writable(self):
        """Make self.app.points writable; update points_colors and vbo_dirty if replaced."""
        if not self.app.points.flags.writeable:
            self.app.points = self.app.points.copy()
            if hasattr(self.app, 'points_colors') and self.app.points_colors is not None:
                self.app.points_colors[:, :3] = self.app.points
            self.app.vbo_dirty = True

    def _recompute_normals_if_needed(self):
        """Recompute vertex normals from faces after scaling."""
        if self.app.has_faces and self.app.faces is not None:
            try:
                from tools.engineering_tool import _recompute_normals
                self.app.normals = _recompute_normals(self.app.points, self.app.faces)
                self.app.has_normals = True
            except ImportError:
                pass

    def on_mouse_button(self, button, action, mods, x, y):
        y_gl = self.app.height - y 
        
        if self.mode == 1: # Target Distance Mode
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not imgui.get_io().want_capture_mouse:
                picked = self.app.pick_single_point(x, y_gl)
                if picked is not None:
                    if len(self.picked_indices) >= 2:
                        self.picked_indices = [picked] # Reset to start new line
                    else:
                        self.picked_indices.append(picked)
                return True
            return False

        # Gizmo Mode
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS and not imgui.get_io().want_capture_mouse:
                if self.gizmo.hovered_handle != -1:
                    self._ensure_points_writable()
                    self.app.save_state("3D Scale")
                    self.gizmo.start_drag(self.app.points, self.app.selected, x, y_gl) 
                    return True
            elif action == glfw.RELEASE:
                if self.gizmo.active_handle != -1:
                    if self.pending_scale:
                        self.gizmo.apply_to_points(self.app.points, getattr(self.app, 'points_colors', None))
                        self.pending_scale = False
                        
                    # Capture the final transform BEFORE end_drag resets current_scale
                    _final_center      = self.gizmo.center.copy()
                    _final_rotation    = self.gizmo.rotation.copy()
                    _final_scale       = self.gizmo.current_scale.copy()
                    _final_pivot_local = self.gizmo.active_pivot_local.copy()

                    self.gizmo.end_drag(self.app.points, self.app.selected)
                    self.app.vbo_dirty = True
                    self.app._rebuild_kdtree()          # rebuild app KD‑tree
                    self.app.sync_sculptor()             # rebuild sculptor
                    self._recompute_normals_if_needed()  # update normals

                    # --- Update area and volume shapes with the captured transform ---
                    if hasattr(self.app, 'tool_manager'):
                        for tool in self.app.tool_manager.tools.values():
                            if isinstance(tool, (AreaMeasureTool, VolumeMeasureTool)):
                                tool.transform_shapes(
                                    _final_center,
                                    _final_rotation,
                                    _final_scale,
                                    _final_pivot_local
                                )
                    # --------------------------------------------------------------

                    return True
        return False

    def on_mouse_move(self, x, y):
        if self.mode == 1: return False # Disable gizmo interactions in 2-point mode
        
        y_gl = self.app.height - y
        
        if self.gizmo.active_handle != -1:
            scale_val = self.gizmo.update_drag(x, y_gl)
            if scale_val is not None:
                self.pending_scale = True 
            return True
            
        elif self.gizmo.active:
            ray_o, ray_d = self._get_ray(x, y_gl)
            dist = getattr(self.app, 'distance', 10.0)
            threshold = max(dist * 0.02, 0.2) 
            self.gizmo.check_hover(ray_o, ray_d, threshold)
            return False
            
        return False

    def update(self):
        if self.mode == 1: return
        
        if self.pending_scale:
            self.gizmo.apply_to_points(self.app.points, getattr(self.app, 'points_colors', None))
            self.app.vbo_dirty = True
            self.pending_scale = False

    def _get_ray(self, x, y):
        try:
            mv = glGetDoublev(GL_MODELVIEW_MATRIX)
            pj = glGetDoublev(GL_PROJECTION_MATRIX)
            vp = glGetIntegerv(GL_VIEWPORT)
            near = gluUnProject(x, y, 0.0, mv, pj, vp)
            far = gluUnProject(x, y, 1.0, mv, pj, vp)
            d = np.array(far) - np.array(near)
            n = np.linalg.norm(d)
            return np.array(near), d / n if n > 0 else d
        except: return np.zeros(3), np.array([0,0,1])

    def render_3d(self):
        if self.mode == 0:
            self.gizmo.render()
        elif self.mode == 1:
            if not self.picked_indices: return
            
            glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_CURRENT_BIT)
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            
            glPointSize(12.0)
            glEnable(GL_POINT_SMOOTH)
            glBegin(GL_POINTS)
            glColor3f(0.2, 0.8, 1.0)
            for idx in self.picked_indices:
                glVertex3f(*self.app.points[idx])
            glEnd()
            
            if len(self.picked_indices) == 2:
                glLineWidth(3.0)
                glBegin(GL_LINES)
                glColor3f(1.0, 0.8, 0.2)
                glVertex3f(*self.app.points[self.picked_indices[0]])
                glVertex3f(*self.app.points[self.picked_indices[1]])
                glEnd()
                
            glPopAttrib()
        
    def draw_options(self):
        imgui.text("Scale Options")
        
        if imgui.radio_button("Gizmo (Drag)", self.mode == 0):
            self.mode = 0
        imgui.same_line()
        if imgui.radio_button("Target Distance", self.mode == 1):
            self.mode = 1
        
        imgui.separator()
        
        if self.mode == 0:
            imgui.text_disabled("Drag Handles to Scale")
            
            changed, self.gizmo.use_local = imgui.checkbox("Local Alignment (PCA)", self.gizmo.use_local)
            if changed:
                self.gizmo.calculate_bounds(self.app.points, self.app.selected)
                
            if imgui.button("Recalculate Bounds"):
                self.gizmo.calculate_bounds(self.app.points, self.app.selected)
        
        elif self.mode == 1:
            imgui.text_disabled("Click 2 points on the mesh.")
            imgui.text(f"Picked points: {len(self.picked_indices)}/2")
            
            if imgui.button("Clear Points"):
                self.picked_indices = []
                
            imgui.separator()
                
            if len(self.picked_indices) == 2:
                p1 = self.app.points[self.picked_indices[0]]
                p2 = self.app.points[self.picked_indices[1]]
                raw_dist = np.linalg.norm(p2 - p1)
                
                current_disp = self.app.format_distance(raw_dist)
                imgui.text(f"Current Distance: {current_disp}")
                
                imgui.text(f"Target Distance ({self.app.display_unit}):")
                
                changed, self.target_distance_str = imgui.input_text("##target_dist", self.target_distance_str, 256)
                
                imgui.spacing()
                if imgui.button("Apply Uniform Scale", width=-1):
                    try:
                        val = float(self.target_distance_str)
                        if val > 0 and raw_dist > 0:
                            unit_mm = getattr(self.app, '_UNIT_TO_MM', {"mm": 1.0})
                            disp_mm = unit_mm.get(getattr(self.app, 'display_unit', 'mm'), 1.0)
                            coord_mm = unit_mm.get(getattr(self.app, 'coord_unit', 'mm'), 1.0)
                            
                            raw_target = val * (disp_mm / coord_mm)
                            factor = raw_target / raw_dist
                            
                            self.app.save_state("Scale to Target Distance")
                            
                            mask = self.app.selected if np.any(self.app.selected) else slice(None)
                            center = np.mean(self.app.points[mask], axis=0)
                            
                            # Ensure points are writable before applying
                            self._ensure_points_writable()
                            
                            self.app.points[mask] = center + (self.app.points[mask] - center) * factor
                            if hasattr(self.app, 'points_colors'):
                                self.app.points_colors[mask, 0:3] = self.app.points[mask]
                                
                            self.app.vbo_dirty = True
                            if hasattr(self.app, 'sync_sculptor'):
                                self.app.sync_sculptor()
                            if hasattr(self.app, '_rebuild_kdtree'):
                                self.app._rebuild_kdtree()
                            self._recompute_normals_if_needed()
                                
                            self.gizmo.calculate_bounds(self.app.points, self.app.selected)
                            self.picked_indices = []
                            
                    except ValueError:
                        pass
