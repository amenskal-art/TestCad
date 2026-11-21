# gizmo_scale.py
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

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
        """
        Applies scale in a LOCAL coordinate system defined by rotation matrix R.
        Math: P_world -> P_local -> Scale -> P_local_scaled -> P_world
        Pivot: The scale occurs relative to a local pivot (p_ox, p_oy, p_oz).
        """
        n = src_points.shape[0]
        for i in prange(n):
            # 1. Translate to Center
            px = src_points[i, 0] - cx
            py = src_points[i, 1] - cy
            pz = src_points[i, 2] - cz
            
            # 2. Rotate to Local Space (Dot product with rows of Transpose(R))
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

# -------------------------------------------

class ScaleGizmo:
    def __init__(self):
        self.active = False
        self.use_local = True # Default to Dental/Local mode
        
        # Bounding Box Data
        self.bounds_local_min = np.zeros(3) # In local space
        self.bounds_local_max = np.zeros(3)
        self.center = np.zeros(3)           # World space center
        self.rotation = np.eye(3)           # World orientation vectors (cols)
        
        # Buffers
        self.original_points_cache = None 
        self.indices_cache = None         
        self.temp_buffer = None
        
        # Interaction
        self.hovered_handle = -1
        self.active_handle = -1
        self.handle_positions = [] # World space positions
        self.drag_start_screen = None 
        self.drag_screen_axis = None  
        self.current_scale = np.ones(3)
        self.active_pivot_local = np.zeros(3) # Local space pivot

    def calculate_bounds(self, points, selection_mask=None):
        if selection_mask is not None and np.any(selection_mask):
            pts = points[selection_mask]
        else:
            pts = points
            
        if len(pts) == 0: self.active = False; return
        self.active = True

        # 1. Calculate Centroid
        self.center = np.mean(pts, axis=0)
        
        if self.use_local and len(pts) > 3:
            # 2. PCA for Auto-Alignment (Dental Mode)
            centered = pts - self.center
            step = max(1, len(pts) // 10000)
            subset = centered[::step]
            cov = np.cov(subset, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            
            x_axis = evecs[:, 2] # Largest eigenvalue
            y_axis = evecs[:, 1]
            z_axis = np.cross(x_axis, y_axis)
            
            # Rotation Matrix (Columns are axes)
            self.rotation = np.column_stack((x_axis, y_axis, z_axis))
            
            # Transform points to Local Space to find min/max
            local_pts = np.dot(centered, self.rotation)
            
        else:
            # Global Mode (World Axis)
            self.rotation = np.eye(3)
            local_pts = pts - self.center

        self.bounds_local_min = np.min(local_pts, axis=0)
        self.bounds_local_max = np.max(local_pts, axis=0)
        self._update_handles()

    def _update_handles(self):
        # We calculate handle positions in LOCAL space, then transform to WORLD
        l_min = self.bounds_local_min
        l_max = self.bounds_local_max
        lc = (l_min + l_max) * 0.5 

        # Define Handles in Local Space
        local_pos = [
            # Faces
            [l_max[0], lc[1], lc[2]],   # 0: +X
            [l_min[0], lc[1], lc[2]],   # 1: -X
            [lc[0], l_max[1], lc[2]],   # 2: +Y
            [lc[0], l_min[1], lc[2]],   # 3: -Y
            [lc[0], lc[1], l_max[2]],   # 4: +Z
            [lc[0], lc[1], l_min[2]],   # 5: -Z
            # Center
            [lc[0], lc[1], lc[2]],      # 6: Uniform
            # Corners
            [l_min[0], l_min[1], l_min[2]], # 7
            [l_max[0], l_min[1], l_min[2]], # 8
            [l_min[0], l_max[1], l_min[2]], # 9
            [l_max[0], l_max[1], l_min[2]], # 10
            [l_min[0], l_min[1], l_max[2]], # 11
            [l_max[0], l_min[1], l_max[2]], # 12
            [l_min[0], l_max[1], l_max[2]], # 13
            [l_max[0], l_max[1], l_max[2]], # 14
        ]
        
        # Transform to World: P_world = Center + Rotation @ P_local
        self.handle_positions = []
        for p in local_pos:
            p_vec = np.array(p)
            world_pos = self.center + np.dot(self.rotation, p_vec)
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

        # --- LOCAL PIVOT LOGIC ---
        if idx <= 5: 
            # Faces: Pivot is opposite face projection
            if idx == 0:   self.active_pivot_local = np.array([l_min[0], lc[1], lc[2]])
            elif idx == 1: self.active_pivot_local = np.array([l_max[0], lc[1], lc[2]])
            elif idx == 2: self.active_pivot_local = np.array([lc[0], l_min[1], lc[2]])
            elif idx == 3: self.active_pivot_local = np.array([lc[0], l_max[1], lc[2]])
            elif idx == 4: self.active_pivot_local = np.array([lc[0], lc[1], l_min[2]])
            elif idx == 5: self.active_pivot_local = np.array([lc[0], lc[1], l_max[2]])
        elif idx == 6:
            self.active_pivot_local = lc
        else:
            # Corners: Pivot is diagonally opposite corner
            i = idx - 7
            px = l_max[0] if not (i & 1) else l_min[0]
            py = l_max[1] if not (i & 2) else l_min[1]
            pz = l_max[2] if not (i & 4) else l_min[2]
            self.active_pivot_local = np.array([px, py, pz])

        # --- MOUSE AXIS ---
        screen_center = self._project_point(self.center)
        screen_handle = self._project_point(self.handle_positions[idx])
        axis_vec = screen_handle - screen_center
        norm = np.linalg.norm(axis_vec)
        self.drag_screen_axis = axis_vec / norm if norm > 0.1 else np.array([1.0, 1.0]) / 1.414

        # --- CACHE ---
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
        
        if idx <= 5: # Axis
            axis = idx // 2
            self.current_scale[axis] = scale_val
        else: # Uniform
            self.current_scale[:] = scale_val
            
        return self.current_scale

    def apply_to_points(self, points, points_colors=None):
        """Apply the current scale transformation to the cached selection."""
        if self.original_points_cache is None:
            return

        # Ensure the destination is writable
        if not points.flags.writeable:
            raise RuntimeError("ScaleGizmo.apply_to_points: points array is read-only")

        src = self.original_points_cache
        dst = self.temp_buffer
        
        sx, sy, sz = self.current_scale
        cx, cy, cz = self.center
        
        # Unpack rotation matrix for Numba
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
            # Numpy Fallback (Slower but works)
            centered = src - self.center
            local = np.dot(centered, self.rotation)
            local = (local - self.active_pivot_local) * self.current_scale + self.active_pivot_local
            world = np.dot(local, self.rotation.T) + self.center
            np.copyto(dst, world)
        
        points[self.indices_cache] = dst
        
        if points_colors is not None:
            points_colors[self.indices_cache, 0:3] = dst

    def end_drag(self):
        self.active_handle = -1
        self.original_points_cache = None
        self.indices_cache = None
        self.temp_buffer = None
        self.current_scale = np.ones(3)
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

        # Draw box by connecting handles (corners are indices 7-14)
        corners = self.handle_positions[7:15]
        
        glColor3f(1.0, 0.6, 0.0) # Orange Box
        glBegin(GL_LINES)
        
        indices = [
            (0,1), (1,3), (3,2), (2,0), # Bottom
            (4,5), (5,7), (7,6), (6,4), # Top
            (0,4), (1,5), (2,6), (3,7)  # Sides
        ]
        for a, b in indices:
            glVertex3f(*corners[a]); glVertex3f(*corners[b])
        glEnd()

        # Draw Local Axes at Center
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

        # Draw Handles
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
