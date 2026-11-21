from tool_system import BaseTool
import imgui
import glfw
import numpy as np
from scipy.spatial import KDTree
from OpenGL.GL import *
import math
import time





class MarginLineTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Margin Line Pro"
        self.icon_key = 'margin'
        self.hotkey = 'M'
        
     
        

        # --- State Management ---
        self.current_step_key = "MARGIN" # Start at the base level
        self.current_subtool = None      # The active subtool object

        # --- Core Margin Data ---
        self.points = []
        self.active_curve_indices = []
        self.active_curve_segments = [] 
        self.curve_closed = False
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Interaction State
        self.dragging_point = None
        self.drag_point_idx = None
        self.drag_offset = np.array([0.0, 0.0, 0.0])
        
        # Point insertion
        self.hover_segment = None
        self.hover_segment_t = 0.0
        
        # Preview Data
        self.preview_segment = []
        self.last_preview_update = 0
        self.hover_start_point = False
        
        # Settings
        self.snap_radius = 0.2
        self.curve_width = 4.0
        self.resolution = 0.15        
        self.smoothness = 10          
        self.lift_strength = 0.8      
        self.normal_offset = 0.05
        self.point_size = 12.0        
        
        # --- Colors ---
        self.point_color = [1.0, 0.2, 0.2, 1.0]
        self.active_point_color = [1.0, 0.6, 0.0, 1.0]
        self.active_curve_color = [0.0, 0.6, 1.0, 1.0]
        self.closed_curve_color = [0.2, 1.0, 0.4, 1.0] 
        self.preview_color = [1.0, 1.0, 0.0, 0.7]
        self.close_loop_color = [0.0, 1.0, 0.0, 1.0]
        self.insert_point_color = [1.0, 0.0, 1.0, 1.0]
        self.occluded_color = [1.0, 0.0, 0.0, 0.6]
        
        self.is_drawing = False
        self.mesh_vertices = None
        self.mesh_normals = None  
        self.kd_tree = None
        self.status_text = "Margin Line: Click to place points"

        # --- Matrix Cache ---
        self.cached_view_matrix = np.eye(4, dtype=np.float32)
        self.cached_proj_matrix = np.eye(4, dtype=np.float32)
        self.matrices_initialized = False
        
        # --- SOLUTION 1: Visual Indicator States ---
        self.show_curve_quality = True
        self.curve_quality = 0.0  # 0-100%
        self.quality_color = [0.2, 0.8, 0.2, 1.0]  # Green by default
        
        # --- SOLUTION 2: Action Indicator ---
        self.show_action_guide = True
        self.current_action = "Click to place points"
        self.action_progress = 0.0  # For progress bars
        self.last_action_time = 0.0
        self.action_timeout = 3.0  # Show action for 3 seconds
        
        # --- SOLUTION 3: Numerical Feedback ---
        self.show_numerical_feedback = True
        self.num_points = 0
        self.curve_length = 0.0
        self.avg_point_spacing = 0.0
        self.curve_continuity = "Open"

    def on_activate(self):
        # Always reset to Margin step when tool is activated
        self.switch_step("MARGIN")
        return self._initialize_from_point_cloud()
        
    def _initialize_from_point_cloud(self):
        print("\n" + "="*60)
        print("Initializing Professional Margin Line Tool")
        
        if hasattr(self.app, 'sculptor') and self.app.sculptor:
             if hasattr(self.app.sculptor, 'points'):
                 self.mesh_vertices = np.array(self.app.sculptor.points, dtype=np.float32)
             if hasattr(self.app.sculptor, 'normals'):
                 self.mesh_normals = np.array(self.app.sculptor.normals, dtype=np.float32)
        elif hasattr(self.app, 'points_colors'):
             try:
                self.mesh_vertices = self.app.points_colors[:, :3].copy()
             except: pass
        
        if self.mesh_vertices is None or len(self.mesh_vertices) == 0:
            print("✗ No point cloud data found")
            return False

        if self.mesh_normals is None or len(self.mesh_normals) != len(self.mesh_vertices):
             self.mesh_normals = np.zeros_like(self.mesh_vertices)
             self.mesh_normals[:, 1] = 1.0 
        
        print(f"✓ Loaded {len(self.mesh_vertices):,} points")
        self.kd_tree = KDTree(self.mesh_vertices)
        print("="*60)
        return True

    # ==========================================
    # SOLUTION 1: Visual Indicator Functions
    # ==========================================
    def _calculate_curve_quality(self):
        """Calculate the quality of the current curve"""
        if len(self.active_curve_indices) < 3:
            return 0.0
        
        total_quality = 0.0
        factors = 0
        
        # Factor 1: Point spacing consistency
        if len(self.active_curve_segments) > 0:
            lengths = []
            for segment in self.active_curve_segments:
                if segment is not None and len(segment) >= 2:
                    segment_length = 0.0
                    for i in range(len(segment) - 1):
                        segment_length += np.linalg.norm(segment[i+1] - segment[i])
                    lengths.append(segment_length)
            
            if lengths:
                avg_length = np.mean(lengths)
                if avg_length > 0:
                    cv = np.std(lengths) / avg_length  # Coefficient of variation
                    spacing_quality = max(0, 1.0 - cv * 2.0)  # Less variation = higher quality
                    total_quality += spacing_quality * 0.4
                    factors += 0.4
        
        # Factor 2: Curve smoothness (angle consistency)
        if len(self.active_curve_indices) >= 3:
            angles = []
            for i in range(1, len(self.active_curve_indices) - 1):
                prev_idx = self.active_curve_indices[i-1]
                curr_idx = self.active_curve_indices[i]
                next_idx = self.active_curve_indices[i+1]
                
                v1 = self.points[curr_idx]['pos'] - self.points[prev_idx]['pos']
                v2 = self.points[next_idx]['pos'] - self.points[curr_idx]['pos']
                
                if np.linalg.norm(v1) > 0.001 and np.linalg.norm(v2) > 0.001:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
                    angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                if avg_angle > 0:
                    angle_variation = np.std(angles) / avg_angle
                    smoothness_quality = max(0, 1.0 - angle_variation * 0.5)
                    total_quality += smoothness_quality * 0.4
                    factors += 0.4
        
        # Factor 3: Completeness (closed curve bonus)
        if self.curve_closed:
            total_quality += 0.2
            factors += 0.2
        
        if factors > 0:
            quality = (total_quality / factors) * 100.0
        else:
            quality = 0.0
        
        # Update quality color
        if quality > 80:
            self.quality_color = [0.2, 0.8, 0.2, 1.0]  # Green
        elif quality > 60:
            self.quality_color = [0.8, 0.8, 0.2, 1.0]  # Yellow
        elif quality > 40:
            self.quality_color = [0.8, 0.5, 0.2, 1.0]  # Orange
        else:
            self.quality_color = [0.8, 0.2, 0.2, 1.0]  # Red
        
        return min(100.0, quality)
    
    def _update_visual_indicators(self):
        """Update all visual indicators based on current state"""
        # Update curve quality
        self.curve_quality = self._calculate_curve_quality()
        
        # Update numerical feedback
        self.num_points = len(self.active_curve_indices)
        
        # Calculate curve length
        self.curve_length = 0.0
        for segment in self.active_curve_segments:
            if segment is not None and len(segment) >= 2:
                for i in range(len(segment) - 1):
                    self.curve_length += np.linalg.norm(segment[i+1] - segment[i])
        
        # Calculate average point spacing
        if self.num_points > 1 and self.curve_length > 0:
            self.avg_point_spacing = self.curve_length / self.num_points
        else:
            self.avg_point_spacing = 0.0
        
        # Update curve continuity
        self.curve_continuity = "Closed" if self.curve_closed else "Open"

    # ==========================================
    # SOLUTION 2: Action Guide Functions
    # ==========================================
    def _update_action_guide(self, action_text, show_timeout=True):
        """Update the current action being performed"""
        self.current_action = action_text
        if show_timeout:
            self.last_action_time = time.time()
    
    def _get_suggested_action(self):
        """Get suggested action based on current state"""
        if self.current_subtool:
            return self.current_subtool.name
        
        if self.curve_closed:
            return "Ready for next step"
        elif len(self.active_curve_indices) == 0:
            return "Click to place first point"
        elif len(self.active_curve_indices) < 3:
            return "Add more points to define curve"
        elif self.hover_start_point:
            return "Click to close curve"
        elif self.hover_segment:
            return "SHIFT+Click to insert point"
        elif self.dragging_point is not None:
            return "Drag point to new position"
        else:
            return "Click to add point or close loop"
    
    def _check_action_timeout(self):
        """Check if action guide should be hidden due to timeout"""
        if time.time() - self.last_action_time > self.action_timeout:
            self.current_action = self._get_suggested_action()

    # ==========================================
    # WORKFLOW SWITCHING (GENERIC)
    # ==========================================
    def switch_step(self, step_key):
        """Generic function to move between tools"""
        # Update action guide
        self._update_action_guide(f"Switching to {step_key}")
        
        # 1. Handle reset to base
        if step_key == "MARGIN":
            self.current_step_key = "MARGIN"
            self.current_subtool = None
            return

        if step_key not in self.workflow_registry:
            print(f"Error: Step {step_key} not found")
            return

        if not self.curve_closed:
            print("Cannot proceed: Curve not closed")
            self._update_action_guide("Close curve to proceed", show_timeout=True)
            return

        # 2. Prepare Data Packet (Context) to send to the sub-tool
        raw_margin_points = [p['pos'] for p in self.points if isinstance(p, dict)]
        if not raw_margin_points and self.points:
             raw_margin_points = self.points

        context_data = {
            "margin_points": raw_margin_points,
            "margin_indices": self.active_curve_indices,
            "mesh_vertices": self.mesh_vertices,
            "mesh_normals": self.mesh_normals,
            "previous_tool_data": self.current_subtool
        }

        # 3. Activate New Tool
        self.current_step_key = step_key
        self.current_subtool = self.workflow_registry[step_key]
        
        # 4. Inject Data
        if hasattr(self.current_subtool, 'on_enter'):
            self.current_subtool.on_enter(context_data)
        
        # 5. Update visual indicators
        self._update_visual_indicators()

    # ==========================================
    # UNDO/REDO SYSTEM
    # ==========================================
    def _save_state(self):
        state = {
            'points': [p.copy() for p in self.points],
            'active_curve_indices': self.active_curve_indices.copy(),
            'active_curve_segments': [s.copy() for s in self.active_curve_segments],
            'curve_closed': self.curve_closed
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()
        
        # Update indicators
        self._update_visual_indicators()
        
    def _undo(self):
        if not self.undo_stack: return
        self._update_action_guide("Undo last action")
        
        current_state = {
            'points': [p.copy() for p in self.points],
            'active_curve_indices': self.active_curve_indices.copy(),
            'active_curve_segments': [s.copy() for s in self.active_curve_segments],
            'curve_closed': self.curve_closed
        }
        self.redo_stack.append(current_state)
        state = self.undo_stack.pop()
        self.points = [p.copy() for p in state['points']]
        self.active_curve_indices = state['active_curve_indices'].copy()
        self.active_curve_segments = [s.copy() for s in state['active_curve_segments']]
        self.curve_closed = state['curve_closed']
        
        # Update indicators
        self._update_visual_indicators()
        
    def _redo(self):
        if not self.redo_stack: return
        self._update_action_guide("Redo last action")
        
        current_state = {
            'points': [p.copy() for p in self.points],
            'active_curve_indices': self.active_curve_indices.copy(),
            'active_curve_segments': [s.copy() for s in self.active_curve_segments],
            'curve_closed': self.curve_closed
        }
        self.undo_stack.append(current_state)
        state = self.redo_stack.pop()
        self.points = [p.copy() for p in state['points']]
        self.active_curve_indices = state['active_curve_indices'].copy()
        self.active_curve_segments = [s.copy() for s in state['active_curve_segments']]
        self.curve_closed = state['curve_closed']
        
        # Update indicators
        self._update_visual_indicators()

    # ==========================================
    # GPU DEPTH PICKING
    # ==========================================
    def get_world_pos_from_mouse(self, mouse_x, mouse_y):
        if not self.matrices_initialized: return None

        gl_y = self.app.height - mouse_y
        try:
            z_val_buffer = glReadPixels(mouse_x, gl_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        except: return None
        
        if isinstance(z_val_buffer, np.ndarray): z_val = z_val_buffer[0][0]
        else: z_val = z_val_buffer

        if z_val == 1.0: return None 

        x_ndc = (2.0 * mouse_x) / self.app.width - 1.0
        y_ndc = (2.0 * gl_y) / self.app.height - 1.0
        z_ndc = 2.0 * z_val - 1.0
        
        clip_coords = np.array([x_ndc, y_ndc, z_ndc, 1.0])
        
        view_mx = self.cached_view_matrix.T
        proj_mx = self.cached_proj_matrix.T
        view_proj = np.dot(proj_mx, view_mx)
        
        try:
            inv_view_proj = np.linalg.inv(view_proj)
        except np.linalg.LinAlgError:
            return None
        
        world_coords = np.dot(inv_view_proj, clip_coords)
        if world_coords[3] != 0:
            world_coords /= world_coords[3]
            
        return world_coords[:3]

    def _find_surface_point(self, cursor_pos):
        if self.kd_tree is None or cursor_pos is None: return None
        try:
            distance, vertex_idx = self.kd_tree.query(cursor_pos)
            if vertex_idx < len(self.mesh_vertices):
                return {'pos': self.mesh_vertices[vertex_idx], 'idx': vertex_idx}
        except: pass
        return None

    # ==========================================
    # POINT INTERACTION
    # ==========================================
    def _find_nearest_point_screen(self, mouse_x, mouse_y, max_pixels=30):
        if not self.points or not self.matrices_initialized:
            return None
        
        view_mx = self.cached_view_matrix.T
        proj_mx = self.cached_proj_matrix.T
        view_proj = np.dot(proj_mx, view_mx)
        
        gl_y = self.app.height - mouse_y
        
        min_dist = float('inf')
        nearest_idx = None
        
        for i, p in enumerate(self.points):
            if i not in self.active_curve_indices and len(self.active_curve_indices) > 0:
                 pass 

            pos_4d = np.array([p['pos'][0], p['pos'][1], p['pos'][2], 1.0])
            clip_coords = np.dot(view_proj, pos_4d)
            
            if clip_coords[3] == 0: continue
            
            ndc = clip_coords[:3] / clip_coords[3]
            screen_x = (ndc[0] + 1.0) * 0.5 * self.app.width
            screen_y = (ndc[1] + 1.0) * 0.5 * self.app.height
            
            dx = screen_x - mouse_x
            dy = screen_y - gl_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist and dist < max_pixels:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _find_nearest_segment(self, world_pos, max_distance=0.2):
        if not self.active_curve_segments or world_pos is None:
            return None
        
        min_dist = float('inf')
        best_segment = None
        best_t = 0.0
        
        for seg_idx, segment in enumerate(self.active_curve_segments):
            if segment is None or len(segment) < 2: continue
            
            for i in range(len(segment) - 1):
                p1 = segment[i]; p2 = segment[i + 1]
                v = p2 - p1; u = world_pos - p1
                length_sq = np.dot(v, v)
                
                if length_sq < 1e-6: continue
                
                t = np.clip(np.dot(u, v) / length_sq, 0.0, 1.0)
                closest = p1 + t * v
                dist = np.linalg.norm(world_pos - closest)
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_segment = (seg_idx, i)
                    best_t = t
        
        return (best_segment, best_t) if best_segment is not None else None

    # ==========================================
    # CURVE MATH
    # ==========================================
    def _generate_curved_path(self, start_pos, end_pos):
        if self.kd_tree is None: return [start_pos, end_pos]
        
        dist = np.linalg.norm(end_pos - start_pos)
        if dist < 0.001: return [start_pos, end_pos]
        
        samples = max(20, int(dist / self.resolution))
        t = np.linspace(0, 1, samples)
        linear_path = np.outer(1 - t, start_pos) + np.outer(t, end_pos)
        
        dists, indices = self.kd_tree.query(linear_path)
        snapped_path = self.mesh_vertices[indices]
        
        path = self._resample_path(snapped_path, self.resolution)
        smoothed_path = self._smooth_path(path, iterations=self.smoothness, weight=0.5)
        
        dists, indices = self.kd_tree.query(smoothed_path)
        surface_points = self.mesh_vertices[indices]
        surface_normals = self.mesh_normals[indices]
        
        raw_lift = surface_points - smoothed_path
        clean_lift = self._smooth_vectors(raw_lift, iterations=15)
        
        final_path = smoothed_path + (clean_lift * self.lift_strength) + (surface_normals * self.normal_offset)
        final_path = self._smooth_path(final_path, iterations=2, weight=0.2)
        
        return final_path

    def _resample_path(self, points, resolution=0.1):
        if len(points) < 2: return points
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_length = cum_dist[-1]
        if total_length < 1e-6: return points
        num_points = max(10, int(total_length / resolution))
        new_dists = np.linspace(0, total_length, num_points)
        new_points = np.zeros((num_points, 3))
        for i in range(3):
            new_points[:, i] = np.interp(new_dists, cum_dist, points[:, i])
        return new_points

    def _smooth_path(self, points, iterations=10, weight=0.5):
        if len(points) < 3: return points
        path = points.copy()
        start_pt = path[0].copy(); end_pt = path[-1].copy()
        for _ in range(iterations):
            prev_pts = path[:-2]; curr_pts = path[1:-1]; next_pts = path[2:]
            neighbor_avg = (prev_pts + next_pts) * 0.5
            path[1:-1] = curr_pts * (1.0 - weight) + neighbor_avg * weight
        path[0] = start_pt; path[-1] = end_pt
        return path

    def _smooth_vectors(self, vectors, iterations=10):
        if len(vectors) < 3: return vectors
        v = vectors.copy()
        for _ in range(iterations):
            v[1:-1] = (v[:-2] + v[1:-1] + v[2:]) / 3.0
        return v
    
    def _generate_surface_curve(self, start_point, end_point):
        pt_a = start_point['pos'] if isinstance(start_point, dict) else start_point
        pt_b = end_point['pos'] if isinstance(end_point, dict) else end_point
        return self._generate_curved_path(pt_a, pt_b)

    # ==========================================
    # MISSING METHOD: UPDATE CONNECTED SEGMENTS
    # ==========================================
    def _update_connected_segments(self, point_idx):
        """Update the curve segments connected to a moved point"""
        if point_idx not in self.active_curve_indices: 
            return
        
        idx_pos = self.active_curve_indices.index(point_idx)
        total_points = len(self.active_curve_indices)
        
        def update_seg(seg_idx, p_start_idx, p_end_idx):
            if 0 <= seg_idx < len(self.active_curve_segments):
                pt_a = self.points[p_start_idx]
                pt_b = self.points[p_end_idx]
                self.active_curve_segments[seg_idx] = self._generate_surface_curve(pt_a, pt_b)

        # Update segment to the right (next point)
        if idx_pos < total_points - 1:
            next_idx = self.active_curve_indices[idx_pos + 1]
            update_seg(idx_pos, point_idx, next_idx)
        elif self.curve_closed:
            # If it's the last point and curve is closed, update closing segment
            first_idx = self.active_curve_indices[0]
            update_seg(idx_pos, point_idx, first_idx)

        # Update segment to the left (previous point)
        if idx_pos > 0:
            prev_idx = self.active_curve_indices[idx_pos - 1]
            update_seg(idx_pos - 1, prev_idx, point_idx)
        elif self.curve_closed:
            # If it's the first point and curve is closed, update last segment
            last_idx = self.active_curve_indices[-1]
            last_seg_idx = len(self.active_curve_segments) - 1
            if last_seg_idx >= 0:
                update_seg(last_seg_idx, last_idx, point_idx)

    # ==========================================
    # POINT INTERACTION (with action feedback)
    # ==========================================
    def _insert_point_at_segment(self):
        if not self.hover_segment: return
        
        self._update_action_guide("Inserting point")
        self._save_state()
        
        (seg_idx, line_idx), t = self.hover_segment
        # Make sure seg_idx is within bounds
        if seg_idx >= len(self.active_curve_segments):
            return
            
        segment = self.active_curve_segments[seg_idx]
        if segment is None or len(segment) <= line_idx + 1:
            return
        
        p1 = segment[line_idx]
        p2 = segment[line_idx + 1]
        insert_pos = p1 + t * (p2 - p1)
        
        surface_point = self._find_surface_point(insert_pos)
        if not surface_point: return
        
        self.points.append(surface_point)
        new_point_idx = len(self.points) - 1
        
        insert_at_index = seg_idx + 1
        self.active_curve_indices.insert(insert_at_index, new_point_idx)
        
        is_closing_segment = (self.curve_closed and seg_idx == len(self.active_curve_segments) - 1)
        
        if is_closing_segment:
             prev_pt_idx = self.active_curve_indices[-2]
             next_pt_idx = self.active_curve_indices[0]
        else:
             prev_pt_idx = self.active_curve_indices[insert_at_index - 1]
             next_pt_idx = self.active_curve_indices[insert_at_index + 1]
             
        seg_a = self._generate_surface_curve(self.points[prev_pt_idx], surface_point)
        seg_b = self._generate_surface_curve(surface_point, self.points[next_pt_idx])
        
        self.active_curve_segments[seg_idx] = seg_a
        self.active_curve_segments.insert(seg_idx + 1, seg_b)
        
        self.hover_segment = None
        
        # Update indicators
        self._update_visual_indicators()

    def _undo_last_point(self):
        if self.active_curve_indices and not self.curve_closed:
            self._update_action_guide("Undo last point")
            self._save_state()
            self.active_curve_indices.pop()
            if self.active_curve_segments: 
                self.active_curve_segments.pop()
            
            # Update indicators
            self._update_visual_indicators()

    def _add_margin_point(self):
        cursor_pos = self.app.cursor_world_pos
        surface_point = self._find_surface_point(cursor_pos)
        if not surface_point: return False
        
        if self.points:
            pos_array = np.array([p['pos'] for p in self.points])
            dists = np.linalg.norm(pos_array - surface_point['pos'], axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < self.snap_radius:
                if len(self.active_curve_indices) > 0 and min_idx != self.active_curve_indices[-1]:
                    if min_idx == self.active_curve_indices[0] and len(self.active_curve_indices) > 2:
                        self._extend_active_curve(min_idx)
                        self._complete_curve(close_loop=True)
                        return True
        
        self.points.append(surface_point)
        new_idx = len(self.points) - 1
        self._extend_active_curve(new_idx)
        
        # Update action guide
        point_num = len(self.active_curve_indices)
        self._update_action_guide(f"Added point {point_num}")
        
        # Update indicators
        self._update_visual_indicators()
        
        return True

    def _extend_active_curve(self, new_idx):
        if len(self.active_curve_indices) > 0:
            last_idx = self.active_curve_indices[-1]
            pt_a = self.points[last_idx]
            pt_b = self.points[new_idx]
            segment = self._generate_surface_curve(pt_a, pt_b)
            self.active_curve_segments.append(segment)
        self.active_curve_indices.append(new_idx)

    def _complete_curve(self, close_loop=False):
        if len(self.active_curve_indices) < 3: return
        
        if close_loop:
            self._update_action_guide("Closing curve loop")
        else:
            self._update_action_guide("Completing open curve")
            
        self._save_state()
        
        if close_loop:
            last_point = self.points[self.active_curve_indices[-1]]
            first_point = self.points[self.active_curve_indices[0]]
            closing_segment = self._generate_surface_curve(last_point, first_point)
            
            self.active_curve_segments.append(closing_segment)
            self.curve_closed = True
            
            print(f"✓ Curve Loop Closed: {len(self.active_curve_indices)} points")
        
        self.is_drawing = False
        self.preview_segment = []
        self.hover_start_point = False
        
        # Update indicators
        self._update_visual_indicators()

    # ==========================================
    # INPUT HANDLERS
    # ==========================================
    def on_mouse_move(self, x, y):
        # Update action timeout check
        self._check_action_timeout()
        
        # 1. Block Main Logic if Sub-Tool is active
        if self.current_subtool:
            return

        # --- Standard Margin Logic ---
        gpu_pos = self.get_world_pos_from_mouse(x, y)
        
        if gpu_pos is None:
            self.hover_start_point = False
            self.hover_segment = None
            return

        # --- DRAGGING LOGIC ---
        if self.dragging_point is not None:
            surface_point = self._find_surface_point(gpu_pos)
            if surface_point:
                self.points[self.drag_point_idx]['pos'] = surface_point['pos'].copy()
                self.points[self.drag_point_idx]['idx'] = surface_point['idx']
                self._update_connected_segments(self.drag_point_idx)
                
                # Update indicators while dragging
                self._update_visual_indicators()
            return

        # Normal hovering logic
        has_active_curve = len(self.active_curve_indices) > 0
        
        if self.is_drawing and not self.curve_closed and len(self.active_curve_indices) > 2:
             start_pos = self.points[self.active_curve_indices[0]]['pos']
             dist = np.linalg.norm(gpu_pos - start_pos)
             self.hover_start_point = (dist < self.snap_radius * 1.5)
        else:
             self.hover_start_point = False

        if has_active_curve and not self.hover_start_point:
            # Check bounds before accessing hover segment
            if self.hover_segment:
                (seg_idx, line_idx), t = self.hover_segment
                if seg_idx < len(self.active_curve_segments):
                    self.hover_segment = self._find_nearest_segment(gpu_pos)
                else:
                    self.hover_segment = None
            else:
                self.hover_segment = self._find_nearest_segment(gpu_pos)
        else:
            self.hover_segment = None

        if self.is_drawing and has_active_curve and not self.curve_closed:
            if time.time() - self.last_preview_update < 0.016: return
            
            surface_point = self._find_surface_point(gpu_pos)
            if surface_point:
                last_point = self.points[self.active_curve_indices[-1]]
                self.preview_segment = self._generate_surface_curve(last_point, surface_point)
                self.last_preview_update = time.time()
        else:
            self.preview_segment = []

    def on_mouse_button(self, button, action, mods, x, y):
        # Update action guide based on interaction
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if mods & glfw.MOD_ALT:
                self._update_action_guide("Ready to drag point")
            elif mods & glfw.MOD_SHIFT:
                self._update_action_guide("Ready to insert point")
            elif mods & glfw.MOD_CONTROL:
                self._update_action_guide("Completing open curve")
        
        # 1. Block Main Logic if Sub-Tool is active
        if self.current_subtool:
            return False

        if button == glfw.MOUSE_BUTTON_LEFT and not imgui.get_io().want_capture_mouse:
            if action == glfw.PRESS:
                gpu_pos = self.get_world_pos_from_mouse(x, y)
                if gpu_pos is None: return False
                
                self.app.cursor_world_pos = gpu_pos
                
                # 1. Point Dragging (Alt + Click)
                if mods & glfw.MOD_ALT:
                    point_idx = self._find_nearest_point_screen(x, y, max_pixels=30)
                    if point_idx is not None:
                        self._update_action_guide("Dragging point")
                        self._save_state()
                        self.dragging_point = self.points[point_idx]
                        self.drag_point_idx = point_idx
                        return True
                
                # 2. Point Insertion (Shift + Click)
                if mods & glfw.MOD_SHIFT and self.hover_segment:
                    self._insert_point_at_segment()
                    return True
                
                # 3. Drawing Logic
                if self.curve_closed:
                    return False
                
                if len(self.active_curve_indices) > 0:
                    if self.hover_start_point: 
                        self._complete_curve(close_loop=True)
                    elif mods & glfw.MOD_CONTROL: 
                        self._complete_curve(close_loop=False)
                    else: 
                        self._save_state()
                        self._add_margin_point()
                    return True
                else:
                    self.is_drawing = True
                    self._save_state()
                    self._add_margin_point()
                    return True
            
            elif action == glfw.RELEASE:
                if self.dragging_point is not None:
                    self.dragging_point = None
                    self.drag_point_idx = None
                    self._update_action_guide("Point moved")
                    return True
                    
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
             if self.current_step_key == "MARGIN" and not self.curve_closed:
                self._undo_last_point()
                return True
        return False

    # ==========================================
    # RENDERING
    # ==========================================
    def render_3d(self):
        try:
            view = glGetFloatv(GL_MODELVIEW_MATRIX)
            proj = glGetFloatv(GL_PROJECTION_MATRIX)
            self.cached_view_matrix = np.array(view, dtype=np.float32).reshape(4, 4)
            self.cached_proj_matrix = np.array(proj, dtype=np.float32).reshape(4, 4)
            self.matrices_initialized = True
        except: pass
        
        if self.kd_tree is None: return

        # 1. Render Margin Geometry (Always visible foundation)
        self._render_margin_geometry()
        
        # 2. Render Quality Indicator (Visual feedback)
        self._render_quality_indicator()
        
        # 3. Render Sub-Tool (If Active)
        if self.current_subtool:
            self.current_subtool.render_3d()

    def _render_quality_indicator(self):
        """Render visual quality indicator in 3D space"""
        if not self.show_curve_quality or len(self.active_curve_indices) < 3:
            return
        
        # Find center of curve for indicator placement
        curve_points = []
        for segment in self.active_curve_segments:
            if segment is not None and len(segment) > 0:
                curve_points.extend(segment)
        
        if not curve_points:
            return
        
        # Calculate center
        curve_points_array = np.array(curve_points)
        center = np.mean(curve_points_array, axis=0)
        
        # Add slight offset above the curve
        center[1] += 0.1  # Small Y offset
        
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw quality ring (circle representing 0-100%)
        radius = 0.05 + (self.curve_quality / 100.0) * 0.05
        
        glColor4f(*self.quality_color)
        glLineWidth(3.0)
        
        # Draw filled circle for quality percentage
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*center)
        for i in range(33):  # 32 segments for smooth circle
            angle = (i / 32.0) * 2.0 * math.pi
            x = center[0] + radius * math.cos(angle)
            z = center[2] + radius * math.sin(angle)
            glVertex3f(x, center[1], z)
        glEnd()
        
        # Draw outline
        glColor4f(1.0, 1.0, 1.0, 0.8)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = (i / 32.0) * 2.0 * math.pi
            x = center[0] + radius * math.cos(angle)
            z = center[2] + radius * math.sin(angle)
            glVertex3f(x, center[1], z)
        glEnd()
        
        glPopAttrib()

    def _render_margin_geometry(self):
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        def draw_scene_geometry(use_color_override=None):
            # Draw Segments
            seg_color = self.closed_curve_color if self.curve_closed else self.active_curve_color
            if use_color_override: seg_color = use_color_override
            
            for segment in self.active_curve_segments:
                if segment is not None and len(segment) > 1:
                    self._draw_tube_strip(segment, seg_color, self.curve_width)
                
            # Preview Segment
            if len(self.preview_segment) > 0 and len(self.active_curve_indices) > 0 and not self.curve_closed:
                p_color = self.close_loop_color if self.hover_start_point else self.preview_color
                if use_color_override: p_color = use_color_override
                
                width = self.curve_width * 1.5 if self.hover_start_point else self.curve_width
                self._draw_tube_strip(self.preview_segment, p_color, width)

        # PASS 1: OCCLUDED PARTS (RED)
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_FALSE)
        draw_scene_geometry(use_color_override=self.occluded_color)

        # PASS 2: VISIBLE PARTS
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0) 
        draw_scene_geometry(use_color_override=None)
        
        # POINTS (Always Visible on Top)
        glPolygonOffset(-2.0, -2.0)
        if self.points:
            for i, p in enumerate(self.points):
                if i not in self.active_curve_indices and len(self.active_curve_indices) > 0:
                    continue

                pos = p['pos']
                
                # Visual style based on state
                is_start = (len(self.active_curve_indices) > 0 and i == self.active_curve_indices[0])
                is_hovered = (i == self.drag_point_idx)
                
                pt_size = self.point_size
                pt_color = self.active_point_color
                
                if self.curve_closed:
                    pt_color = self.active_point_color
                
                if is_start and self.hover_start_point:
                      pt_size *= 1.5
                      pt_color = self.close_loop_color
                elif is_hovered:
                      pt_size *= 1.3
                      pt_color = [1.0, 1.0, 1.0, 1.0]

                glPointSize(pt_size)
                glColor4f(*pt_color)
                glBegin(GL_POINTS); glVertex3f(*pos); glEnd()
        
        # Draw insertion point preview - WITH BOUNDS CHECKING
        if self.hover_segment and not self.dragging_point and self.current_step_key == "MARGIN":
            (seg_idx, line_idx), t = self.hover_segment
            # Check bounds before accessing
            if seg_idx < len(self.active_curve_segments):
                segment = self.active_curve_segments[seg_idx]
                if segment is not None and len(segment) > line_idx + 1:
                    p1 = segment[line_idx]
                    p2 = segment[line_idx + 1]
                    insert_pos = p1 + t * (p2 - p1)
                    
                    glPointSize(self.point_size * 1.2)
                    glColor4f(*self.insert_point_color)
                    glBegin(GL_POINTS); glVertex3f(*insert_pos); glEnd()
            
        glPopAttrib()

    def _draw_tube_strip(self, points, color, width):
        if len(points) < 2: return
        glLineWidth(width + 2.0)
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_LINE_STRIP)
        for p in points: glVertex3f(*p)
        glEnd()
        
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINE_STRIP)
        for p in points: glVertex3f(*p)
        glEnd()

    def on_key(self, key, scancode, action, mods):
        # 1. Block Main Logic if Sub-Tool is active
        if self.current_subtool:
            return False

        if action == glfw.PRESS:
            if key == glfw.KEY_ENTER and not self.curve_closed: 
                self._update_action_guide("Closing curve with ENTER")
                self._complete_curve(close_loop=True); return True
            elif key == glfw.KEY_ESCAPE: 
                self._update_action_guide("Reset curve")
                # Reset tool
                self.active_curve_indices = []
                self.active_curve_segments = []
                self.curve_closed = False
                self.preview_segment = []
                self._update_visual_indicators()
                return True
            elif key == glfw.KEY_Z and mods & glfw.MOD_CONTROL:
                if mods & glfw.MOD_SHIFT: 
                    self._redo()
                else: 
                    self._undo()
                return True
            # Toggle indicators
            elif key == glfw.KEY_Q:
                self.show_curve_quality = not self.show_curve_quality
                self._update_action_guide(f"Quality indicator: {'ON' if self.show_curve_quality else 'OFF'}")
                return True
            elif key == glfw.KEY_A:
                self.show_action_guide = not self.show_action_guide
                self._update_action_guide(f"Action guide: {'ON' if self.show_action_guide else 'OFF'}")
                return True
            elif key == glfw.KEY_N:
                self.show_numerical_feedback = not self.show_numerical_feedback
                self._update_action_guide(f"Numerical feedback: {'ON' if self.show_numerical_feedback else 'OFF'}")
                return True
        return False
            
    # ==========================================
    # GUI
    # ==========================================
    def draw_options(self):
        # Always update visual indicators before drawing
        self._update_visual_indicators()
        self._check_action_timeout()
        
        # ------------------------------------
        # STEP 1: MARGIN LINE (Base)
        # ------------------------------------
        if self.current_step_key == "MARGIN":
            imgui.text_colored("Step 1: Margin Line", 0.0, 0.8, 1.0)
            imgui.separator()
            
            # ===== SOLUTION 2: ACTION GUIDE =====
            if self.show_action_guide:
                imgui.text_colored("Current Action:", 1.0, 1.0, 0.0)
                imgui.same_line()
                imgui.text_wrapped(self.current_action)
                imgui.separator()
            
            # ===== SOLUTION 1: QUALITY INDICATOR =====
            if self.show_curve_quality and self.num_points > 0:
                imgui.text("Curve Quality:")
                imgui.same_line()
                # Color-coded quality indicator
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *self.quality_color[:3])
                imgui.progress_bar(self.curve_quality / 100.0, (100, 15), f"{self.curve_quality:.1f}%")
                imgui.pop_style_color(1)
                
                # Quality description
                if self.curve_quality > 80:
                    imgui.text_colored("Excellent", 0.2, 0.8, 0.2)
                elif self.curve_quality > 60:
                    imgui.text_colored("Good", 0.8, 0.8, 0.2)
                elif self.curve_quality > 40:
                    imgui.text_colored("Fair", 0.8, 0.5, 0.2)
                else:
                    imgui.text_colored("Needs Improvement", 0.8, 0.2, 0.2)
                imgui.separator()
            
            # ===== SOLUTION 3: NUMERICAL FEEDBACK =====
            if self.show_numerical_feedback:
                imgui.text("Statistics:")
                imgui.columns(2, "stats", False)
                imgui.text("Points:"); imgui.next_column()
                imgui.text(f"{self.num_points}"); imgui.next_column()
                
                imgui.text("Length:"); imgui.next_column()
                imgui.text(f"{self.curve_length:.2f} mm"); imgui.next_column()
                
                imgui.text("Avg Spacing:"); imgui.next_column()
                imgui.text(f"{self.avg_point_spacing:.2f} mm"); imgui.next_column()
                
                imgui.text("Continuity:"); imgui.next_column()
                imgui.text_colored(self.curve_continuity, 
                                 0.2 if self.curve_closed else 0.8,
                                 0.8 if self.curve_closed else 0.5,
                                 0.2 if self.curve_closed else 0.5)
                imgui.columns(1)
                imgui.separator()
            
            # Control buttons
            if imgui.button("Undo", width=145): self._undo()
            imgui.same_line()
            if imgui.button("Redo", width=145): self._redo()
            
            imgui.separator()
            
            # Status based on curve state
            if self.curve_closed:
                imgui.text_colored("✓ Margin Closed", 0.2, 1.0, 0.4)
                imgui.text_wrapped("Hold ALT to drag points.\nSHIFT+Click to insert points.")
                if imgui.button("Clear / Delete", width=-1):
                    self._update_action_guide("Clearing curve")
                    self._save_state()
                    self.active_curve_indices = []
                    self.active_curve_segments = []
                    self.curve_closed = False
                    self._update_visual_indicators()
            elif len(self.active_curve_indices) > 0:
                imgui.text_colored(f"Drawing... ({self.num_points} points)", 1.0, 0.6, 0.0)
                # Progress indicator for minimum points
                if self.num_points < 3:
                    imgui.progress_bar(self.num_points / 3.0, (100, 6), f"{self.num_points}/3 min points")
            else:
                imgui.text_colored("Status: Ready to Draw", 0.5, 0.5, 0.5)
            
            imgui.separator()
            
            # Settings
            imgui.text("Settings:")
            _, self.point_size = imgui.slider_float("Point Size", self.point_size, 6.0, 24.0)
            _, self.resolution = imgui.slider_float("Resolution", self.resolution, 0.05, 0.5)
            _, self.smoothness = imgui.slider_int("Smoothness", self.smoothness, 1, 30)
            _, self.lift_strength = imgui.slider_float("Lift Strength", self.lift_strength, 0.0, 2.0)
            
            # Toggle buttons for solutions
            imgui.separator()
            imgui.text("Display Options:")
            _, self.show_curve_quality = imgui.checkbox("Quality Indicator (Q)", self.show_curve_quality)
            _, self.show_action_guide = imgui.checkbox("Action Guide (A)", self.show_action_guide)
            _, self.show_numerical_feedback = imgui.checkbox("Numerical Stats (N)", self.show_numerical_feedback)
            
            # Hotkey help
            imgui.separator()
            imgui.text("Hotkeys:")
            imgui.text_wrapped("Q: Toggle Quality\nA: Toggle Action Guide\nN: Toggle Numerical Stats\nESC: Reset\nCTRL+Z: Undo\nCTRL+SHIFT+Z: Redo")
            
            # --- THE NEXT BUTTON (Dynamic) ---
            imgui.spacing()
            imgui.separator()
            
            if self.curve_closed:
                # Get the first tool in the registry
                first_tool_key = list(self.workflow_registry.keys())[0]
                if imgui.button(f"NEXT: {first_tool_key} >", width=-1, height=35):
                    self.switch_step(first_tool_key)
            else:
                imgui.text_colored("Close curve to proceed", 0.5, 0.5, 0.5)
                # Show what's needed
                if self.num_points < 3:
                    imgui.text_colored(f"Need {3 - self.num_points} more points", 1.0, 0.5, 0.5)

        # ------------------------------------
        # GENERIC SUB-TOOL HANDLER
        # ------------------------------------
        elif self.current_subtool:
            # 1. Back Button Logic
            if imgui.button("< Back", width=120):
                self._update_action_guide("Returning to previous step")
                keys = list(self.workflow_registry.keys())
                curr_idx = keys.index(self.current_step_key)
                if curr_idx == 0:
                    self.switch_step("MARGIN")
                else:
                    self.switch_step(keys[curr_idx - 1])
            
            imgui.separator()
            
            # 2. Show current workflow progress
            keys = list(self.workflow_registry.keys())
            curr_idx = keys.index(self.current_step_key)
            total_steps = len(keys)
            imgui.text(f"Step {curr_idx + 2} of {total_steps + 1}")
            imgui.progress_bar((curr_idx + 1) / (total_steps + 1), (200, 10))
            
            imgui.separator()
            
            # 3. Delegate GUI
            self.current_subtool.draw_options()

            # 4. Next Button Logic
            imgui.separator()
            
            # If there is a next tool, show Next button
            if curr_idx < len(keys) - 1:
                next_key = keys[curr_idx + 1]
                if imgui.button(f"NEXT: {next_key} >", width=-1):
                    self.switch_step(next_key)
            else:
                # End of workflow
                imgui.text_colored("✓ Workflow Complete", 0.2, 1.0, 0.4)
                if imgui.button("Return to Margin Line", width=-1):
                    self.switch_step("MARGIN")
