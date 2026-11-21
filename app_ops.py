# app_ops.py (modified)
import numpy as np
import copy
import ctypes
import glfw
import math  
import struct
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from matplotlib.path import Path
from utils import save_ply_selection, load_mesh_data


def _make_default_rotation():
    """
    Build a 4×4 rotation matrix that gives an isometric top-down view:
      - 45° yaw  (rotate around world Y so we see the scene at an angle)
      - 60° pitch (tilt camera down so the grid is clearly visible from above)
    """
    yaw   = np.radians(45.0)
    pitch = np.radians(60.0)

    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [           0, 1,           0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float32)

    Rx = np.array([
        [1,             0,              0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ], dtype=np.float32)

    R3 = Rx @ Ry
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R3
    return mat

try:
    from scipy.spatial import cKDTree as KDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[WARNING] scipy not found – fast pick unavailable. Run: pip install scipy")

class AppOpsMixin:

    # =================================================================
    # --- FAST POINT PICKING (KD-Tree) ---
    # =================================================================
    def _rebuild_kdtree(self):
        """Build/rebuild the KD-tree from current self.points."""
        if _HAS_SCIPY and getattr(self, 'points', None) is not None and len(self.points) > 0:
            self._kdtree = KDTree(self.points)
        else:
            self._kdtree = None

    def pick_single_point(self, x, y_gl):
        """Fast O(log n) point pick via depth-buffer unproject + KD-tree."""
        if not hasattr(self, '_kdtree') or self._kdtree is None:
            self._rebuild_kdtree()
        if self._kdtree is None:
            return None
        world_pos = self.get_world_pos_at_mouse(x, y_gl)
        if world_pos is None:
            return None
        _, idx = self._kdtree.query(world_pos, workers=1)
        return int(idx)

    def save_state(self, action_name="Action"):
        state = {
            'points': self.points.copy() if self.points is not None else None,
            'colors': self.base_colors.copy() if self.base_colors is not None else None,
            'selected': self.selected.copy() if self.selected is not None else None,
            'measurements': copy.deepcopy(self.measurements),
            'center': self.center.copy() if self.center is not None else None,
            'normals': self.normals.copy() if getattr(self, 'normals', None) is not None else None,
            'render_mode_mesh': getattr(self, 'render_mode_mesh', False),
            'has_normals': getattr(self, 'has_normals', False),
            'point_object_ids': getattr(self, 'point_object_ids', np.zeros(len(self.points), dtype=np.int32)).copy() if self.points is not None else None,
            'faces': self.faces.copy() if getattr(self, 'faces', None) is not None else None,
            'has_faces': getattr(self, 'has_faces', False),
            'tool_states': {}
        }
        if hasattr(self, 'tool_manager'):
            for t_id, tool in self.tool_manager.tools.items():
                if hasattr(tool, 'get_state'):
                    state['tool_states'][t_id] = tool.get_state()
        if not hasattr(self, 'undo_stack'): self.undo_stack = []
        if not hasattr(self, 'redo_stack'): self.redo_stack = []
        self.undo_stack.append(state)
        if len(self.undo_stack) > getattr(self, 'max_history', 30):
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not hasattr(self, 'undo_stack') or not self.undo_stack: 
            self.status_text = "Nothing to Undo."
            return
        current_state = self._capture_full_snapshot()
        if not hasattr(self, 'redo_stack'): self.redo_stack = []
        self.redo_stack.append(current_state)
        state = self.undo_stack.pop()
        self._restore_state(state)
        self.status_text = "Undid last action."

    def redo(self):
        if not hasattr(self, 'redo_stack') or not self.redo_stack: 
            self.status_text = "Nothing to Redo."
            return
        current_state = self._capture_full_snapshot()
        if not hasattr(self, 'undo_stack'): self.undo_stack = []
        self.undo_stack.append(current_state)
        state = self.redo_stack.pop()
        self._restore_state(state)
        self.status_text = "Redid action."

    def _capture_full_snapshot(self):
        snap = {
            'points': self.points.copy() if self.points is not None else None,
            'colors': self.base_colors.copy() if self.base_colors is not None else None,
            'selected': self.selected.copy() if self.selected is not None else None,
            'measurements': copy.deepcopy(self.measurements),
            'center': self.center.copy() if self.center is not None else None,
            'normals': self.normals.copy() if getattr(self, 'normals', None) is not None else None,
            'render_mode_mesh': getattr(self, 'render_mode_mesh', False),
            'has_normals': getattr(self, 'has_normals', False),
            'point_object_ids': getattr(self, 'point_object_ids', np.zeros(len(self.points), dtype=np.int32)).copy() if self.points is not None else None,
            'faces': self.faces.copy() if getattr(self, 'faces', None) is not None else None,
            'has_faces': getattr(self, 'has_faces', False),
            'tool_states': {}
        }
        if hasattr(self, 'tool_manager'):
            for t_id, tool in self.tool_manager.tools.items():
                if hasattr(tool, 'get_state'):
                    snap['tool_states'][t_id] = tool.get_state()
        return snap

    def _restore_state(self, state):
        self.points = state['points']
        self.base_colors = state['colors']
        self.selected = state['selected']
        self.measurements = state['measurements']
        self.center = state['center']
        self.normals = state.get('normals', np.zeros_like(self.points) if self.points is not None else None)
        self.render_mode_mesh = state.get('render_mode_mesh', False)
        self.has_normals = state.get('has_normals', False)
        self.faces = state.get('faces')
        if self.faces is not None:
            self.faces = self.faces.copy()
        self.has_faces = state.get('has_faces', False)
        if 'point_object_ids' in state and state['point_object_ids'] is not None:
            self.point_object_ids = state['point_object_ids'].copy()
        tool_states = state.get('tool_states', {})
        if hasattr(self, 'tool_manager'):
            for t_id, t_state in tool_states.items():
                if t_id in self.tool_manager.tools and hasattr(self.tool_manager.tools[t_id], 'set_state'):
                    self.tool_manager.tools[t_id].set_state(t_state)
        if self.points is not None and self.base_colors is not None:
            self.points_colors = np.hstack([self.points, self.base_colors])
            self.rebuild_ids(len(self.points))
            self.update_vbo_full()
            self.update_colors()
        if hasattr(self, 'sync_sculptor'):
            self.sync_sculptor()
        self._rebuild_kdtree()

    def _align_mask_to_triangles(self, mask):
        if not getattr(self, 'render_mode_mesh', False): return mask
        n = len(mask)
        if n % 3 != 0: return mask 
        faces_mask = mask.reshape(-1, 3)
        face_is_selected = np.any(faces_mask, axis=1)
        aligned_mask = np.repeat(face_is_selected, 3)
        return aligned_mask

    def delete_selection(self):
        if self.selected is None: return
        count = np.sum(self.selected)
        if count == 0: self.status_text = "Nothing to delete."; return
        self.save_state("Delete Points")
        mask_to_delete = self.selected
        if getattr(self, 'render_mode_mesh', False) and not getattr(self, 'has_faces', False):
            mask_to_delete = self._align_mask_to_triangles(self.selected)
        self.apply_mask(~mask_to_delete)
        self.status_text = f"Deleted selection."

    def crop_selection(self):
        if self.selected is None: return
        count = np.sum(self.selected)
        if count == 0: return
        self.save_state("Crop Points")
        mask_to_keep = self.selected
        if getattr(self, 'render_mode_mesh', False) and not getattr(self, 'has_faces', False):
            mask_to_keep = self._align_mask_to_triangles(self.selected)
        self.apply_mask(mask_to_keep)
        self.status_text = f"Cropped to selection."

    def apply_mask(self, mask):
        """
        Keep only points where mask is True.
        Also updates measurements and active drag points accordingly.
        """
        # Build mapping from old indices to new indices
        kept_indices = np.where(mask)[0]
        mapping = np.full(len(mask), -1, dtype=np.int32)
        mapping[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)

        # Handle faces if present
        if getattr(self, 'has_faces', False) and getattr(self, 'faces', None) is not None:
            new_faces = mapping[self.faces]
            valid_faces_mask = np.all(new_faces != -1, axis=1)
            self.faces = new_faces[valid_faces_mask].astype(np.uint32)
            if len(self.faces) == 0:
                self.has_faces = False
                self.faces = None

        # --- Update measurements -------------------------------------------------
        if hasattr(self, 'measurements') and self.measurements:
            new_measurements = []
            for i, j in self.measurements:
                if mapping[i] != -1 and mapping[j] != -1:
                    new_measurements.append((mapping[i], mapping[j]))
            self.measurements = new_measurements

        # Clear active drag points if they reference a deleted vertex
        if hasattr(self, 'measure_start_pt') and self.measure_start_pt is not None:
            if self.measure_start_pt >= len(mapping) or mapping[self.measure_start_pt] == -1:
                self.measure_start_pt = None
        if hasattr(self, 'measure_drag_pt') and self.measure_drag_pt is not None:
            if self.measure_drag_pt >= len(mapping) or mapping[self.measure_drag_pt] == -1:
                self.measure_drag_pt = None
        # -------------------------------------------------------------------------

        # Apply mask to point data
        self.points = self.points[mask]
        self.base_colors = self.base_colors[mask]
        if getattr(self, 'has_normals', False) and len(self.normals) == len(mask):
            self.normals = self.normals[mask]
        elif getattr(self, 'normals', None) is not None:
            self.normals = self.normals[:len(self.points)]
        if hasattr(self, 'point_object_ids') and getattr(self, 'point_object_ids', None) is not None and len(self.point_object_ids) == len(mask):
            self.point_object_ids = self.point_object_ids[mask]

        self.selected = np.zeros(len(self.points), dtype=bool)
        self.points_colors = np.hstack([self.points, self.base_colors])
        self.rebuild_ids(len(self.points))
        self.update_vbo_full()
        self.measure_start_pt = None
        self.measure_drag_pt = None
        self.is_measuring_drag = False

        if hasattr(self, 'sync_sculptor'):
            self.sync_sculptor()
        self._rebuild_kdtree()

    def rebuild_ids(self, n):
        indices = np.arange(n, dtype=np.uint32)
        r_id = (indices & 0xFF).astype(np.float32) / 255.0
        g_id = ((indices >> 8) & 0xFF).astype(np.float32) / 255.0
        b_id = ((indices >> 16) & 0xFF).astype(np.float32) / 255.0
        self.points_ids = np.hstack([self.points, np.stack([r_id, g_id, b_id], axis=-1)])

    def rebuild_buffers(self):
        self.points_colors[:, :3] = self.points
        self.rebuild_ids(len(self.points))
        self.update_vbo_full()

    def align_floor(self):
        """
        Align the mesh so its dominant base surface sits flat on Y=0.

        Algorithm
        ---------
        1. Collect the bottom 20 % of points by Y — these represent the
           candidate ground surface.
        2. Fit a plane to those points with SVD (Principal Component Analysis).
           The plane normal is the singular vector with the *smallest* singular
           value (i.e., the axis of least variance in the ground patch).
        3. Build a Rodrigues rotation that maps the fitted normal to world-up
           (0, 1, 0) so the base lies perfectly horizontal.
        4. Apply the rotation to every point (and to normals if present),
           rotating around the mesh centroid so the object stays centred.
        5. Translate so the new minimum Y = 0 (object rests exactly on the
           ground plane without hovering or sinking).
        """
        if self.points is None or len(self.points) == 0:
            return
        self.save_state("Align Floor")

        pts = self.points.astype(np.float64)

        # ── Step 1: identify ground candidates (bottom 20 % by Y) ────────────
        y_vals = pts[:, 1]
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        y_span = y_max - y_min
        y_thresh = y_min + max(0.20 * y_span, 1e-6)
        ground_pts = pts[y_vals <= y_thresh]

        if len(ground_pts) < 3:
            # Degenerate: just translate
            pts[:, 1] -= y_min
            self.points = pts.astype(np.float32)
            self.points_colors[:, :3] = self.points
            self.center = np.mean(self.points, axis=0)
            self.rebuild_buffers()
            self.status_text = "Aligned to Floor (translation only — too few ground points)"
            return

        # ── Step 2: SVD plane fit on ground points ────────────────────────────
        centroid_g = ground_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(ground_pts - centroid_g, full_matrices=False)
        # Last row of Vt = normal of best-fit plane (smallest variance axis)
        normal = Vt[2].copy()

        # Make sure the normal points upward
        if normal[1] < 0:
            normal = -normal

        # ── Step 3: Rodrigues rotation  normal → (0, 1, 0) ───────────────────
        world_up = np.array([0.0, 1.0, 0.0])
        v = np.cross(normal, world_up)       # rotation axis (un-normalised)
        s = float(np.linalg.norm(v))         # sin of angle
        c = float(np.dot(normal, world_up))  # cos of angle

        if s < 1e-9:
            # Already aligned or exactly anti-aligned
            R = np.eye(3, dtype=np.float64)
            if c < 0:
                # 180° flip around the X axis
                R[1, 1] = -1.0
                R[2, 2] = -1.0
        else:
            vx = np.array([[ 0.0,  -v[2],  v[1]],
                           [ v[2],   0.0, -v[0]],
                           [-v[1],  v[0],   0.0]], dtype=np.float64)
            R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

        # ── Step 4: rotate all points around the mesh centroid ────────────────
        mesh_centroid = pts.mean(axis=0)
        pts_r = (R @ (pts - mesh_centroid).T).T + mesh_centroid

        # Rotate normals if present
        if hasattr(self, 'normals') and self.normals is not None and len(self.normals):
            nrm = (R @ self.normals.astype(np.float64).T).T
            self.normals = nrm.astype(np.float32)

        # ── Step 5: translate so the lowest point rests on Y = 0 ─────────────
        pts_r[:, 1] -= float(pts_r[:, 1].min())

        self.points = pts_r.astype(np.float32)
        self.points_colors[:, :3] = self.points
        self.center = np.mean(self.points, axis=0)
        self._rebuild_kdtree()
        self.rebuild_buffers()
        self.status_text = "Aligned to Ground Plane  ✔"

    def rotate_90_x(self):
        """
        Rotate the mesh 90° around its own X axis, then re-seat it on Y=0.

        The rotation is applied about the mesh centroid (not the world origin)
        so the object stays centred in the scene instead of flying off to one
        side.  After rotation the mesh is translated so its lowest point sits
        exactly on Y=0.
        """
        if self.points is None or len(self.points) == 0:
            return
        self.save_state("Rotate 90° X")

        pts = self.points.astype(np.float64)
        centroid = pts.mean(axis=0)

        # Centre → rotate → restore position
        pts_c = pts - centroid
        # R_x(90°): Y → −Z,  Z → +Y
        new_y = -pts_c[:, 2].copy()
        new_z =  pts_c[:, 1].copy()
        pts_c[:, 1] = new_y
        pts_c[:, 2] = new_z
        pts = pts_c + centroid

        # Rotate normals the same way
        if hasattr(self, 'normals') and self.normals is not None and len(self.normals):
            nrm = self.normals.astype(np.float64)
            n_y = -nrm[:, 2].copy()
            n_z =  nrm[:, 1].copy()
            nrm[:, 1] = n_y
            nrm[:, 2] = n_z
            self.normals = nrm.astype(np.float32)

        # Re-seat on Y = 0
        pts[:, 1] -= float(pts[:, 1].min())

        self.points = pts.astype(np.float32)
        self.points_colors[:, :3] = self.points
        self.center = np.mean(self.points, axis=0)
        self._rebuild_kdtree()
        self.rebuild_buffers()
        self.status_text = "Rotated 90° on X-Axis  ✔"

    def rotate_90_z(self):
        if self.points is None or len(self.points) == 0: return
        self.save_state("Rotate Z")
        old_x = self.points[:, 0].copy()
        old_y = self.points[:, 1].copy()
        self.points[:, 0] = -old_y
        self.points[:, 1] = old_x
        self.center = np.mean(self.points, axis=0)
        self.rebuild_buffers()
        self.status_text = "Rotated 90 degrees Z"

    # =================================================================
    # --- NEW GEOMETRY OPERATIONS ---
    # =================================================================
    def rotate_90_y(self):
        """Rotate the mesh 90 deg around its own Y axis, about the centroid."""
        if self.points is None or len(self.points) == 0:
            return
        self.save_state("Rotate 90 deg Y")

        pts = self.points.astype(np.float64)
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        # R_y(90 deg): X -> +Z,  Z -> -X
        new_x =  pts_c[:, 2].copy()
        new_z = -pts_c[:, 0].copy()
        pts_c[:, 0] = new_x
        pts_c[:, 2] = new_z
        pts = pts_c + centroid

        if hasattr(self, 'normals') and self.normals is not None and len(self.normals):
            nrm = self.normals.astype(np.float64)
            n_x =  nrm[:, 2].copy()
            n_z = -nrm[:, 0].copy()
            nrm[:, 0] = n_x
            nrm[:, 2] = n_z
            self.normals = nrm.astype(np.float32)

        self.points = pts.astype(np.float32)
        self.points_colors[:, :3] = self.points
        self.center = np.mean(self.points, axis=0)
        self._rebuild_kdtree()
        self.rebuild_buffers()
        self.status_text = "Rotated 90 deg on Y-Axis"

    def mirror_geometry(self, axis='x'):
        """
        Mirror the geometry across the chosen axis through the centroid.
        Flips the matching normal component and reverses face winding so
        the surface stays correctly oriented.
        """
        if self.points is None or len(self.points) == 0:
            return
        axis = axis.lower()
        idx = {'x': 0, 'y': 1, 'z': 2}.get(axis)
        if idx is None:
            return
        self.save_state(f"Mirror {axis.upper()}")

        centroid = self.points.mean(axis=0)
        pts = self.points.copy()
        pts[:, idx] = 2.0 * centroid[idx] - pts[:, idx]
        self.points = pts.astype(np.float32)
        self.points_colors[:, :3] = self.points

        if hasattr(self, 'normals') and self.normals is not None and len(self.normals):
            self.normals[:, idx] = -self.normals[:, idx]

        # Mirroring inverts handedness; flip face winding to keep normals out.
        if getattr(self, 'has_faces', False) and getattr(self, 'faces', None) is not None and len(self.faces):
            self.faces = self.faces[:, ::-1].copy().astype(np.uint32)

        self.center = np.mean(self.points, axis=0)
        self._rebuild_kdtree()
        self.rebuild_buffers()
        self.status_text = f"Mirrored across {axis.upper()} axis"

    def center_on_origin(self):
        """Translate the geometry so its bounding-box centre sits at (0,0,0)."""
        if self.points is None or len(self.points) == 0:
            return
        self.save_state("Center on Origin")

        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        bbox_center = (mins + maxs) / 2.0
        self.points = (self.points - bbox_center).astype(np.float32)
        self.points_colors[:, :3] = self.points
        self.center = np.mean(self.points, axis=0)
        self._rebuild_kdtree()
        self.rebuild_buffers()
        self.status_text = "Centered on origin"

    def flip_normals(self):
        """Invert all vertex normals and reverse face winding to match."""
        if not getattr(self, 'has_normals', False) or self.normals is None or len(self.normals) == 0:
            self.status_text = "No normals to flip."
            return
        self.save_state("Flip Normals")
        self.normals = (-self.normals).astype(np.float32)
        if getattr(self, 'has_faces', False) and getattr(self, 'faces', None) is not None and len(self.faces):
            self.faces = self.faces[:, ::-1].copy().astype(np.uint32)
        self.rebuild_buffers()
        if hasattr(self, 'sync_sculptor'):
            self.sync_sculptor()
        self.status_text = "Normals flipped"

    def update_colors(self):
        if self.points_colors is not None and self.base_colors is not None:
            self.points_colors[:, 3:6] = self.base_colors
            self.vbo_dirty = True

    def select_sub_object(self, object_id):
        if not hasattr(self, 'point_object_ids'):
            return
        self.save_state("Select Sub-Object")
        self.selected = self.selected | (self.point_object_ids == object_id)
        self.update_colors()
        self.status_text = f"Selected Sub-Object {object_id}"

    def invert_selection(self):
        if self.selected is None: return
        self.save_state("Invert Selection")
        self.selected = ~self.selected
        if getattr(self, 'render_mode_mesh', False):
            self.selected = self._align_mask_to_triangles(self.selected)
        self.update_colors()
        self.status_text = "Selection inverted."

    # =================================================================
    # --- SELECTION OPERATIONS (Edit menu) ---
    # =================================================================
    def select_all(self):
        """Select every point in the scene."""
        if self.points is None or len(self.points) == 0:
            self.status_text = "Nothing to select."
            return
        self.save_state("Select All")
        self.selected = np.ones(len(self.points), dtype=bool)
        self.update_colors()
        self.status_text = f"Selected all {len(self.points):,} points."

    def _selection_neighbor_radius(self):
        """Pick a reasonable neighbour radius for grow/shrink based on point spacing."""
        if self.points is None or len(self.points) < 2:
            return None
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        # ~0.5% of the bounding-box diagonal is a sensible single-step radius.
        return max(1e-6, diag * 0.005)

    def grow_selection(self):
        """
        Expand the selection to include points within a small radius of any
        currently selected point (one ring of growth per call).
        """
        if self.selected is None or self.points is None or not np.any(self.selected):
            self.status_text = "Nothing selected to grow."
            return
        if not _HAS_SCIPY:
            self.status_text = "Grow/Shrink needs scipy."
            return
        radius = self._selection_neighbor_radius()
        if radius is None:
            return
        self.save_state("Grow Selection")
        if not hasattr(self, '_kdtree') or self._kdtree is None:
            self._rebuild_kdtree()
        sel_pts = self.points[self.selected]
        # query_ball_point returns, for each seed, the indices within radius
        neighbor_lists = self._kdtree.query_ball_point(sel_pts, radius)
        new_idx = np.unique(np.concatenate([np.asarray(l, dtype=np.int64)
                                            for l in neighbor_lists if len(l)])) \
                  if any(len(l) for l in neighbor_lists) else np.array([], dtype=np.int64)
        if len(new_idx):
            self.selected[new_idx] = True
        if getattr(self, 'render_mode_mesh', False):
            self.selected = self._align_mask_to_triangles(self.selected)
        self.update_colors()
        self.status_text = f"Grew selection to {int(np.sum(self.selected)):,} points."

    def shrink_selection(self):
        """
        Contract the selection: drop any selected point that has an unselected
        neighbour within the step radius (removes the outer ring).
        """
        if self.selected is None or self.points is None or not np.any(self.selected):
            self.status_text = "Nothing selected to shrink."
            return
        if not _HAS_SCIPY:
            self.status_text = "Grow/Shrink needs scipy."
            return
        radius = self._selection_neighbor_radius()
        if radius is None:
            return
        self.save_state("Shrink Selection")
        if not hasattr(self, '_kdtree') or self._kdtree is None:
            self._rebuild_kdtree()
        sel_idx = np.where(self.selected)[0]
        sel_pts = self.points[sel_idx]
        neighbor_lists = self._kdtree.query_ball_point(sel_pts, radius)
        # A selected point survives only if ALL its neighbours are also selected.
        keep = np.ones(len(sel_idx), dtype=bool)
        for i, nbrs in enumerate(neighbor_lists):
            if len(nbrs) == 0:
                continue
            if not np.all(self.selected[np.asarray(nbrs, dtype=np.int64)]):
                keep[i] = False
        new_selected = np.zeros(len(self.points), dtype=bool)
        new_selected[sel_idx[keep]] = True
        self.selected = new_selected
        self.update_colors()
        self.status_text = f"Shrunk selection to {int(np.sum(self.selected)):,} points."

    def duplicate_selection(self):
        """
        Duplicate the selected points as a new object (new object id), offset
        slightly so the copy is visible, and leave the copy selected.
        """
        if self.selected is None or self.points is None or not np.any(self.selected):
            self.status_text = "Nothing selected to duplicate."
            return
        self.save_state("Duplicate Selection")

        sel = self.selected
        new_pts = self.points[sel].copy()
        new_cols = self.base_colors[sel].copy()

        # Small offset (1% of bbox diagonal) so the duplicate doesn't perfectly overlap.
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        offset = np.array([diag * 0.01, 0.0, 0.0], dtype=np.float32)
        new_pts = (new_pts + offset).astype(np.float32)

        n_new = len(new_pts)
        old_n = len(self.points)

        self.points = np.vstack([self.points, new_pts]).astype(np.float32)
        self.base_colors = np.vstack([self.base_colors, new_cols]).astype(np.float32)

        # Normals
        if getattr(self, 'has_normals', False) and self.normals is not None and len(self.normals) == old_n:
            new_norms = self.normals[sel].copy()
            self.normals = np.vstack([self.normals, new_norms]).astype(np.float32)
        elif self.normals is not None:
            self.normals = np.vstack([self.normals,
                                      np.zeros((n_new, 3), dtype=np.float32)]).astype(np.float32)

        # Object ids — assign a fresh id to the duplicate
        if hasattr(self, 'point_object_ids') and self.point_object_ids is not None and len(self.point_object_ids) == old_n:
            new_obj_id = int(np.max(self.point_object_ids)) + 1 if old_n > 0 else 1
            self.point_object_ids = np.concatenate(
                [self.point_object_ids, np.full(n_new, new_obj_id, dtype=np.int32)])
        else:
            self.point_object_ids = np.zeros(len(self.points), dtype=np.int32)

        # Duplicating loose points does not carry faces (the copy is a point set).
        # Leave existing faces untouched; the new points are unindexed.

        # Select only the new copy
        self.selected = np.zeros(len(self.points), dtype=bool)
        self.selected[old_n:] = True

        self.points_colors = np.hstack([self.points, self.base_colors])
        self.rebuild_ids(len(self.points))
        self.update_vbo_full()
        self.update_colors()
        if hasattr(self, 'sync_sculptor'):
            self.sync_sculptor()
        self._rebuild_kdtree()
        self.status_text = f"Duplicated {n_new:,} points as new object."

    def save_selected(self, fname):
        if self.selected is None or self.points is None: return
        mask = self.selected
        if getattr(self, 'render_mode_mesh', False):
            mask = self._align_mask_to_triangles(mask)
        pts = self.points[mask] if np.sum(mask) > 0 else self.points
        cols = self.base_colors[mask] if np.sum(mask) > 0 else self.base_colors
        save_ply_selection(fname, pts, cols)
        self.status_text = f"Saved {len(pts)} points to {fname}"

    def apply_poly_cut(self):
        if not hasattr(self, 'knife_poly') or len(self.knife_poly) < 3: return
        self.save_state("Polygon Knife")
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = self.width, self.height
        px_ratio = fb_w / win_w if win_w > 0 else 1.0
        glDisable(GL_DITHER); glDisable(GL_MULTISAMPLE); glDisable(GL_LIGHTING)
        glDisable(GL_FOG); glDisable(GL_BLEND); glDisable(GL_COLOR_MATERIAL)
        glClearColor(1.0, 1.0, 1.0, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        self.setup_camera()
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))
        glPointSize(max(1.0, getattr(self, 'point_size', 2.0)))
        if getattr(self, 'render_mode_mesh', False) and len(self.points_ids) % 3 == 0:
            glShadeModel(GL_FLAT); glDrawArrays(GL_TRIANGLES, 0, len(self.points_ids)); glShadeModel(GL_SMOOTH)
        else:
            glDrawArrays(GL_POINTS, 0, len(self.points_ids))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_COLOR_ARRAY)
        glFlush(); glFinish()
        poly = np.array(self.knife_poly) * px_ratio
        min_x, max_x = int(max(0, poly[:,0].min())), int(min(self.width, poly[:,0].max()))
        min_y, max_y = int(max(0, poly[:,1].min())), int(min(self.height, poly[:,1].max()))
        w, h = max_x - min_x, max_y - min_y
        if w > 0 and h > 0:
            pixels = glReadPixels(min_x, min_y, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
            grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
            mask_flat = Path(poly).contains_points(np.vstack((grid_x.ravel(), grid_y.ravel())).T)
            valid_pixels = img[mask_flat.reshape(h, w)]
            if len(valid_pixels) > 0:
                p_r, p_g, p_b = valid_pixels[:, 0].astype(np.uint32), valid_pixels[:, 1].astype(np.uint32), valid_pixels[:, 2].astype(np.uint32)
                recovered = p_r + (p_g << 8) + (p_b << 16)
                valid = recovered[recovered < len(self.points)]
                self.selected[valid] = True
                if getattr(self, 'render_mode_mesh', False):
                    self.selected = self._align_mask_to_triangles(self.selected)
        glEnable(GL_DITHER)
        glDisable(GL_POINT_SMOOTH)
        self.update_colors()
        self.knife_poly = []
        self.is_poly_active = False
        self.status_text = f"Polygon Select Applied."

    def lasso_select_pixels(self):
        if not hasattr(self, 'lasso') or len(self.lasso) < 3: return
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = self.width, self.height
        px_ratio = fb_w / win_w if win_w > 0 else 1.0
        glDisable(GL_DITHER); glDisable(GL_MULTISAMPLE); glDisable(GL_LIGHTING)
        glDisable(GL_FOG); glDisable(GL_BLEND); glDisable(GL_COLOR_MATERIAL)
        glClearColor(1.0, 1.0, 1.0, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        self.setup_camera()
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, 24, ctypes.c_void_p(12))
        glPointSize(max(1.0, getattr(self, 'point_size', 2.0)))
        if getattr(self, 'render_mode_mesh', False) and len(self.points_ids) % 3 == 0:
            glShadeModel(GL_FLAT); glDrawArrays(GL_TRIANGLES, 0, len(self.points_ids)); glShadeModel(GL_SMOOTH)
        else:
            glDrawArrays(GL_POINTS, 0, len(self.points_ids))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_COLOR_ARRAY)
        glFlush(); glFinish()
        poly = np.array(self.lasso) * px_ratio
        min_x, max_x = int(max(0, poly[:,0].min())), int(min(self.width, poly[:,0].max()))
        min_y, max_y = int(max(0, poly[:,1].min())), int(min(self.height, poly[:,1].max()))
        w, h = max_x - min_x, max_y - min_y
        if w > 0 and h > 0:
            pixels = glReadPixels(min_x, min_y, w, h, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
            grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
            mask_flat = Path(poly).contains_points(np.vstack((grid_x.ravel(), grid_y.ravel())).T)
            valid_pixels = img[mask_flat.reshape(h, w)]
            if len(valid_pixels) > 0:
                p_r, p_g, p_b = valid_pixels[:, 0].astype(np.uint32), valid_pixels[:, 1].astype(np.uint32), valid_pixels[:, 2].astype(np.uint32)
                recovered = p_r + (p_g << 8) + (p_b << 16)
                valid = recovered[recovered < len(self.points)]
                if getattr(self, 'lasso_mode', 1) == 1:
                    self.selected[valid] = True
                    if getattr(self, 'render_mode_mesh', False):
                        self.selected = self._align_mask_to_triangles(self.selected)
                elif getattr(self, 'lasso_mode', 1) == -1:
                    self.selected[valid] = False
                    if getattr(self, 'render_mode_mesh', False):
                        self.selected = self._align_mask_to_triangles(self.selected)
        glEnable(GL_DITHER)
        glDisable(GL_POINT_SMOOTH)
        self.update_colors()
        self.status_text = f"Selected: {np.sum(self.selected)}"

    # ---------- Clear Scene ----------
    def clear_scene(self):
        """Reset the entire scene to empty, discarding all data."""
        try:
            self.save_state("Clear Scene")
            self.points = np.zeros((0, 3), dtype=np.float32)
            self.base_colors = np.zeros((0, 3), dtype=np.float32)
            self.normals = np.zeros((0, 3), dtype=np.float32)
            self.selected = np.zeros(0, dtype=bool)
            self.point_object_ids = np.zeros(0, dtype=np.int32)
            self.faces = None
            self.has_faces = False
            self.has_normals = False
            self.render_mode_mesh = False
            self.measurements = []
            self.undo_stack = []
            self.redo_stack = []
            self.points_colors = np.hstack([self.points, self.base_colors])
            self.rebuild_ids(0)
            self.center = np.array([0.0, 0.0, 0.0])
            self.distance = 2.5
            self.rotation_matrix = _make_default_rotation()
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._kdtree = None
            self.sculptor = None
            self.update_vbo_full()
            self._invalidate_lod_cache()
            if hasattr(self, 'tool_manager'):
                self.tool_manager.set_active_tool(0)
            self.status_text = "New scene created."
            # Reset brush sizes to a sensible default (using a placeholder diagonal of 10.0)
            if hasattr(self, 'adjust_brush_sizes'):
                self.adjust_brush_sizes(10.0)
        except Exception as e:
            print(f"clear_scene error: {e}")

    # =================================================================
    # --- PROPRIETARY FILE FORMAT (.gcad) ---
    # =================================================================
    def save_project_as_dialog(self):
        """Open a Save As dialog and save the current project."""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        filepath = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".gcad",
            filetypes=[("GridCad Project", "*.gcad"), ("All Files", "*.*")]
        )
        root.destroy()
        if filepath:
            self.current_project_path = filepath
            self.save_project()

    def save_project(self):
        """Save the current scene to the current project path."""
        if not hasattr(self, 'current_project_path') or not self.current_project_path:
            self.save_project_as_dialog()
            return
        try:
            # Collect data
            points = self.points.astype(np.float32)
            colors = self.base_colors.astype(np.float32)
            normals = self.normals.astype(np.float32) if self.has_normals else None
            faces = self.faces.astype(np.uint32) if self.has_faces else None
            obj_ids = self.point_object_ids.astype(np.int32) if hasattr(self, 'point_object_ids') else np.zeros(len(points), dtype=np.int32)
            selected = self.selected.astype(np.uint8)
            measurements = np.array(self.measurements, dtype=np.uint64).flatten() if self.measurements else np.array([], dtype=np.uint64)

            # Camera
            cam = np.array([
                self.center[0], self.center[1], self.center[2],
                self.distance,
                *self.rotation_matrix.flatten(),
                self.pan_x, self.pan_y,
                1.0 if self.is_orthographic else 0.0
            ], dtype=np.float32)

            with open(self.current_project_path, 'wb') as f:
                # Header
                f.write(b'GRIDCAD1')         # 8 bytes magic
                f.write(struct.pack('<I', 1)) # version 1
                flags = 0
                if normals is not None: flags |= 1
                if faces is not None:   flags |= 2
                f.write(struct.pack('<I', flags))

                # Counts
                f.write(struct.pack('<Q', len(points)))          # num_points
                f.write(struct.pack('<Q', len(measurements) // 2)) # num_measurements
                f.write(struct.pack('<Q', len(faces) if faces is not None else 0)) # num_faces

                # Offsets (will be filled later)
                offset_pos = f.tell()
                f.write(struct.pack('<Q', 0) * 8)  # 8 offsets

                # Write sections
                offsets = {}
                def write_section(name, data):
                    offsets[name] = f.tell()
                    f.write(data.tobytes())

                write_section('points', points)
                write_section('colors', colors)
                if normals is not None:
                    write_section('normals', normals)
                if faces is not None:
                    write_section('faces', faces)
                write_section('obj_ids', obj_ids)
                write_section('selected', selected)
                if len(measurements) > 0:
                    write_section('measurements', measurements)
                write_section('camera', cam)

                # Go back and write offsets
                f.seek(offset_pos)
                for name in ['points', 'colors', 'normals', 'faces', 'obj_ids', 'selected', 'measurements', 'camera']:
                    off = offsets.get(name, 0)
                    f.write(struct.pack('<Q', off))

            self.status_text = f"Project saved: {os.path.basename(self.current_project_path)}"
        except Exception as e:
            self.status_text = f"Save failed: {e}"

    def open_project_dialog(self):
        """Open a file dialog for selecting a .gcad project and start loading."""
        import threading, tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("GridCad Project", "*.gcad"), ("All Files", "*.*")]
        )
        root.destroy()
        if filepath:
            self.pending_file = filepath
            self.pending_file_mode = 'project'
            # No need to start thread here; check_file_loader will do it.

    def _start_project_load_thread(self, filepath):
        """Background thread for loading a .gcad file."""
        import threading, time
        self._load_thread_running = True
        self._loaded_result = None
        self.status_text = f"Loading project {os.path.basename(filepath)} …"

        def _worker():
            try:
                t0 = time.perf_counter()
                with open(filepath, 'rb') as f:
                    magic = f.read(8)
                    # Accept the current magic and the legacy one so old
                    # project files still open in this build.
                    if magic not in (b'GRIDCAD1', b'DENTAL3D'):
                        raise ValueError("Not a valid GridCad (.gcad) project file")
                    version = struct.unpack('<I', f.read(4))[0]
                    if version != 1:
                        raise ValueError(f"Unsupported version {version}")
                    flags = struct.unpack('<I', f.read(4))[0]
                    num_points = struct.unpack('<Q', f.read(8))[0]
                    num_measurements = struct.unpack('<Q', f.read(8))[0]
                    num_faces = struct.unpack('<Q', f.read(8))[0]
                    offsets = struct.unpack('<8Q', f.read(8*8))

                    # Seek and read sections
                    def read_array(offset, dtype, count, shape=None):
                        if offset == 0:
                            return None
                        f.seek(offset)
                        data = np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype)
                        if shape:
                            data = data.reshape(shape)
                        return data

                    points = read_array(offsets[0], np.float32, num_points*3, (num_points,3))
                    colors = read_array(offsets[1], np.float32, num_points*3, (num_points,3))
                    normals = read_array(offsets[2], np.float32, num_points*3, (num_points,3)) if (flags & 1) else None
                    faces = read_array(offsets[3], np.uint32, num_faces*3, (num_faces,3)) if (flags & 2) else None
                    obj_ids = read_array(offsets[4], np.int32, num_points)
                    selected = read_array(offsets[5], np.uint8, num_points).astype(bool)
                    measurements_raw = read_array(offsets[6], np.uint64, num_measurements*2)
                    measurements = measurements_raw.reshape(-1,2).tolist() if measurements_raw is not None else []
                    # Camera count: center(3) + distance(1) + rot(16) + pan(2) + ortho(1) = 23
                    camera = read_array(offsets[7], np.float32, 3+1+16+2+1)
                    if camera is not None:
                        camera = camera.flatten()

                print(f"[Loader] project parse: {time.perf_counter()-t0:.2f}s")

                # Rebuild KD‑tree and sculptor in background
                t0 = time.perf_counter()
                try:
                    from scipy.spatial import cKDTree
                    kdtree = cKDTree(points)
                except Exception:
                    kdtree = None
                try:
                    from sculpt_core import SculptSession
                    sculptor = SculptSession(points, normals if normals is not None else np.zeros_like(points))
                except Exception:
                    sculptor = None
                print(f"[Loader] spatial indices: {time.perf_counter()-t0:.2f}s")

                self._loaded_result = (points, colors, normals, faces, obj_ids, selected,
                                        measurements, camera, kdtree, sculptor, None)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._loaded_result = (None, None, None, None, None, None, None, None, None, None, str(e))
            finally:
                self._load_thread_running = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _commit_project_result(self):
        """Apply the loaded project data to the scene."""
        (points, colors, normals, faces, obj_ids, selected,
         measurements, camera, kdtree, sculptor, err) = self._loaded_result
        self._loaded_result = None
        if err:
            self.status_text = f"Project load failed: {err}"
            return
        if points is None or len(points) == 0:
            self.status_text = "Project contains no points."
            return

        self.points = points
        self.base_colors = colors
        self.normals = normals if normals is not None else np.zeros_like(points)
        self.has_normals = normals is not None
        self.render_mode_mesh = self.has_normals
        self.faces = faces
        self.has_faces = faces is not None and len(faces) > 0
        self.point_object_ids = obj_ids
        self.selected = selected
        self.measurements = measurements

        # --- Fix: ensure EBO exists if faces are present ---
        if self.has_faces:
            if not hasattr(self, 'ebo') or self.ebo is None:
                from OpenGL.GL import glGenBuffers
                self.ebo = glGenBuffers(1)
        else:
            # If no faces, delete any existing EBO
            if hasattr(self, 'ebo') and self.ebo is not None:
                try:
                    glDeleteBuffers(1, [self.ebo])
                except:
                    pass
                self.ebo = None
        # ----------------------------------------------------

        # Camera
        if camera is not None:
            self.center = camera[0:3]
            self.distance = camera[3]
            self.rotation_matrix = camera[4:20].reshape(4,4).copy()
            self.pan_x = camera[20]
            self.pan_y = camera[21]
            self.is_orthographic = camera[22] > 0.5

        # Rebuild GPU data
        self.points_colors = np.hstack([self.points, self.base_colors])
        self.rebuild_ids(len(self.points))
        self.update_vbo_full()
        self.update_colors()
        self._invalidate_lod_cache()

        # Restore spatial indices
        if kdtree is not None:
            self._kdtree = kdtree
        else:
            self._kdtree = None
        if sculptor is not None:
            self.sculptor = sculptor
        else:
            self.sculptor = None

        self.status_text = f"Project loaded: {len(points):,} points"

        # ----- NEW: Adjust brush sizes to match the loaded model -----
        if self.points is not None and len(self.points) > 0:
            bbox_min = np.min(self.points, axis=0)
            bbox_max = np.max(self.points, axis=0)
            diag = float(np.linalg.norm(bbox_max - bbox_min))
            self.adjust_brush_sizes(diag)
        # --------------------------------------------------------------

    def auto_fit_camera(self):
        if getattr(self, 'points', None) is None or len(self.points) == 0:
            return
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        center = (mins + maxs) / 2.0
        radius = np.linalg.norm(maxs - center)
        fov = 60.0 
        distance = radius / math.tan(math.radians(fov / 2.0)) * 1.2
        self.distance = max(distance, 0.1)
        self.center = center
        self.pan_x = 0.0
        self.pan_y = 0.0

    # ------------------------------------------------------------------
    # Background file loader (extended for project mode)
    # ------------------------------------------------------------------
    def _start_file_load_thread(self, filepath, mode):
        """Spawn a background thread for mesh or project loading."""
        import threading, time
        self._load_thread_running = True
        self._loaded_result = None
        fname = os.path.basename(filepath)
        self.status_text = f"Loading {fname} …"

        if mode == 'project':
            self._start_project_load_thread(filepath)
            return

        # Mesh loading (existing)
        def _worker():
            try:
                t0 = time.perf_counter()
                if str(filepath).lower().endswith(('.glb', '.gltf')):
                    # GLB needs normals for mesh mode (render_mode_mesh is
                    # gated on has_normals) — utils.load_mesh_data doesn't
                    # provide them, the dedicated loader does.
                    from glb_loader import load_glb_mesh_data
                    pts, cols, norms, faces = load_glb_mesh_data(filepath)
                else:
                    pts, cols, norms, faces = load_mesh_data(filepath)
                print(f"[Loader] parse:      {time.perf_counter()-t0:.2f}s  ({len(pts) if pts is not None else 0:,} verts)")
                if pts is None or len(pts) == 0:
                    self._loaded_result = (pts, cols, norms, faces, mode, None, None, None)
                    return
                t0 = time.perf_counter()
                try:
                    from scipy.spatial import cKDTree
                    kdtree = cKDTree(pts)
                except Exception:
                    kdtree = None
                print(f"[Loader] KD-tree:    {time.perf_counter()-t0:.2f}s")
                t0 = time.perf_counter()
                try:
                    from sculpt_core import SculptSession
                    _norms = norms if norms is not None else np.zeros((len(pts), 3), dtype=np.float32)
                    sculptor = SculptSession(pts, _norms)
                except Exception:
                    sculptor = None
                print(f"[Loader] sculptor:   {time.perf_counter()-t0:.2f}s")
                self._loaded_result = (pts, cols, norms, faces, mode, None, kdtree, sculptor)
            except Exception as e:
                import traceback; traceback.print_exc()
                self._loaded_result = (None, None, None, None, mode, str(e), None, None)
            finally:
                self._load_thread_running = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _commit_loaded_result(self):
        """Called from main thread when loading finishes."""
        if self._loaded_result is None:
            return
        # Check if it's a project result (has 11 elements) or mesh result (8 elements)
        if len(self._loaded_result) == 11:
            self._commit_project_result()
            return

        pts, cols, norms, faces, mode, err, kdtree, sculptor = self._loaded_result
        self._loaded_result = None
        if err is not None:
            self.status_text = f"Error loading file: {err}"
            return
        if pts is None or len(pts) == 0:
            self.status_text = "File contained no vertices."
            return

        default_color = getattr(self, 'default_mesh_color', [0.75, 0.75, 0.75])
        has_incoming_faces = faces is not None and len(faces) > 0
        incoming_faces = faces.astype(np.uint32) if has_incoming_faces else None

        if mode == 'append':
            self.save_state("Import Object")
            if getattr(self, 'points', None) is None or len(self.points) == 0:
                self.points = pts
                self.base_colors = cols if cols is not None else np.tile(default_color, (len(pts), 1)).astype(np.float32)
                self.normals = norms if norms is not None else np.zeros((len(pts), 3), dtype=np.float32)
                self.selected = np.zeros(len(pts), dtype=bool)
                self.point_object_ids = np.zeros(len(pts), dtype=np.int32)
                self.faces = incoming_faces
                self.has_faces = has_incoming_faces
                if has_incoming_faces and (not hasattr(self, 'ebo') or self.ebo is None):
                    from OpenGL.GL import glGenBuffers
                    self.ebo = glGenBuffers(1)
            else:
                offset = len(self.points)
                min_b, max_b = np.min(pts, axis=0), np.max(pts, axis=0)
                incoming_center = (min_b + max_b) / 2.0
                target_center = self.center if getattr(self, 'center', None) is not None else np.zeros(3, dtype=np.float32)
                pts = pts + (target_center - incoming_center)
                self.points = np.vstack([self.points, pts])
                self.base_colors = np.vstack([self.base_colors, cols if cols is not None else np.tile(default_color, (len(pts), 1)).astype(np.float32)])
                self.normals = np.vstack([self.normals, norms if norms is not None else np.zeros((len(pts), 3), dtype=np.float32)])
                if has_incoming_faces:
                    shifted = (incoming_faces + offset).astype(np.uint32)
                    self.faces = np.vstack([self.faces, shifted]) if getattr(self, 'faces', None) is not None else shifted
                    self.has_faces = True
                    if not hasattr(self, 'ebo') or self.ebo is None:
                        from OpenGL.GL import glGenBuffers
                        self.ebo = glGenBuffers(1)
                if not hasattr(self, 'point_object_ids'):
                    self.point_object_ids = np.zeros(offset, dtype=np.int32)
                new_obj_id = int(np.max(self.point_object_ids)) + 1 if len(self.point_object_ids) > 0 else 1
                self.point_object_ids = np.concatenate([self.point_object_ids, np.full(len(pts), new_obj_id, dtype=np.int32)])
                new_sel = np.zeros(len(self.points), dtype=bool)
                new_sel[-len(pts):] = True
                self.selected = new_sel
                kdtree = None
                sculptor = None
            self.has_normals = getattr(self, 'has_normals', False) or (norms is not None and len(norms) > 0)
            self.render_mode_mesh = self.has_normals
            if hasattr(self, 'tool_manager'):
                self.tool_manager.set_active_tool(4)
                self.active_tool = 4
        else:  # replace
            self.points = pts
            self.base_colors = cols if cols is not None else np.tile(default_color, (len(pts), 1)).astype(np.float32)
            self.normals = norms if norms is not None else np.zeros((len(pts), 3), dtype=np.float32)
            self.faces = incoming_faces
            self.has_faces = has_incoming_faces
            self.has_normals = norms is not None and len(norms) > 0
            self.render_mode_mesh = self.has_normals
            self.selected = np.zeros(len(pts), dtype=bool)
            self.point_object_ids = np.zeros(len(pts), dtype=np.int32)
            self.rotation_matrix = _make_default_rotation()
            self.measurements = []
            self.undo_stack = []
            self.redo_stack = []
            if has_incoming_faces and (not hasattr(self, 'ebo') or self.ebo is None):
                from OpenGL.GL import glGenBuffers
                self.ebo = glGenBuffers(1)
            elif not has_incoming_faces:
                self.ebo = None
            self.auto_fit_camera()

        import time
        t0 = time.perf_counter()
        self.points_colors = np.hstack([self.points, self.base_colors])
        self.rebuild_ids(len(self.points))
        self.update_vbo_full()
        self.update_colors()
        print(f"[Loader] GL commit:  {time.perf_counter()-t0:.2f}s")

        if kdtree is not None:
            self._kdtree = kdtree
        else:
            self._kdtree = None
        if sculptor is not None:
            self.sculptor = sculptor

        n_total = len(self.points)
        n_new = len(pts)
        self.status_text = f"Loaded {n_new:,} verts (total {n_total:,})"

        # ----- NEW: Adjust brush sizes to match the loaded model -----
        if self.points is not None and len(self.points) > 0:
            bbox_min = np.min(self.points, axis=0)
            bbox_max = np.max(self.points, axis=0)
            diag = float(np.linalg.norm(bbox_max - bbox_min))
            self.adjust_brush_sizes(diag)
        # --------------------------------------------------------------

    def check_file_loader(self):
        """Check for pending files and start/commit loading."""
        if getattr(self, 'pending_file', None) and not getattr(self, '_load_thread_running', False) and getattr(self, '_loaded_result', None) is None:
            filepath = self.pending_file
            mode = self.pending_file_mode if getattr(self, 'pending_file_mode', None) else ('replace' if not getattr(self, 'append_mode', False) else 'append')
            self.pending_file = None
            self.pending_file_mode = None
            self._start_file_load_thread(filepath, mode)
        elif not getattr(self, '_load_thread_running', False) and getattr(self, '_loaded_result', None) is not None:
            self._commit_loaded_result()

    # =================================================================
    # --- NEW: Automatic brush size scaling ---
    # =================================================================
    def adjust_brush_sizes(self, bbox_diag: float):
        """
        Set brush radii of all tools proportional to the model's bounding‑box diagonal.
        Called after loading a new mesh or project.
        """
        if bbox_diag <= 0.0:
            return

        # Tool IDs: 3 = Sculpt, 5 = Brush, 17 = Smooth (Engineering)
        tool_map = {
            3:  'sculpt_radius',      # attribute name in app (legacy)
            5:  'brush_size',         # attribute in BrushTool
            17: 'brush_radius',       # attribute in SmoothTool
        }

        # Default relative sizes (as fraction of diagonal)
        defaults = {
            3:  0.05,   # sculpt: 5%
            5:  0.02,   # paint: 2%
            17: 0.05,   # engineering brush: 5%
        }

        if not hasattr(self, 'tool_manager'):
            return

        for tool_id, attr_name in tool_map.items():
            if tool_id not in self.tool_manager.tools:
                continue
            tool = self.tool_manager.tools[tool_id]
            new_val = max(0.1, bbox_diag * defaults[tool_id])
            setattr(tool, attr_name, new_val)

        # Also update the legacy sculpt_radius on the app itself (used by draw_brush_cursor etc.)
        if 3 in self.tool_manager.tools:
            self.sculpt_radius = getattr(self.tool_manager.tools[3], 'sculpt_radius', bbox_diag * 0.05)

    # =================================================================
    # --- UI BUTTON TRANSFORMS ---
    # =================================================================
    def translate_selection(self, dx, dy, dz, snapshot=False):
        if getattr(self, 'selected', None) is None or not np.any(self.selected): return
        if snapshot: self.save_state("Translate Object")
        sel = self.selected
        self.points[sel] += np.array([dx, dy, dz], dtype=np.float32)
        self.points_colors[sel, :3] = self.points[sel]
        self.vbo_dirty = True
        if snapshot and hasattr(self, 'sync_sculptor'): self.sync_sculptor()

    def rotate_selection_euler(self, rx_deg, ry_deg, rz_deg, snapshot=False):
        if getattr(self, 'selected', None) is None or not np.any(self.selected): return
        if snapshot: self.save_state("Rotate Object")
        sel = self.selected
        center = np.mean(self.points[sel], axis=0)
        pts = self.points[sel] - center
        rot = np.eye(3, dtype=np.float32)
        if rx_deg != 0:
            a = np.radians(rx_deg); rot = rot @ np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]], dtype=np.float32)
        if ry_deg != 0:
            a = np.radians(ry_deg); rot = rot @ np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]], dtype=np.float32)
        if rz_deg != 0:
            a = np.radians(rz_deg); rot = rot @ np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]], dtype=np.float32)
        self.points[sel] = (pts @ rot.T) + center
        if getattr(self, 'has_normals', False) and len(self.normals) == len(self.points):
            self.normals[sel] = self.normals[sel] @ rot.T
        self.points_colors[sel, :3] = self.points[sel]
        self.vbo_dirty = True
        if snapshot and hasattr(self, 'sync_sculptor'): self.sync_sculptor()

    def scale_selection(self, factor, snapshot=False):
        if getattr(self, 'selected', None) is None or not np.any(self.selected) or factor == 0: return
        if snapshot: self.save_state("Scale Object")
        sel = self.selected
        center = np.mean(self.points[sel], axis=0)
        self.points[sel] = center + (self.points[sel] - center) * factor
        self.points_colors[sel, :3] = self.points[sel]
        self.vbo_dirty = True
        if snapshot and hasattr(self, 'sync_sculptor'): self.sync_sculptor()

    # =================================================================
    # --- ZERO-ALLOCATION MATH (PERFECT 60 FPS DRAGGING) ---
    # =================================================================
    def begin_transform(self):
        if getattr(self, 'selected', None) is None or not np.any(self.selected):
            self._sel_slice = None
            return
        indices = np.where(self.selected)[0]
        if len(indices) == 0:
            self._sel_slice = None
            return
        if len(indices) == (indices[-1] - indices[0] + 1):
            self._sel_slice = slice(indices[0], indices[-1] + 1)
        else:
            self._sel_slice = indices
        self._orig_pts_world = self.points[self._sel_slice].copy()
        self._transform_center = np.mean(self._orig_pts_world, axis=0)
        self._orig_pts_local = self._orig_pts_world - self._transform_center
        if getattr(self, 'has_normals', False) and getattr(self, 'normals', None) is not None and len(self.normals) == len(self.points):
            self._orig_normals = self.normals[self._sel_slice].copy()
        else:
            self._orig_normals = None
        self._fast_math_buffer = np.empty_like(self._orig_pts_world)
        if self._orig_normals is not None:
            self._fast_norm_buffer = np.empty_like(self._orig_normals)

    def end_transform(self, snapshot_name="Transform Object"):
        if not hasattr(self, '_sel_slice') or self._sel_slice is None: return
        self.save_state(snapshot_name)
        if hasattr(self, 'sync_sculptor'): self.sync_sculptor()
        self._sel_slice = None
        self._rebuild_kdtree()

    def translate_selection_fast(self, dx, dy, dz):
        if not hasattr(self, '_sel_slice') or self._sel_slice is None: return
        np.add(self._orig_pts_world, np.array([dx, dy, dz], dtype=np.float32), out=self._fast_math_buffer)
        self.points[self._sel_slice] = self._fast_math_buffer
        self.points_colors[self._sel_slice, 0:3] = self._fast_math_buffer
        self.vbo_dirty = True

    def rotate_selection_fast(self, rx_deg, ry_deg, rz_deg):
        if not hasattr(self, '_sel_slice') or self._sel_slice is None: return
        rot = np.eye(3, dtype=np.float32)
        if rx_deg != 0:
            a = np.radians(rx_deg); c, s = np.cos(a), np.sin(a)
            rot = rot @ np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
        if ry_deg != 0:
            a = np.radians(ry_deg); c, s = np.cos(a), np.sin(a)
            rot = rot @ np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
        if rz_deg != 0:
            a = np.radians(rz_deg); c, s = np.cos(a), np.sin(a)
            rot = rot @ np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        np.dot(self._orig_pts_local, rot.T, out=self._fast_math_buffer)
        self._fast_math_buffer += self._transform_center
        self.points[self._sel_slice] = self._fast_math_buffer
        self.points_colors[self._sel_slice, 0:3] = self._fast_math_buffer
        if self._orig_normals is not None:
            np.dot(self._orig_normals, rot.T, out=self._fast_norm_buffer)
            self.normals[self._sel_slice] = self._fast_norm_buffer
        self.vbo_dirty = True
