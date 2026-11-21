import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import imgui
from matplotlib.path import Path
from tool_system import BaseTool

class SmartSelectionTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Smart Select"
        self.icon_key = 'selection'
        self.hotkey = 'A'

        self.mode = 0        # 0: Lasso Inside,  1: Lasso Outside
        self.operation = 1   # 1: Add,  2: Subtract
        self.select_through = False

        self.is_dragging = False
        self.lasso_pts   = []

        # Depth buffer cache ─────────────────────────────────────────────────
        # _apply_lasso() is triggered by the mouse-release callback which fires
        # inside glfw.poll_events() — AFTER glfw.swap_buffers().  At that point
        # the back-buffer depth data is implementation-defined (zeroed or stale
        # on AMD/Intel drivers).  We therefore cache the depth buffer in
        # render_3d(), which runs during draw_scene() BEFORE the swap, ensuring
        # we always compare against a valid frame.
        self._depth_cache    = None   # numpy (h, w) float32 array
        self._depth_vp_cache = None   # (x0, y0, w, h) tuple

    # ─────────────────────────────────────── lifecycle ──────────────────────

    def on_activate(self):
        self.app.status_text = "Lasso Selection: Ready  (Shift=Add, Ctrl=Sub, Del=Smooth Cut)"
        self.is_dragging  = False
        self.lasso_pts    = []
        self._depth_cache = None

    def on_deactivate(self):
        self.is_dragging  = False
        self.lasso_pts    = []
        self._depth_cache = None

    # ─────────────────────────────────────── input ──────────────────────────

    def on_key(self, key, scancode, action, mods):
        if key == glfw.KEY_DELETE and action == glfw.PRESS:
            self._execute_smooth_cut()
            return True
        return False

    def on_mouse_button(self, button, action, mods, x, y):
        if imgui.get_io().want_capture_mouse:
            return False

        y_gl = self.app.height - y

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                if mods & glfw.MOD_SHIFT:
                    self.operation = 1
                elif mods & glfw.MOD_CONTROL:
                    self.operation = 2
                else:
                    self.operation = 1
                    self.app.save_state("Clear Selection")
                    self.app.selected[:] = False
                    self.app.update_colors()

                self.app.save_state("Smart Selection")
                self.is_dragging = True
                self.lasso_pts   = [(x, y_gl)]
                return True

            elif action == glfw.RELEASE and self.is_dragging:
                self.is_dragging = False
                if len(self.lasso_pts) > 2:
                    self._apply_lasso()
                self.lasso_pts    = []
                self._depth_cache = None
                return True

        return False

    def on_mouse_move(self, x, y):
        if imgui.get_io().want_capture_mouse:
            return False

        if self.is_dragging:
            y_gl = self.app.height - y
            self.lasso_pts.append((x, y_gl))
            return True

        return False

    # ─────────────────────────────────────── depth cache ────────────────────

    def render_3d(self):
        """
        Called every frame from draw_scene() — BEFORE glfw.swap_buffers().

        When a lasso drag is active and X-ray is off we capture the full
        viewport depth buffer here.  This guarantees _apply_lasso() (which
        runs later, after the swap) always has a valid depth image.
        """
        if self.is_dragging and not self.select_through:
            try:
                vp   = glGetIntegerv(GL_VIEWPORT)
                w, h = int(vp[2]), int(vp[3])
                if w > 0 and h > 0:
                    raw = glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT)
                    self._depth_cache    = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
                    self._depth_vp_cache = (int(vp[0]), int(vp[1]), w, h)
            except Exception:
                self._depth_cache = None
        elif not self.is_dragging:
            # Release after each stroke so memory doesn't linger
            self._depth_cache = None

    # ─────────────────────────────────────── lasso ──────────────────────────

    def _apply_lasso(self):
        if not hasattr(self.app, 'points') or len(self.app.points) == 0:
            return

        # ── 1. Build combined MVP and project all points to screen space ──
        viewport = glGetIntegerv(GL_VIEWPORT)
        mv       = glGetDoublev(GL_MODELVIEW_MATRIX).T
        proj     = glGetDoublev(GL_PROJECTION_MATRIX).T
        vp_mat   = np.dot(proj, mv)

        pts_4      = np.hstack([self.app.points, np.ones((len(self.app.points), 1))])
        clip_space = np.dot(pts_4, vp_mat.T)
        w_clip     = clip_space[:, 3]

        # Only process points in front of the camera (w_clip > 0).
        # Points with w_clip <= 0 would produce NaN/Inf NDC values.
        front_mask = w_clip > 0

        # Safe NDC: divide only for front-camera points, leave others at 0
        ndc = np.zeros_like(clip_space[:, :3])
        ndc[front_mask] = clip_space[front_mask, :3] / w_clip[front_mask, np.newaxis]

        win_x = viewport[0] + (ndc[:, 0] + 1.0) * viewport[2] / 2.0
        win_y = viewport[1] + (ndc[:, 1] + 1.0) * viewport[3] / 2.0
        win_z = (ndc[:, 2] + 1.0) / 2.0

        # ── 2. Polygon containment test ───────────────────────────────────
        poly  = np.array(self.lasso_pts)
        min_x = poly[:, 0].min();  max_x = poly[:, 0].max()
        min_y = poly[:, 1].min();  max_y = poly[:, 1].max()

        bbox_mask     = (front_mask
                         & (win_x >= min_x) & (win_x <= max_x)
                         & (win_y >= min_y) & (win_y <= max_y))
        candidate_idxs = np.where(bbox_mask)[0]

        inside_idxs = np.array([], dtype=np.int64)
        if len(candidate_idxs) > 0:
            lasso_path      = Path(poly)
            pts_2d          = np.column_stack((win_x[candidate_idxs],
                                               win_y[candidate_idxs]))
            inside_mask     = lasso_path.contains_points(pts_2d)
            inside_idxs     = candidate_idxs[inside_mask]

        # ── 3. Mode: inside vs outside ────────────────────────────────────
        if self.mode == 0:      # Lasso Inside
            target_idxs = inside_idxs
        else:                   # Lasso Outside  (mode == 1)
            is_inside_full = np.zeros(len(self.app.points), dtype=bool)
            if len(inside_idxs) > 0:
                is_inside_full[inside_idxs] = True
            target_idxs = np.where(front_mask & ~is_inside_full)[0]

        if len(target_idxs) == 0:
            return

        # ── 4. X-Ray OFF: depth visibility filter ─────────────────────────
        if not self.select_through:
            # Prefer the pre-swap cached depth buffer (captured in render_3d).
            # Fall back to a live read only when no cache is available.
            if self._depth_cache is not None:
                depth_img = self._depth_cache
                _, _, dep_w, dep_h = self._depth_vp_cache
            else:
                # Fallback — may be unreliable on some drivers after swap
                dep_w = int(viewport[2])
                dep_h = int(viewport[3])
                if dep_w > 0 and dep_h > 0:
                    raw       = glReadPixels(0, 0, dep_w, dep_h,
                                             GL_DEPTH_COMPONENT, GL_FLOAT)
                    depth_img = np.frombuffer(raw, dtype=np.float32).reshape(dep_h, dep_w)
                else:
                    depth_img = None

            if depth_img is not None and len(target_idxs) > 0:
                # Sample depth at each point's projected screen position.
                # depth_img is stored bottom-row-first (GL convention),
                # matching win_y which is also measured from the bottom.
                px = np.clip(np.round(win_x[target_idxs]).astype(int), 0, dep_w - 1)
                py = np.clip(np.round(win_y[target_idxs]).astype(int), 0, dep_h - 1)

                buffer_z = depth_img[py, px]
                point_z  = win_z[target_idxs]

                # A point is visible when its depth is <= the rendered depth
                # at that pixel (plus a small tolerance for float imprecision).
                visible_mask = point_z <= (buffer_z + 0.002)
                target_idxs  = target_idxs[visible_mask]

            # Front-face culling: discard back-facing points
            if self.app.has_normals and len(target_idxs) > 0:
                cam_pos = self.app.get_camera_pos()
                dirs    = self.app.points[target_idxs] - cam_pos
                norms   = self.app.normals[target_idxs]
                # dot(vec_to_point, normal) <= 0 → normal faces camera (front)
                # threshold 0.1 permits slightly grazing normals
                front_face_mask = np.sum(dirs * norms, axis=1) <= 0.1
                target_idxs     = target_idxs[front_face_mask]

        # ── 5. Apply selection ────────────────────────────────────────────
        if len(target_idxs) > 0:
            if self.operation == 1:
                self.app.selected[target_idxs] = True
            elif self.operation == 2:
                self.app.selected[target_idxs] = False
            self.app.update_colors()

    # ─────────────────────────────────────── smooth cut ─────────────────────

    def _execute_smooth_cut(self):
        if not hasattr(self.app, 'faces') or self.app.faces is None or len(self.app.faces) == 0:
            if hasattr(self.app, 'delete_selection'):
                self.app.delete_selection()
            return

        self.app.save_state("Smooth Cut Delete")

        pts = self.app.points
        faces = self.app.faces
        sel = self.app.selected
        has_normals = hasattr(self.app, 'normals') and self.app.normals is not None

        face_sel   = sel[faces]
        sel_counts = np.sum(face_sel, axis=1)

        faces_to_keep = faces[sel_counts == 0]
        mixed_faces   = faces[(sel_counts == 1) | (sel_counts == 2)]

        new_pts          = []
        new_normals_list = []
        new_faces        = []

        edge_map       = {}
        boundary_edges = []
        base_idx       = len(pts)

        def get_or_create_edge_vertex(vA, vB):
            key = tuple(sorted((vA, vB)))
            if key in edge_map:
                return edge_map[key]

            pA, pB = pts[vA], pts[vB]
            p_new  = pA + 0.5 * (pB - pA)

            idx = base_idx + len(new_pts)
            new_pts.append(p_new)

            if has_normals:
                nA, nB  = self.app.normals[vA], self.app.normals[vB]
                n_new   = nA + 0.5 * (nB - nA)
                norm_len = np.linalg.norm(n_new)
                if norm_len > 1e-6:
                    n_new /= norm_len
                new_normals_list.append(n_new)

            edge_map[key] = idx
            return idx

        for face in mixed_faces:
            v0, v1, v2 = face
            s0, s1, s2 = sel[v0], sel[v1], sel[v2]

            if s1 == s2:
                top_v, bl_v, br_v = v0, v1, v2;  top_s = s0
            elif s0 == s2:
                top_v, bl_v, br_v = v1, v2, v0;  top_s = s1
            else:
                top_v, bl_v, br_v = v2, v0, v1;  top_s = s2

            idx_new1 = get_or_create_edge_vertex(top_v, bl_v)
            idx_new2 = get_or_create_edge_vertex(top_v, br_v)

            boundary_edges.append((idx_new1 - base_idx, idx_new2 - base_idx))

            if top_s == False:
                new_faces.append([top_v, idx_new1, idx_new2])
            else:
                new_faces.append([bl_v, br_v, idx_new1])
                new_faces.append([br_v, idx_new2, idx_new1])

        if new_pts:
            new_pts_arr = np.array(new_pts, dtype=np.float32)

            adj = {i: set() for i in range(len(new_pts_arr))}
            for i1, i2 in boundary_edges:
                adj[i1].add(i2)
                adj[i2].add(i1)

            for _ in range(5):
                temp_pts = new_pts_arr.copy()
                for i in range(len(new_pts_arr)):
                    if len(adj[i]) > 0:
                        neighbors  = list(adj[i])
                        temp_pts[i] = np.mean(new_pts_arr[neighbors], axis=0)
                new_pts_arr = temp_pts

            self.app.points = np.vstack([pts, new_pts_arr])

            if hasattr(self.app, 'base_colors') and self.app.base_colors is not None:
                color_dim      = self.app.base_colors.shape[1]
                new_base_colors = np.zeros((len(new_pts), color_dim), dtype=np.float32)
                default_col    = [0.8, 0.8, 0.8, 1.0]
                for c in range(min(color_dim, 4)):
                    new_base_colors[:, c] = default_col[c]
                self.app.base_colors = np.vstack([self.app.base_colors, new_base_colors])

            if hasattr(self.app, 'points_colors') and self.app.points_colors is not None:
                pc_dim = self.app.points_colors.shape[1]
                new_pc = np.zeros((len(new_pts), pc_dim), dtype=np.float32)
                new_pc[:, 0:3] = new_pts_arr
                if pc_dim >= 6:
                    new_pc[:, 3:6] = (new_base_colors[:, 0:3]
                                      if hasattr(self.app, 'base_colors')
                                      else [0.8, 0.8, 0.8])
                self.app.points_colors = np.vstack([self.app.points_colors, new_pc])

            if has_normals:
                new_normals_arr = np.array(new_normals_list, dtype=np.float32)
                self.app.normals = np.vstack([self.app.normals, new_normals_arr])

            if (hasattr(self.app, 'point_object_ids')
                    and self.app.point_object_ids is not None):
                new_ids = np.zeros(len(new_pts), dtype=np.int32)
                self.app.point_object_ids = np.concatenate(
                    [self.app.point_object_ids, new_ids])

            self.app.selected = np.append(
                self.app.selected, np.zeros(len(new_pts), dtype=bool))

            if len(new_faces) > 0:
                all_kept_faces = np.vstack(
                    [faces_to_keep, np.array(new_faces, dtype=np.int32)])
            else:
                all_kept_faces = faces_to_keep
            self.app.faces = all_kept_faces
        else:
            self.app.faces = faces_to_keep

        keep_mask = ~self.app.selected
        self.app.apply_mask(keep_mask)

    # ─────────────────────────────────────── 2-D overlay ────────────────────

    def render_2d_overlay(self):
        if not self.is_dragging or len(self.lasso_pts) < 2:
            return

        draw_list = imgui.get_background_draw_list()

        if self.operation == 2:
            col = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 0.9)
        else:
            col = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 0.9)

        win_h = self.app.height
        for i in range(len(self.lasso_pts) - 1):
            p1 = self.lasso_pts[i]
            p2 = self.lasso_pts[i + 1]
            draw_list.add_line(p1[0], win_h - p1[1],
                               p2[0], win_h - p2[1], col, 2.0)

        if len(self.lasso_pts) > 2:
            p1 = self.lasso_pts[-1]
            p2 = self.lasso_pts[0]
            draw_list.add_line(p1[0], win_h - p1[1],
                               p2[0], win_h - p2[1], col, 2.0)

    # ─────────────────────────────────────── panel ──────────────────────────

    def draw_options(self):
        imgui.text_colored("Smart Selection Tool", 0.4, 0.85, 1.0)
        imgui.separator()

        imgui.text("Style:")
        if imgui.radio_button("Lasso Inside##m0",   self.mode == 0): self.mode = 0
        imgui.same_line()
        if imgui.radio_button("Lasso Outside##m1",  self.mode == 1): self.mode = 1

        imgui.dummy(0, 5)
        _, self.select_through = imgui.checkbox("Select Through Mesh (X-Ray)",
                                                self.select_through)

        imgui.dummy(0, 10)
        imgui.separator()
        imgui.dummy(0, 5)

        imgui.text_disabled("Controls:")
        imgui.text_disabled("- Left Click + Drag: Replace Selection")
        imgui.text_disabled("- Shift + Drag:       Add to Selection")
        imgui.text_disabled("- Ctrl  + Drag:       Subtract from Selection")
        imgui.text_disabled("- 'Delete' key:       Smooth Cut Delete")
