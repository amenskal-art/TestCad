# tools/area_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *


class AreaMeasureTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Area Measure"
        self.icon_key = 'area'
        self.hotkey = '4'

        self.interaction_mode = 0
        self.shape_type = 0
        self.shape_names = ["Rectangle", "Circle", "Polygon", "Lasso"]

        self.drawing_points = []      # list of 3D world points (NOT vertex indices)
        self.preview_pt = None        # 3D world point under cursor
        self.active_shapes = []
        self.measurements = []

        self.selected_target = None
        self.hover_state = None
        self.drag_mode = None
        self.drag_extra = None

        self.last_drag_screen = None
        self.drag_depth = 0.0
        self.px_ratio = 1.0
        self.surface_offset = 0.1

        # Visibility: shapes hide when the tool is inactive unless this is on.
        self.visible = True
        self.show_when_inactive = False

        # Lasso point spacing in screen pixels — avoids dumping a point every
        # event (which produced jittery, overdense outlines).
        self._last_lasso_screen = None
        self._lasso_min_px = 12.0

    # ------------------------------------------------------------------ units
    def _get_scene_units(self):
        if hasattr(self.app, 'display_unit') and hasattr(self.app, 'coord_unit') and hasattr(self.app, '_UNIT_TO_MM'):
            disp_unit = self.app.display_unit
            factor = self.app._UNIT_TO_MM[self.app.coord_unit] / self.app._UNIT_TO_MM[disp_unit]
            return disp_unit, factor
        return getattr(self.app, 'unit', 'mm'), getattr(self.app, 'unit_scale', 1.0)

    # ------------------------------------------------------------- lifecycle
    def on_activate(self):
        self.visible = True
        self.hover_state = None
        if self.interaction_mode == 0:
            self.app.status_text = f"Add Mode ({self.shape_names[self.shape_type]}): Draw shapes."
        else:
            self.app.status_text = "Adjust Mode: Click shape lines or center to edit."

    def on_deactivate(self):
        self.cancel_drawing()
        # Hide on switch unless the user asked to keep them visible.
        self.visible = self.show_when_inactive

    # ----------------------------------------------------------- world pos
    def _world_at(self, x, y, prev=None):
        """
        Free 3D position under the mouse via depth buffer (NO vertex snapping).

        Rejects implausible samples (background / wrong surface) by comparing to
        `prev`; a sample more than a generous multiple of the scene scale away is
        treated as a miss and discarded, which stops the far-displacement spikes.
        """
        y_gl = self.app.height - y
        wp = None
        if hasattr(self.app, 'get_world_pos_at_mouse'):
            wp = self.app.get_world_pos_at_mouse(x, y_gl)
        if wp is None:
            return None
        wp = np.asarray(wp, dtype=np.float32)

        if prev is not None:
            prev = np.asarray(prev, dtype=np.float32)
            # Reject jumps larger than ~half the camera distance (depth glitch).
            max_jump = max(1.0, getattr(self.app, 'distance', 10.0) * 0.5)
            if np.linalg.norm(wp - prev) > max_jump:
                return None
        return wp

    # ------------------------------------------------------------------ input
    def on_mouse_button(self, button, action, mods, x, y):
        if imgui.get_io().want_capture_mouse:
            return False
        win_w, _ = glfw.get_window_size(self.app.window)
        fb_w, _ = glfw.get_framebuffer_size(self.app.window)
        self.px_ratio = fb_w / win_w if win_w > 0 else 1.0
        mx, my = x * self.px_ratio, (self.app.height - y) * self.px_ratio

        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if self.interaction_mode == 0:
                if self.shape_type == 2 and len(self.drawing_points) >= 3:
                    self._finalize_shape()
                else:
                    self.cancel_drawing()
            else:
                self.selected_target, self.hover_state = None, None
            return True

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                if self.interaction_mode == 1:
                    hit = self._hit_test(mx, my)
                    if hit:
                        mode, grp, shp, extra = hit
                        self.selected_target = (grp, shp)
                        self.drag_mode, self.drag_extra = mode, extra
                        shape = self._get_shape(grp, shp)

                        centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
                        surf_offset = shape['normal'] * self.surface_offset

                        if mode == 'vertex':
                            if shape['type'] == 1:
                                drag_pt = centroid + surf_offset + shape['local_up'] * shape['radius']
                            else:
                                drag_pt = shape['points'][extra] + surf_offset
                        elif mode == 'rotate':
                            drag_pt = centroid + surf_offset + shape['local_up'] * self._rotate_arm(shape, centroid)
                        else:
                            drag_pt = centroid + surf_offset

                        modelview, projection, viewport = glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX), glGetIntegerv(GL_VIEWPORT)
                        _, _, cz = gluProject(drag_pt[0], drag_pt[1], drag_pt[2], modelview, projection, viewport)
                        self.drag_depth, self.last_drag_screen = cz, (mx, my)
                        return True
                    self.selected_target = None
                    return False
                else:
                    wp = self._world_at(x, y)
                    if wp is None:
                        return False
                    if not self.drawing_points:
                        self.app.save_state("Draw Area Shape")

                    if self.shape_type == 0:
                        if len(self.drawing_points) == 0:
                            self.drawing_points.append(wp)
                        elif len(self.drawing_points) == 1:
                            self.drawing_points.append(wp)
                            self._finalize_shape()
                    elif self.shape_type == 1:
                        self.drawing_points, self.preview_pt = [wp], wp
                    elif self.shape_type == 2:
                        self.drawing_points.append(wp)
                    elif self.shape_type == 3:
                        self.drawing_points = [wp]
                        self._last_lasso_screen = (mx, my)
                    return True

            elif action == glfw.RELEASE:
                if self.interaction_mode == 1:
                    if self.drag_mode:
                        if self.selected_target and self.selected_target[0] >= 0:
                            self._recalc_measurement(self.selected_target[0])
                        self.drag_mode, self.app.status_text = None, "Edit applied."
                        self.hover_state = self._hit_test(mx, my)
                        return True
                else:
                    if self.shape_type == 1 and len(self.drawing_points) == 1:
                        if self.preview_pt is not None and np.linalg.norm(self.preview_pt - self.drawing_points[0]) > 1e-4:
                            self.drawing_points.append(self.preview_pt)
                            self._finalize_shape()
                        else:
                            self.cancel_drawing()
                        return True
                    elif self.shape_type == 3 and len(self.drawing_points) >= 3:
                        self._finalize_shape()
                        return True
        return False

    def on_mouse_move(self, x, y):
        mx, my = x * self.px_ratio, (self.app.height - y) * self.px_ratio
        if self.interaction_mode == 1:
            if self.drag_mode and self.selected_target:
                grp, shp_idx = self.selected_target
                shape = self._get_shape(grp, shp_idx)
                modelview, projection, viewport = glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX), glGetIntegerv(GL_VIEWPORT)
                old_wx, old_wy, old_wz = gluUnProject(self.last_drag_screen[0], self.last_drag_screen[1], self.drag_depth, modelview, projection, viewport)
                new_wx, new_wy, new_wz = gluUnProject(mx, my, self.drag_depth, modelview, projection, viewport)
                delta = np.array([new_wx - old_wx, new_wy - old_wy, new_wz - old_wz], dtype=np.float32)

                surf_offset = shape['normal'] * self.surface_offset
                centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)

                if self.drag_mode == 'translate':
                    shape['points'] += delta
                elif self.drag_mode == 'vertex':
                    if shape['type'] == 1:
                        shape['radius'] = max(0.1, np.linalg.norm(np.array([new_wx, new_wy, new_wz]) - (centroid + surf_offset)))
                    else:
                        shape['points'][self.drag_extra] += delta
                elif self.drag_mode == 'rotate':
                    rot_center = centroid + surf_offset
                    vs = np.array([old_wx, old_wy, old_wz]) - rot_center
                    ve = np.array([new_wx, new_wy, new_wz]) - rot_center
                    if np.linalg.norm(vs) > 1e-4 and np.linalg.norm(ve) > 1e-4:
                        vs, ve = vs / np.linalg.norm(vs), ve / np.linalg.norm(ve)
                        axis = shape['normal']
                        vs = vs - np.dot(vs, axis) * axis
                        ve = ve - np.dot(ve, axis) * axis
                        if np.linalg.norm(vs) > 1e-4 and np.linalg.norm(ve) > 1e-4:
                            vs, ve = vs / np.linalg.norm(vs), ve / np.linalg.norm(ve)
                            angle = math.acos(max(-1.0, min(1.0, np.dot(vs, ve))))
                            if np.dot(np.cross(vs, ve), axis) < 0:
                                angle = -angle
                            for i in range(len(shape['points'])):
                                v = shape['points'][i] - centroid
                                shape['points'][i] = centroid + v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * np.dot(axis, v) * (1 - math.cos(angle))
                            v_up = shape['local_up']
                            shape['local_up'] = v_up * math.cos(angle) + np.cross(axis, v_up) * math.sin(angle) + axis * np.dot(axis, v_up) * (1 - math.cos(angle))

                self.last_drag_screen = (mx, my)
                return True
            else:
                self.hover_state = self._hit_test(mx, my)
                return False
        else:
            wp = self._world_at(x, y, prev=self.preview_pt)
            if wp is not None:
                self.preview_pt = wp
                if self.shape_type == 3 and len(self.drawing_points) > 0:
                    # Only add a lasso point if the cursor moved enough on screen.
                    if self._last_lasso_screen is None or \
                       np.linalg.norm(np.array([mx, my]) - np.array(self._last_lasso_screen)) >= self._lasso_min_px:
                        self.drawing_points.append(wp)
                        self._last_lasso_screen = (mx, my)
            return True

    def on_key(self, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                if self.interaction_mode == 0:
                    self.cancel_drawing()
                else:
                    self.selected_target, self.hover_state = None, None
                return True
            elif key == glfw.KEY_BACKSPACE:
                if self.interaction_mode == 0 and self.drawing_points:
                    self.drawing_points.pop()
                elif self.interaction_mode == 1 and self.selected_target:
                    grp, shp = self.selected_target
                    if grp == -1:
                        self.active_shapes.pop(shp)
                    else:
                        self.measurements[grp]['shapes'].pop(shp)
                        if not self.measurements[grp]['shapes']:
                            self.measurements.pop(grp)
                        else:
                            self._recalc_measurement(grp)
                    self.selected_target, self.hover_state = None, None
                return True
            elif key == glfw.KEY_ENTER and self.interaction_mode == 0 and self.shape_type == 2 and len(self.drawing_points) >= 3:
                self._finalize_shape()
                return True
        return False

    # ------------------------------------------------------- hit testing
    def _get_shape(self, grp, shp):
        return self.active_shapes[shp] if grp == -1 else self.measurements[grp]['shapes'][shp]

    def _project_to_screen(self, pt_3d):
        m, p, v = glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX), glGetIntegerv(GL_VIEWPORT)
        sx, sy, sz = gluProject(pt_3d[0], pt_3d[1], pt_3d[2], m, p, v)
        return np.array([sx, sy]), sz

    def _point_to_segment_dist(self, p, a, b):
        l2 = np.sum((a - b) ** 2)
        if l2 == 0:
            return np.linalg.norm(p - a)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        return np.linalg.norm(p - (a + t * (b - a)))

    def _rotate_arm(self, shape, centroid):
        """
        Length of the rotate handle arm, in WORLD units, but capped so it never
        becomes a giant lever on large shapes. Previously this was r_scale*0.4
        with a floor of 2.0 — huge for big shapes — which made rotation twitchy.
        """
        if shape['type'] == 1:
            r = shape['radius']
        else:
            r = max(1e-3, np.max(np.linalg.norm(shape['points'] - centroid, axis=1)))
        # Keep the handle just outside the shape, but clamp its absolute length.
        return min(r * 1.15 + r * 0.05, r + max(1.0, getattr(self.app, 'distance', 10.0) * 0.04))

    def _node_tol_px(self):
        """
        Smart node tolerance: a base pixel radius that grows slightly when zoomed
        out so handles stay grabbable, clamped to a sane range. px_ratio-aware.
        """
        base = 14.0 * self.px_ratio
        return max(10.0 * self.px_ratio, min(base, 22.0 * self.px_ratio))

    def _line_tol_px(self):
        # Edges are lower priority than nodes; keep their grab band tight.
        return 8.0 * self.px_ratio

    def _hit_test(self, mx, my):
        """
        Priority order (nearest-wins within each tier, tiers never cross):
          1. Handles of the currently selected shape (rotate, vertex, center)
          2. Center nodes of all shapes
          3. Edges of all shapes
        Nodes always beat edges, so a tight edge band can't steal a node grab.
        """
        m_pos = np.array([mx, my])
        NODE_TOL = self._node_tol_px()
        LINE_TOL = self._line_tol_px()

        # ---- Tier 1: selected shape handles ----
        if self.selected_target is not None:
            grp, shp = self.selected_target
            shape = self._get_shape(grp, shp)
            centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
            surf_offset = shape['normal'] * self.surface_offset

            best = None
            best_d = NODE_TOL

            # rotate handle
            handle_3d = centroid + surf_offset + shape['local_up'] * self._rotate_arm(shape, centroid)
            h_screen, sz = self._project_to_screen(handle_3d)
            if 0 < sz < 1:
                d = np.linalg.norm(h_screen - m_pos)
                if d < best_d:
                    best_d, best = d, ('rotate', grp, shp, None)

            # vertex / radius handles
            if shape['type'] == 1:
                v_screen, sz = self._project_to_screen(centroid + surf_offset + shape['local_up'] * shape['radius'])
                if 0 < sz < 1:
                    d = np.linalg.norm(v_screen - m_pos)
                    if d < best_d:
                        best_d, best = d, ('vertex', grp, shp, 0)
            else:
                for i, pt in enumerate(shape['points']):
                    v_screen, sz = self._project_to_screen(pt + surf_offset)
                    if 0 < sz < 1:
                        d = np.linalg.norm(v_screen - m_pos)
                        if d < best_d:
                            best_d, best = d, ('vertex', grp, shp, i)

            # center translate (slightly wider)
            c_screen, sz = self._project_to_screen(centroid + surf_offset)
            if 0 < sz < 1:
                d = np.linalg.norm(c_screen - m_pos)
                if d < NODE_TOL + 8.0 and d < best_d + 8.0:
                    if best is None or d < best_d:
                        best_d, best = d, ('translate', grp, shp, None)

            if best is not None:
                return best

        # ---- Tier 2: center nodes of every shape ----
        best_node = None
        node_min = NODE_TOL

        def scan_centers(shapes, g_idx):
            nonlocal best_node, node_min
            for s_idx, shape in enumerate(shapes):
                surf_offset = shape['normal'] * self.surface_offset
                centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
                c_screen, sz = self._project_to_screen(centroid + surf_offset)
                if 0 < sz < 1:
                    d = np.linalg.norm(c_screen - m_pos)
                    if d < node_min:
                        node_min, best_node = d, ('translate', g_idx, s_idx, None)

        scan_centers(self.active_shapes, -1)
        for i, m in enumerate(self.measurements):
            scan_centers(m['shapes'], i)
        if best_node is not None:
            return best_node

        # ---- Tier 3: edges ----
        best_edge = None
        edge_min = LINE_TOL

        def scan_edges(shapes, g_idx):
            nonlocal best_edge, edge_min
            for s_idx, shape in enumerate(shapes):
                surf_offset = shape['normal'] * self.surface_offset
                centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
                c_screen, sz = self._project_to_screen(centroid + surf_offset)
                if shape['type'] == 1:
                    e_screen, ez = self._project_to_screen(shape['points'][0] + surf_offset + shape['local_up'] * shape['radius'])
                    if 0 < sz < 1 and 0 < ez < 1:
                        d_edge = abs(np.linalg.norm(m_pos - c_screen) - np.linalg.norm(e_screen - c_screen))
                        if d_edge < edge_min:
                            edge_min, best_edge = d_edge, ('translate', g_idx, s_idx, None)
                else:
                    pts_2d = [self._project_to_screen(pt + surf_offset) for pt in shape['points']]
                    if all(pt[1] > 0 for pt in pts_2d):
                        n = len(pts_2d)
                        for i in range(n):
                            d_edge = self._point_to_segment_dist(m_pos, pts_2d[i][0], pts_2d[(i + 1) % n][0])
                            if d_edge < edge_min:
                                edge_min, best_edge = d_edge, ('translate', g_idx, s_idx, None)

        scan_edges(self.active_shapes, -1)
        for i, m in enumerate(self.measurements):
            scan_edges(m['shapes'], i)
        return best_edge

    # ------------------------------------------------------- shape logic
    def _finalize_shape(self):
        if not self.drawing_points:
            return
        pts = np.array(self.drawing_points, dtype=np.float32)

        # Normal from the surface near the first point (depth pos has no normal),
        # fall back to averaged scene normal at nearest vertex if available.
        avg_normal = self._estimate_normal(pts)

        world_up = np.array([1, 0, 0] if abs(np.dot(avg_normal, [0, 1, 0])) > 0.99 else [0, 1, 0], dtype=np.float32)
        local_up = np.cross(avg_normal, world_up)
        local_up /= np.linalg.norm(local_up)

        shape_data = {'type': self.shape_type, 'normal': avg_normal, 'local_up': local_up, 'radius': 0.0}

        if self.shape_type == 0:
            v = pts[1] - pts[0]
            v_proj = v - np.dot(v, avg_normal) * avg_normal
            u1, u2 = local_up, np.cross(avg_normal, local_up)
            d1, d2 = np.dot(v_proj, u1), np.dot(v_proj, u2)
            shape_data['points'] = np.array([pts[0], pts[0] + d1 * u1, pts[0] + d1 * u1 + d2 * u2, pts[0] + d2 * u2])
        elif self.shape_type == 1:
            shape_data['points'], shape_data['radius'] = np.array([pts[0]]), float(np.linalg.norm(pts[1] - pts[0]))
        else:
            shape_data['points'] = pts

        self.active_shapes.append(shape_data)
        self.interaction_mode = 1
        self.selected_target = (-1, len(self.active_shapes) - 1)
        self.hover_state = None
        self.drawing_points = []
        self.preview_pt = None
        self._last_lasso_screen = None

    def _estimate_normal(self, pts):
        """Estimate a plane normal for the drawn points via PCA; default to +Z."""
        if len(pts) >= 3:
            c = pts.mean(axis=0)
            u, s, vt = np.linalg.svd(pts - c)
            n = vt[2]
            nl = np.linalg.norm(n)
            if nl > 1e-6:
                return (n / nl).astype(np.float32)
        return np.array([0, 0, 1], dtype=np.float32)

    def transform_shapes(self, center, rotation, scale, pivot_local=None):
        if pivot_local is None:
            pivot_local = np.zeros(3, dtype=np.float32)
        pivot_local = np.asarray(pivot_local, dtype=np.float64)

        def transform_point(p):
            local = rotation.T @ (np.asarray(p, dtype=np.float64) - center)
            local_scaled = (local - pivot_local) * scale + pivot_local
            return center + rotation @ local_scaled

        def transform_dir(d):
            return rotation @ d

        for shape in self.active_shapes:
            if shape['type'] != 1:
                shape['points'] = np.array([transform_point(p) for p in shape['points']])
            else:
                verts = self.get_shape_verts(shape)
                shape['type'] = 2
                shape['points'] = np.array([transform_point(v) for v in verts])
                del shape['radius']
            shape['normal'] = transform_dir(shape['normal'])
            shape['local_up'] = transform_dir(shape['local_up'])

        for meas in self.measurements:
            for shape in meas['shapes']:
                if shape['type'] != 1:
                    shape['points'] = np.array([transform_point(p) for p in shape['points']])
                else:
                    verts = self.get_shape_verts(shape)
                    shape['type'] = 2
                    shape['points'] = np.array([transform_point(v) for v in verts])
                    del shape['radius']
                shape['normal'] = transform_dir(shape['normal'])
                shape['local_up'] = transform_dir(shape['local_up'])
            self._recalc_measurement(self.measurements.index(meas))

    def apply_translation(self, offset):
        offset = np.asarray(offset, dtype=np.float32)
        for shape in self.active_shapes:
            shape['points'] = shape['points'] + offset
        for meas in self.measurements:
            for shape in meas['shapes']:
                shape['points'] = shape['points'] + offset

    def apply_rotation(self, pivot, R):
        pivot = np.asarray(pivot, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        def _rot_pt(p):
            return pivot + R @ (p - pivot)

        def _rot_dir(d):
            nd = R @ d
            n = np.linalg.norm(nd)
            return nd / n if n > 1e-8 else nd

        def _apply(shape):
            shape['points'] = np.array([_rot_pt(p) for p in shape['points']])
            shape['normal'] = _rot_dir(shape['normal'])
            shape['local_up'] = _rot_dir(shape['local_up'])
            lu = shape['local_up'] - np.dot(shape['local_up'], shape['normal']) * shape['normal']
            ln = np.linalg.norm(lu)
            if ln > 1e-8:
                shape['local_up'] = lu / ln

        for shape in self.active_shapes:
            _apply(shape)
        for meas in self.measurements:
            for shape in meas['shapes']:
                _apply(shape)
            self._recalc_measurement(self.measurements.index(meas))

    def cancel_drawing(self):
        self.drawing_points, self.preview_pt, self.drag_mode, self.hover_state = [], None, None, None
        self._last_lasso_screen = None

    def save_measurement(self):
        if not self.active_shapes:
            return
        total_area = sum(self.compute_shape_area(s) for s in self.active_shapes)
        self.measurements.append({'shapes': [s.copy() for s in self.active_shapes], 'total_area': total_area})
        self.active_shapes, self.selected_target, self.interaction_mode = [], None, 0

    def _recalc_measurement(self, group_idx):
        m = self.measurements[group_idx]
        m['total_area'] = sum(self.compute_shape_area(s) for s in m['shapes'])

    def get_normal(self, idx):
        if hasattr(self.app, 'normals') and self.app.normals is not None and idx < len(self.app.normals):
            n = self.app.normals[idx]
            if np.linalg.norm(n) > 1e-6:
                return n / np.linalg.norm(n)
        return np.array([0, 0, 1], dtype=np.float32)

    def compute_shape_area(self, shape):
        if shape['type'] == 1:
            return math.pi * (shape['radius'] ** 2)
        pts = shape['points']
        if len(pts) < 3:
            return 0.0
        normal = [0.0, 0.0, 0.0]
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            normal[0] += (p1[1] * p2[2] - p1[2] * p2[1])
            normal[1] += (p1[2] * p2[0] - p1[0] * p2[2])
            normal[2] += (p1[0] * p2[1] - p1[1] * p2[0])
        return 0.5 * math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

    def get_shape_verts(self, shape):
        if shape['type'] == 1:
            r, c, u1, u2 = shape['radius'], shape['points'][0], shape['local_up'], np.cross(shape['normal'], shape['local_up'])
            return [c + u1 * (r * math.cos(2 * math.pi * i / 32)) + u2 * (r * math.sin(2 * math.pi * i / 32)) for i in range(32)]
        return list(shape['points'])

    # ------------------------------------------------------------- render
    def render_3d(self):
        # Hide when the tool is inactive unless the user opted to keep visible.
        if not self.is_active and not self.show_when_inactive:
            return
        if not self.visible and not self.is_active:
            return

        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        unit, scale = self._get_scene_units()
        interactive = self.is_active  # only show handles when this tool is active

        def draw_shape_wireframe(shape, is_saved, is_selected, grp_idx, shp_idx):
            verts = [p + shape['normal'] * self.surface_offset for p in self.get_shape_verts(shape)]
            if not verts:
                return
            is_hovered = self.hover_state and self.hover_state[1] == grp_idx and self.hover_state[2] == shp_idx
            glLineWidth(3.0 if is_selected else 2.5 if is_hovered else 2.0)
            glColor3f(*((1.0, 0.5, 1.0) if is_selected else (1.0, 1.0, 1.0) if is_hovered else (0.0, 1.0, 1.0) if is_saved else (1.0, 1.0, 0.0)))
            glBegin(GL_LINE_LOOP)
            for p in verts:
                glVertex3f(*p)
            glEnd()
            glColor3f(*((0.0, 1.0, 1.0) if is_saved else (0.0, 1.0, 0.0)))
            glPointSize(4.0)
            glBegin(GL_POINTS)
            for p in verts:
                glVertex3f(*p)
            glEnd()

            if is_selected and interactive and self.interaction_mode == 1:
                centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
                centroid = centroid + shape['normal'] * self.surface_offset
                is_trans = self.hover_state and self.hover_state[0] == 'translate' and self.hover_state[1] == grp_idx
                is_rot = self.hover_state and self.hover_state[0] == 'rotate' and self.hover_state[1] == grp_idx

                glColor3f(1.0, 1.0, 0.5) if is_trans else glColor3f(1.0, 1.0, 1.0)
                glPointSize(16.0 if is_trans else 10.0)
                glBegin(GL_POINTS)
                glVertex3f(*centroid)
                glEnd()

                base_centroid = shape['points'][0] if shape['type'] == 1 else np.mean(shape['points'], axis=0)
                handle = centroid + shape['local_up'] * self._rotate_arm(shape, base_centroid)
                glColor3f(0.0, 1.0, 1.0) if is_rot else glColor3f(0.0, 0.7, 0.8)
                glLineWidth(4.0 if is_rot else 2.5)
                glBegin(GL_LINES)
                glVertex3f(*centroid)
                glVertex3f(*handle)
                glEnd()
                glPointSize(14.0 if is_rot else 9.0)
                glBegin(GL_POINTS)
                glVertex3f(*handle)
                glEnd()

                glColor3f(0.4, 1.0, 0.4)
                if shape['type'] == 1:
                    glPointSize(14.0 if self.hover_state and self.hover_state[0] == 'vertex' else 8.0)
                    glBegin(GL_POINTS)
                    glVertex3f(*(centroid + shape['local_up'] * shape['radius']))
                    glEnd()
                else:
                    for i, pt in enumerate(shape['points']):
                        glPointSize(14.0 if self.hover_state and self.hover_state[0] == 'vertex' and self.hover_state[3] == i else 8.0)
                        glBegin(GL_POINTS)
                        glVertex3f(*(pt + shape['normal'] * self.surface_offset))
                        glEnd()

        for i, shape in enumerate(self.active_shapes):
            draw_shape_wireframe(shape, False, self.selected_target == (-1, i), -1, i)
        for g, m in enumerate(self.measurements):
            for s, shape in enumerate(m['shapes']):
                draw_shape_wireframe(shape, True, self.selected_target == (g, s), g, s)
            if m['shapes']:
                centroid = np.mean(self.get_shape_verts(m['shapes'][0]), axis=0) + m['shapes'][0]['normal'] * self.surface_offset
                self._draw_label(centroid, f"{m['total_area'] * (scale ** 2):.3f} {unit}\u00b2", (0.0, 1.0, 1.0))

        if interactive and self.interaction_mode == 0 and self.drawing_points:
            pts = [p + n * self.surface_offset for p, n in
                   zip(self.drawing_points, [self._estimate_normal(np.array(self.drawing_points))] * len(self.drawing_points))]
            glColor3f(1.0, 1.0, 0.0)
            if self.shape_type in (2, 3) and len(pts) > 1:
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    glVertex3f(*p)
                glEnd()
            if self.preview_pt is not None and pts:
                glColor3f(1.0, 0.5, 0.0)
                glBegin(GL_LINES)
                glVertex3f(*pts[-1])
                glVertex3f(*self.preview_pt)
                glEnd()
            glColor3f(0.0, 1.0, 0.0)
            glPointSize(6.0)
            glBegin(GL_POINTS)
            for p in pts:
                glVertex3f(*p)
            glEnd()
        glPopAttrib()

    def _draw_label(self, world_pos, text, color=(1.0, 1.0, 1.0)):
        sx, sy, sz = self.app.project_point_to_screen(world_pos)
        if 0.0 <= sz <= 1.0:
            draw_list = imgui.get_background_draw_list()
            tw = imgui.calc_text_size(text).x
            draw_list.add_rect_filled(sx, sy, sx + tw + 8, sy + 20, imgui.get_color_u32_rgba(0, 0, 0, 0.7))
            draw_list.add_text(sx + 4, sy + 2, imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0), text)

    # ------------------------------------------------------------- ui
    def draw_options(self):
        unit, scale = self._get_scene_units()
        imgui.text("Area Measurement")
        imgui.separator()

        changed_add = imgui.radio_button("Add Shape", self.interaction_mode == 0)
        if changed_add and self.interaction_mode != 0:
            self.interaction_mode, self.selected_target, self.hover_state = 0, None, None
        imgui.same_line()
        changed_adj = imgui.radio_button("Adjust Shape", self.interaction_mode == 1)
        if changed_adj and self.interaction_mode != 1:
            self.interaction_mode = 1
            self.cancel_drawing()

        if self.interaction_mode == 0:
            changed, self.shape_type = imgui.combo("Shape Profile", self.shape_type, self.shape_names)
            if changed:
                self.cancel_drawing()
        else:
            imgui.text_disabled(f"Shape Profile: {self.shape_names[self.shape_type]}")

        changed, new_off = imgui.slider_float("Surface Lift", self.surface_offset * scale, 0.0, 1.0 * scale, f"%.2f {unit}")
        if changed:
            self.surface_offset = new_off / scale if scale > 0 else 0.0

        # Visibility toggle
        changed, self.show_when_inactive = imgui.checkbox("Show shapes when tool inactive", self.show_when_inactive)
        if changed:
            self.visible = True if self.show_when_inactive else self.is_active

        if self.active_shapes or self.selected_target:
            imgui.dummy(0, 10)
            if self.selected_target and self.selected_target[0] >= 0:
                area = self.measurements[self.selected_target[0]]['total_area'] * (scale ** 2)
                imgui.text(f"Selected Area: {area:.3f} {unit}\u00b2")
            else:
                area = sum(self.compute_shape_area(s) for s in self.active_shapes) * (scale ** 2)
                imgui.text(f"Active Area: {area:.3f} {unit}\u00b2")
                if imgui.button("Save New Measurement", width=-1):
                    self.save_measurement()

        if self.measurements:
            imgui.dummy(0, 5)
            if imgui.button("Clear All Measurements", width=-1):
                self.measurements, self.selected_target = [], None
