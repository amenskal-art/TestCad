# tools/volume_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *


class VolumeMeasureTool(BaseTool):
    """
    XYZ volume measurement. Mirrors AreaMeasureTool's plug-and-play model
    (Add / Adjust modes, BaseTool interface, scene-transform hooks) but builds
    3D primitives.

    Shape types:
        0 Polygon   - draw N-point base footprint, extruded along +normal
        1 Triangle  - draw 3-point base footprint, extruded along +normal
        2 Sphere    - center + radius
        3 Cylinder  - center + radius, extruded along +normal
        4 Function  - area-under-curve solid of revolution of y=f(x) over [a,b]

    Every non-sphere shape is stored as an explicit VERTEX CAGE so any corner
    can be dragged individually; the edges meeting at that corner follow while
    all other corners stay static. Volume is computed from the cage via the
    divergence theorem (signed-tetra sum over a triangulated hull), so it stays
    correct after free dragging, GUI rotation and scaling.
    """

    def __init__(self, app):
        super().__init__(app)
        self.name = "Volume Measure"
        self.icon_key = 'volume'
        self.hotkey = '5'

        self.interaction_mode = 0          # 0 = Add, 1 = Adjust
        self.shape_type = 0
        self.shape_names = ["Polygon", "Triangle", "Sphere", "Cylinder", "Function"]

        self.drawing_points = []
        self.preview_pt = None
        self.active_shapes = []
        self.measurements = []

        self.selected_target = None
        self.hover_state = None
        self.drag_mode = None              # 'corner' | 'translate' | 'radius' | 'scale'
        self.drag_extra = None             # corner index for 'corner'

        self.last_drag_screen = None
        self.drag_depth = 0.0
        self.px_ratio = 1.0
        self.surface_offset = 0.1

        self.default_height = 1.0
        self.rotate_step_deg = 15.0
        self.scale_step = 1.1

        # Function-shape parameters (area-under-curve solid of revolution).
        self.func_names = ["x^2", "sin(x)", "sqrt(x)", "e^(-x)", "1/(1+x^2)"]
        self.func_index = 0
        self.func_a = 0.0
        self.func_b = 2.0
        self.func_samples = 24

        self.visible = True
        self.show_when_inactive = False

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
            self.app.status_text = f"Add Mode ({self.shape_names[self.shape_type]}): Draw volumes."
        else:
            self.app.status_text = "Adjust Mode: Drag corners freely; use panel to rotate/scale."

    def on_deactivate(self):
        self.cancel_drawing()
        self.visible = self.show_when_inactive

    # ----------------------------------------------------------- world pos
    def _world_at(self, x, y, prev=None):
        y_gl = self.app.height - y
        wp = None
        if hasattr(self.app, 'get_world_pos_at_mouse'):
            wp = self.app.get_world_pos_at_mouse(x, y_gl)
        if wp is None:
            return None
        wp = np.asarray(wp, dtype=np.float32)
        if prev is not None:
            prev = np.asarray(prev, dtype=np.float32)
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
                if self.shape_type in (0, 1) and len(self.drawing_points) >= 3:
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
                        anchor = self._handle_world_pos(shape, mode, extra)
                        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                        projection = glGetDoublev(GL_PROJECTION_MATRIX)
                        viewport = glGetIntegerv(GL_VIEWPORT)
                        _, _, cz = gluProject(anchor[0], anchor[1], anchor[2], modelview, projection, viewport)
                        self.drag_depth, self.last_drag_screen = cz, (mx, my)
                        return True
                    self.selected_target = None
                    return False
                else:
                    wp = self._world_at(x, y)
                    if wp is None:
                        return False
                    if not self.drawing_points:
                        self.app.save_state("Draw Volume Shape")

                    if self.shape_type in (0, 1):          # Polygon / Triangle
                        self.drawing_points.append(wp)
                        if self.shape_type == 1 and len(self.drawing_points) == 3:
                            self._finalize_shape()
                    elif self.shape_type in (2, 3):        # Sphere / Cylinder
                        self.drawing_points, self.preview_pt = [wp], wp
                    elif self.shape_type == 4:             # Function solid
                        self.drawing_points = [wp]
                        self._finalize_shape()
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
                    if self.shape_type in (2, 3) and len(self.drawing_points) == 1:
                        if self.preview_pt is not None and np.linalg.norm(self.preview_pt - self.drawing_points[0]) > 1e-4:
                            self.drawing_points.append(self.preview_pt)
                            self._finalize_shape()
                        else:
                            self.cancel_drawing()
                        return True
        return False

    def on_mouse_move(self, x, y):
        mx, my = x * self.px_ratio, (self.app.height - y) * self.px_ratio
        if self.interaction_mode == 1:
            if self.drag_mode and self.selected_target:
                grp, shp_idx = self.selected_target
                shape = self._get_shape(grp, shp_idx)
                modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                projection = glGetDoublev(GL_PROJECTION_MATRIX)
                viewport = glGetIntegerv(GL_VIEWPORT)
                old_w = np.array(gluUnProject(self.last_drag_screen[0], self.last_drag_screen[1], self.drag_depth, modelview, projection, viewport), dtype=np.float32)
                new_w = np.array(gluUnProject(mx, my, self.drag_depth, modelview, projection, viewport), dtype=np.float32)
                delta = new_w - old_w
                self._apply_drag(shape, delta, old_w, new_w)
                self.last_drag_screen = (mx, my)
                return True
            else:
                self.hover_state = self._hit_test(mx, my)
                return False
        else:
            wp = self._world_at(x, y, prev=self.preview_pt)
            if wp is not None:
                self.preview_pt = wp
            return True

    def _apply_drag(self, shape, delta, old_w, new_w):
        if self.drag_mode == 'translate':
            self._shape_center_move(shape, delta)
        elif self.drag_mode == 'corner':
            if shape['type'] != 2:
                shape['verts'][self.drag_extra] = shape['verts'][self.drag_extra] + delta
        elif self.drag_mode == 'radius':
            shape['radius'] = max(0.05, float(np.linalg.norm(new_w - shape['center'])))
        elif self.drag_mode == 'scale':
            c = self._shape_center(shape)
            d_old = np.linalg.norm(old_w - c)
            d_new = np.linalg.norm(new_w - c)
            if d_old > 1e-5:
                self._scale_shape(shape, d_new / d_old)

    # ------------------------------------------------------------------- keys
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
            elif key == glfw.KEY_ENTER and self.interaction_mode == 0 and self.shape_type == 0 and len(self.drawing_points) >= 3:
                self._finalize_shape()
                return True
        return False

    # ------------------------------------------------------- shape helpers
    def _get_shape(self, grp, shp):
        return self.active_shapes[shp] if grp == -1 else self.measurements[grp]['shapes'][shp]

    def _shape_center(self, shape):
        if shape['type'] == 2:
            return shape['center']
        return shape['verts'].mean(axis=0)

    def _shape_center_move(self, shape, delta):
        if shape['type'] == 2:
            shape['center'] = shape['center'] + delta
        else:
            shape['verts'] = shape['verts'] + delta

    def _scale_shape(self, shape, factor):
        factor = max(0.01, float(factor))
        if shape['type'] == 2:
            shape['radius'] = max(0.05, shape['radius'] * factor)
        else:
            c = shape['verts'].mean(axis=0)
            shape['verts'] = c + (shape['verts'] - c) * factor

    def _scale_handle_pt(self, shape):
        c = self._shape_center(shape)
        if shape['type'] == 2:
            return c + shape['local_up'] * (shape['radius'] * 1.3 + 0.1)
        ext = float(np.max(np.linalg.norm(shape['verts'] - c, axis=1)))
        return c + shape['local_up'] * (ext * 1.25 + 0.1)

    def _radius_arm_pt(self, shape):
        return shape['center'] + shape['local_up'] * shape['radius']

    def _handle_world_pos(self, shape, mode, extra):
        if mode == 'translate':
            return self._shape_center(shape).copy()
        if mode == 'radius':
            return self._radius_arm_pt(shape)
        if mode == 'scale':
            return self._scale_handle_pt(shape)
        if mode == 'corner':
            return shape['verts'][extra].copy()
        return self._shape_center(shape).copy()

    # ------------------------------------------------------- hit testing
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

    def _node_tol_px(self):
        base = 14.0 * self.px_ratio
        return max(10.0 * self.px_ratio, min(base, 22.0 * self.px_ratio))

    def _line_tol_px(self):
        return 8.0 * self.px_ratio

    def _hit_test(self, mx, my):
        m_pos = np.array([mx, my])
        NODE_TOL = self._node_tol_px()
        LINE_TOL = self._line_tol_px()

        # ---- Tier 1: handles of the selected shape ----
        if self.selected_target is not None:
            grp, shp = self.selected_target
            shape = self._get_shape(grp, shp)
            best = None
            best_d = NODE_TOL

            def consider(world_pt, payload):
                nonlocal best, best_d
                scr, sz = self._project_to_screen(world_pt)
                if 0 < sz < 1:
                    d = np.linalg.norm(scr - m_pos)
                    if d < best_d:
                        best_d, best = d, payload

            consider(self._scale_handle_pt(shape), ('scale', grp, shp, None))

            if shape['type'] == 2:
                consider(self._radius_arm_pt(shape), ('radius', grp, shp, None))
            else:
                for i in range(len(shape['verts'])):
                    consider(shape['verts'][i], ('corner', grp, shp, i))

            c_screen, sz = self._project_to_screen(self._shape_center(shape))
            if 0 < sz < 1:
                d = np.linalg.norm(c_screen - m_pos)
                if d < NODE_TOL + 8.0 and d < best_d + 8.0:
                    if best is None or d < best_d:
                        best_d, best = d, ('translate', grp, shp, None)

            if best is not None:
                return best

        # ---- Tier 2: center nodes ----
        best_node = None
        node_min = NODE_TOL

        def scan_centers(shapes, g_idx):
            nonlocal best_node, node_min
            for s_idx, shape in enumerate(shapes):
                c_screen, sz = self._project_to_screen(self._shape_center(shape))
                if 0 < sz < 1:
                    d = np.linalg.norm(c_screen - m_pos)
                    if d < node_min:
                        node_min, best_node = d, ('translate', g_idx, s_idx, None)

        scan_centers(self.active_shapes, -1)
        for i, m in enumerate(self.measurements):
            scan_centers(m['shapes'], i)
        if best_node is not None:
            return best_node

        # ---- Tier 3: silhouette edges ----
        best_edge = None
        edge_min = LINE_TOL

        def scan_edges(shapes, g_idx):
            nonlocal best_edge, edge_min
            for s_idx, shape in enumerate(shapes):
                for a3, b3 in self._silhouette_segments(shape):
                    a2, az = self._project_to_screen(a3)
                    b2, bz = self._project_to_screen(b3)
                    if 0 < az < 1 and 0 < bz < 1:
                        d_edge = self._point_to_segment_dist(m_pos, a2, b2)
                        if d_edge < edge_min:
                            edge_min, best_edge = d_edge, ('translate', g_idx, s_idx, None)

        scan_edges(self.active_shapes, -1)
        for i, m in enumerate(self.measurements):
            scan_edges(m['shapes'], i)
        return best_edge

    # ------------------------------------------------------- shape build
    def _basis_from_normal(self, normal):
        world_up = np.array([1, 0, 0] if abs(np.dot(normal, [0, 1, 0])) > 0.99 else [0, 1, 0], dtype=np.float32)
        up = np.cross(normal, world_up)
        ln = np.linalg.norm(up)
        up = up / ln if ln > 1e-8 else np.array([1, 0, 0], dtype=np.float32)
        right = np.cross(normal, up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-8 else right
        return up, right

    def _make_cage(self, base_ring, normal, height):
        base = np.array(base_ring, dtype=np.float32)
        top = base + normal * height
        return np.vstack([base, top]).astype(np.float32)

    def _finalize_shape(self):
        if not self.drawing_points:
            return
        pts = np.array(self.drawing_points, dtype=np.float32)
        normal = self._estimate_normal(pts)
        local_up, right = self._basis_from_normal(normal)
        h = float(self.default_height)

        shape = {'type': self.shape_type, 'normal': normal, 'local_up': local_up}

        if self.shape_type in (0, 1):
            base_ring = [p for p in pts]
            shape['verts'] = self._make_cage(base_ring, normal, h)
            shape['n_ring'] = len(base_ring)
        elif self.shape_type == 3:
            center = pts[0]
            radius = max(0.05, float(np.linalg.norm(pts[1] - pts[0])))
            seg = 24
            base_ring = [center + local_up * (radius * math.cos(2 * math.pi * i / seg))
                         + right * (radius * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
            shape['verts'] = self._make_cage(base_ring, normal, h)
            shape['n_ring'] = seg
        elif self.shape_type == 4:
            shape = self._build_function_shape(pts[0], normal, local_up, right)
        else:
            shape['center'] = pts[0].astype(np.float32)
            shape['radius'] = max(0.05, float(np.linalg.norm(pts[1] - pts[0])))

        self.active_shapes.append(shape)
        self.interaction_mode = 1
        self.selected_target = (-1, len(self.active_shapes) - 1)
        self.hover_state = None
        self.drawing_points = []
        self.preview_pt = None

    def _eval_func(self, idx, x):
        if idx == 0:
            return x * x
        if idx == 1:
            return max(0.0, math.sin(x))
        if idx == 2:
            return math.sqrt(x) if x >= 0 else 0.0
        if idx == 3:
            return math.exp(-x)
        return 1.0 / (1.0 + x * x)

    def _build_function_shape(self, center, normal, local_up, right):
        a, b = self.func_a, self.func_b
        if b < a:
            a, b = b, a
        n = max(2, int(self.func_samples))
        xs = [a + (b - a) * i / n for i in range(n + 1)]
        ring_seg = 16
        verts = []
        for x in xs:
            r = abs(self._eval_func(self.func_index, x))
            cx = center + right * x
            for j in range(ring_seg):
                a_ang = 2 * math.pi * j / ring_seg
                off = local_up * (r * math.cos(a_ang)) + normal * (r * math.sin(a_ang))
                verts.append(cx + off)
        return {
            'type': 4,
            'normal': normal,
            'local_up': local_up,
            'verts': np.array(verts, dtype=np.float32),
            'n_ring': ring_seg,
            'func_index': self.func_index,
            'func_a': a,
            'func_b': b,
        }

    def _estimate_normal(self, pts):
        if len(pts) >= 3:
            c = pts.mean(axis=0)
            u, s, vt = np.linalg.svd(pts - c)
            n = vt[2]
            nl = np.linalg.norm(n)
            if nl > 1e-6:
                return (n / nl).astype(np.float32)
        if hasattr(self.app, 'view_normal'):
            n = np.asarray(self.app.view_normal, dtype=np.float32)
            nl = np.linalg.norm(n)
            if nl > 1e-6:
                return n / nl
        return np.array([0, 0, 1], dtype=np.float32)

    # ------------------------------------------------- GUI rotate / scale
    def _rotate_selected(self, axis_world, deg):
        if not (self.interaction_mode == 1 and self.selected_target):
            return
        grp, shp = self.selected_target
        shape = self._get_shape(grp, shp)
        ang = math.radians(deg)
        axis = np.asarray(axis_world, dtype=np.float32)
        an = np.linalg.norm(axis)
        if an < 1e-8:
            return
        axis = axis / an
        c = self._shape_center(shape)
        ca, sa = math.cos(ang), math.sin(ang)

        def rot(v):
            return v * ca + np.cross(axis, v) * sa + axis * np.dot(axis, v) * (1 - ca)

        if shape['type'] != 2:
            shape['verts'] = np.array([c + rot(p - c) for p in shape['verts']], dtype=np.float32)
        shape['normal'] = rot(shape['normal'])
        nl = np.linalg.norm(shape['normal'])
        if nl > 1e-8:
            shape['normal'] /= nl
        shape['local_up'] = rot(shape['local_up'])
        lu = shape['local_up'] - np.dot(shape['local_up'], shape['normal']) * shape['normal']
        ln = np.linalg.norm(lu)
        if ln > 1e-8:
            shape['local_up'] = lu / ln
        if grp >= 0:
            self._recalc_measurement(grp)

    def _scale_selected(self, factor):
        if not (self.interaction_mode == 1 and self.selected_target):
            return
        grp, shp = self.selected_target
        self._scale_shape(self._get_shape(grp, shp), factor)
        if grp >= 0:
            self._recalc_measurement(grp)

    # ----------------------------------------------------- transforms
    def transform_shapes(self, center, rotation, scale, pivot_local=None):
        if pivot_local is None:
            pivot_local = np.zeros(3, dtype=np.float32)
        pivot_local = np.asarray(pivot_local, dtype=np.float64)
        s_iso = float(np.mean(scale)) if np.ndim(scale) else float(scale)

        def transform_point(p):
            local = rotation.T @ (np.asarray(p, dtype=np.float64) - center)
            local_scaled = (local - pivot_local) * scale + pivot_local
            return (center + rotation @ local_scaled).astype(np.float32)

        def transform_dir(d):
            nd = rotation @ np.asarray(d, dtype=np.float64)
            n = np.linalg.norm(nd)
            return (nd / n).astype(np.float32) if n > 1e-8 else nd.astype(np.float32)

        def apply(shape):
            if shape['type'] == 2:
                shape['center'] = transform_point(shape['center'])
                shape['radius'] *= s_iso
            else:
                shape['verts'] = np.array([transform_point(p) for p in shape['verts']], dtype=np.float32)
            shape['normal'] = transform_dir(shape['normal'])
            shape['local_up'] = transform_dir(shape['local_up'])

        for shape in self.active_shapes:
            apply(shape)
        for meas in self.measurements:
            for shape in meas['shapes']:
                apply(shape)
            self._recalc_measurement(self.measurements.index(meas))

    def apply_translation(self, offset):
        offset = np.asarray(offset, dtype=np.float32)
        for shape in self.active_shapes:
            self._shape_center_move(shape, offset)
        for meas in self.measurements:
            for shape in meas['shapes']:
                self._shape_center_move(shape, offset)

    def apply_rotation(self, pivot, R):
        pivot = np.asarray(pivot, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        def _rot_dir(d):
            nd = R @ d
            n = np.linalg.norm(nd)
            return nd / n if n > 1e-8 else nd

        def _apply(shape):
            if shape['type'] == 2:
                shape['center'] = pivot + R @ (shape['center'] - pivot)
            else:
                shape['verts'] = np.array([pivot + R @ (p - pivot) for p in shape['verts']], dtype=np.float32)
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

    def save_measurement(self):
        if not self.active_shapes:
            return
        total_vol = sum(self.compute_shape_volume(s) for s in self.active_shapes)
        self.measurements.append({'shapes': [self._copy_shape(s) for s in self.active_shapes], 'total_volume': total_vol})
        self.active_shapes, self.selected_target, self.interaction_mode = [], None, 0

    def _copy_shape(self, s):
        c = dict(s)
        c['normal'] = s['normal'].copy()
        c['local_up'] = s['local_up'].copy()
        if s['type'] == 2:
            c['center'] = s['center'].copy()
        else:
            c['verts'] = s['verts'].copy()
        return c

    def _recalc_measurement(self, group_idx):
        m = self.measurements[group_idx]
        m['total_volume'] = sum(self.compute_shape_volume(s) for s in m['shapes'])

    # --------------------------------------------------- volume
    def compute_shape_volume(self, shape):
        if shape['type'] == 2:
            return (4.0 / 3.0) * math.pi * (shape['radius'] ** 3)
        if shape['type'] == 4 and 'func_index' in shape:
            a, b, fi = shape['func_a'], shape['func_b'], shape['func_index']
            n = 200
            h = (b - a) / n if n else 0.0
            total = 0.0
            for i in range(n):
                x0, x1 = a + i * h, a + (i + 1) * h
                f0, f1 = self._eval_func(fi, x0), self._eval_func(fi, x1)
                total += 0.5 * (f0 * f0 + f1 * f1) * h
            return math.pi * total
        return self._cage_volume(shape)

    def _cage_volume(self, shape):
        verts = shape['verts']
        nr = shape.get('n_ring', len(verts) // 2)
        if len(verts) < 6 or nr < 3:
            return 0.0
        is_stacked = (shape['type'] == 4) or (len(verts) > 2 * nr)
        tris = []
        if not is_stacked:
            base = verts[:nr]
            top = verts[nr:2 * nr]
            bc = base.mean(axis=0)
            tc = top.mean(axis=0)
            for i in range(nr):
                j = (i + 1) % nr
                tris.append((bc, base[j], base[i]))
                tris.append((tc, top[i], top[j]))
                tris.append((base[i], base[j], top[j]))
                tris.append((base[i], top[j], top[i]))
        else:
            rings = len(verts) // nr
            for r in range(rings - 1):
                a0, a1 = r * nr, (r + 1) * nr
                for i in range(nr):
                    j = (i + 1) % nr
                    tris.append((verts[a0 + i], verts[a0 + j], verts[a1 + j]))
                    tris.append((verts[a0 + i], verts[a1 + j], verts[a1 + i]))
            c0 = verts[:nr].mean(axis=0)
            base_e = (rings - 1) * nr
            cE = verts[base_e:base_e + nr].mean(axis=0)
            for i in range(nr):
                j = (i + 1) % nr
                tris.append((c0, verts[j], verts[i]))
                tris.append((cE, verts[base_e + i], verts[base_e + j]))

        vol6 = 0.0
        for a, b, c in tris:
            vol6 += np.dot(a, np.cross(b, c))
        return abs(vol6) / 6.0

    # --------------------------------------------------- geometry for render
    def _silhouette_segments(self, shape):
        segs = []
        if shape['type'] == 2:
            c, r = shape['center'], shape['radius']
            up, right = self._basis_from_normal(shape['normal'])
            n = shape['normal']
            for a_axis, b_axis in ((up, right), (up, n), (right, n)):
                ring = [c + a_axis * (r * math.cos(2 * math.pi * i / 16))
                        + b_axis * (r * math.sin(2 * math.pi * i / 16)) for i in range(16)]
                for i in range(16):
                    segs.append((ring[i], ring[(i + 1) % 16]))
            return segs
        verts = shape['verts']
        nr = shape.get('n_ring', len(verts) // 2)
        if shape['type'] == 4 and 'func_index' in shape:
            rings = len(verts) // nr
            for r in range(rings):
                for i in range(0, nr, 4):
                    segs.append((verts[r * nr + i], verts[r * nr + (i + 1) % nr]))
            for i in range(0, nr, 4):
                for r in range(rings - 1):
                    segs.append((verts[r * nr + i], verts[(r + 1) * nr + i]))
            return segs
        base = verts[:nr]
        top = verts[nr:2 * nr]
        for i in range(nr):
            segs.append((base[i], base[(i + 1) % nr]))
            segs.append((top[i], top[(i + 1) % nr]))
            segs.append((base[i], top[i]))
        return segs

    # ------------------------------------------------------------- render
    def render_3d(self):
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
        interactive = self.is_active

        for i, shape in enumerate(self.active_shapes):
            self._draw_shape(shape, False, self.selected_target == (-1, i), -1, i, interactive)
        for g, m in enumerate(self.measurements):
            for s, shape in enumerate(m['shapes']):
                self._draw_shape(shape, True, self.selected_target == (g, s), g, s, interactive)
            if m['shapes']:
                lbl_pt = self._shape_center(m['shapes'][0]) + m['shapes'][0]['normal'] * self.surface_offset
                self._draw_label(lbl_pt, f"{m['total_volume'] * (scale ** 3):.3f} {unit}\u00b3", (0.0, 1.0, 1.0))

        if interactive and self.interaction_mode == 0 and self.drawing_points:
            glColor3f(1.0, 1.0, 0.0)
            if self.shape_type in (0, 1) and len(self.drawing_points) > 1:
                glBegin(GL_LINE_STRIP)
                for p in self.drawing_points:
                    glVertex3f(*p)
                glEnd()
            glPointSize(6.0)
            glBegin(GL_POINTS)
            for p in self.drawing_points:
                glVertex3f(*p)
            glEnd()
            if self.preview_pt is not None:
                glColor3f(1.0, 0.5, 0.0)
                glBegin(GL_LINES)
                glVertex3f(*self.drawing_points[-1])
                glVertex3f(*self.preview_pt)
                glEnd()
                if self.shape_type in (2, 3):
                    self._draw_preview_ring(self.drawing_points[0], float(np.linalg.norm(self.preview_pt - self.drawing_points[0])))
        glPopAttrib()

    def _draw_preview_ring(self, center, radius, segments=32):
        n = self._estimate_normal(np.array([center]))
        up, right = self._basis_from_normal(n)
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            a = 2 * math.pi * i / segments
            glVertex3f(*(center + up * (radius * math.cos(a)) + right * (radius * math.sin(a))))
        glEnd()

    def _draw_shape(self, shape, is_saved, is_selected, grp_idx, shp_idx, interactive):
        is_hovered = self.hover_state and self.hover_state[1] == grp_idx and self.hover_state[2] == shp_idx
        glLineWidth(3.0 if is_selected else 2.5 if is_hovered else 2.0)
        glColor3f(*((1.0, 0.5, 1.0) if is_selected
                    else (1.0, 1.0, 1.0) if is_hovered
                    else (0.0, 1.0, 1.0) if is_saved
                    else (1.0, 1.0, 0.0)))

        for a, b in self._silhouette_segments(shape):
            glBegin(GL_LINES)
            glVertex3f(*a)
            glVertex3f(*b)
            glEnd()

        if shape['type'] != 2:
            glColor3f(0.0, 1.0, 1.0) if is_saved else glColor3f(0.0, 1.0, 0.0)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            for p in shape['verts']:
                glVertex3f(*p)
            glEnd()

        if is_selected and interactive and self.interaction_mode == 1:
            self._draw_handles(shape, grp_idx)

    def _draw_handles(self, shape, grp_idx):
        hs = self.hover_state
        center = self._shape_center(shape)

        is_trans = hs and hs[0] == 'translate' and hs[1] == grp_idx
        glColor3f(1.0, 1.0, 0.5) if is_trans else glColor3f(1.0, 1.0, 1.0)
        glPointSize(16.0 if is_trans else 10.0)
        glBegin(GL_POINTS)
        glVertex3f(*center)
        glEnd()

        is_sc = hs and hs[0] == 'scale' and hs[1] == grp_idx
        sp = self._scale_handle_pt(shape)
        glColor3f(0.2, 1.0, 0.6) if is_sc else glColor3f(0.1, 0.7, 0.4)
        glLineWidth(4.0 if is_sc else 2.5)
        glBegin(GL_LINES)
        glVertex3f(*center)
        glVertex3f(*sp)
        glEnd()
        glPointSize(15.0 if is_sc else 9.0)
        glBegin(GL_POINTS)
        glVertex3f(*sp)
        glEnd()

        if shape['type'] == 2:
            is_r = hs and hs[0] == 'radius'
            glColor3f(0.4, 1.0, 0.4)
            glPointSize(14.0 if is_r else 8.0)
            glBegin(GL_POINTS)
            glVertex3f(*self._radius_arm_pt(shape))
            glEnd()
        else:
            for i, p in enumerate(shape['verts']):
                hovered = hs and hs[0] == 'corner' and hs[3] == i
                glColor3f(1.0, 0.9, 0.2) if hovered else glColor3f(0.4, 1.0, 0.4)
                glPointSize(14.0 if hovered else 7.0)
                glBegin(GL_POINTS)
                glVertex3f(*p)
                glEnd()

    def _draw_label(self, world_pos, text, color=(1.0, 1.0, 1.0)):
        sx, sy, sz = self.app.project_point_to_screen(world_pos)
        if 0.0 <= sz <= 1.0:
            draw_list = imgui.get_background_draw_list()
            tw = imgui.calc_text_size(text).x
            draw_list.add_rect_filled(sx, sy, sx + tw + 8, sy + 20, imgui.get_color_u32_rgba(0, 0, 0, 0.7))
            draw_list.add_text(sx + 4, sy + 2, imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0), text)

    # ------------------------------------------------------- gui icons
    def _half_circle_button(self, label, direction, size=34.0):
        """
        Half-circle arc icon button. `direction` in {'up','down','left','right'}
        controls which way the arc opens (and the implied rotation sense).
        Returns True on click.
        """
        clicked = imgui.button(f"##{label}", width=size, height=size)
        dl = imgui.get_window_draw_list()
        rx_min, ry_min = imgui.get_item_rect_min()
        cx = rx_min + size * 0.5
        cy = ry_min + size * 0.5
        rad = size * 0.30
        col = imgui.get_color_u32_rgba(0.2, 0.9, 1.0, 1.0)
        base = {'up': math.pi, 'down': 0.0, 'left': math.pi * 0.5, 'right': -math.pi * 0.5}[direction]
        segs = 16
        pts = []
        for i in range(segs + 1):
            a = base + math.pi * i / segs
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        for i in range(segs):
            dl.add_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], col, 2.0)
        ex, ey = pts[-1]
        dl.add_circle_filled(ex, ey, 3.0, col)
        return clicked

    # ------------------------------------------------------------- ui
    def draw_options(self):
        unit, scale = self._get_scene_units()
        imgui.text("Volume Measurement (XYZ)")
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
            changed, self.shape_type = imgui.combo("Volume Profile", self.shape_type, self.shape_names)
            if changed:
                self.cancel_drawing()
            if self.shape_type == 0:
                imgui.text_disabled("Click points; Right-click / Enter to close.")
            elif self.shape_type == 1:
                imgui.text_disabled("Click 3 points to form a triangle prism.")
            elif self.shape_type == 4:
                ch, self.func_index = imgui.combo("f(x)", self.func_index, self.func_names)
                ch2, self.func_a = imgui.input_float("a (from)", self.func_a)
                ch3, self.func_b = imgui.input_float("b (to)", self.func_b)
                imgui.text_disabled("Click once to place the solid of revolution.")
        else:
            imgui.text_disabled(f"Volume Profile: {self.shape_names[self.shape_type]}")

        changed, new_h = imgui.slider_float("Default Height", self.default_height * scale, 0.05 * scale, 20.0 * scale, f"%.2f {unit}")
        if changed:
            self.default_height = new_h / scale if scale > 0 else 0.05

        changed, new_off = imgui.slider_float("Surface Lift", self.surface_offset * scale, 0.0, 1.0 * scale, f"%.2f {unit}")
        if changed:
            self.surface_offset = new_off / scale if scale > 0 else 0.0

        # ---- Rotate / Scale controls (replace the 3D lever) ----
        if self.interaction_mode == 1 and self.selected_target:
            grp, shp = self.selected_target
            shape = self._get_shape(grp, shp)
            n = shape['normal']
            up = shape['local_up']
            right = np.cross(n, up)

            imgui.dummy(0, 8)
            imgui.text("Rotate")
            step = self.rotate_step_deg

            if self._half_circle_button("rot_up", 'up'):
                self._rotate_selected(right, step)
            imgui.same_line()
            if self._half_circle_button("rot_down", 'down'):
                self._rotate_selected(right, -step)
            imgui.same_line()
            if self._half_circle_button("rot_left", 'left'):
                self._rotate_selected(n, step)
            imgui.same_line()
            if self._half_circle_button("rot_right", 'right'):
                self._rotate_selected(n, -step)

            if imgui.button("Roll +", width=78):
                self._rotate_selected(up, step)
            imgui.same_line()
            if imgui.button("Roll -", width=78):
                self._rotate_selected(up, -step)

            ch, self.rotate_step_deg = imgui.slider_float("Step", self.rotate_step_deg, 1.0, 90.0, "%.0f deg")

            imgui.dummy(0, 6)
            imgui.text("Scale")
            if imgui.button("Scale +", width=78):
                self._scale_selected(self.scale_step)
            imgui.same_line()
            if imgui.button("Scale -", width=78):
                self._scale_selected(1.0 / self.scale_step)
            ch, self.scale_step = imgui.slider_float("Factor", self.scale_step, 1.01, 2.0, "x%.2f")
            imgui.text_disabled("Or drag the green scale knob on the shape.")

        changed, self.show_when_inactive = imgui.checkbox("Show shapes when tool inactive", self.show_when_inactive)
        if changed:
            self.visible = True if self.show_when_inactive else self.is_active

        if self.active_shapes or self.selected_target:
            imgui.dummy(0, 10)
            if self.selected_target and self.selected_target[0] >= 0:
                vol = self.measurements[self.selected_target[0]]['total_volume'] * (scale ** 3)
                imgui.text(f"Selected Volume: {vol:.3f} {unit}\u00b3")
            else:
                vol = sum(self.compute_shape_volume(s) for s in self.active_shapes) * (scale ** 3)
                imgui.text(f"Active Volume: {vol:.3f} {unit}\u00b3")
                if imgui.button("Save New Measurement", width=-1):
                    self.save_measurement()

        if self.measurements:
            imgui.dummy(0, 5)
            if imgui.button("Clear All Measurements", width=-1):
                self.measurements, self.selected_target = [], None
