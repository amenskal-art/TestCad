# tools/polyline_angle_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np
import math
from OpenGL.GL import *

class AngleMeasureTool(BaseTool):
    """
    Tool for measuring angles along a connected polyline.
    Left click to add points – each new point connects to the previous one.
    Right click finishes and displays the angle at each interior vertex.
    An arc is drawn between the two arms of each angle so it is unambiguous
    which angle is being reported.
    """
    def __init__(self, app):
        super().__init__(app)
        self.name = "Polyline Angle"
        self.icon_key = 'angle'
        self.hotkey = '6'

        # Drawing state
        self.points = []                    # list of point indices
        self.preview_point = None            # current mouse position index

        # Saved measurements: each is a dict with keys: points, angles
        self.measurements = []

        # Small offset to lift lines above the surface (avoids z‑fighting)
        self.surface_offset = 0.1            # mm

        # Arc display controls
        self.arc_steps   = 32               # smoothness of the arc
        self.arc_fraction = 0.25            # arc radius = segment_len * this fraction

    # ----------------------------------------------------------------------
    # Undo/Redo support
    # ----------------------------------------------------------------------
    def get_state(self):
        return {
            'points': self.points.copy(),
            'preview_point': self.preview_point,
            'measurements': [m.copy() for m in self.measurements]
        }

    def set_state(self, state):
        self.points = state['points'].copy()
        self.preview_point = state['preview_point']
        self.measurements = [m.copy() for m in state['measurements']]

    # ----------------------------------------------------------------------
    # Tool lifecycle
    # ----------------------------------------------------------------------
    def on_activate(self):
        self.app.status_text = "Polyline Angle: Left click to add points, right‑click to finish"

    def on_deactivate(self):
        self.cancel_drawing()

    # ----------------------------------------------------------------------
    # Input handling
    # ----------------------------------------------------------------------
    def on_mouse_button(self, button, action, mods, x, y):
        if imgui.get_io().want_capture_mouse:
            return False

        win_h = self.app.height
        y_gl = win_h - y

        # --- Right click: finish and compute angles ---
        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if len(self.points) >= 3:
                self.finish_measurement()
            else:
                self.cancel_drawing()
            return True

        # --- Left click: add point ---
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            picked = self.app.pick_single_point(x, y_gl)
            if picked is None:
                return False

            if not self.points:
                self.app.save_state("Start Polyline Angle")
            self.points.append(picked)
            self.app.status_text = f"Point {len(self.points)} added"
            return True
        return False

    def on_mouse_move(self, x, y):
        if not self.points:
            return False

        win_h = self.app.height
        y_gl = win_h - y
        picked = self.app.pick_single_point(x, y_gl)
        if picked is not None:
            self.preview_point = picked
        return True

    def on_key(self, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.cancel_drawing()
                return True
            elif key == glfw.KEY_BACKSPACE:
                if self.points:
                    self.points.pop()
                    self.app.status_text = f"Last point removed, {len(self.points)} points left"
                    if not self.points and self.app.undo_stack:
                        self.app.undo_stack.pop()
                return True
        return False

    # ----------------------------------------------------------------------
    # Measurement operations
    # ----------------------------------------------------------------------
    def finish_measurement(self):
        """Compute angles at interior vertices and save."""
        if len(self.points) < 3:
            self.app.status_text = "Need at least 3 points to compute angles"
            self.cancel_drawing()
            return

        angles = []
        for i in range(1, len(self.points) - 1):
            angle = self._compute_angle(
                self.app.points[self.points[i - 1]],
                self.app.points[self.points[i]],
                self.app.points[self.points[i + 1]]
            )
            angles.append(angle)

        self.measurements.append({
            'points': self.points.copy(),
            'angles': angles
        })
        self.app.status_text = f"Saved {len(angles)} angles"
        self.points = []
        self.preview_point = None

    def cancel_drawing(self):
        if self.points:
            if self.app.undo_stack:
                self.app.undo_stack.pop()
            self.app.status_text = "Measurement cancelled"
        self.points = []
        self.preview_point = None

    # ----------------------------------------------------------------------
    # Angle math helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _compute_angle(p_prev, p_curr, p_next):
        """Angle (degrees) between the two arms meeting at p_curr.
        Vectors point OUTWARD from the vertex (same convention as _arc_points)
        so the result matches the arc that is drawn."""
        v1 = p_prev - p_curr   # arm pointing back toward previous point
        v2 = p_next - p_curr   # arm pointing forward toward next point
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            return math.degrees(math.acos(cos_a))
        return 0.0

    def _arc_points(self, p_prev, p_curr, p_next):
        """
        Return a list of 3-D points that trace the arc at p_curr, plus the
        bisector direction (unit vector pointing from p_curr toward the arc midpoint).

        The arc lies in the plane of the two segments and sweeps from the
        'backward' ray (p_curr→p_prev direction) to the 'forward' ray
        (p_curr→p_next direction) using Rodrigues rotation.

        Arc radius = arc_fraction * min(|seg1|, |seg2|) so it always fits
        inside the shorter segment.
        """
        v1 = p_prev - p_curr          # ray toward previous point
        v2 = p_next - p_curr          # ray toward next point
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return [], np.zeros(3)

        r1 = v1 / n1                  # unit ray toward prev
        r2 = v2 / n2                  # unit ray toward next

        # Rotation axis (perpendicular to both rays)
        axis = np.cross(r1, r2)
        axis_len = np.linalg.norm(axis)

        # Arc radius scales with the shorter segment so it always fits
        radius = min(n1, n2) * self.arc_fraction

        if axis_len < 1e-6:
            # Collinear points – return a degenerate arc of two points only
            return [p_curr + r1 * radius, p_curr + r2 * radius], r1
        axis /= axis_len

        # Total angle to sweep
        cos_a = np.clip(np.dot(r1, r2), -1.0, 1.0)
        total_angle = math.acos(cos_a)

        # Rodrigues rotation: rotate r1 by theta around axis
        def rodrigues(v, k, theta):
            return (v * math.cos(theta)
                    + np.cross(k, v) * math.sin(theta)
                    + k * np.dot(k, v) * (1.0 - math.cos(theta)))

        arc_pts = []
        for step in range(self.arc_steps + 1):
            t = step / self.arc_steps
            rotated = rodrigues(r1, axis, total_angle * t)
            arc_pts.append(p_curr + rotated * radius)

        # Bisector direction = direction at the arc midpoint
        bisector = rodrigues(r1, axis, total_angle * 0.5)
        if np.linalg.norm(bisector) > 1e-6:
            bisector = bisector / np.linalg.norm(bisector)

        return arc_pts, bisector

    # ----------------------------------------------------------------------
    # Model-matrix helpers
    # ----------------------------------------------------------------------
    def _get_model_matrix(self):
        for attr in ('model_matrix', 'object_transform', 'mesh_transform',
                     'transform_matrix', 'obj_matrix'):
            m = getattr(self.app, attr, None)
            if m is not None and isinstance(m, np.ndarray) and m.shape == (4, 4):
                return m
        return None

    def _local_to_world(self, local_pos):
        m = self._get_model_matrix()
        if m is None:
            return local_pos
        p = np.array([local_pos[0], local_pos[1], local_pos[2], 1.0], dtype=np.float64)
        return (m @ p)[:3]

    # ----------------------------------------------------------------------
    # 3D rendering
    # ----------------------------------------------------------------------
    def render_3d(self):
        """Draw current polyline and all saved polylines with arc angle indicators."""

        def get_pt(idx):
            return self.app.points[idx] if isinstance(idx, (int, np.integer)) else idx

        def get_normal(idx):
            if hasattr(self.app, 'normals') and idx < len(self.app.normals):
                n = self.app.normals[idx]
                if np.linalg.norm(n) > 1e-6:
                    return n / np.linalg.norm(n)
            return np.array([0, 0, 1], dtype=np.float32)

        def offset_point(idx):
            pt = get_pt(idx)
            return pt + get_normal(idx) * self.surface_offset

        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glPointSize(6.0)

        glPushMatrix()
        model_mat = self._get_model_matrix()
        if model_mat is not None:
            glMultMatrixf(model_mat.T.astype(np.float32))

        # ----------------------------------------------------------------
        # Draw in-progress polyline
        # ----------------------------------------------------------------
        if self.points:
            pts = [offset_point(i) for i in self.points]

            glColor3f(1.0, 1.0, 0.0)       # yellow segments
            glBegin(GL_LINE_STRIP)
            for p in pts:
                glVertex3f(*p)
            glEnd()

            glColor3f(0.0, 1.0, 0.0)       # green vertices
            glBegin(GL_POINTS)
            for p in pts:
                glVertex3f(*p)
            glEnd()

            # Preview segment and live angle
            if self.preview_point is not None:
                preview_pt = offset_point(self.preview_point)
                glColor3f(1.0, 0.5, 0.0)   # orange preview
                glBegin(GL_LINES)
                glVertex3f(*pts[-1])
                glVertex3f(*preview_pt)
                glEnd()
                glColor3f(1.0, 1.0, 0.0)
                glBegin(GL_POINTS)
                glVertex3f(*preview_pt)
                glEnd()

                if len(pts) >= 2:
                    p_prev = pts[-2]
                    p_curr = pts[-1]
                    p_next = preview_pt
                    angle = self._compute_angle(p_prev, p_curr, p_next)
                    arc_pts, bisector = self._arc_points(p_prev, p_curr, p_next)
                    self._gl_draw_arc(arc_pts, color=(1.0, 0.8, 0.0))
                    if np.linalg.norm(bisector) > 1e-6:
                        label_offset = min(
                            np.linalg.norm(p_curr - p_prev),
                            np.linalg.norm(p_next - p_curr)
                        ) * self.arc_fraction * 1.6
                        label_pos = p_curr + bisector * label_offset
                        self._draw_label(label_pos, f"{angle:.1f}°",
                                         color=(1.0, 0.8, 0.0))

        # ----------------------------------------------------------------
        # Draw saved measurements
        # ----------------------------------------------------------------
        for m in self.measurements:
            pts = [offset_point(i) for i in m['points']]
            if len(pts) < 2:
                continue

            glColor3f(0.0, 1.0, 1.0)       # cyan polyline
            glBegin(GL_LINE_STRIP)
            for p in pts:
                glVertex3f(*p)
            glEnd()

            glColor3f(0.0, 1.0, 0.0)       # green vertices
            glBegin(GL_POINTS)
            for p in pts:
                glVertex3f(*p)
            glEnd()

            # Arc + label at each interior vertex
            for i, angle in enumerate(m['angles']):
                p_prev = pts[i]
                p_curr = pts[i + 1]
                p_next = pts[i + 2]

                arc_pts, bisector = self._arc_points(p_prev, p_curr, p_next)
                self._gl_draw_arc(arc_pts, color=(0.0, 1.0, 1.0))

                # Draw the two small "tick" lines from vertex along each arm so
                # it is crystal-clear which angle is spanned by the arc.
                self._gl_draw_arm_ticks(p_prev, p_curr, p_next,
                                        color=(0.4, 1.0, 1.0))

                # Place label along the bisector, just outside the arc
                if np.linalg.norm(bisector) > 1e-6:
                    label_offset = min(
                        np.linalg.norm(p_curr - p_prev),
                        np.linalg.norm(p_next - p_curr)
                    ) * self.arc_fraction * 1.6
                    label_pos = p_curr + bisector * label_offset
                    self._draw_label(label_pos, f"{angle:.1f}°")

        glPopMatrix()
        glPopAttrib()

    # ----------------------------------------------------------------------
    # GL helpers for arc + arm ticks
    # ----------------------------------------------------------------------
    def _gl_draw_arc(self, arc_pts, color=(1.0, 1.0, 1.0)):
        """Draw a list of 3-D points as a GL_LINE_STRIP arc."""
        if len(arc_pts) < 2:
            return
        glColor3f(*color)
        glLineWidth(1.5)
        glBegin(GL_LINE_STRIP)
        for p in arc_pts:
            glVertex3f(*p)
        glEnd()
        glLineWidth(2.0)  # restore

    def _gl_draw_arm_ticks(self, p_prev, p_curr, p_next, color=(0.4, 1.0, 1.0)):
        """
        Draw two short lines along each arm of the angle starting from p_curr.
        This makes each arm explicitly visible so the user knows exactly which
        angle the arc and label refer to.
        """
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return

        tick_len = min(n1, n2) * self.arc_fraction  # same radius as arc

        glColor3f(*color)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        # Arm 1: vertex → direction of previous point
        glVertex3f(*p_curr)
        glVertex3f(*(p_curr + (v1 / n1) * tick_len))
        # Arm 2: vertex → direction of next point
        glVertex3f(*p_curr)
        glVertex3f(*(p_curr + (v2 / n2) * tick_len))
        glEnd()
        glLineWidth(2.0)  # restore

    # ----------------------------------------------------------------------
    # 2-D label overlay
    # ----------------------------------------------------------------------
    def _draw_label(self, local_pos, text, color=(1.0, 1.0, 1.0)):
        world_pos = self._local_to_world(local_pos)
        sx, sy, sz = self.app.project_point_to_screen(world_pos)
        if 0.0 <= sz <= 1.0:
            draw_list = imgui.get_background_draw_list()
            tw = imgui.calc_text_size(text).x
            draw_list.add_rect_filled(
                sx, sy, sx + tw + 8, sy + 20,
                imgui.get_color_u32_rgba(0, 0, 0, 0.7)
            )
            draw_list.add_text(
                sx + 4, sy + 2,
                imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0),
                text
            )

    # ----------------------------------------------------------------------
    # UI panel
    # ----------------------------------------------------------------------
    def draw_options(self):
        imgui.text("Polyline Angle Measurement")
        imgui.separator()

        changed, self.surface_offset = imgui.slider_float(
            "Surface Offset", self.surface_offset, 0.0, 1.0, "%.2f mm"
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Distance to lift lines above the mesh surface")

        changed2, self.arc_fraction = imgui.slider_float(
            "Arc Size", self.arc_fraction, 0.05, 0.45, "%.2f"
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Arc radius as a fraction of the shorter segment length")

        if imgui.button("Clear All Measurements", width=-1):
            self.measurements = []

        imgui.dummy(0, 10)
        imgui.separator()
        imgui.text("Instructions:")
        imgui.text_disabled("Left click: add point (automatically connects)")
        imgui.text_disabled("Right click: finish and show angles")
        imgui.text_disabled("Backspace: remove last point")
        imgui.text_disabled("Esc: cancel current")
