# tools/brush_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np
import math
from OpenGL.GL import *

class BrushTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Vertex Paint"
        self.icon_key = 'brush'
        self.hotkey = 'B'

        self.brush_size = 1.5
        self.brush_strength = 1.0
        self.brush_color = [0.0, 1.0, 0.0]

        # Mode: 'paint' applies brush_color, 'erase' restores original colors.
        self.mode = 'paint'

        # Tip profile: 'soft' uses a smooth radial falloff; 'pen' is a hard-edged
        # tip (full strength inside the radius, nothing outside) for crisp lines.
        self.tip = 'soft'

        # Simulated pen pressure (0..1). GLFW exposes no stylus pressure, so this
        # defaults to 1.0 and can be driven by a UI slider. If you later add a
        # tablet input layer, set self.app.pen_pressure and it is used here.
        self.pressure = 1.0

        self.is_painting = False

        # Snapshot of the colors as loaded, used by erase. Lazily captured the
        # first time the tool touches the mesh, and re-captured if the point
        # count changes (e.g. a new model is loaded).
        self._original_colors = None
        self._orig_n = -1

    def on_activate(self):
        self.app.status_text = "Paint Mode: Color the mesh vertices"
        self._ensure_original_snapshot()

    # ------------------------------------------------------------------ helpers
    def _ensure_original_snapshot(self):
        """
        Capture the original per-vertex colors once, so erase can restore them.

        Re-captures when the vertex count changes (new model loaded). This is the
        only place original colors are stored, since the app keeps just a single
        flat default_mesh_color, not per-vertex originals.
        """
        bc = getattr(self.app, 'base_colors', None)
        if bc is None:
            return
        n = len(bc)
        if self._original_colors is None or self._orig_n != n:
            self._original_colors = bc[:, :3].copy()
            self._orig_n = n

    def _effective_pressure(self):
        """Pressure used for this stroke: external tablet value if present, else UI value."""
        p = getattr(self.app, 'pen_pressure', None)
        if p is None:
            p = self.pressure
        return float(max(0.0, min(1.0, p)))

    def _sample_world_pos(self, x, y):
        """
        Resolve a world-space position under the mouse for painting.

        Uses the same depth path as the ring (update_cursor_depth ->
        cursor_world_pos), which accounts for the window/framebuffer pixel ratio.
        Falls back to get_world_pos_at_mouse() only if that yields nothing.
        """
        if hasattr(self.app, 'update_cursor_depth'):
            self.app.update_cursor_depth()
        world_pos = getattr(self.app, 'cursor_world_pos', None)
        if world_pos is not None:
            return world_pos

        y_gl = self.app.height - y
        return self.app.get_world_pos_at_mouse(x, y_gl)

    # ------------------------------------------------------------------ render
    def render_3d(self):
        # Refresh cursor world position from the depth buffer so the ring tracks
        # the mouse for this tool (draw_scene's own call is gated on a stale id).
        if hasattr(self.app, 'update_cursor_depth'):
            self.app.update_cursor_depth()

        if self.app.cursor_world_pos is not None:
            glPushAttrib(GL_ENABLE_BIT)
            glDisable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            center = self.app.cursor_world_pos
            radius = self.brush_size

            # Ring color hints the mode: brush color when painting, white when erasing.
            if self.mode == 'erase':
                glColor3f(1.0, 1.0, 1.0)
            else:
                glColor3f(*self.brush_color)

            # Pen tip draws a thicker ring so the hard edge reads clearly.
            glLineWidth(3.0 if self.tip == 'pen' else 2.0)

            for axis in range(3):
                glBegin(GL_LINE_LOOP)
                for i in range(24):
                    theta = 2.0 * math.pi * float(i) / 24.0
                    dx = radius * math.cos(theta)
                    dy = radius * math.sin(theta)
                    if axis == 0: glVertex3f(center[0], center[1]+dx, center[2]+dy)
                    elif axis == 1: glVertex3f(center[0]+dx, center[1], center[2]+dy)
                    else: glVertex3f(center[0]+dx, center[1]+dy, center[2])
                glEnd()
            glPopAttrib()

    # ------------------------------------------------------------------ input
    def on_mouse_button(self, button, action, mods, x, y):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS and not imgui.get_io().want_capture_mouse:
                world_pos = self._sample_world_pos(x, y)
                if world_pos is not None:
                    self._ensure_original_snapshot()
                    self.app.save_state("Vertex Paint" if self.mode == 'paint' else "Vertex Erase")
                    self.is_painting = True
                    self._apply_paint_at(world_pos)
                    return True
            elif action == glfw.RELEASE:
                self.is_painting = False
                return True
        return False

    def on_mouse_move(self, x, y):
        if self.is_painting:
            world_pos = self._sample_world_pos(x, y)
            if world_pos is not None:
                self._apply_paint_at(world_pos)
            return True
        return False

    # ------------------------------------------------------------------ apply
    def _apply_paint_at(self, center):
        """Paint or erase at an explicit world-space position."""
        if not hasattr(self.app, 'sculptor') or not self.app.sculptor:
            return

        idxs = self.app.sculptor.query_radius(center, self.brush_size)
        if len(idxs) == 0:
            return

        # Ensure base_colors is writable, keeping points_colors in sync.
        if not self.app.base_colors.flags.writeable:
            self.app.base_colors = self.app.base_colors.copy()
            self.app.points_colors[:, 3:6] = self.app.base_colors

        self._ensure_original_snapshot()

        pts = self.app.points[idxs]
        dists = np.linalg.norm(pts - center, axis=1)
        t = np.clip(dists / max(self.brush_size, 1e-6), 0.0, 1.0)

        if self.tip == 'pen':
            # Hard edge: full weight inside the radius, zero outside.
            weights = (t < 1.0).astype(np.float32)
        else:
            # Soft edge: smooth quadratic falloff (matches the sculpt brush feel).
            weights = (1.0 - t * t) ** 2

        # Pressure scales how strongly the target color is blended in.
        weights = weights * self._effective_pressure() * float(self.brush_strength)
        weights = np.clip(weights, 0.0, 1.0)[:, None]

        if self.mode == 'erase':
            target = self._original_colors[idxs]
        else:
            target = np.array(self.brush_color, dtype=np.float32)[None, :]

        current = self.app.base_colors[idxs, :3]
        blended = (current * (1.0 - weights) + target * weights).astype(np.float32)

        self.app.base_colors[idxs, :3] = blended
        self.app.points_colors[idxs, 3:6] = blended
        self.app.vbo_dirty = True

    # ------------------------------------------------------------------ ui
    def draw_options(self):
        imgui.text("Vertex Paint Settings")
        imgui.separator()

        # Mode: Paint vs Erase
        imgui.text("Mode:")
        if imgui.radio_button("Paint", self.mode == 'paint'):
            self.mode = 'paint'
            self.app.status_text = "Paint Mode: Color the mesh vertices"
        imgui.same_line()
        if imgui.radio_button("Erase", self.mode == 'erase'):
            self.mode = 'erase'
            self.app.status_text = "Erase Mode: Restore original colors"

        imgui.dummy(0, 4)

        # Tip: Soft brush vs hard-edged pen
        imgui.text("Tip:")
        if imgui.radio_button("Brush (soft)", self.tip == 'soft'):
            self.tip = 'soft'
        imgui.same_line()
        if imgui.radio_button("Pen (hard)", self.tip == 'pen'):
            self.tip = 'pen'

        imgui.dummy(0, 6)
        imgui.separator()

        imgui.push_item_width(150)
        _, self.brush_size = imgui.slider_float("Radius", self.brush_size, 0.1, 10.0)
        _, self.brush_strength = imgui.slider_float("Flow", self.brush_strength, 0.0, 1.0)

        # Pressure slider simulates pen pressure. If an external tablet layer sets
        # self.app.pen_pressure, that value is used instead at paint time.
        _, self.pressure = imgui.slider_float("Pressure", self.pressure, 0.0, 1.0)
        imgui.pop_item_width()

        if self.mode == 'paint':
            _, self.brush_color = imgui.color_edit3("Color", *self.brush_color)

        if getattr(self.app, 'pen_pressure', None) is not None:
            imgui.text_disabled("Tablet pressure active (overrides slider)")
