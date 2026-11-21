# tools/sculpt_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np

class SculptTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Sculpt"
        self.icon_key = 'sculpt'
        self.hotkey = '4'
        self._sculptor_synced = False

    def on_activate(self):
        self.app.status_text = "Wax Knife Mode"
        self.app.is_sculpting = False
        self._sculptor_synced = False   # force re-sync on next render frame

    def on_deactivate(self):
        if getattr(self.app, 'is_sculpting', False):
            self._finish_stroke()

    # ---------- Helper to ensure points array is writable ----------
    def _ensure_points_writable(self):
        if not self.app.points.flags.writeable:
            self.app.points = self.app.points.copy()
            self.app.points_colors[:, :3] = self.app.points
            if hasattr(self.app, 'sculptor') and self.app.sculptor is not None:
                self.app.sculptor.points = self.app.points
            self.app.vbo_dirty = True

    # ---------- Single exit point for every stroke end ----------
    def _finish_stroke(self):
        self.app.is_sculpting = False
        self.app.status_text = "Sculpt finished."

        sculptor = getattr(self.app, 'sculptor', None)
        if sculptor is not None:
            sculptor.points = self.app.points

            # Step 1: Rebuild sculptor KDTree on post-stroke geometry first.
            #         recompute_dirty_normals() queries this tree, so order matters.
            sculptor.rebuild_tree()

            # Step 2: Recompute normals for every point moved this stroke + neighbors.
            #         Uses vectorized PCA — fast even for large brushes.
            sculptor.recompute_dirty_normals(k=16)

            # Step 3: Sync fresh normals back to the app so the VBO upload
            #         (triggered by vbo_dirty below) pushes correct data to the GPU.
            self.app.normals = sculptor.normals

            # Step 4: Rebuild the app-level KDTree used by pick_single_point().
            #         Without this the measure tool queries stale pre-sculpt positions
            #         and returns a vertex index offset from where the user clicked.
            if hasattr(self.app, '_rebuild_kdtree'):
                self.app._rebuild_kdtree()

        # Mark VBO dirty after all data is consistent so the next render frame
        # uploads both the updated points_colors and normals buffers together.
        self.app.vbo_dirty = True

        # Force re-sync so the next render frame starts clean
        self._sculptor_synced = False
    # ----------------------------------------------------------------

    def on_mouse_button(self, button, action, mods, x, y):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if not imgui.get_io().want_capture_mouse:
                hit = self.app.cursor_world_pos
                if hit is not None:
                    self._ensure_points_writable()
                    self.app.save_state("Sculpt Stroke")
                    self.app.is_sculpting = True
                    self.app.status_text = "Sculpting..."
                    self._apply_sculpt(hit, mods)
                    return True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            if getattr(self.app, 'is_sculpting', False):
                self._finish_stroke()
                return True

        return False

    def on_mouse_move(self, x, y):
        if getattr(self.app, 'is_sculpting', False):
            hit = self.app.cursor_world_pos
            if hit is not None:
                is_shift = glfw.get_key(self.app.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                           glfw.get_key(self.app.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
                is_ctrl  = glfw.get_key(self.app.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or \
                           glfw.get_key(self.app.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS

                mods = 0
                if is_shift: mods |= glfw.MOD_SHIFT
                if is_ctrl:  mods |= glfw.MOD_CONTROL

                self._apply_sculpt(hit, mods)
            return True
        return False

    def _apply_sculpt(self, hit, mods):
        mode     = self.app.sculpt_mode
        is_shift = (mods & glfw.MOD_SHIFT)
        is_ctrl  = (mods & glfw.MOD_CONTROL)

        if is_shift:
            mode = 'smooth'
        elif is_ctrl:
            if mode == 'add':      mode = 'remove'
            elif mode == 'remove': mode = 'add'

        self.app.sculptor.sculpt(hit, self.app.sculpt_radius,
                                 self.app.sculpt_strength, mode)
        self.app.vbo_dirty = True

    def update(self):
        # Safety net: catches RELEASE events swallowed by ImGui.
        if getattr(self.app, 'is_sculpting', False):
            if glfw.get_mouse_button(self.app.window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
                self._finish_stroke()

    def render_3d(self):
        # ── Sync sculptor (mirrors engineering_tool.py pattern) ──────────────
        # Ensures the SculptSession exists and holds a live reference to the
        # current points array before any cursor or draw call uses it.
        if not self._sculptor_synced:
            if not hasattr(self.app, 'sculptor') or self.app.sculptor is None:
                if hasattr(self.app, 'sync_sculptor'):
                    self.app.sync_sculptor()
            self._sculptor_synced = True

        # ── Re-sample depth buffer every frame ───────────────────────────────
        # Gives a fresh world position from the actual depth buffer every render
        # frame so draw_brush_cursor queries get_nearest_normal from the correct
        # surface position rather than a stale mouse-event position.
        if hasattr(self.app, 'update_cursor_depth'):
            self.app.update_cursor_depth()

        if getattr(self.app, 'cursor_world_pos', None) is None:
            return

        self.app.draw_brush_cursor()

    def draw_options(self):
        imgui.text("Brush Settings")
        imgui.push_item_width(150)

        _, self.app.sculpt_radius = imgui.slider_float(
            "Size", self.app.sculpt_radius, 0.1, 50.0
        )
        _, self.app.sculpt_strength = imgui.slider_float(
            "Pressure", self.app.sculpt_strength, 0.01, 1.0
        )

        imgui.pop_item_width()
        imgui.separator()

        imgui.text("Operation:")
        if imgui.radio_button("Add (Inflate)", self.app.sculpt_mode == 'add'):
            self.app.sculpt_mode = 'add'
        if imgui.radio_button("Remove (Deflate)", self.app.sculpt_mode == 'remove'):
            self.app.sculpt_mode = 'remove'
        if imgui.radio_button("Smooth (Melt)", self.app.sculpt_mode == 'smooth'):
            self.app.sculpt_mode = 'smooth'
        if imgui.radio_button("Flatten (Grind)", self.app.sculpt_mode == 'flatten'):
            self.app.sculpt_mode = 'flatten'

        imgui.dummy(0, 5)
        imgui.separator()
        imgui.text("Modifiers:")
        imgui.text_disabled("Shift: Smooth mode")
        imgui.text_disabled("Ctrl: Invert Add/Remove")
        imgui.text_disabled("Left Drag: Sculpt")

        imgui.dummy(0, 5)

        if imgui.button("Reset Brush to Defaults", width=-1):
            self.app.sculpt_radius = 0.1
            self.app.sculpt_strength = 0.3
            self.app.sculpt_mode = 'add'
