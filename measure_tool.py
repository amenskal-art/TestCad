# tools/measure_tool.py
from tool_system import BaseTool
import imgui
import glfw


class MeasureTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Measure"
        self.icon_key = 'measure'
        self.hotkey = '20'
        # Opt out of any brush/cursor overlay the base system might draw.
        self.uses_brush = False
        self.show_brush_cursor = False

    def on_activate(self):
        self.app.status_text = "Measure Mode"
        # Explicitly force straight-line measurement
        self.app.measure_on_surface = False
        # Suppress the brush overlay while measuring
        self.app.show_brush_cursor = False

    def on_deactivate(self):
        if self.app.is_measuring_drag:
            self.app.is_measuring_drag = False
            self.app.measure_start_pt = None
        # Restore brush overlay for brush-based tools
        self.app.show_brush_cursor = True

    def draw_cursor(self):
        # Override so no brush cursor is drawn for this tool.
        pass

    def on_mouse_button(self, button, action, mods, x, y):
        win_h = self.app.height
        y_gl = win_h - y

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if not imgui.get_io().want_capture_mouse:
                picked_pt = self.app.pick_single_point(x, y_gl)
                if picked_pt is not None:
                    self.app.save_state("Add Measurement")
                    self.app.measure_start_pt = picked_pt
                    self.app.measure_drag_pt = picked_pt
                    self.app.is_measuring_drag = True
                    self.app.status_text = "Measuring..."
                    return True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            if self.app.is_measuring_drag:
                final_pt = self.app.pick_single_point(x, y_gl)
                if final_pt is not None and self.app.measure_start_pt is not None:
                    self.app.measurements.append((self.app.measure_start_pt, final_pt))
                    self.app.status_text = "Measurement added."
                else:
                    if self.app.undo_stack:
                        self.app.undo_stack.pop()
                    self.app.status_text = "Measurement Cancelled"

                self.app.measure_start_pt = None
                self.app.measure_drag_pt = None
                self.app.is_measuring_drag = False
                return True

        return False

    def on_mouse_move(self, x, y):
        if self.app.is_measuring_drag:
            win_h = self.app.height
            pt = self.app.pick_single_point(x, win_h - y)
            if pt is not None:
                self.app.measure_drag_pt = pt
            return True
        return False

    def on_key(self, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            if self.app.is_measuring_drag:
                self.app.is_measuring_drag = False
                self.app.measure_start_pt = None
                if self.app.undo_stack:
                    self.app.undo_stack.pop()
                return True
        return False

    def draw_options(self):
        imgui.text("Measurement Settings")
        imgui.separator()

        if imgui.button("Clear All Measurements", width=-1):
            self.app.measurements = []

        imgui.dummy(0, 5)
        imgui.separator()
        imgui.text("Instructions:")
        imgui.text_disabled("Click: Start measurement")
        imgui.text_disabled("Drag: Adjust endpoint")
        imgui.text_disabled("Release: Save measurement")
        imgui.text_disabled("Esc: Cancel")
