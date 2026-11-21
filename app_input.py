# app_input.py
import glfw
import imgui
import numpy as np
import math
import threading
from OpenGL.GL import *
from OpenGL.GLU import *
from utils import open_file_dialog, create_rotation_matrix, normalize

try:
    from scipy.spatial import cKDTree as KDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

class AppInputMixin:

    def pick_single_point(self, x, y_gl):
        if not hasattr(self, '_kdtree') or self._kdtree is None:
            self._rebuild_kdtree()
        if self._kdtree is None:
            return None
        world_pos = self.get_world_pos_at_mouse(x, y_gl)
        if world_pos is None:
            return None
        _, idx = self._kdtree.query(world_pos, workers=1)
        return int(idx)

    def _rebuild_kdtree(self):
        if _HAS_SCIPY and self.points is not None and len(self.points) > 0:
            self._kdtree = KDTree(self.points)
        else:
            self._kdtree = None

    def on_resize(self, wnd, w, h):
        self.width, self.height = w, h
        glViewport(0, 0, self.width, self.height)

    def on_scroll(self, wnd, x_offset, y_offset):
        if imgui.get_io().want_capture_mouse: return
        if y_offset > 0:
            x, y = glfw.get_cursor_pos(self.window)
            y_gl = self.height - y
            hit_pos = self.get_world_pos_at_mouse(x, y_gl)
            if hit_pos is not None:
                self.center = hit_pos
                self.pan_x = 0.0
                self.pan_y = 0.0
                mv_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
                mv_matrix = np.array(mv_matrix).T
                try:
                    inv_mv = np.linalg.inv(mv_matrix)
                    cam_pos = inv_mv[:3, 3]
                    self.distance = np.linalg.norm(cam_pos - hit_pos)
                except:
                    pass
                if hasattr(self, 'width') and hasattr(self, 'height'):
                    center_x, center_y = self.width / 2.0, self.height / 2.0
                    glfw.set_cursor_pos(self.window, center_x, center_y)
        zoom_speed = 0.1
        if y_offset > 0:
            self.distance *= (1.0 - zoom_speed)
        else:
            self.distance *= (1.0 + zoom_speed)
        self.distance = max(0.1, min(self.distance, 500.0))

    def gui_open_file_thread(self):
        threading.Thread(target=lambda: self._set_pending(open_file_dialog(), 'replace'), daemon=True).start()

    def gui_import_file_thread(self):
        threading.Thread(target=lambda: self._set_pending(open_file_dialog(), 'replace'), daemon=True).start()

    def gui_export_file_thread(self):
        def _export_dialog():
            import tkinter as tk
            from tkinter import filedialog
            import os
            from utils import export_mesh_data
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            filepath = filedialog.asksaveasfilename(
                title="Export Edited Object",
                defaultextension=".obj",
                filetypes=[("OBJ Files", "*.obj"), ("PLY Files", "*.ply"), ("STL Files", "*.stl"), ("All Files", "*.*")]
            )
            root.destroy()
            if filepath:
                self.is_exporting = True
                try:
                    pts = getattr(self, 'points', np.array([]))
                    if hasattr(self, 'points_colors') and self.points_colors.shape[1] >= 6:
                        cols = self.points_colors[:, 3:6]
                    elif hasattr(self, 'colors'):
                        cols = self.colors
                    else:
                        cols = np.ones_like(pts) * 0.75
                    fcs = None
                    if getattr(self, 'render_mode_mesh', False) and len(pts) % 3 == 0:
                        fcs = np.arange(len(pts), dtype=np.int32).reshape(-1, 3)
                    else:
                        for attr_name in ['faces', 'faces_array', 'triangles', 'mesh_faces']:
                            if hasattr(self, attr_name):
                                f = getattr(self, attr_name)
                                if f is not None and len(f) > 0:
                                    fcs = f
                                    break
                    export_mesh_data(filepath, pts, cols, fcs)
                    self.status_text = f"Exported: {os.path.basename(filepath)}"
                finally:
                    self.is_exporting = False
        threading.Thread(target=_export_dialog, daemon=True).start()

    def _set_pending(self, path, mode):
        if path:
            self.pending_file = path
            self.pending_file_mode = mode

    def update_vbo_subset(self):
        self.points_colors[:, :3] = self.points
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points_colors.nbytes, self.points_colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def get_world_pos_at_mouse(self, mx, my_gl):
        try:
            z_depth = glReadPixels(int(mx), int(my_gl), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
            if z_depth == 1.0: return None
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            wx, wy, wz = gluUnProject(mx, my_gl, z_depth, modelview, projection, viewport)
            return np.array([wx, wy, wz], dtype=np.float32)
        except:
            return None

    def on_mouse_button(self, wnd, button, action, mods):
        if imgui.get_io().want_capture_mouse: return
        x, y = glfw.get_cursor_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        y_gl = win_h - y 
        is_ctrl = (mods & glfw.MOD_CONTROL)
        is_shift = (mods & glfw.MOD_SHIFT)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            current_time = glfw.get_time()
            if not hasattr(self, '_last_click_time'): self._last_click_time = 0
            is_double_click = (current_time - self._last_click_time) < 0.3
            self._last_click_time = current_time
            hit_pos = self.get_world_pos_at_mouse(x, y_gl)
            if self.active_tool == 1:
                if hit_pos is None:
                    self.save_state("Clear Selection")
                    self.selected[:] = False
                    self.update_colors()
                    self.status_text = "Selection Cleared."
                elif is_double_click and hasattr(self, 'point_object_ids'):
                    dists = np.sum((self.points - hit_pos)**2, axis=1)
                    closest_idx = np.argmin(dists)
                    obj_id = self.point_object_ids[closest_idx]
                    if not is_shift:
                        self.selected[:] = False
                    self.select_sub_object(obj_id)
                    return

        if hasattr(self, 'tool_manager') and self.tool_manager.handle_event('mouse_button', button, action, mods, x, y): return

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                if self.active_tool == 0:
                    new_pivot = self.get_world_pos_at_mouse(x, y_gl)
                    if new_pivot is not None:
                        self.center = new_pivot
                        self.pan_x = 0.0
                        self.pan_y = 0.0
                    self._dragging_rotate = True
                    self._last_dxdy = (x, y_gl)
                    return
                if self.active_tool == 1:
                    if self.selection_submode == 0:
                        self.save_state("Lasso Selection"); self.lasso = [(x, y_gl)]; self.is_lassoing = True; self.lasso_mode = -1 if is_ctrl else 1
                    elif self.selection_submode == 1:
                        self.is_poly_active = True; self.knife_poly.append((x, y_gl)); self.status_text = f"Poly: {len(self.knife_poly)} pts."
                    return
                if self.active_tool == 2:
                    picked_pt = self.pick_single_point(x, y_gl)
                    if picked_pt is not None:
                        self.save_state("Add Measurement"); self.measure_start_pt = picked_pt; self.measure_drag_pt = picked_pt
                        self.is_measuring_drag = True; self.status_text = "Measuring..."
                    return 
                if self.active_tool == 3:
                    hit = self.cursor_world_pos
                    if hit is not None:
                        self.save_state("Sculpt Stroke"); self.is_sculpting = True; self.status_text = "Sculpting..."
                        mode = 'smooth' if is_shift else ('remove' if is_ctrl and self.sculpt_mode == 'add' else ('add' if is_ctrl else self.sculpt_mode))
                        self.sculptor.sculpt(hit, self.sculpt_radius, self.sculpt_strength, mode); self.vbo_dirty = True
                    return
            elif action == glfw.RELEASE:
                self._dragging_rotate = False
                if self.is_sculpting: self.is_sculpting = False; self.vbo_dirty = True; self.status_text = "Sculpt finished."; return
                if self.is_measuring_drag:
                    final_pt = self.pick_single_point(x, y_gl)
                    if final_pt is not None and self.measure_start_pt is not None: self.measurements.append((self.measure_start_pt, final_pt)); self.status_text = "Measurement added."
                    else: 
                        if self.undo_stack: self.undo_stack.pop(); self.status_text = "Measurement Cancelled"
                    self.measure_start_pt = None; self.measure_drag_pt = None; self.is_measuring_drag = False; return
                if self.is_lassoing: self.lasso.append((x, y_gl)); self.is_lassoing = False; self.lasso_select_pixels()
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                if self.active_tool == 1 and self.selection_submode == 1 and self.is_poly_active: self.apply_poly_cut(); return
                self._dragging_pan = True; self._last_dxdy = (x, y_gl)
            elif action == glfw.RELEASE:
                self._dragging_pan = False

    def on_mouse_move(self, wnd, xpos, ypos):
        if imgui.get_io().want_capture_mouse: return
        _PICK_THRESHOLD_SQ = 9
        _PICK_TOOLS = {2, 13, 14, 15}
        active_id = getattr(self, 'active_tool', -1)
        if active_id in _PICK_TOOLS:
            lx, ly = getattr(self, '_last_pick_pos', (None, None))
            if lx is not None:
                dx, dy = xpos - lx, ypos - ly
                if dx * dx + dy * dy < _PICK_THRESHOLD_SQ:
                    return
            self._last_pick_pos = (xpos, ypos)
        else:
            self._last_pick_pos = (None, None)

        if hasattr(self, 'tool_manager') and self.tool_manager.handle_event('mouse_move', xpos, ypos): return

        win_w, win_h = glfw.get_window_size(self.window)
        y_gl = win_h - ypos
        if self.is_lassoing:
            self.lasso.append((xpos, y_gl))
        elif self._dragging_rotate and self._last_dxdy:
            dx, dy = xpos - self._last_dxdy[0], y_gl - self._last_dxdy[1]
            speed = self.rot_speed * 1.5
            lock_x = self.key_constraints.get(glfw.KEY_X, False)
            lock_y = self.key_constraints.get(glfw.KEY_Y, False)
            lock_z = self.key_constraints.get(glfw.KEY_Z, False)
            if lock_x:
                rot = create_rotation_matrix([1,0,0], -dy * speed)
                self.rotation_matrix = self.rotation_matrix @ rot 
            elif lock_y:
                rot = create_rotation_matrix([0,1,0], dx * speed)
                self.rotation_matrix = self.rotation_matrix @ rot 
            elif lock_z:
                rot = create_rotation_matrix([0,0,1], -dx * speed)
                self.rotation_matrix = self.rotation_matrix @ rot 
            else:
                rot_y = create_rotation_matrix([0,1,0], dx * speed) 
                rot_x = create_rotation_matrix([1,0,0], -dy * speed) 
                self.rotation_matrix = rot_x @ rot_y @ self.rotation_matrix
            self._orthonormalize_matrix()
            self._last_dxdy = (xpos, y_gl)
        elif self._dragging_pan and self._last_dxdy:
            dx, dy = xpos - self._last_dxdy[0], y_gl - self._last_dxdy[1]
            fov_rad = math.radians(60.0)
            visible_h = 2.0 * self.distance * math.tan(fov_rad / 2.0)
            pixel_scale = visible_h / self.height
            if not hasattr(self, 'pan_x'): self.pan_x = 0.0
            if not hasattr(self, 'pan_y'): self.pan_y = 0.0
            self.pan_x += dx * pixel_scale
            self.pan_y += dy * pixel_scale
            self._last_dxdy = (xpos, y_gl)

    def _orthonormalize_matrix(self):
        m = self.rotation_matrix
        x = normalize(m[:3, 0])
        y = normalize(m[:3, 1] - np.dot(x, m[:3, 1]) * x)
        m[:3, 0], m[:3, 1], m[:3, 2] = x, y, np.cross(x, y)
        self.rotation_matrix = m

    def on_key(self, wnd, key, scancode, action, mods):
        # FIX: When imgui owns the keyboard (a textbox is focused), forward the
        # key event to imgui so Backspace, Delete, arrow keys, Ctrl+A, etc. all
        # work correctly inside input_text widgets, then return so app hotkeys
        # (number keys, R, L, Delete, etc.) don't fire while the user is typing.
        if imgui.get_io().want_capture_keyboard:
            self.impl.keyboard_callback(wnd, key, scancode, action, mods)
            return

        if key in [glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z]:
            if not hasattr(self, 'key_constraints'): self.key_constraints = {}
            self.key_constraints[key] = (action != glfw.RELEASE)
        if hasattr(self, 'tool_manager') and self.tool_manager.handle_event('key', key, scancode, action, mods): return
        if action == glfw.PRESS:
            # Ctrl+N for New Scene
            if key == glfw.KEY_N and (mods & glfw.MOD_CONTROL):
                self.new_dialog_requested = True
                return True
            # Ctrl+S for Save Project
            if key == glfw.KEY_S and (mods & glfw.MOD_CONTROL):
                self.save_project()
                return True
            # Ctrl+A for Select All
            if key == glfw.KEY_A and (mods & glfw.MOD_CONTROL):
                if hasattr(self, 'select_all'):
                    self.select_all()
                return True

            # --- Number-key tool switching (position-based, matches toolbar order) ---
            # Keys map BY POSITION into the toolbar/registration order read live
            # from tool_manager, so every registered tool is reachable and the
            # mapping stays correct if tools are added or reordered.
            _number_keys = [
                glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5,
                glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9, glfw.KEY_0,
            ]
            if key in _number_keys and hasattr(self, 'tool_manager'):
                slot = _number_keys.index(key)                   # 0-based position
                tool_ids = list(self.tool_manager.tools.keys())  # toolbar order
                if slot < len(tool_ids):
                    tid = tool_ids[slot]
                    self.tool_manager.set_active_tool(tid)
                    self.active_tool = tid
                    tool = self.tool_manager.tools[tid]
                    self.status_text = f"Tool: {tool.name}"
                return True

            # --- Letter-key shortcuts for the overflow tools (past the 10
            #     number keys). These map directly to specific tool IDs.
            #     Letters chosen to avoid clashes with R/L and X/Y/Z. ---
            _letter_tool_map = {
                glfw.KEY_V: 11,   # Volume
                glfw.KEY_P: 12,   # Poisson
                glfw.KEY_O: 13,   # Smooth
                glfw.KEY_G: 14,   # GeoMap
                glfw.KEY_T: 19,   # Thickness
            }
            if (not (mods & glfw.MOD_CONTROL)) and key in _letter_tool_map and hasattr(self, 'tool_manager'):
                tid = _letter_tool_map[key]
                if tid in self.tool_manager.tools:
                    self.tool_manager.set_active_tool(tid)
                    self.active_tool = tid
                    self.status_text = f"Tool: {self.tool_manager.tools[tid].name}"
                    return True

            if key == glfw.KEY_ESCAPE:
                if self.is_measuring_drag: self.is_measuring_drag = False; self.measure_start_pt = None; self.undo_stack.pop() if self.undo_stack else None
                elif self.is_poly_active: self.is_poly_active = False; self.knife_poly = []; self.status_text = "Polygon Cancelled."
                else: glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_ENTER or key == glfw.KEY_KP_ENTER:
                if self.active_tool == 1 and self.selection_submode == 1: self.apply_poly_cut()
            elif key == glfw.KEY_BACKSPACE:
                if self.active_tool == 1 and self.selection_submode == 1 and len(self.knife_poly) > 0: self.knife_poly.pop()
                else: self.delete_selection()
            elif key == glfw.KEY_DELETE: self.delete_selection()
            elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL): self.undo()
            elif key == glfw.KEY_Y and (mods & glfw.MOD_CONTROL): self.redo()
            elif key == glfw.KEY_I and (mods & glfw.MOD_CONTROL): self.invert_selection()
            elif key == glfw.KEY_E and (mods & glfw.MOD_CONTROL): self.gui_export_file_thread()
            elif key == glfw.KEY_R: self.save_state("Clear Selection"); self.selected[:] = False; self.update_colors()
            elif key == glfw.KEY_L: self.gui_open_file_thread()
