# tools/navigation_tool.py
from tool_system import BaseTool
import imgui
import glfw
import numpy as np
import math


def _make_rotation_matrix(axis, angle_rad):
    """
    3×3 rotation matrix (float32) via the matrix form of Rodrigues' formula.
    """
    k  = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    t    = 1.0 - c
    kx, ky, kz = k
    return np.array([
        [t*kx*kx + c,     t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz,  t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky,  t*ky*kz + s*kx, t*kz*kz + c   ],
    ], dtype=np.float32)


_AXIS_LOCK_THRESHOLD_PX = 8


class NavigationTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name     = "Navigation"
        self.icon_key = 'nav'
        self.hotkey   = '1'

        self.drag_start_x = 0
        self.drag_start_y = 0

        # Mesh-only transform modes
        self._is_mesh_pan  = False
        self._is_mesh_rot  = False
        self._temp_sel_all = False

        # Pivot-based rotation (Ctrl + LMB)
        self._rot_pivot       = None
        self._rot_snapshot    = None
        self._rot_axis_locked = None

        # Camera-aligned pan (Shift + LMB)
        self._pan_snapshot = None

        # Per-frame accumulator
        self.active_dx    = 0
        self.active_dy    = 0
        self.needs_update = False

        # ── Coordinate inspector state ─────────────────────────────────────
        # Mouse position in GL coords (bottom-left origin), updated every move.
        self._mouse_x_gl = 0
        self._mouse_y_gl = 0
        # Cached pick result: (vertex_index, np.array xyz) or None.
        # Refreshed once per frame so we never hit depth-buffer twice.
        self._hovered_vtx_idx = None   # int | None
        self._hovered_vtx_xyz = None   # np.float32[3] | None
        self._pick_dirty      = True   # flag: re-pick on next overlay draw

    # =========================================================================
    #  Internal helpers
    # =========================================================================

    def _ensure_selection(self):
        if not np.any(self.app.selected):
            self.app.selected[:] = True
            self._temp_sel_all = True
        else:
            self._temp_sel_all = False

    def _restore_selection(self):
        if self._temp_sel_all:
            self.app.selected[:] = False
            self._temp_sel_all = False

    def _camera_axes(self):
        R = self.app.rotation_matrix[:3, :3].astype(np.float32)
        return R[0], R[1]

    def _begin_mesh_rot(self, x, y_gl):
        pivot = self.app.get_world_pos_at_mouse(x, y_gl)
        if pivot is None:
            sel = self.app.points[self.app.selected]
            pivot = sel.mean(axis=0) if len(sel) else np.zeros(3, dtype=np.float32)
        self._rot_pivot       = np.array(pivot, dtype=np.float32)
        self._rot_snapshot    = self.app.points[self.app.selected].copy()
        self._rot_axis_locked = None

    def _begin_mesh_pan(self):
        self._pan_snapshot = self.app.points[self.app.selected].copy()

    def _clear_mesh_rot(self):
        self._rot_pivot       = None
        self._rot_snapshot    = None
        self._rot_axis_locked = None

    def _clear_mesh_pan(self):
        self._pan_snapshot = None

    def _reset_camera_drag_state(self):
        self.app._last_dxdy = None

    # ── Measurement tool notifications ────────────────────────────────────────

    def _notify_measurement_translation(self, offset):
        """Tell every measurement tool that selected points moved by offset."""
        if not hasattr(self.app, 'tool_manager'):
            return
        for tool in self.app.tool_manager.tools.values():
            if hasattr(tool, 'apply_translation'):
                tool.apply_translation(offset)

    def _notify_measurement_rotation(self, pivot, R):
        """Tell every measurement tool that selected points rotated by R around pivot."""
        if not hasattr(self.app, 'tool_manager'):
            return
        for tool in self.app.tool_manager.tools.values():
            if hasattr(tool, 'apply_rotation'):
                tool.apply_rotation(pivot, R)

    # ── Georef tool lookup ────────────────────────────────────────────────────

    def _get_georef_tool(self):
        """Return the GeoMapTool instance if it exists and has an active anchor."""
        if not hasattr(self.app, 'tool_manager'):
            return None
        for tool in self.app.tool_manager.tools.values():
            if type(tool).__name__ == 'GeoMapTool' and getattr(tool, 'georef_active', False):
                return tool
        return None

    # ── Vertex picking ────────────────────────────────────────────────────────

    def _pick_vertex_under_cursor(self):
        """
        Unproject the mouse ray via the depth buffer, then run a single
        KD-tree query to find the nearest mesh vertex.

        Returns (idx: int, xyz: np.float32[3]) or (None, None).
        Only valid while an OpenGL context is current (call from overlay draw).
        """
        if not hasattr(self.app, '_kdtree') or self.app._kdtree is None:
            return None, None
        if not hasattr(self.app, 'points') or len(self.app.points) == 0:
            return None, None

        world_pos = self.app.get_world_pos_at_mouse(self._mouse_x_gl, self._mouse_y_gl)
        if world_pos is None:
            return None, None

        _, idx = self.app._kdtree.query(world_pos, workers=1)
        idx = int(idx)
        return idx, self.app.points[idx].copy()

    # =========================================================================
    #  Tool lifecycle
    # =========================================================================

    def on_activate(self):
        self.app.status_text = (
            "Navigation | Shift+drag: pan mesh | Ctrl+drag: rotate mesh (axis-locks) | "
            "Hover vertex: inspect coordinates"
        )
        self._pick_dirty = True

    def on_deactivate(self):
        self._is_mesh_pan         = False
        self._is_mesh_rot         = False
        self.app._dragging_rotate = False
        self.app._dragging_pan    = False
        self.needs_update         = False
        self._hovered_vtx_idx     = None
        self._hovered_vtx_xyz     = None
        self._clear_mesh_rot()
        self._clear_mesh_pan()
        self._restore_selection()
        self._reset_camera_drag_state()

    # =========================================================================
    #  Mouse events
    # =========================================================================

    def on_mouse_button(self, button, action, mods, x, y):
        if imgui.get_io().want_capture_mouse:
            return False

        win_h = self.app.height
        y_gl  = win_h - y

        is_shift = bool(mods & glfw.MOD_SHIFT)
        is_ctrl  = bool(mods & glfw.MOD_CONTROL)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.drag_start_x = x
                self.drag_start_y = y_gl
                self.active_dx    = 0
                self.active_dy    = 0

                if is_shift:
                    self._ensure_selection()
                    self.app.begin_transform()
                    self._begin_mesh_pan()
                    self._is_mesh_pan = True
                    self.app.status_text = "Panning mesh (camera-aligned)…"

                elif is_ctrl:
                    self._ensure_selection()
                    self.app.begin_transform()
                    self._begin_mesh_rot(x, y_gl)
                    self._is_mesh_rot = True
                    self.app.status_text = "Rotating mesh — drag to lock axis…"

                else:
                    self.app._dragging_rotate = True
                    self.app._last_dxdy       = (x, y_gl)

                return True

            elif action == glfw.RELEASE:
                if self._is_mesh_pan:
                    # Notify measurement tools before clearing snapshot
                    if self._pan_snapshot is not None and np.any(self.app.selected):
                        sel_pts = self.app.points[self.app.selected]
                        if len(sel_pts) > 0 and len(self._pan_snapshot) > 0:
                            offset = (sel_pts[0] - self._pan_snapshot[0]).astype(np.float32)
                            self._notify_measurement_translation(offset)
                    self.app.end_transform("Pan Mesh")
                    self._restore_selection()
                    self._clear_mesh_pan()
                    self._is_mesh_pan = False
                    self.app.status_text = (
                        "Navigation | Shift+drag: pan mesh | Ctrl+drag: rotate mesh (axis-locks)"
                    )
                elif self._is_mesh_rot:
                    # Compute final rotation matrix and notify measurement tools
                    if self._rot_pivot is not None and self._rot_snapshot is not None:
                        speed = 0.5
                        if self._rot_axis_locked == 'yaw':
                            angle_yaw = math.radians(self.active_dx * speed)
                            angle_pit = 0.0
                        elif self._rot_axis_locked == 'pitch':
                            angle_yaw = 0.0
                            angle_pit = math.radians(-self.active_dy * speed)
                        else:
                            angle_yaw = math.radians(self.active_dx * speed)
                            angle_pit = math.radians(-self.active_dy * speed)
                        cam_right, cam_up = self._camera_axes()
                        R_yaw = _make_rotation_matrix(cam_up,    angle_yaw)
                        R_pit = _make_rotation_matrix(cam_right, angle_pit)
                        R = (R_yaw @ R_pit).astype(np.float32)
                        self._notify_measurement_rotation(self._rot_pivot, R)
                    self.app.end_transform("Rotate Mesh")
                    self._restore_selection()
                    self._clear_mesh_rot()
                    self._is_mesh_rot = False
                    self.app.status_text = (
                        "Navigation | Shift+drag: pan mesh | Ctrl+drag: rotate mesh (axis-locks)"
                    )

                self.app._dragging_rotate = False
                self.app._last_dxdy       = None
                self.needs_update         = False
                return True

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.drag_start_x      = x
                self.drag_start_y      = y_gl
                self.active_dx         = 0
                self.active_dy         = 0
                self.app._dragging_pan = True
                self.app._last_dxdy    = (x, y_gl)
                return True
            elif action == glfw.RELEASE:
                self.app._dragging_pan = False
                self.app._last_dxdy    = None
                self.needs_update      = False
                return True

        return False

    # ------------------------------------------------------------------

    def on_mouse_move(self, x, y):
        if imgui.get_io().want_capture_mouse:
            return False

        win_h = self.app.height
        y_gl  = win_h - y

        # Always keep GL-space mouse position fresh for the inspector.
        self._mouse_x_gl = x
        self._mouse_y_gl = y_gl
        self._pick_dirty = True   # depth pick needs a refresh

        any_mesh_drag = self._is_mesh_pan or self._is_mesh_rot
        if any_mesh_drag:
            self.active_dx    = x    - self.drag_start_x
            self.active_dy    = y_gl - self.drag_start_y
            self.needs_update = True
            return True

        return False

    # =========================================================================
    #  Per-frame mesh transform math
    # =========================================================================

    def update(self):
        if not self.needs_update:
            return

        # --- Ensure the points array is writable (fix for "assignment destination is read-only") ---
        if not self.app.points.flags.writeable:
            try:
                self.app.points.flags.writeable = True
            except ValueError:
                # If it's a view of read‑only data (e.g. memory‑mapped), create a writable copy
                self.app.points = self.app.points.copy()

        cam_right, cam_up = self._camera_axes()

        if self._is_mesh_pan and self._pan_snapshot is not None:
            pixel_scale  = self.app.distance * 0.002
            world_offset = (cam_right * (self.active_dx * pixel_scale) +
                            cam_up    * (self.active_dy * pixel_scale))
            self.app.points[self.app.selected] = self._pan_snapshot + world_offset
            self.app.update_vbo_subset()
            self.needs_update = False

        elif self._is_mesh_rot and self._rot_pivot is not None and self._rot_snapshot is not None:
            speed = 0.5
            dist  = math.sqrt(self.active_dx ** 2 + self.active_dy ** 2)

            if self._rot_axis_locked is None and dist >= _AXIS_LOCK_THRESHOLD_PX:
                if abs(self.active_dx) >= abs(self.active_dy):
                    self._rot_axis_locked = 'yaw'
                    self.app.status_text = "Rotating mesh — YAW (horizontal) locked"
                else:
                    self._rot_axis_locked = 'pitch'
                    self.app.status_text = "Rotating mesh — PITCH (vertical) locked"

            if self._rot_axis_locked == 'yaw':
                angle_yaw = math.radians(self.active_dx * speed);  angle_pit = 0.0
            elif self._rot_axis_locked == 'pitch':
                angle_yaw = 0.0;  angle_pit = math.radians(-self.active_dy * speed)
            else:
                angle_yaw = math.radians(self.active_dx * speed)
                angle_pit = math.radians(-self.active_dy * speed)

            R_yaw = _make_rotation_matrix(cam_up,    angle_yaw)
            R_pit = _make_rotation_matrix(cam_right, angle_pit)
            R     = R_yaw @ R_pit
            pts   = (self._rot_snapshot - self._rot_pivot) @ R.T
            self.app.points[self.app.selected] = pts + self._rot_pivot
            self.app.update_vbo_subset()
            self.needs_update = False

    # =========================================================================
    #  2-D overlay  —  two-line cursor label only, no windows
    # =========================================================================

    def render_2d_overlay(self):
        io = imgui.get_io()

        # Refresh vertex pick once per frame when the mouse moved.
        if self._pick_dirty and not (self._is_mesh_pan or self._is_mesh_rot):
            self._hovered_vtx_idx, self._hovered_vtx_xyz = self._pick_vertex_under_cursor()
            self._pick_dirty = False

        if self._hovered_vtx_idx is None or self._hovered_vtx_xyz is None:
            return

        x, y, z = (float(v) for v in self._hovered_vtx_xyz)

        # Line 1: local XYZ
        line1 = f"X {x:+.4f}  Y {y:+.4f}  Z {z:+.4f}"

        # Line 2: WGS84 if georef is active, otherwise blank
        georef = self._get_georef_tool()
        line2  = ""
        if georef:
            lat, lon, _ = georef._mesh_point_to_latlon(x, y, z)
            if lat is not None:
                line2 = f"{lat:.6f},  {lon:.6f}"

        # Draw directly on the foreground draw list — no imgui window at all.
        dl   = imgui.get_foreground_draw_list()
        mx   = io.mouse_pos.x + 16
        my   = io.mouse_pos.y + 10
        lh   = 15                               # line height in pixels
        shadow = imgui.get_color_u32_rgba(0, 0, 0, 0.7)
        white  = imgui.get_color_u32_rgba(1, 1, 1, 1)
        green  = imgui.get_color_u32_rgba(0.35, 1, 0.6, 1)

        # Shadow offset (+1,+1) keeps text readable on any background.
        dl.add_text(mx + 1, my + 1,          shadow, line1)
        dl.add_text(mx,     my,               white,  line1)
        if line2:
            dl.add_text(mx + 1, my + lh + 1, shadow, line2)
            dl.add_text(mx,     my + lh,      green,  line2)
