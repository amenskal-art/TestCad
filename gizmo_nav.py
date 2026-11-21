# gizmo_nav.py
import imgui
import numpy as np
import math
import glfw
from utils import create_rotation_matrix, normalize

class NavGizmo:
    def __init__(self):
        self.is_dragging = False
        self.is_panning = False
        self.last_mouse_pos = None
        
        # State variables for smart click-vs-drag
        self.waiting_for_snap = False
        self.snap_axis = None
        self.drag_start_pos = (0, 0)
        self.drag_threshold = 5.0 

    def _orthonormalize(self, m):
        x = normalize(m[:3, 0])
        y = normalize(m[:3, 1] - np.dot(x, m[:3, 1]) * x)
        m[:3, 0], m[:3, 1], m[:3, 2] = x, y, np.cross(x, y)
        return m

    def render(self, app):
        io = imgui.get_io()
        win_w, win_h = io.display_size 
        
        # --- Configuration ---
        size = 140 * app.ui_scale
        extra_h = 40 * app.ui_scale 
        margin = 10 * app.ui_scale
        padding_top = 30 * app.ui_scale 
        
        pos_x = win_w - size - margin
        pos_y = margin + padding_top
        center_x = pos_x + size / 2.0
        center_y = pos_y + size / 2.0
        radius = size * 0.35
        hit_radius = 15 * app.ui_scale 

        pan_btn_radius = 14 * app.ui_scale
        pan_btn_y = center_y + radius + 28 * app.ui_scale

        # --- 1. Axis Data ---
        rm = app.rotation_matrix 
        axes_data = [
            {'label': 'X',  'vec': rm[:3, 0],  'col': (0.9, 0.2, 0.2), 'neg': False},
            {'label': '-X', 'vec': -rm[:3, 0], 'col': (0.9, 0.2, 0.2), 'neg': True},
            {'label': 'Y',  'vec': rm[:3, 1],  'col': (0.2, 0.8, 0.2), 'neg': False},
            {'label': '-Y', 'vec': -rm[:3, 1], 'col': (0.2, 0.8, 0.2), 'neg': True},
            {'label': 'Z',  'vec': rm[:3, 2],  'col': (0.2, 0.4, 0.9), 'neg': False},
            {'label': '-Z', 'vec': -rm[:3, 2], 'col': (0.2, 0.4, 0.9), 'neg': True},
        ]
        axes_data.sort(key=lambda d: d['vec'][2])

        mx, my = imgui.get_mouse_pos()
        hovered_axis = None
        for ax in axes_data:
            ax['px'] = center_x + ax['vec'][0] * radius
            ax['py'] = center_y - ax['vec'][1] * radius
            dist = math.sqrt((mx - ax['px'])**2 + (my - ax['py'])**2)
            if dist < hit_radius: hovered_axis = ax['label']

        # --- 2. UI Elements ---
        imgui.set_next_window_position(pos_x, pos_y)
        imgui.set_next_window_size(size, size + extra_h)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BACKGROUND | 
                 imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)
        
        imgui.begin("##NavGizmo", False, flags)
        draw_list = imgui.get_window_draw_list()
        
        # Rotation Hitbox
        imgui.set_cursor_pos((0,0))
        imgui.invisible_button("##gizmo_rot_hitbox", size, size)
        is_rot_clicked = imgui.is_item_clicked(0) 
        is_rot_active = imgui.is_item_active()    

        # Pan Hitbox
        pan_btn_local_y = (pan_btn_y - pos_y) - pan_btn_radius
        pan_btn_local_x = (size / 2.0) - pan_btn_radius
        imgui.set_cursor_pos((pan_btn_local_x, pan_btn_local_y))
        imgui.invisible_button("##gizmo_pan_hitbox", pan_btn_radius*2, pan_btn_radius*2)
        is_pan_active = imgui.is_item_active()
        is_pan_hovered = imgui.is_item_hovered()

        # --- 3. Logic: Pan ---
        if is_pan_active:
            self.is_panning = True
            if self.last_mouse_pos is not None:
                dx = mx - self.last_mouse_pos[0]
                dy = my - self.last_mouse_pos[1]
                
                fov_rad = math.radians(60.0)
                visible_h = 2.0 * app.distance * math.tan(fov_rad / 2.0)
                pixel_scale = visible_h / app.height
                
                if not hasattr(app, 'pan_x'): app.pan_x = 0.0
                if not hasattr(app, 'pan_y'): app.pan_y = 0.0
                
                app.pan_x += dx * pixel_scale
                app.pan_y -= dy * pixel_scale

            # Infinite scroll — wrap cursor at screen edges, same as trackball
            if hasattr(app, 'window'):
                w_glfw, h_glfw = glfw.get_window_size(app.window)
                wrap_margin = 2

                new_x, new_y = mx, my
                warped = False

                if mx <= wrap_margin:
                    new_x = w_glfw - wrap_margin - 1
                    warped = True
                elif mx >= w_glfw - wrap_margin:
                    new_x = wrap_margin + 1
                    warped = True

                if my <= wrap_margin:
                    new_y = h_glfw - wrap_margin - 1
                    warped = True
                elif my >= h_glfw - wrap_margin:
                    new_y = wrap_margin + 1
                    warped = True

                if warped:
                    glfw.set_cursor_pos(app.window, new_x, new_y)
                    self.last_mouse_pos = (new_x, new_y)
                else:
                    self.last_mouse_pos = (mx, my)
            else:
                self.last_mouse_pos = (mx, my)
            
        elif not is_rot_active: 
            self.is_panning = False
            if not self.is_dragging: self.last_mouse_pos = None

        # --- 4. Logic: Rotation & Snap (with Infinite Scroll) ---
        if not self.is_panning:
            
            # A. Mouse Down
            if is_rot_clicked:
                self.drag_start_pos = (mx, my)
                if hovered_axis:
                    self.waiting_for_snap = True
                    self.snap_axis = hovered_axis
                    self.is_dragging = False
                else:
                    self.waiting_for_snap = False
                    self.is_dragging = True
                    self.last_mouse_pos = (mx, my)

            # B. Mouse Held
            if is_rot_active:
                if self.waiting_for_snap:
                    drag_dist = math.sqrt((mx - self.drag_start_pos[0])**2 + (my - self.drag_start_pos[1])**2)
                    if drag_dist > self.drag_threshold:
                        self.waiting_for_snap = False
                        self.is_dragging = True
                        self.last_mouse_pos = (mx, my) 

                if self.is_dragging and self.last_mouse_pos is not None:
                    # 1. Calculate Delta
                    dx = mx - self.last_mouse_pos[0]
                    dy = my - self.last_mouse_pos[1]
                    
                    # 2. Apply Rotation
                    if dx != 0 or dy != 0:
                        speed = app.rot_speed * 1.5
                        delta_rot = create_rotation_matrix([1,0,0], dy * speed) @ \
                                    create_rotation_matrix([0,1,0], dx * speed)
                        app.rotation_matrix = delta_rot @ app.rotation_matrix
                        self._orthonormalize(app.rotation_matrix)

                    # 3. Infinite Scroll (Wrap Mouse)
                    if hasattr(app, 'window'):
                        # Get actual OS window size
                        w_glfw, h_glfw = glfw.get_window_size(app.window)
                        wrap_margin = 2 # distance from edge to trigger wrap
                        
                        new_x, new_y = mx, my
                        warped = False

                        # Check X bounds
                        if mx <= wrap_margin:
                            new_x = w_glfw - wrap_margin - 1
                            warped = True
                        elif mx >= w_glfw - wrap_margin:
                            new_x = wrap_margin + 1
                            warped = True

                        # Check Y bounds
                        if my <= wrap_margin:
                            new_y = h_glfw - wrap_margin - 1
                            warped = True
                        elif my >= h_glfw - wrap_margin:
                            new_y = wrap_margin + 1
                            warped = True

                        if warped:
                            # Warp cursor physically
                            glfw.set_cursor_pos(app.window, new_x, new_y)
                            # Reset `last_mouse_pos` to the NEW warped location 
                            # so the next frame calculates delta from the other side
                            self.last_mouse_pos = (new_x, new_y)
                        else:
                            # Normal tracking
                            self.last_mouse_pos = (mx, my)
                    else:
                        # Fallback if app.window is missing
                        self.last_mouse_pos = (mx, my)

            # C. Mouse Released
            if self.waiting_for_snap and imgui.is_mouse_released(0):
                self.snap_view(app, self.snap_axis)
                self.waiting_for_snap = False
            
            if not is_rot_active:
                self.is_dragging = False
                self.waiting_for_snap = False

        # --- 5. Draw Geometry ---
        pan_col = (0.8, 0.8, 0.8) if is_pan_hovered or self.is_panning else (0.5, 0.5, 0.5)
        draw_list.add_circle_filled(center_x, pan_btn_y, pan_btn_radius, imgui.get_color_u32_rgba(*pan_col, 1.0))
        draw_list.add_circle(center_x, pan_btn_y, pan_btn_radius, 0xFF000000, thickness=1.5)
        
        icon_sz = pan_btn_radius * 0.5
        draw_list.add_line(center_x - icon_sz, pan_btn_y, center_x + icon_sz, pan_btn_y, 0xFF000000, 1.5)
        draw_list.add_line(center_x, pan_btn_y - icon_sz, center_x, pan_btn_y + icon_sz, 0xFF000000, 1.5)

        for ax in axes_data:
            p_x, p_y = ax['px'], ax['py']

            # FIX 1 — negative axes: draw smaller, dimmer filled circle so they're
            # visually distinct from positive axes but still readable
            if ax['neg']:
                col_dim = imgui.get_color_u32_rgba(ax['col'][0]*0.6, ax['col'][1]*0.6, ax['col'][2]*0.6, 1.0)
                circle_radius = 10 * app.ui_scale if ax['label'] == hovered_axis else 8 * app.ui_scale
                draw_list.add_circle_filled(p_x, p_y, circle_radius, col_dim)
                draw_list.add_circle(p_x, p_y, circle_radius, imgui.get_color_u32_rgba(*ax['col'], 0.5), thickness=1.0)
            else:
                col_u32 = imgui.get_color_u32_rgba(*ax['col'], 1.0)
                draw_list.add_line(center_x, center_y, p_x, p_y, col_u32, 2.0)
                circle_radius = 12 * app.ui_scale if ax['label'] == hovered_axis else 10 * app.ui_scale
                draw_list.add_circle_filled(p_x, p_y, circle_radius, col_u32)

            # FIX 2 — draw label text for ALL axes (positive white, negative grey)
            # Previously wrapped in `if not ax['neg']:` which hid all negative labels
            text_col = 0xFFFFFFFF if not ax['neg'] else 0xFFAAAAAA
            tw, th = imgui.calc_text_size(ax['label'])
            draw_list.add_text(p_x - tw/2, p_y - th/2, text_col, ax['label'])

        imgui.end()

    def snap_view(self, app, axis_label):
        # FIX 3a — derive the identity via rot_y(0) so the returned matrix type
        # (3×3 or 4×4) always matches what every other branch returns from
        # create_rotation_matrix.  The old `np.eye(4)` hard-coded 4×4 which could
        # crash matrix-multiply if create_rotation_matrix returns 3×3.
        #
        # FIX 3b — rot_x / rot_y now pass degrees directly to create_rotation_matrix
        # instead of pre-converting to radians.  Previously math.radians(90) ≈ 1.57
        # was passed as if it were already in the unit that create_rotation_matrix
        # expects (degrees), producing only a ~1.57° snap instead of 90°.
        if axis_label == 'Z':    target = self.rot_y(0)    # identity — looking down +Z
        elif axis_label == '-Z': target = self.rot_y(180)
        elif axis_label == 'X':  target = self.rot_y(-90)
        elif axis_label == '-X': target = self.rot_y(90)
        elif axis_label == 'Y':  target = self.rot_x(90)
        elif axis_label == '-Y': target = self.rot_x(-90)
        else:                    target = self.rot_y(0)

        app.rotation_matrix = target
        app.status_text = f"View Snapped: {axis_label}"

    def rot_x(self, deg):
        # Pass degrees directly — create_rotation_matrix handles the conversion
        # internally.  Do NOT pre-convert with math.radians() here.
        return create_rotation_matrix([1, 0, 0], deg)

    def rot_y(self, deg):
        # Same as rot_x — degrees only, no pre-conversion.
        return create_rotation_matrix([0, 1, 0], deg)
