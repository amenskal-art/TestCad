# gui_layout.py

import imgui
import numpy as np
import glfw
import math

def draw_gui(app):
    # --- One-time UI state defaults (panel visibility + dialogs) ---
    if not hasattr(app, 'show_scene_browser'): app.show_scene_browser = True
    if not hasattr(app, 'show_tool_options'):  app.show_tool_options  = True
    if not hasattr(app, 'show_shortcuts'):     app.show_shortcuts     = False
    if not hasattr(app, 'show_about'):         app.show_about         = False

    TOP_BAR_H = int(28 * app.ui_scale)
    STATUS_H = int(32 * app.ui_scale)
    LEFT_PANEL_W = int(72 * app.ui_scale)

    panel_h = app.height - TOP_BAR_H - STATUS_H
    if hasattr(app, 'tool_manager'):
        num_tools = len(app.tool_manager.tools)
    else:
        num_tools = 5
        
    reserved = int(130 * app.ui_scale)
    spacing = int(8 * app.ui_scale)
    available = panel_h - reserved - (num_tools * spacing)
    
    ideal_btn_size = int(42 * app.ui_scale)
    max_btn_width = LEFT_PANEL_W - int(20 * app.ui_scale) 
    btn_size = min(ideal_btn_size, max_btn_width)
    
    if available > 0 and (btn_size * num_tools) > available:
        btn_size = max(int(24 * app.ui_scale), available // num_tools)
        
    app._tool_btn_size = btn_size  

    _draw_main_menu(app)

    imgui.set_next_window_position(0, TOP_BAR_H)
    imgui.set_next_window_size(LEFT_PANEL_W, panel_h)
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
             imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)

    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 1.0)
    imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0, spacing))
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (int(4 * app.ui_scale), int(4 * app.ui_scale)))
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.08, 0.08, 0.09, 1.0)
    imgui.begin("Tools", False, flags)
    
    if hasattr(app, 'tool_manager'):
        for tool_id, tool in app.tool_manager.tools.items():
            _draw_icon_tool_button(app, tool, tool_id)
    else:
        _draw_icon_tool_button_fallback(app, 'nav', 0, "Navigation")
        _draw_icon_tool_button_fallback(app, 'select', 1, "Selection")
        _draw_icon_tool_button_fallback(app, 'measure', 2, "Measure")
        _draw_icon_tool_button_fallback(app, 'sculpt', 3, "Surface Sculpt")
        _draw_icon_tool_button_fallback(app, 'scale', 4, "Scale / Transform")

    imgui.dummy(0, int(15 * app.ui_scale))
    imgui.separator()
    imgui.dummy(0, int(15 * app.ui_scale))
    
    imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.2, 0.2, 0.8)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.25, 0.25, 1.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.95, 0.3, 0.3, 1.0)
    if imgui.button("Del", width=-1, height=int(38 * app.ui_scale)):
        app.delete_selection()
    imgui.pop_style_color(3)
    
    if imgui.is_item_hovered():
        imgui.set_tooltip("Delete Selected Points  (Del)")
    
    imgui.dummy(0, int(6 * app.ui_scale))
    
    if imgui.button("Undo", width=-1, height=int(38 * app.ui_scale)):
        app.undo()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Undo Last Action  (Ctrl+Z)")

    imgui.end()
    imgui.pop_style_color(1)
    imgui.pop_style_var(4)
    
    if hasattr(app, 'tool_manager'):
        active_tool = app.tool_manager.get_active_tool()
        if active_tool and active_tool.name != "Navigation" and app.show_tool_options:
            imgui.set_next_window_position(
                LEFT_PANEL_W + int(20 * app.ui_scale),
                TOP_BAR_H + int(24 * app.ui_scale),
                imgui.FIRST_USE_EVER
            )
            win_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE
            imgui.begin(f"{active_tool.name} Options", False, win_flags)
            active_tool.draw_options()
            imgui.end()
    else:
        if app.active_tool != 0:
            _draw_legacy_tool_options(app)
            
    if app.show_scene_browser:
        _draw_scene_browser(app, TOP_BAR_H)
    
    _draw_status_bar(app, STATUS_H, LEFT_PANEL_W)
    
    if hasattr(app, 'nav_gizmo'):
        app.nav_gizmo.render(app)
        
    if getattr(app, 'is_exporting', False):
        _draw_export_overlay(app)

    # Help windows
    if app.show_shortcuts:
        _draw_shortcuts_window(app)
    if app.show_about:
        _draw_about_window(app)

    # New Scene confirmation popup
    if hasattr(app, 'new_dialog_requested') and app.new_dialog_requested:
        app.new_dialog_requested = False
        imgui.open_popup("New Scene?")

    if imgui.begin_popup_modal("New Scene?", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text("All unsaved changes will be lost. Continue?")
        imgui.separator()
        if imgui.button("Confirm", width=120):
            app.clear_scene()
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", width=120):
            imgui.close_current_popup()
        imgui.end_popup()

def _draw_scene_browser(app, top_offset):
    if not hasattr(app, 'point_object_ids') or len(app.points) == 0:
        return

    unique_ids = np.unique(app.point_object_ids)
    imported_ids = [uid for uid in unique_ids if uid != 0]
    
    if len(imported_ids) == 0:
        return

    win_w = int(280 * app.ui_scale)
    imgui.set_next_window_position(app.width - win_w - int(20 * app.ui_scale), top_offset + int(20 * app.ui_scale), imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(win_w, int(420 * app.ui_scale), imgui.FIRST_USE_EVER)
    
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.08, 0.08, 0.09, 0.95)
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, int(8 * app.ui_scale))
    
    imgui.begin("Scene Objects", True, imgui.WINDOW_NO_COLLAPSE)
    
    imgui.text_colored("Objects in Scene", 0.0, 1.0, 1.0, 1.0)
    imgui.separator()
    
    for obj_id in imported_ids:
        name = f"Part {obj_id}"
        obj_mask = (app.point_object_ids == obj_id)
        is_selected = np.all(app.selected[obj_mask]) and np.any(obj_mask)
        
        changed, active = imgui.selectable(name, is_selected)
        if changed:
            if not imgui.get_io().key_shift:
                app.selected[:] = False  
            app.select_sub_object(obj_id)
            
    imgui.dummy(0, int(15 * app.ui_scale))
    
    imgui.text_colored("Precision Transform", 0.0, 1.0, 1.0, 1.0)
    imgui.separator()
    
    if np.any(app.selected):
        if not hasattr(app, 'ui_step_size'): app.ui_step_size = 1.0
        changed, app.ui_step_size = imgui.slider_float("Step Size", app.ui_step_size, 0.1, 10.0)
        imgui.dummy(0, int(5 * app.ui_scale))
        
        imgui.text("Translate:")
        if imgui.button("-X", width=int(30*app.ui_scale)): app.translate_selection(-app.ui_step_size, 0, 0, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+X", width=int(30*app.ui_scale)): app.translate_selection(app.ui_step_size, 0, 0, snapshot=True)
        imgui.same_line(spacing=int(10*app.ui_scale))
        
        if imgui.button("-Y", width=int(30*app.ui_scale)): app.translate_selection(0, -app.ui_step_size, 0, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+Y", width=int(30*app.ui_scale)): app.translate_selection(0, app.ui_step_size, 0, snapshot=True)
        imgui.same_line(spacing=int(10*app.ui_scale))
        
        if imgui.button("-Z", width=int(30*app.ui_scale)): app.translate_selection(0, 0, -app.ui_step_size, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+Z", width=int(30*app.ui_scale)): app.translate_selection(0, 0, app.ui_step_size, snapshot=True)
        
        imgui.dummy(0, int(5 * app.ui_scale))
        
        imgui.text("Rotate (deg):")
        rot_step = max(1.0, app.ui_step_size * 5) 
        
        if imgui.button("-Xr", width=int(30*app.ui_scale)): app.rotate_selection_euler(-rot_step, 0, 0, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+Xr", width=int(30*app.ui_scale)): app.rotate_selection_euler(rot_step, 0, 0, snapshot=True)
        imgui.same_line(spacing=int(10*app.ui_scale))
        
        if imgui.button("-Yr", width=int(30*app.ui_scale)): app.rotate_selection_euler(0, -rot_step, 0, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+Yr", width=int(30*app.ui_scale)): app.rotate_selection_euler(0, rot_step, 0, snapshot=True)
        imgui.same_line(spacing=int(10*app.ui_scale))
        
        if imgui.button("-Zr", width=int(30*app.ui_scale)): app.rotate_selection_euler(0, 0, -rot_step, snapshot=True)
        imgui.same_line(); 
        if imgui.button("+Zr", width=int(30*app.ui_scale)): app.rotate_selection_euler(0, 0, rot_step, snapshot=True)
        
        imgui.dummy(0, int(5 * app.ui_scale))
        
        imgui.text("Uniform Scale:")
        scale_step = 0.05
        if imgui.button("Scale Down", width=int(80*app.ui_scale)): app.scale_selection(1.0 - scale_step, snapshot=True)
        imgui.same_line()
        if imgui.button("Scale Up", width=int(80*app.ui_scale)): app.scale_selection(1.0 + scale_step, snapshot=True)

    else:
        imgui.text_colored("Select an object to enable transforms.", 0.5, 0.5, 0.5, 1.0)
        
    imgui.end()
    imgui.pop_style_color()
    imgui.pop_style_var()

def _draw_icon_tool_button(app, tool, tool_id):
    texture_info = app.icons.get(tool.icon_key)
    is_active = (app.tool_manager.active_tool_id == tool_id) if hasattr(app, 'tool_manager') else (app.active_tool == tool_id)
    if is_active:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.04, 0.52, 1.0, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.18, 0.59, 1.0, 1.0)
    else:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
    size = getattr(app, '_tool_btn_size', int(42 * app.ui_scale))
    clicked = False
    avail_width = imgui.get_content_region_available().x
    if avail_width > size:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - size) / 2)
    if texture_info and texture_info[0] is not None:
        tex_id, _, _ = texture_info
        clicked = imgui.image_button(tex_id, size, size, (0,1), (1,0), (1,1,1,1), (0,0,0,0))
    else:
        txt = tool.name[:3].upper() if len(tool.name) >= 3 else f"T{tool_id}"
        clicked = imgui.button(txt, width=size, height=size)
    if clicked and hasattr(app, 'tool_manager'):
        app.tool_manager.set_active_tool(tool_id)
        app.active_tool = tool_id
    imgui.pop_style_color(2 if is_active else 1)
    if imgui.is_item_hovered():
        hotkey_info = f" ({tool.hotkey})" if tool.hotkey else ""
        imgui.set_tooltip(f"{tool.name}{hotkey_info}")

def _draw_icon_tool_button_fallback(app, icon_key, tool_id, tooltip):
    texture_info = app.icons.get(icon_key)
    is_active = (app.active_tool == tool_id)
    if is_active:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.04, 0.52, 1.0, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.18, 0.59, 1.0, 1.0)
    else:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
    size = getattr(app, '_tool_btn_size', int(42 * app.ui_scale))
    clicked = False
    avail_width = imgui.get_content_region_available().x
    if avail_width > size:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - size) / 2)
    if texture_info and texture_info[0] is not None:
        tex_id, _, _ = texture_info
        clicked = imgui.image_button(tex_id, size, size, (0,1), (1,0), (1,1,1,1), (0,0,0,0))
    else:
        txt = "T" + str(tool_id) if tool_id != 4 else "SCL"
        clicked = imgui.button(txt, width=size, height=size)
    if clicked:
        app.active_tool = tool_id
    imgui.pop_style_color(2 if is_active else 1)
    if imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)

def _draw_legacy_tool_options(app):
    tool_names = ["Navigation", "Selection", "Measurements",
                  "Surface Sculpt", "Scale / Transform"]
    title = tool_names[app.active_tool] if app.active_tool < len(tool_names) else "Tool Options"
    imgui.set_next_window_position(
        app.width - int(280 * app.ui_scale),
        int(28 * app.ui_scale + 24 * app.ui_scale),
        imgui.FIRST_USE_EVER
    )
    win_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE
    imgui.begin(title, False, win_flags)
    if app.active_tool == 1:
        imgui.text("Selection Mode")
        imgui.separator()
        if imgui.radio_button("Lasso", app.selection_submode == 0):
            app.selection_submode = 0
        imgui.same_line()
        if imgui.radio_button("Polygon", app.selection_submode == 1):
            app.selection_submode = 1
    elif app.active_tool == 2:
        if imgui.button("Clear All Measurements", width=-1):
            app.measurements = []
    elif app.active_tool == 3:
        imgui.text("Brush Settings")
        imgui.push_item_width(int(160 * app.ui_scale))
        _, app.sculpt_radius = imgui.slider_float("Radius", app.sculpt_radius, 0.1, 50.0)
        _, app.sculpt_strength = imgui.slider_float("Strength", app.sculpt_strength, 0.01, 1.0)
        imgui.pop_item_width()
    elif app.active_tool == 4:
        imgui.text("Scale Transform")
        imgui.text_wrapped("Drag the colored handles to scale the mesh.")
    imgui.end()

_EXPORT_FMTS = [
    ("LAS 1.4",   ".las", "Universal 3D Software & Total Station compatible (CloudCompare, Civil 3D, QGIS, etc.)"),
]

_EXPORT_CRS = ["WGS84 lat/lon", "UTM metres  * Civil 3D", "ENU metres from anchor", "Local mesh units"]

def _get_georef(app):
    if not hasattr(app, 'tool_manager'):
        return None
    for t in app.tool_manager.tools.values():
        if type(t).__name__ == 'GeoMapTool' and getattr(t, 'georef_active', False):
            return t
    return None

def _get_export_tool(app):
    return getattr(app, '_geo_exporter', None)

def _trigger_export(app, fmt_idx):
    tool = _get_export_tool(app)
    if tool is not None:
        tool._fmt_idx   = fmt_idx
        tool._coord_idx = app._menu_export_crs
        tool._thin_step = app._menu_export_thin
        tool._do_export()
        app._menu_export_status = tool._status
        app._menu_export_ok     = tool._status_ok
    else:
        app._menu_export_status = "[!] GeoExportTool not registered."
        app._menu_export_ok     = False

def _draw_export_submenu(app):
    georef = _get_georef(app)
    n_pts  = len(app.points) if hasattr(app, 'points') else 0
    if georef:
        imgui.text_colored("[ON] Georeferencing Active", 0.3, 1.0, 0.5, 1.0)
        imgui.text(f"  Anchor: {georef.georef_anchor_lat:.6f}, {georef.georef_anchor_lon:.6f}")
    else:
        imgui.text_colored("[OFF] No Georeferencing - Exporting in Local Units", 0.85, 0.65, 0.2, 1.0)
    imgui.text(f"  {n_pts:,} points loaded")
    imgui.separator()
    imgui.text_colored("Coordinate Reference System", 0.75, 0.75, 0.55, 1.0)
    imgui.push_item_width(220)
    if not georef:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.45, 0.45, 0.45, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.18, 0.18, 0.20, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.18, 0.18, 0.20, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.18, 0.18, 0.20, 1.0)
        imgui.combo("##crs", app._menu_export_crs, _EXPORT_CRS)
        imgui.pop_style_color(4)
        app._menu_export_crs = 3
    else:
        changed, app._menu_export_crs = imgui.combo("##crs", app._menu_export_crs, _EXPORT_CRS)
    imgui.pop_item_width()
    imgui.text_colored("Point Decimation", 0.75, 0.75, 0.55, 1.0)
    imgui.push_item_width(220)
    _, app._menu_export_thin = imgui.slider_int("Every N-th##thin", app._menu_export_thin, 1, 20)
    imgui.pop_item_width()
    out_pts = max(1, n_pts // max(1, app._menu_export_thin))
    imgui.text_colored(f"  -> {out_pts:,} pts exported", 0.55, 0.55, 0.55, 1.0)
    imgui.separator()
    imgui.text_colored("Output Format", 0.75, 0.75, 0.55, 1.0)
    for idx, (label, ext, compat) in enumerate(_EXPORT_FMTS):
        if imgui.menu_item(f"{label}  ({ext})")[0]:
            _trigger_export(app, idx)
        if imgui.is_item_hovered():
            imgui.set_tooltip(compat)
    if app._menu_export_status:
        imgui.separator()
        c = (0.3, 1.0, 0.5, 1.0) if app._menu_export_ok else (1.0, 0.45, 0.3, 1.0)
        imgui.text_colored(app._menu_export_status, *c)
    imgui.separator()
    imgui.text_colored("Quick Reference", 0.4, 0.75, 1.0, 1.0)
    if imgui.begin_menu("Open in Third-Party Applications"):
        imgui.text_colored("CloudCompare", 0.4, 0.85, 1.0, 1.0)
        imgui.text("  File -> Open -> .las")
        imgui.separator()
        imgui.text_colored("AutoCAD Civil 3D  (recommended)", 0.4, 0.85, 1.0, 1.0)
        imgui.text("  LAS UTM: Insert -> Point Cloud -> attach .las")
        imgui.separator()
        imgui.text_colored("Revit", 0.4, 0.85, 1.0, 1.0)
        imgui.text("  Use ReCap to index .las -> .rcp")
        imgui.text("  Then Insert -> Point Cloud in Revit")
        imgui.end_menu()

def _draw_save3d_submenu(app):
    """Submenu content for Save 3D File As."""
    n_pts   = len(app.points) if hasattr(app, 'points') else 0
    has_f   = getattr(app, 'has_faces', False) and app.faces is not None and len(app.faces) > 0
    has_n   = getattr(app, 'has_normals', False)

    # Info header
    imgui.text_colored(f"{n_pts:,} points", 0.55, 0.55, 0.55, 1.0)
    if has_f:
        imgui.same_line()
        imgui.text_colored(f"  {len(app.faces):,} faces", 0.55, 0.55, 0.55, 1.0)
    if has_n:
        imgui.same_line()
        imgui.text_colored("  normals [OK]", 0.3, 0.85, 0.5, 1.0)
    imgui.separator()

    # OBJ — always available
    if imgui.menu_item("OBJ Mesh  (.obj)")[0]:
        app.save_3d_file_dialog('obj')
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Wavefront OBJ with vertex colours.\n"
            "Open in: MeshLab | Blender | RealityCapture | CloudCompare"
        )

    # STL — only useful with faces, but allow attempt and let writer explain
    stl_label = "STL Mesh  (.stl)" if has_f else "STL Mesh  (.stl)  [!] needs faces"
    if imgui.menu_item(stl_label)[0]:
        app.save_3d_file_dialog('stl')
    if imgui.is_item_hovered():
        tip = (
            "Binary STL triangular mesh.\n"
            "Open in: CAD tools | Slicer | MeshLab | Blender"
            if has_f else
            "Binary STL requires triangulated faces.\n"
            "Run Poisson Reconstruction first to generate a mesh."
        )
        imgui.set_tooltip(tip)

    # XYZ — always available
    if imgui.menu_item("Point Cloud  (.xyz)")[0]:
        app.save_3d_file_dialog('xyz')
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "ASCII x y z r g b point cloud.\n"
            "Open in: CloudCompare | QGIS | Excel | any text tool"
        )

    # GLB — always available
    if imgui.menu_item("GLB / glTF Binary  (.glb)")[0]:
        app.save_3d_file_dialog('glb')
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "glTF 2.0 binary with colours"
            + (", normals" if has_n else "")
            + (" and mesh faces" if has_f else " (point cloud mode)")
            + ".\nOpen in: Blender | three.js | Babylon.js | Sketchfab"
        )


def _draw_main_menu(app):
    if not hasattr(app, '_menu_export_fmt'):    app._menu_export_fmt    = 0
    if not hasattr(app, '_menu_export_crs'):    app._menu_export_crs    = 0
    if not hasattr(app, '_menu_export_thin'):   app._menu_export_thin   = 1
    if not hasattr(app, '_menu_export_status'): app._menu_export_status = ""
    if not hasattr(app, '_menu_export_ok'):     app._menu_export_ok     = True

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            if imgui.menu_item("New Scene", "Ctrl+N")[0]:
                app.new_dialog_requested = True
            if imgui.menu_item("Open Project...", "Ctrl+O")[0]:
                app.open_project_dialog()
            if imgui.menu_item("Save Project", "Ctrl+S")[0]:
                app.save_project()
            imgui.separator()
            if imgui.menu_item("Load 3D File...", "Ctrl+I")[0]:
                app.gui_import_file_thread()
            if imgui.begin_menu("Export"):
                _draw_export_submenu(app)
                imgui.end_menu()
            if imgui.begin_menu("Save 3D File As"):
                _draw_save3d_submenu(app)
                imgui.end_menu()
            imgui.separator()
            if imgui.menu_item("Exit", "Esc")[0]:
                glfw.set_window_should_close(app.window, True)
            imgui.end_menu()
        
        if imgui.begin_menu("Edit"):
            if imgui.menu_item("Undo", "Ctrl+Z")[0]:
                app.undo()
            if imgui.menu_item("Redo", "Ctrl+Y")[0]:
                app.redo()
            imgui.separator()
            if imgui.menu_item("Select All", "Ctrl+A")[0]:
                app.select_all()
            if imgui.menu_item("Select None", "R")[0]:
                app.selected[:] = False
                app.update_colors()
            if imgui.menu_item("Invert Selection", "Ctrl+I")[0]:
                app.invert_selection()
            imgui.separator()
            if imgui.menu_item("Grow Selection")[0]:
                app.grow_selection()
            if imgui.menu_item("Shrink Selection")[0]:
                app.shrink_selection()
            if imgui.menu_item("Duplicate Selection")[0]:
                app.duplicate_selection()
            imgui.separator()
            if imgui.menu_item("Clear All Measurements")[0]:
                app.measurements = []
                app.status_text = "Measurements cleared."
            imgui.end_menu()
        
        if imgui.begin_menu("View"):
            _, app.show_grid = imgui.menu_item("Show Grid", "", app.show_grid)
            _, app.show_axes = imgui.menu_item("Show World Axes", "", app.show_axes)
            _, app.is_orthographic = imgui.menu_item("Orthographic Projection", "", app.is_orthographic)
            imgui.separator()
            imgui.text("Point Size:")
            imgui.push_item_width(int(140 * app.ui_scale))
            _, app.point_size = imgui.slider_float("##ptsize", app.point_size, 1.0, 15.0)
            imgui.pop_item_width()
            imgui.separator()
            imgui.text_colored("Unit System", 0.4, 0.9, 1.0, 1.0)
            imgui.text("Coordinate Unit:")
            imgui.same_line()
            imgui.push_item_width(int(90 * app.ui_scale))
            unit_labels = ["mm", "cm", "m", "km"]
            coord_idx   = unit_labels.index(app.coord_unit) if app.coord_unit in unit_labels else 0
            changed, coord_idx = imgui.combo("##coord_unit", coord_idx, unit_labels)
            if changed:
                app.set_coord_unit(unit_labels[coord_idx])
            imgui.pop_item_width()
            imgui.text("Display Unit: ")
            imgui.same_line()
            imgui.push_item_width(int(90 * app.ui_scale))
            disp_idx = unit_labels.index(app.display_unit) if app.display_unit in unit_labels else 0
            changed2, disp_idx = imgui.combo("##disp_unit", disp_idx, unit_labels)
            if changed2:
                app.set_display_unit(unit_labels[disp_idx])
            imgui.pop_item_width()
            if imgui.button("Auto-Detect Unit"):
                app._auto_detect_unit()
            if hasattr(app, 'points') and len(app.points) > 0:
                try:
                    import numpy as _np
                    _diag = float(_np.linalg.norm(_np.max(app.points, axis=0) - _np.min(app.points, axis=0)))
                    imgui.text_colored(f"  Bounding Box Diagonal: {_diag:.4g} {app.coord_unit}", 0.6, 0.6, 0.6, 1.0)
                except:
                    pass
            imgui.separator()
            imgui.text_colored("Interface Scaling", 0.4, 0.9, 1.0, 1.0)
            changed, new_scale = imgui.slider_float("UI Size", app.ui_scale, 0.5, 3.0, "%.2f")
            if changed:
                app.apply_ui_scale(new_scale)
            if imgui.button("Reset to 1.0x", width=-1):
                app.apply_ui_scale(1.0)
            imgui.end_menu()
        
        if imgui.begin_menu("Geometry"):
            if imgui.menu_item("Align to Ground Plane")[0]:
                app.align_floor()
            imgui.separator()
            if imgui.begin_menu("Rotate 90 deg"):
                if imgui.menu_item("X-Axis")[0]:
                    app.rotate_90_x()
                if imgui.menu_item("Y-Axis")[0]:
                    app.rotate_90_y()
                if imgui.menu_item("Z-Axis")[0]:
                    app.rotate_90_z()
                imgui.end_menu()
            if imgui.begin_menu("Mirror"):
                if imgui.menu_item("Across X")[0]:
                    app.mirror_geometry('x')
                if imgui.menu_item("Across Y")[0]:
                    app.mirror_geometry('y')
                if imgui.menu_item("Across Z")[0]:
                    app.mirror_geometry('z')
                imgui.end_menu()
            imgui.separator()
            if imgui.menu_item("Center on Origin")[0]:
                app.center_on_origin()
            if app.has_normals:
                if imgui.menu_item("Flip Normals")[0]:
                    app.flip_normals()
            imgui.end_menu()
        
        if imgui.begin_menu("Tools"):
            if hasattr(app, 'tool_manager'):
                for tool_id, tool in app.tool_manager.tools.items():
                    hotkey_str = f"  ({tool.hotkey})" if tool.hotkey else ""
                    if imgui.menu_item(f"{tool.name}{hotkey_str}")[0]:
                        app.tool_manager.set_active_tool(tool_id)
            imgui.end_menu()
        
        if imgui.begin_menu("Window"):
            _, app.show_scene_browser = imgui.menu_item("Scene Objects Panel", "", app.show_scene_browser)
            _, app.show_tool_options  = imgui.menu_item("Tool Options Panel", "", app.show_tool_options)
            imgui.separator()
            _, app.show_grid = imgui.menu_item("Show Grid", "", app.show_grid)
            _, app.show_axes = imgui.menu_item("Show World Axes", "", app.show_axes)
            imgui.end_menu()
        
        if imgui.begin_menu("Help"):
            if imgui.menu_item("Keyboard Shortcuts")[0]:
                app.show_shortcuts = True
            if imgui.menu_item("About")[0]:
                app.show_about = True
            imgui.end_menu()
        
        imgui.dummy(int(24 * app.ui_scale), 0)
        imgui.text("|")
        imgui.dummy(int(24 * app.ui_scale), 0)
        imgui.text(f"UI Scale: {app.ui_scale:.2f}x")
        
        imgui.end_main_menu_bar()

def _draw_status_bar(app, height, left_w):
    imgui.set_next_window_position(0, app.height - height)
    imgui.set_next_window_size(app.width, height)
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (8, 0))  # zero vertical padding so text fits the strip
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.06, 0.06, 0.07, 1.0)
    flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR
    imgui.begin("Status", False, flags)
    imgui.set_cursor_pos_y((height - imgui.get_text_line_height()) / 2)
    imgui.text(f"  {app.status_text}")
    if hasattr(app, 'tool_manager'):
        active_tool = app.tool_manager.get_active_tool()
        tool_info = f" | Tool: {active_tool.name}" if active_tool else ""
    else:
        tool_info = ""
    right_text = f"Points: {len(app.points):,}{tool_info}  |  {getattr(app,'coord_unit','mm')} -> {getattr(app,'display_unit','mm')}"
    text_w = imgui.calc_text_size(right_text).x
    imgui.same_line(app.width - text_w - int(24 * app.ui_scale))
    imgui.text(right_text)
    imgui.end()
    imgui.pop_style_color()
    imgui.pop_style_var(3)  # rounding + bordersize + padding

def _draw_shortcuts_window(app):
    imgui.set_next_window_position(app.width / 2, app.height / 2,
                                   imgui.FIRST_USE_EVER, pivot_x=0.5, pivot_y=0.5)
    imgui.set_next_window_size(int(360 * app.ui_scale), 0, imgui.FIRST_USE_EVER)
    expanded, app.show_shortcuts = imgui.begin("Keyboard Shortcuts", True,
                                               imgui.WINDOW_NO_COLLAPSE)
    if expanded:
        shortcuts = [
            ("Tools", None),
            ("Navigation", "1"),
            ("Selection", "2"),
            ("Measure", "3"),
            ("Wax Knife / Sculpt", "4"),
            ("Scale", "5"),
            ("Tools 6-10", "6 7 8 9 0"),
            ("Volume", "V"),
            ("Poisson", "P"),
            ("Smooth", "O"),
            ("GeoMap", "G"),
            ("Thickness", "T"),
            ("Edit", None),
            ("Undo", "Ctrl+Z"),
            ("Redo", "Ctrl+Y"),
            ("Select All", "Ctrl+A"),
            ("Select None", "R"),
            ("Invert Selection", "Ctrl+I"),
            ("Delete Selection", "Del / Backspace"),
            ("File", None),
            ("New Scene", "Ctrl+N"),
            ("Save Project", "Ctrl+S"),
            ("Load 3D File", "L"),
            ("Export", "Ctrl+E"),
            ("General", None),
            ("Cancel / Exit", "Esc"),
            ("Constrain Axis (while rotating)", "X / Y / Z"),
        ]
        for label, key in shortcuts:
            if key is None:
                imgui.dummy(0, int(4 * app.ui_scale))
                imgui.text_colored(label, 0.4, 0.9, 1.0, 1.0)
                imgui.separator()
            else:
                imgui.text(label)
                kw = imgui.calc_text_size(key).x
                imgui.same_line(imgui.get_window_width() - kw - int(20 * app.ui_scale))
                imgui.text_colored(key, 0.75, 0.75, 0.75, 1.0)
    imgui.end()

def _draw_about_window(app):
    imgui.set_next_window_position(app.width / 2, app.height / 2,
                                   imgui.FIRST_USE_EVER, pivot_x=0.5, pivot_y=0.5)
    imgui.set_next_window_size(int(320 * app.ui_scale), 0, imgui.FIRST_USE_EVER)
    expanded, app.show_about = imgui.begin("About", True,
                                           imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
    if expanded:
        imgui.text_colored("GridCad", 0.0, 1.0, 1.0, 1.0)
        version = getattr(app, 'app_version', '1.0.0')
        imgui.text(f"Version {version}")
        imgui.separator()
        n_pts = len(app.points) if hasattr(app, 'points') else 0
        imgui.text(f"Points loaded: {n_pts:,}")
        imgui.text(f"Coordinate unit: {getattr(app, 'coord_unit', 'mm')}")
        imgui.text(f"Display unit: {getattr(app, 'display_unit', 'mm')}")
        imgui.separator()
        imgui.text_colored("A 3D point cloud & mesh editing tool", 0.6, 0.6, 0.6, 1.0)
        imgui.dummy(0, int(8 * app.ui_scale))
        if imgui.button("Close", width=-1):
            app.show_about = False
    imgui.end()

def _draw_export_overlay(app):
    imgui.set_next_window_position(app.width / 2, app.height / 2, pivot_x=0.5, pivot_y=0.5)
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
             imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE |
             imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_FOCUS_ON_APPEARING |
             imgui.WINDOW_NO_INPUTS)
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.02, 0.02, 0.03, 0.95)
    imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 1.0, 1.0, 0.4)
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 2.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (int(45 * app.ui_scale), int(35 * app.ui_scale)))
    imgui.begin("Exporting_Overlay_HUD", True, flags)
    draw_list = imgui.get_window_draw_list()
    pos_x, pos_y = imgui.get_cursor_screen_pos()
    hud_radius = int(45 * app.ui_scale)
    center = (pos_x + imgui.get_window_width() / 2, pos_y + hud_radius)
    t = glfw.get_time()
    cyan_solid = imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0)
    cyan_glow = imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.4)
    magenta_solid = imgui.get_color_u32_rgba(0.8, 0.1, 1.0, 1.0)
    outer_r = hud_radius * 1.0
    segments = 36
    rot_outer = t * 1.5
    for i in range(segments):
        if i % 2 == 0:
            a1 = rot_outer + (i / segments) * math.pi * 2
            a2 = rot_outer + ((i + 0.8) / segments) * math.pi * 2
            p1_x, p1_y = center[0] + math.cos(a1) * outer_r, center[1] + math.sin(a1) * outer_r
            p2_x, p2_y = center[0] + math.cos(a2) * outer_r, center[1] + math.sin(a2) * outer_r
            draw_list.add_line(p1_x, p1_y, p2_x, p2_y, cyan_glow, 2.0)
    mid_r = hud_radius * 0.70
    rot_mid = -t * 4.0
    arc_length = math.pi * 1.2
    mid_points = []
    arc_segments = 20
    for i in range(arc_segments + 1):
        a = rot_mid + (i / arc_segments) * arc_length
        mid_points.append((center[0] + math.cos(a) * mid_r, center[1] + math.sin(a) * mid_r))
    draw_list.add_polyline(mid_points, cyan_solid, False, 4.0)
    inner_r = hud_radius * 0.35 + math.sin(t * 8.0) * (hud_radius * 0.08)
    rot_inner = t * 1.0
    hex_points = []
    for i in range(7):
        a = rot_inner + (i / 6.0) * math.pi * 2
        hex_points.append((center[0] + math.cos(a) * inner_r, center[1] + math.sin(a) * inner_r))
    draw_list.add_polyline(hex_points, magenta_solid, False, 2.0)
    cross_size = outer_r * 1.25
    draw_list.add_line(center[0] - cross_size, center[1], center[0] - outer_r, center[1], cyan_glow, 1.0)
    draw_list.add_line(center[0] + outer_r, center[1], center[0] + cross_size, center[1], cyan_glow, 1.0)
    draw_list.add_line(center[0], center[1] - cross_size, center[0], center[1] - outer_r, cyan_glow, 1.0)
    draw_list.add_line(center[0], center[1] + outer_r, center[0], center[1] + cross_size, cyan_glow, 1.0)
    imgui.dummy(hud_radius * 2.5, hud_radius * 2)
    imgui.dummy(0, int(15 * app.ui_scale))
    cursor = "_" if int(t * 4) % 2 == 0 else " "
    text1 = "SYS.EXPORT // DATA_STREAM_ACTIVE"
    text2 = f"> ENCODING MESH GEOMETRY {cursor}"
    t1_w = imgui.calc_text_size(text1).x
    t2_w = imgui.calc_text_size(text2).x
    imgui.set_cursor_pos_x((imgui.get_window_width() - t1_w) / 2)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 1.0, 0.8)
    imgui.text(text1)
    imgui.pop_style_color()
    imgui.set_cursor_pos_x((imgui.get_window_width() - t2_w) / 2)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.8, 0.1, 1.0, 1.0)
    imgui.text(text2)
    imgui.pop_style_color()
    imgui.end()
    imgui.pop_style_var(3)
    imgui.pop_style_color(2)
