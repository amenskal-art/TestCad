# tools/geo_map_tool.py
import imgui
import glfw
import numpy as np
import math
import threading
import requests
import io
import asyncio
import concurrent.futures
from PIL import Image
from OpenGL.GL import *
import winrt.windows.devices.geolocation as wdg

try:
    from pyproj import Transformer as _ProjTransformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

from tool_system import BaseTool


class GeoMapTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Geo Map & Geofence"
        self.icon_key = 'gps'
        self.hotkey = 'M'

        self.lat = 33.3152
        self.lon = 44.3661
        self.alt = 35.0
        self.zoom = 18

        self.device_lat = None
        self.device_lon = None

        # Slippy Map Cache
        self.tile_textures = {}
        self.pending_tiles = set()
        self.texture_queue = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        # Map navigation state
        self.shape_type = 0
        self.pan_start_pos = None
        self.map_screen_size = (640, 640)

        # ── GEOREFERENCING STATE ──────────────────────────────────────────────
        self.georef_anchor_lat  = None   
        self.georef_anchor_lon  = None
        self.georef_anchor_alt  = 0.0
        self.georef_scale       = 1.0    
        self.georef_heading_deg = 0.0    
        self.georef_active      = False  
        self._georef_status_msg = ""

    # =========================================================================
    #  LIFECYCLE
    # =========================================================================
    def on_activate(self):
        self.app.status_text = "Geo Map: Set an anchor on the map to geolocate your mesh."
        self._fetch_gps()

    def reset_georef_state(self):
        """Clear all mesh-specific georeferencing state on new scene."""
        self.georef_anchor_lat  = None
        self.georef_anchor_lon  = None
        self.georef_anchor_alt  = 0.0
        self.georef_scale       = 1.0
        self.georef_heading_deg = 0.0
        self.georef_active      = False
        self._georef_status_msg = ""

    # =========================================================================
    #  SLIPPY-MAP MATH
    # =========================================================================
    def latlon_to_global_px(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = (lon + 180.0) / 360.0 * n * 256.0
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256.0
        return x, y

    def global_px_to_latlon(self, x, y, zoom):
        n = 2.0 ** zoom
        lon_deg = x / 256.0 / n * 360.0 - 180.0
        n_pi = math.pi * (1 - 2 * y / 256.0 / n)
        n_pi = max(-math.pi * 0.999, min(math.pi * 0.999, n_pi))
        lat_rad = math.atan(math.sinh(n_pi))
        return math.degrees(lat_rad), lon_deg

    def latlon_to_screen(self, lat, lon, w, h):
        px, py = self.latlon_to_global_px(lat, lon, self.zoom)
        cpx, cpy = self.latlon_to_global_px(self.lat, self.lon, self.zoom)
        return w / 2 + (px - cpx), h / 2 + (py - cpy)

    def screen_to_latlon(self, sx, sy, w, h):
        cpx, cpy = self.latlon_to_global_px(self.lat, self.lon, self.zoom)
        px = cpx + (sx - w / 2)
        py = cpy + (sy - h / 2)
        return self.global_px_to_latlon(px, py, self.zoom)

    # =========================================================================
    #  GEOREFERENCING MATH
    # =========================================================================
    def _fallback_latlon_to_utm(self, lat, lon):
        a = 6378137.0
        eccSquared = 0.00669437999014
        k0 = 0.9996
        zone = int((lon + 180) / 6) + 1
        lonOrigin = (zone - 1) * 6 - 180 + 3
        eccPrimeSquared = (eccSquared) / (1 - eccSquared)
        N = a / math.sqrt(1 - eccSquared * math.sin(math.radians(lat))**2)
        T = math.tan(math.radians(lat))**2
        C = eccPrimeSquared * math.cos(math.radians(lat))**2
        A = math.cos(math.radians(lat)) * math.radians(lon - lonOrigin)
        M = a * ((1 - eccSquared/4 - 3*eccSquared**2/64 - 5*eccSquared**3/256) * math.radians(lat) 
                 - (3*eccSquared/8 + 3*eccSquared**2/32 + 45*eccSquared**3/1024) * math.sin(2 * math.radians(lat))
                 + (15*eccSquared**2/256 + 45*eccSquared**3/1024) * math.sin(4 * math.radians(lat)) 
                 - (35*eccSquared**3/3072) * math.sin(6 * math.radians(lat)))
        easting = (k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * eccPrimeSquared) * A**5 / 120) + 500000.0)
        northing = (k0 * (M + N * math.tan(math.radians(lat)) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24 + (61 - 58 * T + T**2 + 600 * C - 330 * eccPrimeSquared) * A**6 / 720)))
        if lat < 0:
            northing += 10000000.0
        return easting, northing

    def _fallback_utm_to_latlon(self, easting, northing, zone, northern_hemi=True):
        a = 6378137.0
        eccSquared = 0.00669437999014
        k0 = 0.9996
        e1 = (1 - math.sqrt(1 - eccSquared)) / (1 + math.sqrt(1 - eccSquared))
        x = easting - 500000.0
        y = northing if northern_hemi else northing - 10000000.0
        m = y / k0
        mu = m / (a * (1 - eccSquared/4 - 3*eccSquared**2/64 - 5*eccSquared**3/256))
        phi1Rad = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu) + (151*e1**3/96)*math.sin(6*mu)
        N1 = a / math.sqrt(1 - eccSquared * math.sin(phi1Rad)**2)
        T1 = math.tan(phi1Rad)**2
        C1 = (eccSquared / (1 - eccSquared)) * math.cos(phi1Rad)**2
        R1 = a * (1 - eccSquared) / math.pow(1 - eccSquared * math.sin(phi1Rad)**2, 1.5)
        D = x / (N1 * k0)
        lat = phi1Rad - (N1 * math.tan(phi1Rad) / R1) * (D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*eccSquared)*D**4/24 + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*eccSquared - 3*C1**2)*D**6/720)
        lonOrigin = (zone - 1) * 6 - 180 + 3
        lon = (D - (1 + 2*T1 + C1)*D**3/6 + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*eccSquared + 24*T1**2)*D**5/120) / math.cos(phi1Rad)
        return math.degrees(lat), lonOrigin + math.degrees(lon)

    def _mesh_xyz_to_enu(self, x, y, z):
        h = math.radians(self.georef_heading_deg)
        fwd = -z
        right = x
        east  = (right * math.cos(h) + fwd * math.sin(h)) * self.georef_scale
        north = (-right * math.sin(h) + fwd * math.cos(h)) * self.georef_scale
        up    = y * self.georef_scale
        return east, north, up

    def _get_utm_epsg(self, lat, lon):
        zone = int((lon + 180) / 6) + 1
        base = 32600 if lat >= 0 else 32700
        return f"EPSG:{base + zone}"

    def _latlon_to_utm(self, lat, lon):
        if _HAS_PYPROJ:
            epsg = self._get_utm_epsg(lat, lon)
            t = _ProjTransformer.from_crs("EPSG:4326", epsg, always_xy=True)
            return t.transform(lon, lat)
        return self._fallback_latlon_to_utm(lat, lon)

    def _utm_to_latlon(self, easting, northing, epsg=None):
        if _HAS_PYPROJ and epsg:
            t = _ProjTransformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            lon, lat = t.transform(easting, northing)
            return lat, lon
        zone = int((self.georef_anchor_lon + 180) / 6) + 1
        hemi = self.georef_anchor_lat >= 0
        return self._fallback_utm_to_latlon(easting, northing, zone, hemi)

    def _enu_to_latlon(self, east_m, north_m):
        if not self.georef_active:
            return None, None
        ae, an = self._latlon_to_utm(self.georef_anchor_lat, self.georef_anchor_lon)
        epsg = self._get_utm_epsg(self.georef_anchor_lat, self.georef_anchor_lon) if _HAS_PYPROJ else None
        return self._utm_to_latlon(ae + east_m, an + north_m, epsg)

    def _latlon_to_enu(self, lat, lon):
        if not self.georef_active:
            return 0.0, 0.0
        ae, an = self._latlon_to_utm(self.georef_anchor_lat, self.georef_anchor_lon)
        if _HAS_PYPROJ:
            epsg = self._get_utm_epsg(self.georef_anchor_lat, self.georef_anchor_lon)
            t = _ProjTransformer.from_crs("EPSG:4326", epsg, always_xy=True)
            te, tn = t.transform(lon, lat)
        else:
            te, tn = self._fallback_latlon_to_utm(lat, lon)
        return te - ae, tn - an

    def _mesh_point_to_latlon(self, x, y, z):
        east, north, up = self._mesh_xyz_to_enu(x, y, z)
        lat, lon = self._enu_to_latlon(east, north)
        return lat, lon, self.georef_anchor_alt + up

    def _get_mesh_footprint_latlons(self):
        if not self.georef_active or not hasattr(self.app, 'points') or len(self.app.points) == 0:
            return []
        pts = self.app.points
        min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
        min_z, max_z = float(pts[:, 2].min()), float(pts[:, 2].max())
        corners_mesh = [(min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z)]
        footprint = []
        for cx, cz in corners_mesh:
            lat, lon, _ = self._mesh_point_to_latlon(cx, 0.0, cz)
            if lat is not None: footprint.append((lat, lon))
        return footprint

    # =========================================================================
    #  APPLY GEOREF
    # =========================================================================
    def _apply_georef_to_mesh(self):
        if not self.georef_active:
            self._georef_status_msg = "⚠ Set an anchor first."
            return
        if not hasattr(self.app, 'points') or len(self.app.points) == 0:
            self._georef_status_msg = "⚠ No mesh loaded."
            return

        self.app.save_state("Georeferencing")
        pts = self.app.points.astype(np.float64)
        centroid = pts.mean(axis=0)
        pts -= centroid
        pts *= self.georef_scale

        h = math.radians(self.georef_heading_deg)
        cos_h, sin_h = math.cos(h), math.sin(h)
        Ry = np.array([[ cos_h, 0, sin_h], [0, 1, 0], [-sin_h, 0, cos_h]], dtype=np.float64)
        pts = (Ry @ pts.T).T

        self.app.points = pts.astype(np.float32)
        self.app.points_colors[:, 0:3] = self.app.points

        if hasattr(self.app, 'normals') and self.app.normals is not None and len(self.app.normals):
            nrm = (Ry @ self.app.normals.astype(np.float64).T).T
            self.app.normals = nrm.astype(np.float32)

        self.app.update_vbo_full()
        self.app._rebuild_kdtree()

        self._georef_status_msg = (
            f"✔ Applied — centroid at anchor ({self.georef_anchor_lat:.6f}, "
            f"{self.georef_anchor_lon:.6f}, alt {self.georef_anchor_alt:.1f} m)."
        )

    # =========================================================================
    #  BACKGROUND WORKERS 
    # =========================================================================
    def _fetch_gps(self):
        def worker():
            try:
                async def get_loc():
                    locator = wdg.Geolocator()
                    locator.desired_accuracy = wdg.PositionAccuracy.HIGH
                    access_status = await wdg.Geolocator.request_access_async()
                    if access_status == wdg.GeolocationAccessStatus.ALLOWED:
                        pos = await locator.get_geoposition_async()
                        return pos.coordinate.latitude, pos.coordinate.longitude, pos.coordinate.altitude
                    return None, None, None
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                lat, lon, alt = loop.run_until_complete(get_loc())
                if lat and lon:
                    self.device_lat, self.device_lon = lat, lon
                    self.lat, self.lon = lat, lon
                    if alt is not None:
                        self.alt = alt
            except Exception as e:
                print(f"GPS Error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _request_tile(self, z, x, y):
        if (z, x, y) in self.tile_textures or (z, x, y) in self.pending_tiles:
            return
        self.pending_tiles.add((z, x, y))
        def fetch():
            url = f"https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
                    self.texture_queue.append((z, x, y, img))
            except: pass
            finally: self.pending_tiles.discard((z, x, y))
        self.executor.submit(fetch)

    def _process_texture_queue(self):
        while self.texture_queue:
            z, x, y, img = self.texture_queue.pop(0)
            img_data = np.array(img)
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            self.tile_textures[(z, x, y)] = tex
        if len(self.tile_textures) > 300:
            keys_to_remove = [k for k in self.tile_textures if k[0] != self.zoom]
            for k in keys_to_remove[:50]:
                tex = self.tile_textures.pop(k)
                glDeleteTextures([tex])

    # =========================================================================
    #  SIDEBAR OPTIONS
    # =========================================================================
    def draw_options(self):
        imgui.text_colored("Mesh Georeferencing", 0.4, 0.85, 1.0)
        imgui.separator()

        if imgui.button("Center Map on My Location"):
            if self.device_lat is not None and self.device_lon is not None:
                self.lat, self.lon = self.device_lat, self.device_lon
            else:
                self._fetch_gps()

        imgui.dummy(0, 10)
        imgui.separator()
        imgui.text_colored("Anchor & Alignment", 0.4, 1.0, 0.7)
        imgui.separator()

        if self.shape_type == 1:
            imgui.text_colored("Click on the map to place the anchor.", 1.0, 1.0, 0.3)
        else:
            if imgui.button("Set Anchor on Map"):
                self.shape_type = 1

        if self.georef_active:
            imgui.text_colored(f"Anchor: {self.georef_anchor_lat:.6f}, {self.georef_anchor_lon:.6f}", 0.4, 1.0, 0.4)
            imgui.text(f"        Alt: {self.georef_anchor_alt:.2f} m")
            changed, val = imgui.input_float("Anchor Alt (m)", self.georef_anchor_alt, 0.5, 5.0, "%.2f")
            if changed: self.georef_anchor_alt = val
        else:
            imgui.text_colored("No anchor set.", 0.7, 0.7, 0.7)

        imgui.dummy(0, 4)
        
        # Heading slider for manual visual alignment
        imgui.text("Mesh Rotation:")
        _, self.georef_heading_deg = imgui.slider_float("Heading (°)", self.georef_heading_deg, -180.0, 180.0, "%.1f°")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rotate the mesh footprint to align with map imagery.")

        imgui.dummy(0, 10)
        imgui.separator()

        full_w = imgui.get_content_region_available()[0]

        if self.georef_active:
            imgui.text_colored("Ready to apply", 0.4, 1.0, 0.4)
        else:
            imgui.text_colored("Waiting for: anchor", 0.6, 0.6, 0.6)

        imgui.dummy(0, 4)
        if imgui.button("Apply to Mesh", full_w, 36):
            self._apply_georef_to_mesh()

        if self._georef_status_msg:
            col = (0.4, 1.0, 0.4) if self._georef_status_msg.startswith("✔") else (1.0, 0.5, 0.3)
            imgui.text_colored(self._georef_status_msg, *col)

    # =========================================================================
    #  MAP WINDOW  (2-D overlay)
    # =========================================================================
    def render_2d_overlay(self):
        self._process_texture_queue()

        imgui.set_next_window_size(700, 750, imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("Live GPS Map", True)
        if not expanded:
            imgui.end()
            return

        imgui.text(f"Location: {self.lat:.6f}, {self.lon:.6f} | Alt: {self.alt:.1f}m | Zoom: {self.zoom}")
        imgui.same_line()
        if imgui.button("Zoom In +"):  self.zoom = min(21, self.zoom + 1)
        imgui.same_line()
        if imgui.button("Zoom Out -"): self.zoom = max(2,  self.zoom - 1)

        w, h = imgui.get_content_region_available()
        w, h = max(100, w), max(100, h)
        self.map_screen_size = (w, h)
        cursor_pos = imgui.get_cursor_screen_pos()

        draw_list = imgui.get_window_draw_list()
        imgui.invisible_button("map_canvas", w, h)
        is_hovered = imgui.is_item_hovered()
        is_active  = imgui.is_item_active()
        io = imgui.get_io()

        mx = io.mouse_pos.x - cursor_pos[0]
        my = io.mouse_pos.y - cursor_pos[1]
        hover_lat, hover_lon = self.screen_to_latlon(mx, my, w, h)

        if is_hovered and io.mouse_wheel != 0:
            self.zoom += 1 if io.mouse_wheel > 0 else -1
            self.zoom = max(2, min(21, self.zoom))

        if self.shape_type == 1:
            if is_hovered and imgui.is_mouse_clicked(0):
                self.georef_anchor_lat = hover_lat
                self.georef_anchor_lon = hover_lon
                self.georef_active     = True
                self._georef_status_msg = f"Anchor set: {hover_lat:.6f}, {hover_lon:.6f}"
                self.lat, self.lon = hover_lat, hover_lon
                self.shape_type = 0
        elif self.shape_type == 0:
            if is_hovered and imgui.is_mouse_clicked(0):
                self.pan_start_pos = (io.mouse_pos.x, io.mouse_pos.y)
            if is_active and imgui.is_mouse_dragging(0) and self.pan_start_pos:
                dx = io.mouse_pos.x - self.pan_start_pos[0]
                dy = io.mouse_pos.y - self.pan_start_pos[1]
                if dx != 0 or dy != 0:
                    cpx, cpy = self.latlon_to_global_px(self.lat, self.lon, self.zoom)
                    cpx -= dx;  cpy -= dy
                    self.lat, self.lon = self.global_px_to_latlon(cpx, cpy, self.zoom)
                    self.pan_start_pos = (io.mouse_pos.x, io.mouse_pos.y)
            if imgui.is_mouse_released(0):
                self.pan_start_pos = None

        draw_list.push_clip_rect(cursor_pos[0], cursor_pos[1], cursor_pos[0] + w, cursor_pos[1] + h, True)
        cpx, cpy   = self.latlon_to_global_px(self.lat, self.lon, self.zoom)
        view_x, view_y = cpx - w / 2, cpy - h / 2
        start_tx, start_ty = int(view_x // 256), int(view_y // 256)
        end_tx,   end_ty   = int((view_x + w) // 256), int((view_y + h) // 256)

        for ty in range(start_ty, end_ty + 1):
            for tx in range(start_tx, end_tx + 1):
                sx = cursor_pos[0] + (tx * 256 - view_x)
                sy = cursor_pos[1] + (ty * 256 - view_y)
                key = (self.zoom, tx, ty)
                if key in self.tile_textures:
                    draw_list.add_image(self.tile_textures[key], (sx, sy), (sx + 256, sy + 256))
                else:
                    self._request_tile(self.zoom, tx, ty)
                    draw_list.add_rect_filled(sx, sy, sx+256, sy+256, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
                    draw_list.add_rect(sx, sy, sx+256, sy+256, imgui.get_color_u32_rgba(0.25, 0.25, 0.25, 1.0))

        if self.georef_active:
            ax, ay = self.latlon_to_screen(self.georef_anchor_lat, self.georef_anchor_lon, w, h)
            ax += cursor_pos[0];  ay += cursor_pos[1]
            arm = 14
            anchor_col  = imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 1.0)
            anchor_fill = imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.25)
            draw_list.add_circle_filled(ax, ay, 10, anchor_fill)
            draw_list.add_circle(ax, ay, 10, anchor_col, thickness=2)
            draw_list.add_line(ax - arm, ay, ax + arm, ay, anchor_col, 2)
            draw_list.add_line(ax, ay - arm, ax, ay + arm, anchor_col, 2)
            draw_list.add_text(ax + 12, ay - 10, anchor_col, "ANCHOR")

            # Draw Footprint (Visually rotated by heading slider)
            footprint = self._get_mesh_footprint_latlons()
            if len(footprint) == 4:
                fp_screen = []
                for fla, flo in footprint:
                    fx, fy = self.latlon_to_screen(fla, flo, w, h)
                    fp_screen.append((cursor_pos[0] + fx, cursor_pos[1] + fy))
                fp_col  = imgui.get_color_u32_rgba(0.4, 1.0, 0.7, 1.0)
                fp_fill = imgui.get_color_u32_rgba(0.4, 1.0, 0.7, 0.12)
                for i in range(4):
                    j = (i + 1) % 4
                    draw_list.add_line(fp_screen[i][0], fp_screen[i][1], fp_screen[j][0], fp_screen[j][1], fp_col, 2)
                draw_list.add_quad_filled(
                    fp_screen[0][0], fp_screen[0][1], fp_screen[1][0], fp_screen[1][1],
                    fp_screen[2][0], fp_screen[2][1], fp_screen[3][0], fp_screen[3][1], fp_fill
                )
                if hasattr(self.app, 'points') and len(self.app.points):
                    pts = self.app.points
                    dx_m = (pts[:, 0].max() - pts[:, 0].min()) * self.georef_scale
                    dz_m = (pts[:, 2].max() - pts[:, 2].min()) * self.georef_scale
                    lx = (fp_screen[0][0] + fp_screen[2][0]) / 2 - 40
                    ly = (fp_screen[0][1] + fp_screen[2][1]) / 2 - 7
                    draw_list.add_text(lx, ly, fp_col, f"Mesh: {dx_m:.2f} x {dz_m:.2f} m")

            arrow_len = 30
            h_rad = math.radians(self.georef_heading_deg)
            nx = ax + arrow_len * math.sin(h_rad)
            ny = ay - arrow_len * math.cos(h_rad)   
            north_col = imgui.get_color_u32_rgba(1.0, 1.0, 0.2, 1.0)
            draw_list.add_line(ax, ay, nx, ny, north_col, 2)
            draw_list.add_circle_filled(nx, ny, 4, north_col)
            draw_list.add_text(nx + 5, ny - 6, north_col, "N")

        if self.device_lat is not None and self.device_lon is not None:
            dev_x, dev_y = self.latlon_to_screen(self.device_lat, self.device_lon, w, h)
            scr_x = cursor_pos[0] + dev_x
            scr_y = cursor_pos[1] + dev_y
            draw_list.add_circle_filled(scr_x, scr_y, 12, imgui.get_color_u32_rgba(0.2, 0.6, 1.0, 0.3))
            draw_list.add_circle_filled(scr_x, scr_y, 7,  imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0))
            draw_list.add_circle_filled(scr_x, scr_y, 5,  imgui.get_color_u32_rgba(0.1, 0.5, 1.0, 1.0))

        if self.shape_type == 1 and is_hovered:
            hint_col = imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.9)
            draw_list.add_line(io.mouse_pos.x - 12, io.mouse_pos.y, io.mouse_pos.x + 12, io.mouse_pos.y, hint_col, 2)
            draw_list.add_line(io.mouse_pos.x, io.mouse_pos.y - 12, io.mouse_pos.x, io.mouse_pos.y + 12, hint_col, 2)
            draw_list.add_circle(io.mouse_pos.x, io.mouse_pos.y, 8, hint_col, thickness=1)

        draw_list.pop_clip_rect()
        imgui.end()
