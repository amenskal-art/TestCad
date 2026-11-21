# tools/export_geo_tool.py
"""
Georeferenced Export
====================
Plain helper class — NOT a BaseTool.  Attach to app directly in core.py:

    from tools.export_geo_tool import GeoExportTool
    self._geo_exporter = GeoExportTool(self)

All UI lives in the menu bar (gui_layout._draw_export_submenu).
This module only contains the coordinate math and the LAS file writer.

Supported formats
-----------------
  LAS 1.4    – georeferenced point cloud (Universal 3D Software & Total Station compatible)
"""

import os
import math
import struct
import datetime
import json
import numpy as np

try:
    from pyproj import Transformer as _ProjTransformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False


# =============================================================================
#  COORDINATE HELPERS
# =============================================================================

def _mesh_to_enu(points: np.ndarray, scale: float, heading_deg: float) -> np.ndarray:
    """
    Mesh local XYZ (N x 3)  ->  ENU metres (N x 3, float64).

    Convention (matches GeoMapTool):
      mesh +X = right,  mesh -Z = forward,  mesh +Y = up
      heading = degrees clockwise from North to mesh +X
    """
    pts  = points.astype(np.float64)
    h    = math.radians(heading_deg)
    c, s = math.cos(h), math.sin(h)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fwd   = -z
    east  = (x * c + fwd * s) * scale
    north = (-x * s + fwd * c) * scale
    up    = y * scale
    return np.stack([east, north, up], axis=1)


def _get_utm_epsg(anchor_lat: float, anchor_lon: float) -> str:
    """Return EPSG code string for the UTM zone covering the anchor."""
    zone = int((anchor_lon + 180) / 6) + 1
    base = 32600 if anchor_lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def _utm_wkt(anchor_lat: float, anchor_lon: float) -> str:
    """Return a WKT CRS string for the UTM zone covering the anchor."""
    zone   = int((anchor_lon + 180) / 6) + 1
    hemi   = "N" if anchor_lat >= 0 else "S"
    name   = f"WGS 84 / UTM zone {zone}{hemi}"
    epsg   = int(_get_utm_epsg(anchor_lat, anchor_lon).split(':')[1])
    # Minimal but valid WKT1 that Civil 3D / QGIS / ArcGIS recognise
    return (
        f'PROJCS["{name}",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],'
        f'PARAMETER["latitude_of_origin",0],'
        f'PARAMETER["central_meridian",{(zone - 1) * 6 - 177}],'
        'PARAMETER["scale_factor",0.9996],'
        f'PARAMETER["false_easting",500000],'
        f'PARAMETER["false_northing",{"10000000" if anchor_lat < 0 else "0"}],'
        'UNIT["metre",1]]'
    )


def _enu_to_wgs84(enu: np.ndarray,
                  anchor_lat: float, anchor_lon: float, anchor_alt: float):
    """
    ENU metres (N x 3)  ->  lat (N,), lon (N,), alt (N,) — float64.
    Uses pyproj when available; falls back to flat-earth approximation.
    """
    if _HAS_PYPROJ:
        epsg = _get_utm_epsg(anchor_lat, anchor_lon)
        # anchor in UTM
        t_fwd = _ProjTransformer.from_crs("EPSG:4326", epsg, always_xy=True)
        ae, an = t_fwd.transform(anchor_lon, anchor_lat)
        te = ae + enu[:, 0]
        tn = an + enu[:, 1]
        # back to WGS84
        t_inv = _ProjTransformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        lon_arr, lat_arr = t_inv.transform(te, tn)
        alt = anchor_alt + enu[:, 2]
        return lat_arr, lon_arr, alt
    else:
        cos_lat = math.cos(math.radians(anchor_lat))
        lat = anchor_lat + enu[:, 1] / 111_320.0
        lon = anchor_lon + enu[:, 0] / (111_320.0 * cos_lat)
        alt = anchor_alt + enu[:, 2]
        return lat, lon, alt


def _fallback_latlon_to_utm(lat: float, lon: float):
    """Pure math fallback for WGS84 to UTM when pyproj is missing."""
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
    
    easting = (k0 * N * (A + (1 - T + C) * A**3 / 6
                       + (5 - 18 * T + T**2 + 72 * C - 58 * eccPrimeSquared) * A**5 / 120)
               + 500000.0)
    
    northing = (k0 * (M + N * math.tan(math.radians(lat)) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
                                                           + (61 - 58 * T + T**2 + 600 * C - 330 * eccPrimeSquared) * A**6 / 720)))
    if lat < 0:
        northing += 10000000.0
        
    return easting, northing


def _enu_to_utm(enu: np.ndarray,
                anchor_lat: float, anchor_lon: float, anchor_alt: float):
    """
    ENU metres (N x 3)  ->  easting (N,), northing (N,), elev (N,) in UTM.
    Auto-detects UTM zone from anchor. Uses pyproj if available, else pure math fallback.
    """
    if _HAS_PYPROJ:
        epsg = _get_utm_epsg(anchor_lat, anchor_lon)
        t = _ProjTransformer.from_crs("EPSG:4326", epsg, always_xy=True)
        ae, an = t.transform(anchor_lon, anchor_lat)
    else:
        ae, an = _fallback_latlon_to_utm(anchor_lat, anchor_lon)
        
    easting  = ae + enu[:, 0]
    northing = an + enu[:, 1]
    elev     = anchor_alt + enu[:, 2]
    return easting, northing, elev


# =============================================================================
#  LAS 1.4  (hand-rolled, no laspy)
# =============================================================================

_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def write_las14(path: str,
                x: np.ndarray, y: np.ndarray, z: np.ndarray,
                r: np.ndarray = None, g: np.ndarray = None, b: np.ndarray = None,
                crs_wkt: str = "") -> None:
    """
    Write standard LAS 1.4.
    PDRF 6 (XYZ only, 30 bytes) or PDRF 7 (XYZ + RGB, 36 bytes).
    Scale 1e-7 for geographic degrees, 1e-4 for metric/ENU.
    """
    n       = len(x)
    has_rgb = (r is not None and g is not None and b is not None)
    
    # Strictly conform to LAS 1.4 PDRF 6 and 7
    fmt_id  = 7 if has_rgb else 6
    pt_len  = 36 if has_rgb else 30

    scale_xyz = 1e-7 if (abs(x).max() < 400) else 1e-4
    x_off = float(x.mean())
    y_off = float(y.mean())
    z_off = float(z.mean())
    xi = np.round((x - x_off) / scale_xyz).astype(np.int32)
    yi = np.round((y - y_off) / scale_xyz).astype(np.int32)
    zi = np.round((z - z_off) / scale_xyz).astype(np.int32)

    now = datetime.datetime.now()
    doy = now.timetuple().tm_yday

    vlr_bytes = b''
    vlr_count = 0
    global_encoding = 1  # bit 0: GPS Time Type = Standard
    
    if crs_wkt:
        wkt_b     = crs_wkt.encode('utf-8')
        user_id   = b'LASF_Projection\x00'      # 16 bytes
        vlr_bytes = (user_id
                     + struct.pack('<H', 2112)
                     + struct.pack('<H', len(wkt_b))
                     + b'\x00' * 32
                     + wkt_b)
        vlr_count = 1
        global_encoding = 17  # bit 0 (GPS Time) + bit 4 (WKT active)

    hdr_size  = 375
    pt_offset = hdr_size + len(vlr_bytes)

    # Pack the exact 375 byte structure defined by ASPRS LAS 1.4
    hdr = struct.pack(
        '<'
        '4s HH IHH 8s BB 32s 32s HH ' # 94 bytes (Header part 1)
        'H I I B H I IIIII '          # 37 bytes (Counts, Formats)
        'ddd ddd dddddd '             # 96 bytes (Scales, Offsets, Max/Min)
        'Q Q I Q 15Q',                # 148 bytes (Waveform, EVLRs, Total records)
        b'LASF',
        0, global_encoding,
        0, 0, 0, b'\x00' * 8,
        1, 4,
        b'GeoMapTool'.ljust(32, b'\x00'),
        b'3D Editor GeoExport'.ljust(32, b'\x00'),
        doy, now.year,
        hdr_size,
        pt_offset,
        vlr_count,
        fmt_id, pt_len,
        min(n, 0xFFFFFFFF),
        min(n, 0xFFFFFFFF), 0, 0, 0, 0,
        scale_xyz, scale_xyz, scale_xyz,
        x_off, y_off, z_off,
        float(x.max()), float(x.min()),
        float(y.max()), float(y.min()),
        float(z.max()), float(z.min()),
        0, 0, 0,
        n,
        n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )

    with open(path, 'wb') as f:
        f.write(hdr)
        if vlr_bytes:
            f.write(vlr_bytes)

        # Match exact C-struct alignment for PDRF 6 & 7 to prevent data corruption
        if has_rgb:
            dt = np.dtype([
                ('x', '<i4'), ('y', '<i4'), ('z', '<i4'),
                ('intensity', '<u2'),
                ('return_info', 'u1'), ('class_flags', 'u1'),
                ('classification', 'u1'), ('user_data', 'u1'),
                ('scan_angle', '<i2'), ('point_source_id', '<u2'),
                ('gps_time', '<f8'),
                ('red', '<u2'), ('green', '<u2'), ('blue', '<u2'),
            ], align=False)
        else:
            dt = np.dtype([
                ('x', '<i4'), ('y', '<i4'), ('z', '<i4'),
                ('intensity', '<u2'),
                ('return_info', 'u1'), ('class_flags', 'u1'),
                ('classification', 'u1'), ('user_data', 'u1'),
                ('scan_angle', '<i2'), ('point_source_id', '<u2'),
                ('gps_time', '<f8'),
            ], align=False)
        
        pts = np.zeros(n, dtype=dt)
        pts['x'], pts['y'], pts['z'] = xi, yi, zi
        
        # 17 = 0b00010001 (Return 1, Number of returns 1)
        pts['return_info'] = 17 
        
        if has_rgb:
            pts['red']   = (np.clip(r, 0, 1) * 65535).astype(np.uint16)
            pts['green'] = (np.clip(g, 0, 1) * 65535).astype(np.uint16)
            pts['blue']  = (np.clip(b, 0, 1) * 65535).astype(np.uint16)

        f.write(pts.tobytes())


# =============================================================================
#  GeoExportTool  —  plain class, no BaseTool, no tool strip icon
# =============================================================================

class GeoExportTool:
    """
    Attach to app in core.py __init__:

        from tools.export_geo_tool import GeoExportTool
        self._geo_exporter = GeoExportTool(self)

    gui_layout._draw_export_submenu sets _fmt_idx / _coord_idx / _thin_step
    then calls _do_export().  Reads _status / _status_ok for the result badge.
    """

    def __init__(self, app):
        self.app = app

        # Written by gui_layout before each export call
        self._fmt_idx   = 0   # 0=LAS
        self._coord_idx = 0   # 0=WGS84  1=UTM metres  2=ENU metres  3=local mesh units
        self._thin_step = 1   # export every Nth point (1 = all)

        # Read back by gui_layout to display the result badge
        self._status    = ""
        self._status_ok = True

        self._confirm_large = False   # two-click safety gate for >2M pts

    # ── Georef lookup ─────────────────────────────────────────────────────────

    def _get_georef(self):
        """Return the active GeoMapTool instance, or None."""
        if not hasattr(self.app, 'tool_manager'):
            return None
        for t in self.app.tool_manager.tools.values():
            if type(t).__name__ == 'GeoMapTool' and getattr(t, 'georef_active', False):
                return t
        return None

    # ── Coordinate preparation ────────────────────────────────────────────────

    def _prepare_coords(self, georef):
        """
        Return (x, y, z, r, g, b, faces, normals, meta) ready for the writers.
        All coordinate arrays are float64.  r/g/b are float32 0-1 or None.
        """
        pts  = self.app.points.astype(np.float64)
        step = max(1, self._thin_step)
        if step > 1:
            pts = pts[::step]

        # Colours
        cols = getattr(self.app, 'base_colors', None)
        if cols is not None and len(cols) >= len(self.app.points):
            cols = cols.astype(np.float32)[::step]
            r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
        else:
            r = g = b = None

        # Faces and normals are only valid without thinning
        faces_raw = getattr(self.app, 'faces', None)
        faces = (faces_raw
                 if step == 1 and faces_raw is not None and len(faces_raw)
                 else None)

        nrm_raw = getattr(self.app, 'normals', None)
        normals = (nrm_raw.astype(np.float32)
                   if nrm_raw is not None
                   and len(nrm_raw) == len(self.app.points)
                   and step == 1
                   else None)

        meta = None

        if self._coord_idx == 0 and georef:
            # WGS84 geographic (lat/lon degrees)
            enu = _mesh_to_enu(pts, georef.georef_scale, georef.georef_heading_deg)
            lat, lon, alt = _enu_to_wgs84(
                enu,
                georef.georef_anchor_lat,
                georef.georef_anchor_lon,
                georef.georef_anchor_alt,
            )
            x, y, z = lon, lat, alt
            meta = {
                'crs':              'WGS84',
                'crs_wkt':          _WGS84_WKT,
                'anchor_lat':       georef.georef_anchor_lat,
                'anchor_lon':       georef.georef_anchor_lon,
                'anchor_alt':       georef.georef_anchor_alt,
                'scale_m_per_unit': georef.georef_scale,
                'heading_deg':      georef.georef_heading_deg,
            }

        elif self._coord_idx == 1 and georef:
            # UTM metres (recommended for Civil 3D)
            enu = _mesh_to_enu(pts, georef.georef_scale, georef.georef_heading_deg)
            x, y, z = _enu_to_utm(
                enu,
                georef.georef_anchor_lat,
                georef.georef_anchor_lon,
                georef.georef_anchor_alt,
            )
            epsg    = _get_utm_epsg(georef.georef_anchor_lat, georef.georef_anchor_lon)
            utm_wkt = _utm_wkt(georef.georef_anchor_lat, georef.georef_anchor_lon)
            zone    = int((georef.georef_anchor_lon + 180) / 6) + 1
            hemi    = "N" if georef.georef_anchor_lat >= 0 else "S"
            meta = {
                'crs':              f'UTM Zone {zone}{hemi}',
                'crs_wkt':          utm_wkt,
                'epsg':             epsg,
                'anchor_lat':       georef.georef_anchor_lat,
                'anchor_lon':       georef.georef_anchor_lon,
                'anchor_alt':       georef.georef_anchor_alt,
                'scale_m_per_unit': georef.georef_scale,
                'heading_deg':      georef.georef_heading_deg,
            }

        elif self._coord_idx == 2 and georef:
            # ENU metres from anchor
            enu  = _mesh_to_enu(pts, georef.georef_scale, georef.georef_heading_deg)
            x, y, z = enu[:, 0], enu[:, 1], enu[:, 2]
            meta = {
                'crs':        'ENU_from_anchor',
                'crs_wkt':    '',
                'anchor_lat': georef.georef_anchor_lat,
                'anchor_lon': georef.georef_anchor_lon,
                'anchor_alt': georef.georef_anchor_alt,
            }

        else:
            # Local mesh units, no CRS
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            meta = None

        return x, y, z, r, g, b, faces, normals, meta

    # ── Output path ───────────────────────────────────────────────────────────

    def _make_out_path(self, ext: str) -> str:
        base    = getattr(self.app, 'loaded_file_path', None)
        stem    = os.path.splitext(os.path.basename(base))[0] if base else 'export'
        out_dir = os.path.dirname(base) if base else '.'
        return os.path.join(out_dir, f'{stem}_geo{ext}')

    # ── Export dispatcher (called by gui_layout) ───────────────────────────────

    def _do_export(self):
        georef = self._get_georef()

        if self._coord_idx < 2 and georef is None:
            self._status = (
                "No georef anchor set. "
                "Open Geo Map tool and place an anchor first, "
                "or choose 'Local mesh units'."
            )
            self._status_ok = False
            return

        if not hasattr(self.app, 'points') or len(self.app.points) == 0:
            self._status    = "No mesh loaded."
            self._status_ok = False
            return

        n = len(self.app.points) // max(1, self._thin_step)

        # Two-click safety gate for very large exports
        if n > 2_000_000 and not self._confirm_large:
            self._status = (
                f"{n:,} points — this may take a while. "
                "Click the export format again to confirm."
            )
            self._status_ok = False
            self._confirm_large = True
            return
        self._confirm_large = False

        try:
            x, y, z, r, g, b, faces, normals, meta = self._prepare_coords(georef)
        except Exception as e:
            self._status    = f"Coordinate conversion failed: {e}"
            self._status_ok = False
            return

        ext  = '.las'
        path = self._make_out_path(ext)

        # CRS WKT from meta (set in _prepare_coords for each coord mode)
        crs_wkt = (meta or {}).get('crs_wkt', '')

        try:
            # Embed correct WKT matching the chosen coord system
            write_las14(path, x, y, z, r, g, b, crs_wkt=crs_wkt)

            self._status    = f"Saved: {path}  ({n:,} pts)"
            self._status_ok = True

        except Exception as e:
            self._status    = f"Export failed: {e}"
            self._status_ok = False
