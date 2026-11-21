# tools/synthetic_depth_engine.py
# Standalone, reusable "fake" synthetic-depth engine for the AI Scanner.
#
# This is a trimmed-down version of synthetic_depth.py:
#   * NO camera capture of its own        (the scanner already owns the camera)
#   * NO Open3D / point-cloud viewer       (this is purely a cosmetic 2D view)
#   * NO cv2.imshow / blocking run() loop  (the scanner's UI drives it)
#
# It takes a real BGR frame and returns a stylized "depth" BGR image purely
# for looks. It NEVER touches the frames that get sent to the cloud model.
#
# Usage:
#     from tools.synthetic_depth_engine import SyntheticDepthEngine
#     engine = SyntheticDepthEngine()
#     depth_vis_bgr = engine.process(frame_bgr)   # ready to show / upload to a texture
#
# `process()` is self-contained and thread-safe enough to be called from the
# scanner's stream thread (one engine instance per stream).

import numpy as np

try:
    import cv2
except ImportError:  # cv2 is optional at import time; process() will no-op
    cv2 = None


class SyntheticDepthEngine:
    def __init__(self, min_depth=0.3, max_depth=3.0, depth_scale=1000.0):
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

        # caches (depend only on resolution)
        self._priors = {}
        self._overlay = None
        self._grid = None
        self._vignette = None
        self._hud_frame = None
        self._frame_no = 0
        self._hex_seed = None

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #
    def process(self, color_image_bgr):
        """Real BGR frame in -> stylized depth BGR frame out (display only)."""
        if cv2 is None or color_image_bgr is None:
            return color_image_bgr
        try:
            depth_mm = self._generate_synthetic_depth(color_image_bgr)
            return self._render_depth_techy(depth_mm, color_image_bgr)
        except Exception:
            # Never let the cosmetic view break the real stream.
            return color_image_bgr

    # ------------------------------------------------------------------ #
    #  Geometry priors
    # ------------------------------------------------------------------ #
    def _get_priors(self, h, w):
        if (h, w) in self._priors:
            return self._priors[(h, w)]

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        vert = yy / (h - 1)
        vert = 1.0 - vert                         # 1 far at top, 0 near at bottom

        cx, cy = w / 2.0, h * 0.45
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = r / r.max()
        radial = r

        self._priors[(h, w)] = (vert, radial)
        return vert, radial

    # ------------------------------------------------------------------ #
    #  Synthetic depth generation (quarter-res for speed)
    # ------------------------------------------------------------------ #
    def _generate_synthetic_depth(self, color_image):
        h, w = color_image.shape[:2]
        sw, sh = w // 2, h // 2

        small = cv2.resize(color_image, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        gray = np.power(gray, 1.4)
        gray = np.clip(gray, 0.0, 0.92) / 0.92

        illum = cv2.blur(gray, (31, 31))
        scene = gray - illum + 0.5
        scene = np.clip(scene, 0, 1)
        scene = 1.0 - scene

        vert, radial = self._get_priors(sh, sw)

        depth_n = (0.60 * scene +
                   0.30 * vert +
                   0.10 * radial)

        depth_n = cv2.bilateralFilter(depth_n.astype(np.float32), 7, 0.06, 7)
        depth_n = cv2.normalize(depth_n, None, 0, 1, cv2.NORM_MINMAX)

        blur = cv2.blur(depth_n, (9, 9))
        depth_n = np.clip(depth_n + 0.8 * (depth_n - blur), 0, 1)

        noise = np.random.randn(sh // 6, sw // 6).astype(np.float32)
        noise = cv2.resize(noise, (sw, sh), interpolation=cv2.INTER_LINEAR) * 0.01
        depth_n = np.clip(depth_n + noise, 0, 1)

        depth_n = cv2.resize(depth_n, (w, h), interpolation=cv2.INTER_LINEAR)

        depth = self.min_depth + depth_n * (self.max_depth - self.min_depth)
        depth_mm = (depth * self.depth_scale).astype(np.uint16)
        return depth_mm

    # ------------------------------------------------------------------ #
    #  Static stylized layers (cached per resolution)
    # ------------------------------------------------------------------ #
    def _build_static_layers(self, h, w):
        ov = np.zeros((h, w, 3), np.uint8)
        ov[::3, :] = (6, 6, 6)
        grid = np.zeros((h, w, 3), np.uint8)
        for x in range(0, w, 96):
            cv2.line(grid, (x, 0), (x, h), (28, 28, 28), 1)
        for y in range(0, h, 96):
            cv2.line(grid, (0, y), (w, y), (28, 28, 28), 1)
        self._overlay, self._grid = ov, grid

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r /= r.max()
        vig = np.clip(1.0 - (r ** 2.2) * 0.85, 0, 1)
        self._vignette = cv2.merge([vig, vig, vig]).astype(np.float32)

        hud = np.zeros((h, w, 3), np.uint8)
        C = (0, 220, 180)
        L, m = 38, 14
        for (px, py, dx, dy) in [(m, m, 1, 1), (w - m, m, -1, 1),
                                 (m, h - m, 1, -1), (w - m, h - m, -1, -1)]:
            cv2.line(hud, (px, py), (px + dx * L, py), C, 2, cv2.LINE_AA)
            cv2.line(hud, (px, py), (px, py + dy * L), C, 2, cv2.LINE_AA)
        cv2.rectangle(hud, (m, m), (w - m, h - m), (0, 70, 58), 1, cv2.LINE_AA)
        self._hud_frame = hud

    # ------------------------------------------------------------------ #
    #  Heavily stylized futuristic depth display
    # ------------------------------------------------------------------ #
    def _render_depth_techy(self, depth_mm, color_image=None):
        vis = cv2.normalize(depth_mm.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        h, w = vis.shape[:2]
        self._frame_no += 1
        f = self._frame_no
        C = (0, 230, 190)

        if self._overlay is None or self._overlay.shape[:2] != (h, w):
            self._build_static_layers(h, w)

        vis = cv2.subtract(vis, self._overlay)
        vis = cv2.add(vis, self._grid)
        vis = (vis.astype(np.float32) * self._vignette).astype(np.uint8)

        if f % 47 < 3:
            sh = 4 + (f % 3)
            b, g, r = cv2.split(vis)
            r = np.roll(r, sh, axis=1)
            b = np.roll(b, -sh, axis=1)
            vis = cv2.merge([b, g, r])
            gy = int((f * 53) % h)
            vis[gy:gy + 2, :] = cv2.add(vis[gy:gy + 2, :], 60)

        sweep_y = int((f * 6) % h)
        band = vis[max(0, sweep_y - 2):sweep_y + 2, :]
        if band.size:
            vis[max(0, sweep_y - 2):sweep_y + 2, :] = cv2.add(band, 45)

        flick = 0.7 + 0.3 * abs(np.sin(f * 0.3))
        vis = cv2.add(vis, (self._hud_frame.astype(np.float32) * flick).astype(np.uint8))

        d = depth_mm.astype(np.float32) / self.depth_scale
        cxp, cyp = w // 2, h // 2

        ox = int(40 * np.sin(f * 0.05))
        oy = int(28 * np.cos(f * 0.07))
        tx, ty = cxp + ox, cyp + oy
        s = 26
        for (dx, dy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cv2.line(vis, (tx + dx * s, ty + dy * s),
                     (tx + dx * s, ty + dy * (s - 9)), C, 1, cv2.LINE_AA)
            cv2.line(vis, (tx + dx * s, ty + dy * s),
                     (tx + dx * (s - 9), ty + dy * s), C, 1, cv2.LINE_AA)
        cv2.circle(vis, (tx, ty), 3, C, -1, cv2.LINE_AA)
        tdist = float(d[max(0, ty - 3):ty + 3, max(0, tx - 3):tx + 3].mean())
        cv2.putText(vis, f"LOCK {tdist:.2f}m", (tx + s + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C, 1, cv2.LINE_AA)

        rcx, rcy, rr = w - 60, 80, 40
        cv2.circle(vis, (rcx, rcy), rr, (0, 90, 72), 1, cv2.LINE_AA)
        cv2.circle(vis, (rcx, rcy), rr * 2 // 3, (0, 70, 56), 1, cv2.LINE_AA)
        cv2.circle(vis, (rcx, rcy), rr // 3, (0, 70, 56), 1, cv2.LINE_AA)
        ang = (f * 0.12) % (2 * np.pi)
        ex, ey = int(rcx + rr * np.cos(ang)), int(rcy + rr * np.sin(ang))
        cv2.line(vis, (rcx, rcy), (ex, ey), C, 1, cv2.LINE_AA)
        for k in range(1, 6):
            a = ang - k * 0.12
            fx, fy = int(rcx + rr * np.cos(a)), int(rcy + rr * np.sin(a))
            cv2.line(vis, (rcx, rcy), (fx, fy), (0, 60 - k * 8, 48 - k * 6), 1, cv2.LINE_AA)

        if self._hex_seed is None:
            self._hex_seed = np.random.randint(0, 16, size=200)
        for row in range(0, h, 16):
            idx = (row // 16 + f // 2) % len(self._hex_seed)
            hx = "".join(f"{self._hex_seed[(idx + j) % len(self._hex_seed)]:X}"
                         for j in range(4))
            cv2.putText(vis, hx, (4, row + 12), cv2.FONT_HERSHEY_PLAIN,
                        0.7, (0, 110, 85), 1, cv2.LINE_AA)

        hist = cv2.calcHist([cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
                             .astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.max() + 1e-6)
        hx0, hy0, hh = w - 64 - 18, h - 20, 36
        for i, v in enumerate(hist):
            cv2.line(vis, (hx0 + i, hy0), (hx0 + i, hy0 - int(v * hh)),
                     (0, 200, 165), 1)

        def plate(txt, pos, scale=0.5, col=C):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.rectangle(vis, (pos[0] - 3, pos[1] - th - 4),
                          (pos[0] + tw + 3, pos[1] + 4), (0, 0, 0), -1)
            cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        col, 1, cv2.LINE_AA)

        plate(f"DEPTH  {d.min():.2f}-{d.max():.2f}m", (22, 34), 0.52)
        plate("SYNTH-MONO  v2", (22, 56), 0.42, (0, 180, 150))
        plate(f"FRM {f:05d}", (22, h - 28), 0.42, (0, 180, 150))
        status = "SCANNING" + ("." * (1 + (f // 8) % 3))
        plate(status, (w - 150, h - 28), 0.45)
        return vis
