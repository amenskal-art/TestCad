# tools/sculpt_tool.py
#
# Sculpt Tool — Blender-style mesh sculpting for IQScanner
#
# Brushes:
#   0 - Draw      push / pull vertices along their normals  (Ctrl = invert)
#   1 - Flatten   project vertices toward the local tangent plane
#   2 - Inflate   push along normals, scaled by local curvature
#   3 - Grab      translate a soft-selected region freely (delta-based)
#   4 - Pinch     attract vertices toward the brush centre
#
# All core math is fully vectorised — no Python loops over N vertices.
#
# Drop in tools/, register in core.py _register_tools(). Zero other changes.

from tool_system import BaseTool
import imgui
import glfw
import numpy as np
from OpenGL.GL import *
import threading
import ctypes

try:
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix, diags
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# =============================================================================
#  Vectorised math helpers
# =============================================================================

def _build_laplacian(num_verts: int, faces: np.ndarray):
    """Row-normalised uniform Laplacian as scipy CSR + valence array."""
    f    = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    rows = np.concatenate([f[:,0],f[:,1],f[:,2], f[:,1],f[:,2],f[:,0]])
    cols = np.concatenate([f[:,1],f[:,2],f[:,0], f[:,0],f[:,1],f[:,2]])
    A    = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(num_verts, num_verts), dtype=np.float32)
    A.sum_duplicates(); A.data[:] = 1.0
    degrees      = np.array(A.sum(axis=1), dtype=np.float32).ravel()
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    L            = diags(1.0/degrees_safe, format='csr', dtype=np.float32) @ A
    return L, degrees.astype(np.int32)


def _recompute_normals(verts, faces):
    """Fast area-weighted vertex normals via np.bincount (no Python loop)."""
    f  = faces.reshape(-1, 3).astype(np.int32)
    v0, v1, v2 = verts[f[:,0]], verts[f[:,1]], verts[f[:,2]]
    fn = np.cross(v1-v0, v2-v0)
    nv = len(verts)
    nx = np.bincount(f[:,0],fn[:,0],nv)+np.bincount(f[:,1],fn[:,0],nv)+np.bincount(f[:,2],fn[:,0],nv)
    ny = np.bincount(f[:,0],fn[:,1],nv)+np.bincount(f[:,1],fn[:,1],nv)+np.bincount(f[:,2],fn[:,1],nv)
    nz = np.bincount(f[:,0],fn[:,2],nv)+np.bincount(f[:,1],fn[:,2],nv)+np.bincount(f[:,2],fn[:,2],nv)
    n  = np.stack([nx,ny,nz], axis=1)
    n /= np.where(np.linalg.norm(n,axis=1,keepdims=True)>0,
                  np.linalg.norm(n,axis=1,keepdims=True), 1.0)
    return n.astype(np.float32)


def _compute_curvature(verts, L):
    diff = (L @ verts.astype(np.float64)) - verts.astype(np.float64)
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def _brush_falloff(dists: np.ndarray, radius: float, exp: float = 2.0) -> np.ndarray:
    """
    Smooth radial falloff weight in [0, 1].
    exp=1  → linear,  exp=2 → smooth Gaussian-like (default),  exp≥4 → hard edge.
    """
    t = np.clip(1.0 - dists / (radius + 1e-8), 0.0, 1.0)
    return (t ** exp).astype(np.float32)


# =============================================================================
#  Global (whole-mesh) sculpt operations
# =============================================================================

def _sculpt_draw_global(verts, normals, L, mask, strength, direction, iters):
    """
    Push / pull all masked verts along neighbourhood-blended normals.
    Blending prevents large-scale spikes on convex / concave mesh regions.
    L may be None (falls back to raw normals).
    """
    v = verts.astype(np.float64)
    n = normals.astype(np.float64)
    if L is not None:
        avg_n = np.asarray(L @ n, dtype=np.float64)
        nlen  = np.linalg.norm(avg_n, axis=1, keepdims=True)
        nlen[nlen < 1e-8] = 1.0
        avg_n /= nlen
        n_use = 0.3 * n + 0.7 * avg_n
        blen  = np.linalg.norm(n_use, axis=1, keepdims=True)
        blen[blen < 1e-8] = 1.0
        n_use /= blen
    else:
        n_use = n
    d = float(direction) * float(strength)
    for _ in range(iters):
        v[mask] += d * n_use[mask]
    return v.astype(np.float32)


def _sculpt_flatten_global(verts, L, normals, mask, strength, iters):
    """
    Project masked verts toward the local tangent plane.
    For each vert the Laplacian avg gives an estimate of the local plane centre;
    only the normal component of the displacement is applied, reducing height
    variation while leaving lateral shape intact.
    """
    v = verts.astype(np.float64)
    n = normals.astype(np.float64)
    for _ in range(iters):
        avg   = np.asarray(L @ v, dtype=np.float64)
        delta = avg - v
        n_m   = n[mask]
        d_m   = delta[mask]
        # Normal component only — drives verts onto the local mean plane
        nc = np.sum(d_m * n_m, axis=1)[:, None] * n_m
        v[mask] += strength * nc
    return v.astype(np.float32)


def _sculpt_inflate_global(verts, L, normals, mask, strength, iters):
    """
    Push verts along normals, magnitude proportional to local curvature.
    Highly curved (bumpy) areas inflate more; flat areas barely move.
    Multiply strength by -1 to deflate.
    """
    v = verts.astype(np.float64)
    n = normals.astype(np.float64)
    for _ in range(iters):
        avg  = np.asarray(L @ v, dtype=np.float64)
        curv = np.linalg.norm(avg - v, axis=1, keepdims=True)   # (N,1)
        v[mask] += strength * curv[mask] * n[mask]
    return v.astype(np.float32)


def _sculpt_pinch_global(verts, mask, center, strength, iters):
    """Pull all masked verts toward a world-space centre point."""
    v = verts.astype(np.float64)
    c = np.asarray(center, dtype=np.float64)
    for _ in range(iters):
        diff    = c - v[mask]
        v[mask] += strength * diff
    return v.astype(np.float32)


# =============================================================================
#  Local (brush) sculpt operations   — operate directly on pts array (writable)
# =============================================================================

def _brush_draw(pts, normals, L, hit_arr, falloff, strength, direction):
    """
    Draw brush: push / pull along normals with radial falloff.

    Raw per-vertex normals cause spikes because adjacent vertices can diverge.
    We blend the raw normal (30 pct) with its Laplacian neighbourhood average (70 pct)
    so all vertices in the brush area lean in nearly the same direction, giving
    smooth Blender-style clay strokes instead of spikes.

    If L is None (not yet built) we fall back to raw normals.
    """
    if L is not None:
        avg_n = np.asarray(L[hit_arr, :] @ normals, dtype=np.float32)
        nlen  = np.linalg.norm(avg_n, axis=1, keepdims=True)
        nlen[nlen < 1e-8] = 1.0
        avg_n /= nlen
        blended = 0.3 * normals[hit_arr] + 0.7 * avg_n
        blen = np.linalg.norm(blended, axis=1, keepdims=True)
        blen[blen < 1e-8] = 1.0
        blended /= blen
    else:
        blended = normals[hit_arr]

    d = float(direction) * float(strength)
    pts[hit_arr] += d * falloff[:, None] * blended


def _brush_flatten(pts, normals, hit_arr, falloff, strength):
    """
    Flatten brush: detect the average plane of hit vertices then push each
    vertex toward that plane proportional to its signed distance from it.
    Reduces local height variation (bumps and dips flattened simultaneously).
    """
    sub   = pts[hit_arr]
    avg_p = sub.mean(axis=0)
    avg_n = normals[hit_arr].mean(axis=0)
    nn    = np.linalg.norm(avg_n)
    if nn < 1e-8:
        return
    avg_n /= nn

    # Signed distance above the average plane
    dist = np.einsum('ij,j->i', sub - avg_p, avg_n)          # (k,)
    pts[hit_arr] -= strength * falloff * dist[:, None] * avg_n


def _brush_inflate(pts, normals, L, hit_arr, falloff, strength):
    """
    Inflate brush: push along normals, magnitude proportional to local curvature.
    The full Laplacian row-slice gives the neighbourhood average cheaply.
    Multiply strength by -1 to deflate.
    """
    avg  = np.asarray(L[hit_arr, :] @ pts, dtype=np.float32)
    curv = np.linalg.norm(avg - pts[hit_arr], axis=1, keepdims=True)  # (k,1)
    pts[hit_arr] += strength * falloff[:, None] * curv * normals[hit_arr]


def _brush_pinch(pts, hit_arr, center, falloff, strength):
    """
    Pinch brush: attract verts toward the brush centre with radial falloff.
    Negative strength = magnify (push away from centre).
    """
    diff = center.astype(np.float32) - pts[hit_arr]
    pts[hit_arr] += strength * falloff[:, None] * diff


# =============================================================================
#  Tool
# =============================================================================

class SculptTool(BaseTool):
    """
    Blender-style Sculpt Tool — plug-and-play BaseTool.
    """

    BRUSH_NAMES = ["Draw", "Flatten", "Inflate", "Grab", "Pinch"]
    BRUSH_HINTS = [
        "Push / pull verts along normals  (Ctrl = invert)",
        "Flatten toward local tangent plane",
        "Scale outward by curvature  (bumps amplified, flat stays flat)",
        "Grab and drag a soft-selection freely",
        "Attract verts toward brush centre  (Ctrl = repel)",
    ]

    def __init__(self, app):
        super().__init__(app)
        self.name     = "Sculpt"
        self.icon_key = 'sculpt'
        self.hotkey   = 'B'

        # ── brush / algo ──────────────────────────────────────────────────────
        self.brush_idx  = 0            # active brush index
        self.mode       = 1            # 0 = Global apply,  1 = Brush paint
        self.direction  = 1            # +1 = push outward / attract,  -1 = pull inward

        # ── brush shape ───────────────────────────────────────────────────────
        self.brush_radius   = 5.0
        self.brush_strength = 0.3
        self.brush_falloff  = 2.0      # falloff exponent: 1=linear, 2=smooth, ≥4=hard

        # ── global mode ───────────────────────────────────────────────────────
        self.iterations    = 3
        self.use_selection = False

        # ── symmetry ─────────────────────────────────────────────────────────
        self.sym_x = False
        self.sym_y = False
        self.sym_z = False

        # ── painting state ────────────────────────────────────────────────────
        self.is_painting    = False
        self._grab_verts    = None     # vertex indices locked at grab-press
        self._grab_weights  = None     # per-vert falloff weights for grab
        self._grab_last_pos = None     # previous world pos for grab delta

        # ── Laplacian (Flatten global + Inflate need it) ──────────────────────
        self._L          = None
        self._degrees    = None
        self._L_n_verts  = -1

        self._curvature  = None        # cached curvature for heatmap

        # ── KD-tree ───────────────────────────────────────────────────────────
        self._kd_tree = None
        self._kd_n    = -1

        # ── async build ───────────────────────────────────────────────────────
        self._building     = False
        self._build_result = None

        # ── deferred work ─────────────────────────────────────────────────────
        self._pending_normal_recompute = False
        self._sculptor_synced          = False

        # ── heatmap ───────────────────────────────────────────────────────────
        self.show_heatmap    = False
        self._heatmap_mode   = 0       # 0 = displacement from snapshot, 1 = curvature
        self._heatmap_vmax   = 1.0
        self._base_snapshot  = None    # vertex positions captured on activate

        self.status_msg = "Ready"

    # =========================================================================
    #  Lifecycle
    # =========================================================================

    def on_activate(self):
        self.app.status_text  = "Sculpt"
        self._sculptor_synced = False
        # capture the mesh as it was when the tool opened — used as
        # the zero-reference for the displacement heatmap
        self._base_snapshot   = self.app.points.copy()
        self._start_build_async()

    def on_deactivate(self):
        self.is_painting    = False
        self._grab_verts    = None
        self._grab_weights  = None
        self._grab_last_pos = None
        if self.show_heatmap:
            self.show_heatmap = False
            self._restore_colors()

    # =========================================================================
    #  Async Laplacian build  (identical pattern to smooth_tool)
    # =========================================================================

    def _start_build_async(self):
        if not _SCIPY_OK:
            self.status_msg = "⚠ scipy missing"; return
        if not self.app.has_faces or self.app.faces is None:
            self.status_msg = "⚠ No face topology"; return
        n = len(self.app.points)
        if self._L is not None and self._L_n_verts == n:
            return
        if self._building:
            return
        self._building = True; self._build_result = None
        self.status_msg = "Building mesh graph…"
        faces_copy = self.app.faces.copy()
        def _worker():
            L, deg = _build_laplacian(n, faces_copy)
            self._build_result = (L, deg, n)
        threading.Thread(target=_worker, daemon=True).start()

    def _poll_build(self):
        if self._build_result is not None:
            self._L, self._degrees, self._L_n_verts = self._build_result
            self._build_result = None; self._building = False
            self.status_msg = "Ready"

        if self._pending_normal_recompute:
            self._pending_normal_recompute = False
            if self.app.has_faces and self.app.faces is not None:
                self.app.normals     = _recompute_normals(self.app.points,
                                                          self.app.faces)
                self.app.has_normals = True

    def _ensure_laplacian(self) -> bool:
        if not _SCIPY_OK or not self.app.has_faces or self.app.faces is None:
            return False
        n = len(self.app.points)
        if self._L is not None and self._L_n_verts == n:
            return True
        self._L, self._degrees, self._L_n_verts = _build_laplacian(
            n, self.app.faces)
        return True

    def _ensure_kdtree(self):
        n = len(self.app.points)
        if self._kd_tree is not None and self._kd_n == n:
            return
        if _SCIPY_OK:
            self._kd_tree = KDTree(self.app.points)
            self._kd_n    = n

    def _invalidate(self):
        self._L         = None;  self._L_n_verts = -1
        self._kd_tree   = None;  self._kd_n      = -1
        self._curvature = None
        self._grab_verts    = None
        self._grab_weights  = None
        self._grab_last_pos = None

    # =========================================================================
    #  Small helpers
    # =========================================================================

    def _ensure_points_writable(self):
        """Make self.app.points writable (copy if necessary)."""
        if not self.app.points.flags.writeable:
            self.app.points = self.app.points.copy()
            self.app.points_colors[:, :3] = self.app.points
            self.app.vbo_dirty = True

    def _ensure_normals(self):
        if (not self.app.has_normals or self.app.normals is None
                or len(self.app.normals) != len(self.app.points)):
            if self.app.has_faces and self.app.faces is not None:
                self.app.normals     = _recompute_normals(self.app.points,
                                                          self.app.faces)
                self.app.has_normals = True

    def _make_mask(self, indices=None):
        n = len(self.app.points)
        if indices is not None:
            m = np.zeros(n, dtype=bool); m[indices] = True; return m
        if self.use_selection and np.any(self.app.selected):
            return self.app.selected.copy()
        return np.ones(n, dtype=bool)

    def _mirror_positions(self, world_pos):
        """
        Return a list of world positions: the original plus every symmetry
        mirror.  With X+Y+Z enabled this yields 8 strokes.
        """
        positions = [np.asarray(world_pos, dtype=np.float32)]
        for axis, flag in enumerate((self.sym_x, self.sym_y, self.sym_z)):
            if flag:
                new = []
                for q in positions:
                    m = q.copy(); m[axis] = -m[axis]
                    new.append(m)
                positions.extend(new)
        return positions

    # =========================================================================
    #  Heatmap
    # =========================================================================

    def _analyse(self):
        if not self._ensure_laplacian():
            return
        self._curvature = _compute_curvature(self.app.points, self._L)
        if self.show_heatmap:
            self._push_heatmap()

    def _push_heatmap(self):
        data = self._heatmap_data()
        if data is None:
            return
        self._heatmap_vmax = float(np.percentile(data, 98)) or 1.0
        self._apply_heatmap_all(data)
        glBindBuffer(GL_ARRAY_BUFFER, self.app.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.app.points_colors.nbytes,
                     self.app.points_colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _heatmap_data(self):
        if self._heatmap_mode == 0:
            # displacement from the snapshot taken at on_activate
            if (self._base_snapshot is not None
                    and len(self._base_snapshot) == len(self.app.points)):
                return np.linalg.norm(
                    self.app.points - self._base_snapshot, axis=1
                ).astype(np.float32)
        return self._curvature   # fallback / curvature mode

    def _apply_heatmap_all(self, data):
        t = np.clip(data / (self._heatmap_vmax + 1e-8), 0.0, 1.0).astype(np.float32)
        self.app.points_colors[:, :3] = self.app.points
        self.app.points_colors[:, 3]  = np.clip(2.0 * t,         0.0, 1.0)
        self.app.points_colors[:, 4]  = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
        self.app.points_colors[:, 5]  = 0.0

    def _apply_heatmap_local(self, hit_arr, data):
        # data is already the local slice (len == len(hit_arr)) — do NOT re-index
        t = np.clip(data / (self._heatmap_vmax + 1e-8), 0.0, 1.0).astype(np.float32)
        self.app.points_colors[hit_arr, 3] = np.clip(2.0 * t,         0.0, 1.0)
        self.app.points_colors[hit_arr, 4] = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
        self.app.points_colors[hit_arr, 5] = 0.0

    def _restore_colors(self):
        self.app.points_colors = np.hstack(
            [self.app.points, self.app.base_colors]).astype(np.float32)
        if hasattr(self.app, 'update_vbo_full'):
            self.app.update_vbo_full()

    # =========================================================================
    #  Global sculpt   (mode == 0)
    # =========================================================================

    def _run_sculpt(self):
        self._ensure_normals()

        # Flatten and Inflate need the Laplacian
        if self.brush_idx in (1, 2):
            if not self._ensure_laplacian():
                self.status_msg = "⚠ No face data"; return

        self.app.save_state("Sculpt")
        mask = self._make_mask()
        v    = self.app.points.copy()
        lam  = float(self.brush_strength)
        nrm  = (self.app.normals if self.app.has_normals
                else _recompute_normals(v, self.app.faces))

        if self.brush_idx == 0:          # Draw
            v = _sculpt_draw_global(v, nrm, self._L if self._L is not None else None,
                                    mask, lam, self.direction, self.iterations)

        elif self.brush_idx == 1:        # Flatten
            v = _sculpt_flatten_global(v, self._L, nrm, mask,
                                       lam, self.iterations)

        elif self.brush_idx == 2:        # Inflate
            v = _sculpt_inflate_global(v, self._L, nrm, mask,
                                       lam * float(self.direction),
                                       self.iterations)

        elif self.brush_idx == 3:        # Grab — global: gently pinch toward centroid
            center = v[mask].mean(axis=0)
            v = _sculpt_pinch_global(v, mask, center, lam * 0.5, self.iterations)

        elif self.brush_idx == 4:        # Pinch — global: pinch toward mesh centroid
            center = v.mean(axis=0)
            v = _sculpt_pinch_global(v, mask, center,
                                     lam * float(self.direction),
                                     self.iterations)

        self.app.points      = v
        self.app.normals     = _recompute_normals(v, self.app.faces)
        self.app.has_normals = True
        self.app.points_colors = np.hstack(
            [v, self.app.base_colors]).astype(np.float32)
        self._invalidate()
        self.app.update_vbo_full()
        if self.show_heatmap:
            self._ensure_laplacian(); self._analyse()
        self.status_msg = (f"Done — {int(np.sum(mask)):,} verts, "
                           f"{self.BRUSH_NAMES[self.brush_idx]}, "
                           f"{self.iterations} iter")
        self.app.status_text = self.status_msg
        self._start_build_async()

    # =========================================================================
    #  Brush sculpt   (mode == 1)
    # =========================================================================

    def _sculpt_brush_at(self, world_pos, radius, strength):
        """
        Apply the active brush at world_pos (and all symmetry mirrors).
        Called every mouse-move frame while painting.
        """
        if not _SCIPY_OK:
            return
        self._ensure_kdtree()
        if self._kd_tree is None:
            return

        positions = self._mirror_positions(world_pos)

        self._ensure_points_writable()
        pts = self.app.points

        any_hit = False
        for pos in positions:
            hit = self._kd_tree.query_ball_point(pos, radius)
            if not hit:
                continue

            hit_arr = np.array(hit, dtype=np.int32)
            dists   = np.linalg.norm(
                pts[hit_arr] - pos, axis=1).astype(np.float32)
            falloff = _brush_falloff(dists, radius, self.brush_falloff)

            if self.brush_idx == 0:      # Draw
                self._ensure_normals()
                _brush_draw(pts, self.app.normals, self._L, hit_arr, falloff,
                            strength, self.direction)

            elif self.brush_idx == 1:    # Flatten
                self._ensure_normals()
                _brush_flatten(pts, self.app.normals, hit_arr, falloff, strength)

            elif self.brush_idx == 2:    # Inflate
                if self._L is None:
                    continue             # Laplacian not yet ready
                self._ensure_normals()
                _brush_inflate(pts, self.app.normals, self._L,
                               hit_arr, falloff,
                               strength * float(self.direction))

            elif self.brush_idx == 3:    # Grab — delta-based, handled separately
                pass

            elif self.brush_idx == 4:    # Pinch
                _brush_pinch(pts, hit_arr, pos, falloff,
                             strength * float(self.direction))

            # ── live heatmap update (displacement mode) ──────────────────────
            if self.show_heatmap and self._heatmap_mode == 0:
                snap = self._base_snapshot
                if snap is not None and len(snap) == len(pts):
                    disp = np.linalg.norm(
                        pts[hit_arr] - snap[hit_arr], axis=1
                    ).astype(np.float32)
                    self._apply_heatmap_local(hit_arr, disp)

            any_hit = True

        if not any_hit:
            return

        # ── upload changed vertices ───────────────────────────────────────────
        self.app.points_colors[:, :3] = pts
        glBindBuffer(GL_ARRAY_BUFFER, self.app.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.app.points_colors.nbytes,
                     self.app.points_colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ── Grab brush helpers ────────────────────────────────────────────────────

    def _grab_start(self, world_pos, radius):
        """
        Lock in the vertex set affected by the grab brush at press-time.
        Weights are baked once so the shape of the falloff envelope is stable
        throughout the drag, exactly like Blender's Grab brush.
        """
        self._ensure_kdtree()
        if self._kd_tree is None:
            return
        pos     = np.asarray(world_pos, dtype=np.float32)
        hit     = self._kd_tree.query_ball_point(pos, radius)
        if not hit:
            self._grab_verts   = np.array([], dtype=np.int32)
            self._grab_weights = np.array([], dtype=np.float32)
            return
        hit_arr = np.array(hit, dtype=np.int32)
        dists   = np.linalg.norm(
            self.app.points[hit_arr] - pos, axis=1).astype(np.float32)
        self._grab_verts    = hit_arr
        self._grab_weights  = _brush_falloff(dists, radius, self.brush_falloff)
        self._grab_last_pos = pos

    def _grab_update(self, world_pos):
        """
        Translate grabbed vertices by the mouse delta.
        The weight envelope stays fixed (baked at grab-start).
        """
        if self._grab_verts is None or len(self._grab_verts) == 0:
            return
        pos = np.asarray(world_pos, dtype=np.float32)
        if self._grab_last_pos is None:
            self._grab_last_pos = pos; return

        delta = (pos - self._grab_last_pos) * self.brush_strength
        self._grab_last_pos = pos

        self._ensure_points_writable()
        pts = self.app.points
        pts[self._grab_verts] += self._grab_weights[:, None] * delta

        # live heatmap update
        if (self.show_heatmap and self._heatmap_mode == 0
                and self._base_snapshot is not None
                and len(self._base_snapshot) == len(pts)):
            disp = np.linalg.norm(
                pts[self._grab_verts] - self._base_snapshot[self._grab_verts],
                axis=1).astype(np.float32)
            self._apply_heatmap_local(self._grab_verts, disp)

        self.app.points_colors[:, :3] = pts
        glBindBuffer(GL_ARRAY_BUFFER, self.app.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.app.points_colors.nbytes,
                     self.app.points_colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # =========================================================================
    #  Events
    # =========================================================================

    def on_mouse_button(self, button, action, mods, x, y):
        if self.mode != 1 or imgui.get_io().want_capture_mouse:
            return False

        if button == glfw.MOUSE_BUTTON_LEFT:
            # Ctrl held at press time inverts direction for Draw / Inflate / Pinch
            if mods & glfw.MOD_CONTROL:
                self.direction = -1
            else:
                self.direction = 1

            if action == glfw.PRESS:
                hit = getattr(self.app, 'cursor_world_pos', None)
                if hit is None: return False
                if self._L is None and self.brush_idx in (1, 2):
                    self.status_msg = "⏳ Building mesh graph — wait a moment"
                    return False
                self.app.save_state("Sculpt Brush")
                self.is_painting = True
                self._ensure_kdtree()
                if self.brush_idx == 3:
                    self._grab_start(hit, self.brush_radius)
                else:
                    self._sculpt_brush_at(hit, self.brush_radius,
                                          self.brush_strength)
                return True

            elif action == glfw.RELEASE:
                if self.is_painting:
                    self.is_painting    = False
                    self._grab_verts    = None
                    self._grab_weights  = None
                    self._grab_last_pos = None
                    # invalidate caches and rebuild
                    self._pending_normal_recompute = True
                    self._L = None; self._L_n_verts = -1
                    self._kd_tree = None; self._kd_n = -1
                    self._start_build_async()
                    self._ensure_kdtree()
                    if self.show_heatmap:
                        self._push_heatmap()
                    else:
                        self.app.points_colors = np.hstack(
                            [self.app.points,
                             self.app.base_colors]).astype(np.float32)
                        self.app.update_vbo_full()
                    return True

        return False

    def on_mouse_move(self, x, y):
        if self.mode == 1 and self.is_painting:
            hit = getattr(self.app, 'cursor_world_pos', None)
            if hit is not None:
                if self.brush_idx == 3:
                    self._grab_update(hit)
                else:
                    self._sculpt_brush_at(hit, self.brush_radius,
                                          self.brush_strength)
            return True
        return False

    def on_key(self, key, scancode, action, mods):
        if action != glfw.PRESS: return False
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if self.mode == 0: self._run_sculpt(); return True
        elif key == glfw.KEY_H:
            self.show_heatmap = not self.show_heatmap
            if self.show_heatmap: self._ensure_laplacian(); self._analyse()
            else: self._restore_colors()
            return True
        elif key == glfw.KEY_TAB:
            self.mode = 1 - self.mode
            if self.mode == 1: self._ensure_kdtree()
            return True
        elif key == glfw.KEY_D:
            # Cycle to next brush
            self.brush_idx = (self.brush_idx + 1) % len(self.BRUSH_NAMES)
            self.status_msg = f"Brush: {self.BRUSH_NAMES[self.brush_idx]}"
            return True
        elif key == glfw.KEY_X:
            self.sym_x = not self.sym_x
            self.status_msg = f"X Symmetry: {'ON' if self.sym_x else 'OFF'}"
            return True
        return False

    def update(self):
        self._poll_build()

    # =========================================================================
    #  3-D rendering
    # =========================================================================

    def render_3d(self):
        if self.mode != 1:
            return
        if not self._sculptor_synced:
            if not hasattr(self.app, 'sculptor') or self.app.sculptor is None:
                if hasattr(self.app, 'sync_sculptor'):
                    self.app.sync_sculptor()
            self._sculptor_synced = True
        if hasattr(self.app, 'update_cursor_depth'):
            self.app.update_cursor_depth()
        if getattr(self.app, 'cursor_world_pos', None) is None:
            return
        prev = getattr(self.app, 'sculpt_radius', self.brush_radius)
        self.app.sculpt_radius = self.brush_radius
        if hasattr(self.app, 'draw_brush_cursor'):
            self.app.draw_brush_cursor()
        else:
            self._draw_cursor_fallback()
        self.app.sculpt_radius = prev

    def _draw_cursor_fallback(self):
        pos = getattr(self.app, 'cursor_world_pos', None)
        if pos is None: return
        cx, cy, cz = pos; r = self.brush_radius; steps = 64

        # Colour-coded per brush so the user always knows the active tool at a glance
        #   Draw=cyan  Flatten=lime  Inflate=orange  Grab=yellow  Pinch=magenta
        brush_cols = [
            (0.0, 0.85, 1.0, 1.0),   # Draw    — cyan
            (0.3, 1.0,  0.3, 1.0),   # Flatten — lime
            (1.0, 0.6,  0.1, 1.0),   # Inflate — orange
            (1.0, 1.0,  0.1, 1.0),   # Grab    — yellow
            (1.0, 0.2,  1.0, 1.0),   # Pinch   — magenta
        ]
        col = brush_cols[self.brush_idx % len(brush_cols)]

        n   = np.array([0., 1., 0.])
        tmp = np.array([1., 0., 0.]) if abs(n[1]) > 0.9 else np.array([0., 1., 0.])
        t1  = np.cross(n, tmp); t1 /= (np.linalg.norm(t1) or 1.)
        t2  = np.cross(n, t1);  t2 /= (np.linalg.norm(t2) or 1.)
        ctr = np.array([cx, cy, cz])

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glLineWidth(2.); glColor4f(*col)
        glBegin(GL_LINE_LOOP)
        for i in range(steps):
            a = 2. * np.pi * i / steps
            p = ctr + r * (np.cos(a) * t1 + np.sin(a) * t2)
            glVertex3f(*p)
        glEnd()

        # Draw brush: show a small arrow indicating push/pull direction
        if self.brush_idx == 0:
            up  = np.array([0., 1., 0.])
            tip = ctr + float(self.direction) * r * 0.45 * up
            glLineWidth(3.); glBegin(GL_LINES)
            glVertex3f(cx, cy, cz); glVertex3f(*tip)
            glEnd()

        # Pinch brush: show inward arrows at N/S/E/W of ring
        if self.brush_idx == 4:
            glLineWidth(1.5); glBegin(GL_LINES)
            for frac in (0.9, 0.5):
                for ang in (0., np.pi*0.5, np.pi, np.pi*1.5):
                    outer = ctr + r * frac * (np.cos(ang)*t1 + np.sin(ang)*t2)
                    inner = ctr + r * (frac-0.25) * float(self.direction) * \
                            (np.cos(ang)*t1 + np.sin(ang)*t2)
                    glVertex3f(*outer); glVertex3f(*inner)
            glEnd()

        glPointSize(5.); glBegin(GL_POINTS); glVertex3f(cx, cy, cz); glEnd()
        glPopAttrib()

    # =========================================================================
    #  GUI
    # =========================================================================

    def draw_options(self):
        self._poll_build()

        if self._building:
            imgui.text_colored("  Building mesh graph…", 1.0, 0.8, 0.2)
            imgui.separator()

        if imgui.begin_tab_bar("SculptTabs"):
            if imgui.begin_tab_item("Sculpt")[0]:
                self._draw_sculpt_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Symmetry")[0]:
                self._draw_symmetry_tab()
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def _draw_sculpt_tab(self):
        imgui.dummy(0, 4)
        imgui.text_colored("Sculpt Brush", 0.4, 0.85, 1.0)
        imgui.separator()

        # Mode selector
        imgui.text("Mode:")
        imgui.same_line()
        if imgui.radio_button("Global##sm", self.mode == 0):
            self.mode = 0
        imgui.same_line()
        if imgui.radio_button("Brush##sm", self.mode == 1):
            self.mode = 1; self._ensure_kdtree()
        imgui.text_disabled("Tab: toggle  |  H: heatmap  |  Ctrl: invert")
        imgui.dummy(0, 4); imgui.separator()

        # Brush picker
        imgui.push_item_width(190)
        _, self.brush_idx = imgui.combo("##brush", self.brush_idx,
                                        self.BRUSH_NAMES)
        imgui.pop_item_width()
        imgui.text_disabled(self.BRUSH_HINTS[self.brush_idx])
        imgui.dummy(0, 4)

        # Direction (Draw / Inflate / Pinch)
        if self.brush_idx in (0, 2, 4):
            if self.brush_idx == 4:
                pos_label, neg_label = "+ Attract", "− Repel"
            else:
                pos_label, neg_label = "+ Push out", "− Pull in"
            imgui.text("Direction:")
            imgui.same_line()
            if imgui.radio_button(f"{pos_label}##d1", self.direction == 1):
                self.direction = 1
            imgui.same_line()
            if imgui.radio_button(f"{neg_label}##d0", self.direction == -1):
                self.direction = -1
            imgui.text_disabled("Ctrl+drag also inverts in Brush mode")
            imgui.dummy(0, 4)

        # Brush shape sliders
        imgui.push_item_width(160)
        _, self.brush_radius = imgui.slider_float(
            "Radius##br", self.brush_radius, 0.1, 50., "%.1f")
        _, self.brush_strength = imgui.slider_float(
            "Strength##bs", self.brush_strength, 0.001, 1.0, "%.3f")
        _, self.brush_falloff = imgui.slider_float(
            "Falloff##bf", self.brush_falloff, 0.5, 8.0, "%.1f")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls how quickly the brush fades at the edges.\n"
                "0.5 = very wide falloff (near-constant strength)\n"
                "2.0 = smooth Gaussian-like (default)\n"
                "8.0 = hard edge (almost constant inside, sharp cut-off)")
        imgui.pop_item_width()

        if self.mode == 0:
            imgui.dummy(0, 4); imgui.separator()
            imgui.push_item_width(160)
            _, self.iterations = imgui.slider_int(
                "Iterations##it", self.iterations, 1, 20)
            imgui.pop_item_width()
            _, self.use_selection = imgui.checkbox(
                "Selection Only##sel", self.use_selection)
            imgui.dummy(0, 6)
            if self._building:
                imgui.push_style_var(imgui.STYLE_ALPHA,
                                     imgui.get_style().alpha * 0.4)
            if imgui.button("Apply Sculpt##go", width=-1, height=32):
                if not self._building: self._run_sculpt()
            if self._building:
                imgui.pop_style_var()
            imgui.text_disabled("Enter: quick apply")
        else:
            imgui.text_disabled("Left-drag: sculpt  |  D: cycle brush")

        imgui.dummy(0, 4); imgui.separator()

        # Heatmap controls
        imgui.text("Visualisation:")
        chg, self.show_heatmap = imgui.checkbox("Show Heatmap##hm",
                                                 self.show_heatmap)
        if chg:
            if self.show_heatmap:
                self._ensure_laplacian(); self._analyse()
            else:
                self._restore_colors()
        if self.show_heatmap:
            prev_mode = self._heatmap_mode
            if imgui.radio_button("Displacement##hm0", self._heatmap_mode == 0):
                self._heatmap_mode = 0
            imgui.same_line()
            if imgui.radio_button("Curvature##hm1", self._heatmap_mode == 1):
                self._heatmap_mode = 1
            if self._heatmap_mode != prev_mode:
                self._push_heatmap()
            imgui.text_colored("Green = untouched   Red = sculpted",
                               0.8, 0.8, 0.2)
            if imgui.button("Refresh##rhm", width=-1):
                self._ensure_laplacian(); self._analyse()
            if imgui.button("Reset Snapshot##rsnap", width=-1):
                self._base_snapshot = self.app.points.copy()
                self._analyse()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Resets the zero-displacement reference to the current\n"
                    "mesh state — useful after committing a stage of sculpting.")

        imgui.dummy(0, 4); imgui.separator()
        if imgui.button("Reset Parameters##rp", width=-1):
            self._reset_params()
        imgui.dummy(0, 4)
        imgui.text_disabled(self.status_msg)

    def _draw_symmetry_tab(self):
        imgui.dummy(0, 4)
        imgui.text_colored("Symmetry Configuration", 0.4, 0.85, 1.0)
        imgui.separator()
        imgui.dummy(0, 4)
        imgui.text("Mirror Axis:")
        imgui.dummy(0, 4)

        _, self.sym_x = imgui.checkbox("X Axis  (hotkey: X)##sx", self.sym_x)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Mirror every brush stroke across the X = 0 plane.\n"
                "Hotkey: X")
        _, self.sym_y = imgui.checkbox("Y Axis##sy", self.sym_y)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Mirror every brush stroke across the Y = 0 plane.")
        _, self.sym_z = imgui.checkbox("Z Axis##sz", self.sym_z)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Mirror every brush stroke across the Z = 0 plane.")

        n_active = sum([self.sym_x, self.sym_y, self.sym_z])
        imgui.dummy(0, 4)
        if n_active > 0:
            mult = 2 ** n_active
            imgui.text_colored(f"  {mult}× strokes per drag", 0.6, 1.0, 0.6)
            if n_active > 1:
                imgui.text_disabled(f"  (all {mult} axis-combinations mirrored)")
        else:
            imgui.text_disabled("  No mirroring active")

        imgui.dummy(0, 4); imgui.separator()

        # Show brush colour legend so user knows what the cursor colour means
        imgui.text("Brush Cursor Colours:")
        imgui.dummy(0, 2)
        legend = [
            ((0.0, 0.85, 1.0), "Draw"),
            ((0.3, 1.0,  0.3), "Flatten"),
            ((1.0, 0.6,  0.1), "Inflate"),
            ((1.0, 1.0,  0.1), "Grab"),
            ((1.0, 0.2,  1.0), "Pinch"),
        ]
        for col, name in legend:
            active = (self.BRUSH_NAMES.index(name) == self.brush_idx)
            lbl = f"  ● {name}" + ("  ← active" if active else "")
            imgui.text_colored(lbl, *col)

        imgui.dummy(0, 4); imgui.separator()
        if imgui.button("Reset Parameters##rp2", width=-1):
            self._reset_params()
        imgui.dummy(0, 4)
        imgui.text_disabled(self.status_msg)

    # =========================================================================
    #  Parameter reset
    # =========================================================================

    def _reset_params(self):
        self.brush_idx      = 0
        self.mode           = 1
        self.direction      = 1
        self.brush_radius   = 5.0
        self.brush_strength = 0.3
        self.brush_falloff  = 2.0
        self.iterations     = 3
        self.use_selection  = False
        self.sym_x = self.sym_y = self.sym_z = False
        self.status_msg = "Parameters reset"
