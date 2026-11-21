# tools/engineering_tool.py
#
# Surface Refinement & Smart Smoothing Tool
# For reverse-engineering and 3D-print preparation.
#
# Algorithms:
#   0 - Laplacian            fast, general noise removal
#   1 - Taubin (Volume-Safe)   volume-preserving, best for RE
#   2 - HC Laplacian           feature-preserving
#   3 - Smart (Feature-Aware)  adaptive: flat stays flat, edges stay sharp,
#                              noise removed, round curves stay round —
#                              uses tangential-only projection so no mass
#                              is added or removed
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
from collections import defaultdict

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


def _compute_feature_weights(normals: np.ndarray, L,
                              curv: np.ndarray,
                              edge_thresh: float = 0.6,
                              noise_boost: float = 2.0) -> np.ndarray:
    n    = normals.astype(np.float32)
    avg_n = np.asarray(L @ n, dtype=np.float32)                  
    lengths = np.linalg.norm(avg_n, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    avg_n /= lengths

    alignment = np.clip(np.sum(n * avg_n, axis=1), 0.0, 1.0)    
    lo, hi = edge_thresh, 1.0
    t      = np.clip((alignment - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    base   = t * t * (3.0 - 2.0 * t)   

    local_mean = np.asarray(L @ curv, dtype=np.float32)           
    local_mean = np.maximum(local_mean, 1e-8)
    noise_ratio = np.clip(curv / local_mean, 0.0, 4.0)
    noise_w     = np.clip((noise_ratio - 1.0) * noise_boost * 0.5,
                          0.0, 1.0).astype(np.float32)

    weight = np.clip(base + noise_w * base, 0.0, 1.0).astype(np.float32)
    return weight


def _smart_smooth(verts, L, normals, mask, strength, feature_weights,
                   iters, fill_strength=0.0, fill_mode=0):
    v  = verts.astype(np.float64)
    n  = normals.astype(np.float64)   
    fw = feature_weights.astype(np.float64)   

    lam  = float(strength)
    mu   = -lam / (1.0 - 0.1 * lam)   
    fs   = float(fill_strength)

    masked = np.where(mask)[0]

    for _ in range(iters):
        avg   = np.asarray(L @ v, dtype=np.float64)   
        delta = avg - v                                 

        n_m   = n[masked]          
        d_m   = delta[masked]      
        fw_m  = fw[masked]         

        signed_curv = np.sum(d_m * n_m, axis=1)   

        if fs > 0.0:
            if fill_mode == 1:      
                nc = np.maximum(signed_curv, 0.0)
            elif fill_mode == 2:    
                nc = np.minimum(signed_curv, 0.0)
            else:                   
                nc = signed_curv
            v[masked] += fs * (fw_m * nc)[:, None] * n_m

        d_tang = d_m - signed_curv[:, None] * n_m   
        v[masked] += lam * fw_m[:, None] * d_tang

        avg2   = np.asarray(L @ v, dtype=np.float64)
        d2     = avg2[masked] - v[masked]
        sc2    = np.sum(d2 * n_m, axis=1)
        d2_t   = d2 - sc2[:, None] * n_m
        v[masked] += mu * fw_m[:, None] * d2_t

    return v.astype(np.float32)


def _laplacian_smooth(verts, L, mask, lam, iters):
    v = verts.astype(np.float64)
    for _ in range(iters):
        delta = (L @ v) - v; v[mask] += lam * delta[mask]
    return v.astype(np.float32)


def _taubin_smooth(verts, L, mask, lam, mu, iters):
    v = verts.astype(np.float64)
    for _ in range(iters):
        delta = (L @ v) - v; v[mask] += lam * delta[mask]
        delta = (L @ v) - v; v[mask] += mu  * delta[mask]
    return v.astype(np.float32)


def _hc_smooth(verts, L, mask, alpha, beta, lam, iters):
    v = verts.astype(np.float64); p = v.copy()
    for _ in range(iters):
        q = v.copy(); delta = (L@q)-q; q[mask] += lam*delta[mask]
        b = q-(alpha*p+(1.0-alpha)*v); b_avg = L@b
        q[mask] -= (beta*b[mask]+(1.0-beta)*b_avg[mask]); v = q
    return v.astype(np.float32)


def _readiness_score(verts, degrees, curv):
    diag         = float(np.linalg.norm(np.ptp(verts, axis=0))) or 1.0
    spike_thresh = diag * 0.005
    spike_count  = int(np.sum(curv > spike_thresh))
    bad_valence  = int(np.sum((degrees < 3) | (degrees > 20)))
    roughness    = float(np.mean(curv)) / diag
    max_dev      = float(np.max(curv))  / diag
    score = max(0.0, 100.0
                - min(50, spike_count/max(len(verts),1)*2000)
                - min(20, bad_valence/max(len(verts),1)*1000)
                - min(30, roughness*5000))
    return dict(score=round(score,1), spike_count=spike_count,
                bad_valence=bad_valence, roughness=roughness,
                max_dev=max_dev, spike_thresh=spike_thresh,
                spike_ratio=spike_count/max(len(verts),1))



# =============================================================================
#  Hole-filling helpers
# =============================================================================

def _find_boundary_loops(faces, num_verts):
    """
    Trace boundary edge loops of a triangle mesh.

    A boundary (open) edge is a directed edge (u->v) whose opposite (v->u)
    does not appear anywhere in the mesh.

    IMPORTANT: after operations like the smooth-cut delete, boundary vertices
    can be NON-MANIFOLD — a single vertex may have several outgoing boundary
    edges (T-junctions, shared corners).  The previous implementation used
    `dict(zip(src, dst))`, which keeps only ONE outgoing edge per vertex and
    silently drops the rest.  The walk then jumped across the model and built
    giant phantom loops; when _fill_hole fanned those to their centroid it
    shredded the whole mesh (the corner-to-corner shards you saw).

    This version stores ALL outgoing boundary edges per vertex and consumes
    each directed edge exactly once, so each real loop is traced correctly
    even on a non-manifold boundary.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    N = int(num_verts)
    a, b, c = f[:, 0], f[:, 1], f[:, 2]

    # All directed edges of every triangle
    src_arr = np.concatenate([a, b, c])
    dst_arr = np.concatenate([b, c, a])

    # Boundary edge = its reverse is not present among all directed edges
    fwd_set  = set((src_arr.astype(np.int64) * N + dst_arr).tolist())
    rev_keys = dst_arr.astype(np.int64) * N + src_arr
    is_bnd   = ~np.array([r in fwd_set for r in rev_keys.tolist()], dtype=bool)

    bnd_src = src_arr[is_bnd]
    bnd_dst = dst_arr[is_bnd]
    if len(bnd_src) == 0:
        return []

    # Allow MULTIPLE outgoing edges per vertex (the key fix vs dict(zip(...)))
    out_edges = defaultdict(list)
    remaining = {}
    for u, v in zip(bnd_src.tolist(), bnd_dst.tolist()):
        out_edges[u].append(v)
        remaining[(u, v)] = remaining.get((u, v), 0) + 1

    loops = []
    for start in list(out_edges.keys()):
        # Start new loops from this vertex while it still has unused edges
        while True:
            nxt = None
            for v in out_edges[start]:
                if remaining.get((start, v), 0) > 0:
                    nxt = v
                    break
            if nxt is None:
                break

            loop = [start]
            remaining[(start, nxt)] -= 1
            cur = nxt
            max_steps = len(bnd_src) + 2
            steps = 0
            ok = True
            while cur != start:
                steps += 1
                if steps > max_steps:
                    ok = False
                    break
                loop.append(cur)
                pick = None
                for v in out_edges.get(cur, ()):
                    if remaining.get((cur, v), 0) > 0:
                        pick = v
                        break
                if pick is None:
                    ok = False   # open chain — not a closed loop
                    break
                remaining[(cur, pick)] -= 1
                cur = pick

            if ok and len(loop) >= 3:
                loops.append(np.array(loop, dtype=np.int32))

    loops.sort(key=len, reverse=True)
    return loops


def _loop_perimeter(verts, loop):
    pts  = verts[loop]
    diff = np.roll(pts, -1, axis=0) - pts
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def _loop_is_fillable(verts, loop, mesh_diag, span_frac=0.6):
    """
    Reject loops that are too large/degenerate to fan-fill safely.

    A genuine hole spans a small fraction of the model.  A phantom loop (from a
    bad trace, or the model's own outer silhouette) spans a large fraction of
    the bounding box; fanning it to a centroid produces the spanning shards.
    Returns False for any loop whose bounding-box diagonal exceeds
    span_frac * mesh_diag, or that is geometrically degenerate.
    """
    pts = verts[loop]
    if len(pts) < 3:
        return False
    bb = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if mesh_diag > 0 and bb > span_frac * mesh_diag:
        return False
    return True


def _fill_hole(verts, loop, normals=None, vert_offset=0):
    k      = len(loop)
    pts    = verts[loop].astype(np.float64)  
    center = pts.mean(axis=0)

    cov    = np.cov((pts - center).T)
    eigval, eigvec = np.linalg.eigh(cov)
    plane_n = eigvec[:, 0]          
    if normals is not None:
        avg_n = normals[loop].mean(axis=0).astype(np.float64)
        avg_n /= (np.linalg.norm(avg_n) or 1.0)
        if np.dot(plane_n, avg_n) < 0:
            plane_n = -plane_n

    t1    = eigvec[:, 2]            
    t2    = eigvec[:, 1]            
    rel   = pts - center
    rad2d = np.linalg.norm(rel - np.outer(np.dot(rel, plane_n), plane_n),
                           axis=1)
    mean_r = float(rad2d.mean()) or 1.0

    if k <= 20 or mean_r < 1e-6:
        ring_fracs = []            
    elif k <= 60:
        ring_fracs = [0.40]
    else:
        ring_fracs = [0.55, 0.20]

    new_xyz     = []
    new_tris    = []
    ring_indices = [loop]          

    for frac in ring_fracs:
        prev_loop = ring_indices[-1]
        if len(ring_indices) == 1:
            prev_pts = verts[prev_loop].astype(np.float64)
        else:
            prev_pts = np.array(new_xyz[-len(prev_loop):], dtype=np.float64)
            
        new_ring_pts = center + frac * (prev_pts - center)
        ring_start   = vert_offset + len(new_xyz)
        new_xyz.extend(new_ring_pts.tolist())
        ring_idx = np.arange(ring_start, ring_start + len(prev_loop), dtype=np.int32)
        ring_indices.append(ring_idx)
        _connect_rings(prev_loop, ring_idx, new_tris)

    inner  = ring_indices[-1]
    c_idx  = vert_offset + len(new_xyz)
    new_xyz.append(center.tolist())
    for i in range(len(inner)):
        a = int(inner[i])
        b = int(inner[(i + 1) % len(inner)])
        new_tris.append([a, b, c_idx])

    new_verts = np.array(new_xyz, dtype=np.float32)
    new_faces = np.array(new_tris, dtype=np.int32)

    if len(new_faces) > 0:
        f0 = new_faces[0]
        def get_pt(idx):
            return verts[idx] if idx < len(verts) else new_verts[idx - vert_offset]
            
        p0, p1, p2 = get_pt(f0[0]), get_pt(f0[1]), get_pt(f0[2])
        fn = np.cross(p1 - p0, p2 - p0)
        if np.dot(fn, plane_n) < 0:
            new_faces = new_faces[:, [0, 2, 1]]   

    return new_verts, new_faces


def _connect_rings(outer, inner, tris):
    ko, ki = len(outer), len(inner)
    io_, ii_ = 0, 0
    total = ko + ki
    for _ in range(total):
        a  = int(outer[io_ % ko])
        b  = int(outer[(io_ + 1) % ko])
        c  = int(inner[ii_ % ki])
        d  = int(inner[(ii_ + 1) % ki])
        ao = (io_ + 1) / ko
        ai = (ii_ + 1) / ki
        if ao <= ai:
            tris.append([a, b, c])
            io_ += 1
        else:
            tris.append([a, c, d])
            ii_ += 1
    while io_ < ko:
        tris.append([int(outer[io_ % ko]),
                     int(outer[(io_+1) % ko]),
                     int(inner[ii_ % ki])])
        io_ += 1
    while ii_ < ki:
        tris.append([int(outer[io_ % ko]),
                     int(inner[ii_ % ki]),
                     int(inner[(ii_+1) % ki])])
        ii_ += 1


# =============================================================================
#  Tool
# =============================================================================

class SmoothTool(BaseTool):
    """
    Surface Refinement & Smart Smoothing Tool — plug-and-play BaseTool.
    """

    def __init__(self, app):
        super().__init__(app)
        self.name      = "Surface Smooth"
        self.icon_key = 'smooth'
        self.hotkey   = 'S'

        self.algo_idx    = 3   
        self.algo_names  = ["Laplacian", "Taubin (Volume-Safe)",
                            "HC Laplacian", "Smart (Feature-Aware)"]
        self.iterations  = 3
        self.strength    = 0.5
        self.taubin_mu   = -0.53
        self.hc_alpha    = 0.0
        self.hc_beta     = 0.5
        self.use_selection = False

        self.smart_edge_thresh = 0.6   
        self.smart_noise_boost = 2.0   
        self.smart_fill_strength = 0.3  
        self.smart_fill_mode     = 0    

        self.mode           = 0
        self.brush_radius   = 5.0
        self.brush_strength = 0.3
        self.is_painting    = False

        self._L          = None
        self._degrees    = None
        self._L_n_verts  = -1

        self._feature_weights    = None   
        self._fw_n_verts         = -1

        self.show_heatmap  = False
        self._heatmap_mode = 0    
        self._readiness    = None
        self._curv         = None
        self._heatmap_vmax = 1.0

        self._kd_tree = None
        self._kd_n    = -1

        self._building     = False
        self._build_result = None   

        self._pending_normal_recompute = False
        self._sculptor_synced = False

        self.hole_max_verts   = 100    
        self.hole_smooth_iter = 5      
        self._detected_holes  = None   
        self._hole_selected   = []     
        self._hole_result     = ""     

        self.status_msg = "Ready"

    def on_activate(self):
        self.app.status_text = "Surface Smooth"
        self._sculptor_synced = False
        self._start_build_async()

    def on_deactivate(self):
        self.is_painting = False
        if self.show_heatmap:
            self.show_heatmap = False
            self._restore_colors()

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
            if self.algo_idx == 3 or self.mode == 1:
                self._recompute_feature_weights()

        if self._pending_normal_recompute:
            self._pending_normal_recompute = False
            if self.app.has_faces and self.app.faces is not None:
                self.app.normals     = _recompute_normals(self.app.points,
                                                          self.app.faces)
                self.app.has_normals = True
            self._recompute_feature_weights()

    def _ensure_laplacian(self) -> bool:
        if not _SCIPY_OK or not self.app.has_faces or self.app.faces is None:
            return False
        n = len(self.app.points)
        if self._L is not None and self._L_n_verts == n:
            return True
        self._L, self._degrees = _build_laplacian(n, self.app.faces)
        self._L_n_verts = n
        return True

    def _ensure_kdtree(self):
        n = len(self.app.points)
        if self._kd_tree is not None and self._kd_n == n:
            return
        if _SCIPY_OK:
            self._kd_tree = KDTree(self.app.points); self._kd_n = n

    def _invalidate(self):
        self._L = None; self._L_n_verts = -1
        self._kd_tree = None; self._kd_n = -1
        self._curv = None; self._readiness = None
        self._feature_weights = None; self._fw_n_verts = -1

    def _recompute_feature_weights(self):
        if self._L is None:
            return
        n = len(self.app.points)

        if (not self.app.has_normals or self.app.normals is None
                or len(self.app.normals) != n):
            if self.app.has_faces and self.app.faces is not None:
                self.app.normals     = _recompute_normals(self.app.points,
                                                          self.app.faces)
                self.app.has_normals = True
            else:
                return

        if self._curv is None or len(self._curv) != n:
            self._curv = _compute_curvature(self.app.points, self._L)

        self._feature_weights = _compute_feature_weights(
            self.app.normals, self._L, self._curv,
            self.smart_edge_thresh, self.smart_noise_boost)
        self._fw_n_verts = n

    def _get_feature_weights(self) -> np.ndarray:
        n = len(self.app.points)
        if self._feature_weights is None or self._fw_n_verts != n:
            self._recompute_feature_weights()
        if self._feature_weights is None:
            return np.ones(n, dtype=np.float32)
        return self._feature_weights

    def _analyse(self):
        if not self._ensure_laplacian():
            return
        self._curv      = _compute_curvature(self.app.points, self._L)
        self._readiness = _readiness_score(self.app.points,
                                           self._degrees, self._curv)
        self._recompute_feature_weights()
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
        if self._heatmap_mode == 1 and self._feature_weights is not None:
            return 1.0 - self._feature_weights   
        return self._curv  

    def _apply_heatmap_all(self, data):
        t = np.clip(data / (self._heatmap_vmax + 1e-8), 0.0, 1.0).astype(np.float32)
        self.app.points_colors[:, :3] = self.app.points
        self.app.points_colors[:, 3]  = np.clip(2.0 * t,         0.0, 1.0)
        self.app.points_colors[:, 4]  = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
        self.app.points_colors[:, 5]  = 0.0

    def _apply_heatmap_local(self, hit_arr, data):
        local = data[hit_arr]
        t = np.clip(local / (self._heatmap_vmax + 1e-8), 0.0, 1.0).astype(np.float32)
        self.app.points_colors[hit_arr, 3] = np.clip(2.0 * t,         0.0, 1.0)
        self.app.points_colors[hit_arr, 4] = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
        self.app.points_colors[hit_arr, 5] = 0.0

    def _restore_colors(self):
        self.app.points_colors = np.hstack(
            [self.app.points, self.app.base_colors]).astype(np.float32)
        if hasattr(self.app, 'update_vbo_full'):
            self.app.update_vbo_full()

    def _run_smooth(self):
        if not self._ensure_laplacian():
            self.status_msg = "⚠ No face data"; return
        self.app.save_state("Smooth Surface")

        mask = self._make_mask()
        v    = self.app.points.copy()
        lam  = float(self.strength)

        if self.algo_idx == 0:
            v = _laplacian_smooth(v, self._L, mask, lam, self.iterations)
        elif self.algo_idx == 1:
            v = _taubin_smooth(v, self._L, mask, lam,
                               float(self.taubin_mu), self.iterations)
        elif self.algo_idx == 2:
            v = _hc_smooth(v, self._L, mask, float(self.hc_alpha),
                           float(self.hc_beta), lam, self.iterations)
        else:  
            normals = self.app.normals if self.app.has_normals else \
                      _recompute_normals(v, self.app.faces)
            fw = self._get_feature_weights()
            v  = _smart_smooth(v, self._L, normals, mask, lam, fw,
                               self.iterations,
                               fill_strength=self.smart_fill_strength,
                               fill_mode=self.smart_fill_mode)

        self.app.points = v
        self.app.normals     = _recompute_normals(v, self.app.faces)
        self.app.has_normals = True
        self.app.points_colors = np.hstack(
            [v, self.app.base_colors]).astype(np.float32)
        self._invalidate()
        self.app.update_vbo_full()
        if self.show_heatmap:
            self._ensure_laplacian(); self._analyse()
        self.status_msg = (f"Done — {int(np.sum(mask)):,} verts, "
                           f"{self.algo_names[self.algo_idx]}, "
                           f"{self.iterations} iter")
        self.app.status_text = self.status_msg
        self._start_build_async()

    def _make_mask(self, indices=None):
        n = len(self.app.points)
        if indices is not None:
            m = np.zeros(n, dtype=bool); m[indices] = True; return m
        if self.use_selection and np.any(self.app.selected):
            return self.app.selected.copy()
        return np.ones(n, dtype=bool)

    # ---------- FIX: Ensure points array is writable before modifying ----------
    def _ensure_points_writable(self):
        """Make self.app.points writable (copy if necessary)."""
        if not self.app.points.flags.writeable:
            self.app.points = self.app.points.copy()
            # Also update points_colors to reflect new points array
            self.app.points_colors[:, :3] = self.app.points
            self.app.vbo_dirty = True

    def _smooth_brush_at(self, world_pos, radius, strength):
        if self._L is None or not _SCIPY_OK:
            return
        self._ensure_kdtree()
        if self._kd_tree is None:
            return

        hit = self._kd_tree.query_ball_point(world_pos, radius)
        if not hit:
            return

        hit_arr = np.array(hit, dtype=np.int32)

        # Ensure points array is writable before modification
        self._ensure_points_writable()
        pts = self.app.points   # now writable

        L_local   = self._L[hit_arr, :]
        avg_local = np.asarray(L_local @ pts, dtype=np.float32)
        delta     = avg_local - pts[hit_arr]

        if self.algo_idx == 3 and self.app.has_normals:
            n_local = self.app.normals[hit_arr]
            fw      = self._get_feature_weights()[hit_arr]
            signed_curv = np.sum(delta * n_local, axis=1)
            fs = self.smart_fill_strength
            if fs > 0.0:
                fm = self.smart_fill_mode
                if fm == 1:
                    nc = np.maximum(signed_curv, 0.0)
                elif fm == 2:
                    nc = np.minimum(signed_curv, 0.0)
                else:
                    nc = signed_curv
                pts[hit_arr] += (fs * fw * nc)[:, None] * n_local
            d_tang = delta - signed_curv[:, None] * n_local
            pts[hit_arr] += (strength * fw[:, None] * d_tang)
        else:
            pts[hit_arr] += strength * delta

        # Update normals if needed? Not done here – will be recomputed on release.

        if self.show_heatmap:
            if self._heatmap_mode == 0 and self._curv is not None:
                self._curv[hit_arr] = np.linalg.norm(
                    avg_local - pts[hit_arr], axis=1).astype(np.float32)
                self._apply_heatmap_local(hit_arr, self._curv)
            elif self._heatmap_mode == 1 and self._feature_weights is not None:
                pass

        # Update points_colors and upload
        self.app.points_colors[hit_arr, :3] = pts[hit_arr]
        glBindBuffer(GL_ARRAY_BUFFER, self.app.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.app.points_colors.nbytes,
                     self.app.points_colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    # ---------------------------------------------------------------------------

    def on_mouse_button(self, button, action, mods, x, y):
        if self.mode != 1 or imgui.get_io().want_capture_mouse:
            return False

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            hit = getattr(self.app, 'cursor_world_pos', None)
            if hit is None: return False
            if self._L is None:
                self.status_msg = "⏳ Building — wait a moment"; return False
            self.app.save_state("Smooth Brush")
            self.is_painting = True
            self._ensure_kdtree()
            self._smooth_brush_at(hit, self.brush_radius, self.brush_strength)
            return True

        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            if self.is_painting:
                self.is_painting = False
                self._pending_normal_recompute = True
                self._L = None; self._L_n_verts = -1
                self._kd_tree = None; self._kd_n = -1
                self._start_build_async()
                self._ensure_kdtree()   
                if self.show_heatmap and self._curv is not None:
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
                self._smooth_brush_at(hit, self.brush_radius,
                                      self.brush_strength)
            return True
        return False

    def on_key(self, key, scancode, action, mods):
        if action != glfw.PRESS: return False
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if self.mode == 0: self._run_smooth(); return True
        elif key == glfw.KEY_H:
            self.show_heatmap = not self.show_heatmap
            if self.show_heatmap: self._ensure_laplacian(); self._analyse()
            else: self._restore_colors()
            return True
        elif key == glfw.KEY_TAB:
            self.mode = 1 - self.mode
            if self.mode == 1: self._ensure_kdtree()
            return True
        return False

    def update(self):
        self._poll_build()

    def render_3d(self):
        if self.mode != 1:
            return
        if not self._sculptor_synced:
            if not hasattr(self.app,'sculptor') or self.app.sculptor is None:
                if hasattr(self.app,'sync_sculptor'):
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
        cx,cy,cz = pos; r = self.brush_radius; steps = 64
        n = np.array([0.,1.,0.])
        tmp = np.array([1.,0.,0.]) if abs(n[1])>0.9 else np.array([0.,1.,0.])
        t1 = np.cross(n,tmp); t1/=(np.linalg.norm(t1) or 1.)
        t2 = np.cross(n,t1);  t2/=(np.linalg.norm(t2) or 1.)
        ctr = np.array([cx,cy,cz])
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glLineWidth(2.); glColor4f(0.,0.85,1.,1.)
        glBegin(GL_LINE_LOOP)
        for i in range(steps):
            a = 2.*np.pi*i/steps
            p = ctr + r*(np.cos(a)*t1 + np.sin(a)*t2)
            glVertex3f(*p)
        glEnd()
        glPointSize(5.); glBegin(GL_POINTS); glVertex3f(cx,cy,cz); glEnd()
        glPopAttrib()

    def _detect_holes(self):
        if not self.app.has_faces or self.app.faces is None:
            self._hole_result = "⚠ No face topology"; return
        loops = _find_boundary_loops(self.app.faces, len(self.app.points))
        if not loops:
            self._detected_holes = []
            self._hole_selected  = []
            self._hole_result    = "✓ No holes found — mesh is closed"
            return

        # Mesh bounding-box diagonal — used to reject phantom / outer-silhouette
        # loops that would otherwise be fanned across the whole model.
        mesh_diag = float(np.linalg.norm(
            self.app.points.max(axis=0) - self.app.points.min(axis=0))) or 1.0

        filtered = []
        rejected = 0
        for lp in loops:
            if not _loop_is_fillable(self.app.points, lp, mesh_diag):
                rejected += 1
                continue
            perim = _loop_perimeter(self.app.points, lp)
            filtered.append((lp, perim))

        self._detected_holes = filtered
        self._hole_selected  = [True] * len(filtered)
        if filtered:
            note = f" ({rejected} large loop(s) ignored)" if rejected else ""
            self._hole_result = (f"Found {len(filtered)} hole(s)  "
                                 f"({sum(len(l) for l,_ in filtered):,} boundary verts)"
                                 + note)
        else:
            self._hole_result = (f"No fillable holes — {rejected} loop(s) too large "
                                 f"(likely the open mesh border)")

    def _run_fill_holes(self):
        if not self._detected_holes:
            self._hole_result = "⚠ Run Detect first"; return
        if not self.app.has_faces or self.app.faces is None:
            self._hole_result = "⚠ No face topology"; return

        self.app.save_state("Fill Holes")

        # Mesh diagonal for the per-loop sanity guard
        mesh_diag = float(np.linalg.norm(
            self.app.points.max(axis=0) - self.app.points.min(axis=0))) or 1.0

        # Performance Fix: Accumulate all arrays into lists, then vstack exactly ONCE.
        verts_list = [self.app.points.copy()]
        faces_list = [self.app.faces.copy().reshape(-1, 3).astype(np.int32)]
        norms = self.app.normals if self.app.has_normals else _recompute_normals(self.app.points, self.app.faces)

        current_vert_count = len(self.app.points)
        filled = 0
        skipped = 0
        patch_records = []

        for idx, (loop, perim) in enumerate(self._detected_holes):
            if not self._hole_selected[idx]:
                skipped += 1; continue
            if len(loop) > self.hole_max_verts:
                skipped += 1; continue
            # Safety guard: never fan a loop that spans a large part of the model.
            if not _loop_is_fillable(self.app.points, loop, mesh_diag):
                skipped += 1; continue

            try:
                new_v, new_f = _fill_hole(self.app.points, loop, normals=norms, vert_offset=current_vert_count)
            except Exception as e:
                skipped += 1; continue

            if len(new_f) == 0:
                # Small hole (<=20 verts): no new vertices, just a triangle fan
                # over existing boundary vertices is still valid via _fill_hole's
                # final fan, so an empty new_f means nothing was produced — skip.
                skipped += 1; continue

            if len(new_v) > 0:
                verts_list.append(new_v.astype(np.float32))
            faces_list.append(new_f.astype(np.int32))
            patch_records.append((loop, current_vert_count, current_vert_count + len(new_v)))

            current_vert_count += len(new_v)
            filled += 1

        if filled == 0:
            self._hole_result = (f"No holes filled "
                                 f"({skipped} skipped / too large)")
            return

        # Single, rapid mass-reallocation
        verts = np.vstack(verts_list)
        faces = np.vstack(faces_list)

        patch_union = np.zeros(len(verts), dtype=bool)
        for loop, start_idx, end_idx in patch_records:
            patch_union[loop] = True
            if end_idx > start_idx:
                patch_union[start_idx:end_idx] = True

        # Single normal computation
        norms = _recompute_normals(verts, faces)

        # ── Post-fill smart smooth on the patch ───────────────────────────
        L_new, deg_new = _build_laplacian(len(verts), faces)
        curv_new = _compute_curvature(verts, L_new)
        fw_new   = _compute_feature_weights(
            norms, L_new, curv_new,
            self.smart_edge_thresh, self.smart_noise_boost)

        verts = _smart_smooth(
            verts, L_new, norms, patch_union,
            strength=0.5, feature_weights=fw_new,
            iters=self.hole_smooth_iter,
            fill_strength=0.3, fill_mode=0)

        norms = _recompute_normals(verts, faces)

        # ── Commit (keep faces as (F,3) uint32 — the canonical shape) ──────
        self.app.points        = verts
        self.app.faces         = faces.reshape(-1, 3).astype(np.uint32)
        self.app.normals       = norms
        self.app.has_normals   = True
        self.app.has_faces     = True

        n_new = len(verts) - len(self.app.base_colors)
        if n_new > 0:
            grey = np.full((n_new, 3), 0.7, dtype=np.float32)
            self.app.base_colors = np.vstack(
                [self.app.base_colors, grey]).astype(np.float32)

        self.app.points_colors = np.hstack(
            [verts, self.app.base_colors]).astype(np.float32)

        # Keep selection / object-id arrays in sync with the new vertex count
        self.app.selected = np.zeros(len(verts), dtype=bool)
        if hasattr(self.app, 'point_object_ids'):
            old_ids = self.app.point_object_ids
            if old_ids is not None and len(old_ids) < len(verts):
                pad = np.zeros(len(verts) - len(old_ids), dtype=np.int32)
                self.app.point_object_ids = np.concatenate([old_ids, pad])

        if hasattr(self.app, 'rebuild_ids'):
            self.app.rebuild_ids(len(verts))

        # Ensure the EBO exists so indexed rendering is used
        if (not hasattr(self.app, 'ebo')) or self.app.ebo is None:
            try:
                from OpenGL.GL import glGenBuffers
                self.app.ebo = glGenBuffers(1)
            except Exception:
                pass

        self._invalidate()
        self.app.update_vbo_full()
        if hasattr(self.app, '_invalidate_lod_cache'):
            self.app._invalidate_lod_cache()
        if hasattr(self.app, '_rebuild_kdtree'):
            self.app._rebuild_kdtree()
        if hasattr(self.app, 'sync_sculptor'):
            self.app.sync_sculptor()

        self._detected_holes = None
        self._hole_selected  = []
        self._start_build_async()

        self._hole_result = (f"✓ Filled {filled} hole(s)"
                             + (f", skipped {skipped}" if skipped else ""))
        self.status_msg   = self._hole_result

    # ── GUI ───────────────────────────────────────────────────────────────

    def draw_options(self):
        self._poll_build()

        if self._building:
            imgui.text_colored("  Building mesh graph…", 1.0, 0.8, 0.2)
            imgui.separator()

        if imgui.begin_tab_bar("SmoothHoleTabs"):
            if imgui.begin_tab_item("Smoothing")[0]:
                self._draw_smooth_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Hole Filling")[0]:
                self._draw_hole_tab()
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def _draw_smooth_tab(self):
        imgui.dummy(0, 4)
        imgui.text_colored("Surface Refinement", 0.4, 0.85, 1.0)
        imgui.separator()

        imgui.text("Mode:")
        imgui.same_line()
        if imgui.radio_button("Global##sm", self.mode == 0): self.mode = 0
        imgui.same_line()
        if imgui.radio_button("Brush##sm",  self.mode == 1):
            self.mode = 1; self._ensure_kdtree()
        imgui.text_disabled("Tab: toggle  |  H: heatmap")
        imgui.dummy(0, 4); imgui.separator()

        imgui.push_item_width(190)
        _, self.algo_idx = imgui.combo("##algo", self.algo_idx, self.algo_names)
        imgui.pop_item_width()
        hints = ["Fast — slight shrinkage",
                 "Volume-safe — best for RE",
                 "Feature-preserving (HC)",
                 "Smart — edges preserved, noise removed"]
        imgui.text_disabled(hints[self.algo_idx])
        imgui.dummy(0, 4)

        if self.algo_idx == 3:
            imgui.text_colored("Smart Settings", 0.6, 1.0, 0.6)
            imgui.push_item_width(160)
            chg_et, self.smart_edge_thresh = imgui.slider_float(
                "Edge Threshold##et", self.smart_edge_thresh, 0.0, 1.0, "%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Normal alignment below this = sharp edge (preserved).\n"
                    "Lower = only very sharp edges preserved.\n"
                    "Higher = even gentle curves preserved.")
            chg_nb, self.smart_noise_boost = imgui.slider_float(
                "Noise Boost##nb", self.smart_noise_boost, 0.5, 5.0, "%.1f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "How aggressively scan noise spikes are removed.\n"
                    "Higher = spike verts get extra smoothing force.")
            imgui.pop_item_width()
            imgui.dummy(0, 4)
            imgui.text_colored("Surface Correction (Mass)", 0.6, 1.0, 0.6)
            chg_fs, self.smart_fill_strength = imgui.slider_float(
                "Fill Strength##fs", self.smart_fill_strength, 0.0, 1.0, "%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Strength of normal-direction correction.\n"
                    "0 = tangential smoothing only (no mass change)\n"
                    ">0 = surface height corrected toward local mean:\n"
                    "  concave dips filled, convex bumps reduced.")
            fill_labels = ["Bilateral (fill+cut)", "Fill only", "Cut only"]
            imgui.push_item_width(190)
            _, self.smart_fill_mode = imgui.combo(
                "##fm", self.smart_fill_mode, fill_labels)
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Bilateral: correct both concave AND convex errors\n"
                    "Fill only: only push outward (fix scan dropouts)\n"
                    "Cut only:  only pull inward  (fix bumps/spikes)")
            imgui.text_disabled(
                ["↕ Fill valleys + cut bumps",
                 "↑ Fill concave dropouts only",
                 "↓ Cut convex bumps only"][self.smart_fill_mode])

            if chg_et or chg_nb:
                self._feature_weights = None; self._fw_n_verts = -1
            imgui.dummy(0, 4)

        if self.mode == 0:
            imgui.push_item_width(160)
            _, self.iterations = imgui.slider_int("Iterations##it",
                                                   self.iterations, 1, 20)
            _, self.strength   = imgui.slider_float("Strength##lam",
                                                     self.strength, 0.01, 1.0, "%.2f")
            if self.algo_idx == 1:
                _, self.taubin_mu = imgui.slider_float("Inflate (μ)##mu",
                                                        self.taubin_mu, -1., -.01, "%.3f")
            if self.algo_idx == 2:
                _, self.hc_alpha = imgui.slider_float("HC Alpha##hca",
                                                       self.hc_alpha, 0., 1., "%.2f")
                _, self.hc_beta  = imgui.slider_float("HC Beta##hcb",
                                                       self.hc_beta, 0., 1., "%.2f")
            imgui.pop_item_width()
            _, self.use_selection = imgui.checkbox("Selection Only##sel",
                                                    self.use_selection)
            imgui.dummy(0, 6)
            if self._building:
                imgui.push_style_var(imgui.STYLE_ALPHA,
                                     imgui.get_style().alpha * 0.4)
            if imgui.button("Apply Smooth##go", width=-1, height=32):
                if not self._building: self._run_smooth()
            if self._building: imgui.pop_style_var()
            imgui.text_disabled("Enter: quick apply")
        else:
            imgui.push_item_width(160)
            _, self.brush_radius   = imgui.slider_float("Radius##br",
                                                         self.brush_radius, 0.1, 50., "%.1f")
            _, self.brush_strength = imgui.slider_float("Strength##bs",
                                                         self.brush_strength, 0.01, 1., "%.2f")
            imgui.pop_item_width()
            imgui.text_disabled("Left-drag: paint smooth")

        imgui.dummy(0, 4); imgui.separator()

        imgui.text("Visualisation:")
        chg, self.show_heatmap = imgui.checkbox("Show Heatmap##hm",
                                                 self.show_heatmap)
        if chg:
            if self.show_heatmap: self._ensure_laplacian(); self._analyse()
            else: self._restore_colors()
        if self.show_heatmap:
            prev_mode = self._heatmap_mode
            if imgui.radio_button("Curvature##hm0", self._heatmap_mode == 0):
                self._heatmap_mode = 0
            imgui.same_line()
            if imgui.radio_button("Feature Edges##hm1", self._heatmap_mode == 1):
                self._heatmap_mode = 1
            if self._heatmap_mode != prev_mode:
                self._push_heatmap()
            imgui.text_colored("Green=Smooth/Flat  Red=Edge/Spike",
                               0.8, 0.8, 0.2)
            if imgui.button("Refresh##rhm", width=-1):
                self._ensure_laplacian(); self._analyse()

        imgui.dummy(0, 4); imgui.separator()

        imgui.text("3D Print Readiness:")
        if imgui.button("Analyse Surface##an", width=-1): self._analyse()
        if self._readiness is not None:
            r     = self._readiness
            score = r["score"]
            col   = ((0.2,1.,0.2) if score>=85 else
                     (1.,0.85,0.) if score>=60 else (1.,0.3,0.2))
            lbl   = ("GOOD" if score>=85 else "FAIR" if score>=60 else "POOR")
            imgui.text_colored(f"Score: {score}/100 — {lbl}", *col)
            imgui.push_item_width(-1)
            imgui.progress_bar(score/100., (-1,14), f"{score:.1f}/100")
            imgui.pop_item_width()
            imgui.dummy(0,2)
            imgui.text(f"  Spikes      : {r['spike_count']:,}")
            imgui.text(f"  Bad valence : {r['bad_valence']:,}")
            imgui.text(f"  Roughness   : {r['roughness']*1000:.2f}‰")
            imgui.text(f"  Max dev     : {r['max_dev']*1000:.2f}‰")
            if score < 85:
                imgui.separator()
                imgui.text_colored("Recommended:", 1., 0.85, 0.3)
                if r["spike_count"] > 50:
                    imgui.text_disabled("→ Try Smart mode, 3–5 passes")
                if r["roughness"] > 0.002:
                    imgui.text_disabled("→ Increase iterations")
                if r["bad_valence"] > 0:
                    imgui.text_disabled("→ Check mesh repair first")
                imgui.dummy(0,4)
                if imgui.button("Auto-Fix##af", width=-1, height=28):
                    self._auto_fix(r)

        imgui.dummy(0,4); imgui.separator()

        if self.algo_idx == 3 and self._feature_weights is not None:
            imgui.text("Feature Map:")
            n_edge = int(np.sum(self._feature_weights < 0.2))
            n_flat = int(np.sum(self._feature_weights > 0.8))
            imgui.text(f"  Sharp edges : {n_edge:,} verts")
            imgui.text(f"  Smooth areas: {n_flat:,} verts")
            if imgui.button("Show Feature Edges##fe", width=-1):
                self.show_heatmap    = True
                self._heatmap_mode   = 1
                self._ensure_laplacian(); self._analyse()

        imgui.dummy(0,4); imgui.separator()
        if imgui.button("Reset Parameters##rp", width=-1):
            self._reset_params()
        imgui.dummy(0,4)
        imgui.text_disabled(self.status_msg)

    def _draw_hole_tab(self):
        imgui.dummy(0, 4)
        imgui.text_colored("Hole Filling Configuration", 0.4, 0.85, 1.0)
        imgui.push_item_width(160)
        _, self.hole_max_verts = imgui.slider_int(
            "Max hole verts##mhv", self.hole_max_verts, 3, 500)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Holes with more boundary vertices than this\n"
                "are skipped (probably intentional openings).")
        _, self.hole_smooth_iter = imgui.slider_int(
            "Blend passes##hsi", self.hole_smooth_iter, 1, 12)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Smart smooth passes applied to the filled patch\n"
                "after hole filling, to blend it into surrounding surface.")
        imgui.pop_item_width()
        imgui.dummy(0, 4)

        if imgui.button("Detect Holes##dh", width=-1, height=28):
            self._detect_holes()

        if self._detected_holes is not None:
            total_holes = len(self._detected_holes)
            if total_holes == 0:
                imgui.text_colored("  ✓ Mesh is closed", 0.4, 1.0, 0.4)
            else:
                imgui.text(f"  {total_holes} hole(s) detected:")
                
                if imgui.button("Select All##sa"):
                    self._hole_selected = [True] * total_holes
                imgui.same_line()
                if imgui.button("Deselect All##da"):
                    self._hole_selected = [False] * total_holes
                
                # Professional scrollable child window to hide the massive checkbox list
                imgui.begin_child("holes_list", width=0, height=180, border=True)
                for i, (lp, perim) in enumerate(self._detected_holes):
                    within = len(lp) <= self.hole_max_verts
                    if not within:
                        imgui.push_style_var(imgui.STYLE_ALPHA,
                                             imgui.get_style().alpha * 0.4)
                    chk = self._hole_selected[i] if i < len(self._hole_selected) else True
                    changed, new_chk = imgui.checkbox(
                        f"  Hole {i+1}: {len(lp)} verts, ⌀{perim:.1f}##hc{i}",
                        chk)
                    if changed and i < len(self._hole_selected):
                        self._hole_selected[i] = new_chk
                    if not within:
                        imgui.pop_style_var()
                        imgui.same_line()
                        imgui.text_disabled("(too large)")
                imgui.end_child()

                n_sel = sum(1 for i,(_,__)
                            in enumerate(self._detected_holes)
                            if (i < len(self._hole_selected)
                                and self._hole_selected[i]
                                and len(_) <= self.hole_max_verts))
                imgui.dummy(0, 4)
                lbl = f"Fill {n_sel} Hole(s)##fh" if n_sel else "Fill Holes##fh"
                disabled = n_sel == 0
                if disabled:
                    imgui.push_style_var(imgui.STYLE_ALPHA,
                                         imgui.get_style().alpha * 0.4)
                if imgui.button(lbl, width=-1, height=32):
                    if not disabled:
                        self._run_fill_holes()
                if disabled:
                    imgui.pop_style_var()

        if self._hole_result:
            col = (0.4,1.,0.4) if self._hole_result.startswith("✓") else (1.,0.6,0.3)
            imgui.text_colored(self._hole_result, *col)

        imgui.dummy(0,4); imgui.separator()
        if imgui.button("Reset Parameters##rp2", width=-1):
            self._reset_params()
        imgui.dummy(0,4)
        imgui.text_disabled(self.status_msg)

    def _reset_params(self):
        self.iterations=3; self.strength=0.5; self.taubin_mu=-0.53
        self.hc_alpha=0.; self.hc_beta=0.5; self.algo_idx=3
        self.brush_radius=5.; self.brush_strength=0.3
        self.smart_edge_thresh=0.6; self.smart_noise_boost=2.0
        self.smart_fill_strength=0.3; self.smart_fill_mode=0
        self.hole_max_verts=100; self.hole_smooth_iter=5
        self._detected_holes=None; self._hole_selected=[]; self._hole_result=""

    def _auto_fix(self, r):
        score = r["score"]
        self.algo_idx = 3  
        if score < 40:
            self.iterations=8; self.strength=0.6; self.smart_noise_boost=3.5
            self.smart_fill_strength=0.5; self.smart_fill_mode=0  
        elif score < 70:
            self.iterations=4; self.strength=0.5; self.smart_noise_boost=2.5
            self.smart_fill_strength=0.3; self.smart_fill_mode=0
        else:
            self.iterations=2; self.strength=0.4; self.smart_noise_boost=2.0
            self.smart_fill_strength=0.15; self.smart_fill_mode=0
        self._run_smooth()
        self.status_msg = (f"Auto-fixed (Smart, {self.iterations} iter) "
                           f"— was {score}/100")
