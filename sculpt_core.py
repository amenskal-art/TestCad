# sculpt_core.py
import numpy as np
from typing import Literal

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
    print("Warning: Scipy not installed. Sculpting will be slower.")

class SculptSession:
    def __init__(self, points: np.ndarray, normals: np.ndarray):
        # We hold a reference to the live points array from the App
        self.points = points
        self.normals = normals
        self._dirty_indices = set()   # tracks which point indices were displaced this stroke
        self.tree = None
        self.rebuild_tree()

    def rebuild_tree(self):
        """Builds the KDTree for fast spatial queries."""
        if cKDTree is not None:
            self.tree = cKDTree(self.points)
        else:
            self.tree = None

    def query_radius(self, center, radius):
        """Returns indices of points within radius of center."""
        if self.tree:
            return np.array(self.tree.query_ball_point(center, radius), dtype=np.int64)
        else:
            d2 = np.sum((self.points - center)**2, axis=1)
            return np.nonzero(d2 <= radius**2)[0]

    def get_nearest_normal(self, center, view_dir=None):
        """
        Finds the normal of the point closest to the 3D cursor that also
        faces toward the camera.

        Without a view_dir filter, the nearest point is often on the BACK
        face of thin geometry, whose normal points away from the camera.
        That flips the brush ring to orient against the wrong surface.

        Strategy:
          1. Query the k nearest candidates (not just 1).
          2. Among those, prefer the first one whose normal has a positive
             dot product with the camera-to-point direction (i.e. front-facing
             relative to the viewer).
          3. Fall back to the absolute nearest if every candidate is back-facing
             (degenerate / single-layer cloud).

        Parameters
        ----------
        center   : (3,) world-space position of the brush cursor
        view_dir : (3,) unit vector from camera toward the scene.
                   If None, no facing filter is applied (original behaviour).
        """
        K = 16  # candidates to inspect — cheap, KDTree query is O(log N)

        if self.tree:
            n_pts = len(self.points)
            k = min(K, n_pts)
            dists, idxs = self.tree.query(center, k=k)
            # query returns a scalar when k=1, normalise to array
            if k == 1:
                return self.normals[idxs]
        else:
            # Brute-force fallback
            d2 = np.sum((self.points - center) ** 2, axis=1)
            order = np.argsort(d2)
            idxs = order[:K]

        if view_dir is None:
            # No camera info — just return the nearest as before
            return self.normals[idxs[0]]

        # Normalise view_dir defensively
        vd = np.asarray(view_dir, dtype=np.float64)
        vd_len = np.linalg.norm(vd)
        if vd_len < 1e-9:
            return self.normals[idxs[0]]
        vd /= vd_len

        # A normal is "front-facing" when it opposes the view direction.
        # dot(normal, view_dir) < 0  →  surface faces the camera.
        candidate_normals = self.normals[idxs]           # (k, 3)
        dots = candidate_normals @ vd                    # (k,)
        front_facing = np.where(dots < 0.0)[0]

        if len(front_facing) > 0:
            # Return the closest front-facing candidate
            return self.normals[idxs[front_facing[0]]]
        else:
            # All candidates are back-facing (e.g. inside a cavity looking out).
            # Fall back to the absolute nearest to avoid a dead brush.
            return self.normals[idxs[0]]

    def _ensure_writable(self):
        """
        Guarantee self.points is a writable, owned array.

        The App may hand us a read-only view (e.g. a numpy slice backed by a
        ctypes / OpenGL buffer).  We copy once and own it from then on.
        After every sculpt() call the caller MUST sync back:
            app.points = sculptor.points
        so that the VBO rebuild reads the updated data.
        """
        if not self.points.flags.writeable:
            self.points = self.points.copy()

    def sculpt(self,
               hit_point: np.ndarray,
               radius: float,
               strength: float,
               mode: Literal['add', 'remove', 'smooth', 'flatten'] = 'add'):

        # Guarantee the array is writable. Caller must sync: app.points = sculptor.points
        self._ensure_writable()

        # 1. Spatial Query: Find points inside the brush
        idxs = self.query_radius(hit_point, radius)
        if len(idxs) == 0: return

        pts_subset = self.points[idxs]

        # 2. Falloff Calculation
        dists = np.linalg.norm(pts_subset - hit_point, axis=1)
        t = dists / radius
        alphas = (1.0 - t*t)**2
        alphas[t >= 1.0] = 0.0
        alphas = alphas[:, None]

        # 3. Apply Modes
        if mode == 'smooth':
            local_center = np.average(pts_subset, axis=0, weights=alphas.flatten())
            delta = local_center - pts_subset
            self.points[idxs] += delta * (strength * alphas)

        elif mode == 'flatten':
            local_norm = np.mean(self.normals[idxs], axis=0)
            local_norm /= np.linalg.norm(local_norm) + 1e-9

            centroid = np.mean(pts_subset, axis=0)
            v = pts_subset - centroid
            dist_from_plane = np.sum(v * local_norm, axis=1)[:, None]

            flatten_delta = -local_norm * dist_from_plane
            self.points[idxs] += flatten_delta * (strength * alphas)

        elif mode in ['add', 'remove']:
            # --- FIX: spikes / explosions on repeated inflate-then-deflate ----
            #
            # Previously this branch pushed every vertex along ITS OWN stored
            # normal, scaled by the FULL brush radius:
            #     displacement = per_vertex_normal * (strength * radius * alpha)
            #
            # Two problems combined to produce spikes:
            #   1. The step was proportional to `radius` (often 10–50 units),
            #      so a single drag frame could move a vertex a huge distance.
            #   2. Per-vertex normals become noisy after the surface is bumped,
            #      and are only recomputed at STROKE END. Adjacent vertices then
            #      get pushed in divergent directions, tearing the surface apart
            #      when you deflate the region you just inflated.
            #
            # Fix: use ONE averaged normal for the whole brush footprint (like
            # `flatten` does) so all affected vertices move along the same axis,
            # and scale the step by a small fraction of the radius so no single
            # frame can launch a vertex.
            local_norm = np.mean(self.normals[idxs], axis=0)
            ln = np.linalg.norm(local_norm)
            if ln < 1e-6:
                # Degenerate normals: fall back to radial direction from the hit.
                local_norm = np.mean(pts_subset - hit_point, axis=0)
                ln = np.linalg.norm(local_norm)
                if ln < 1e-6:
                    return
            local_norm = local_norm / ln

            # Bounded step: a small fraction of the radius rather than the full
            # radius. Tune the 0.05 coefficient (0.02–0.08) for faster/slower
            # build-up. `strength` (0.01–1.0 from the UI) still scales it.
            step = 0.05 * radius * strength
            factor = step if mode == 'add' else -step
            displacement = local_norm[None, :] * (factor * alphas)
            self.points[idxs] += displacement
            # ------------------------------------------------------------------

        # 4. Mark all moved points as dirty for normal recomputation at stroke end
        self._dirty_indices.update(idxs.tolist())

    def recompute_dirty_normals(self, k: int = 16):
        """
        Recompute normals for every point touched during this stroke, plus
        their immediate neighbors (so the border of the brushed region blends
        smoothly into the untouched surface).

        Uses vectorized PCA: for each affected point, the normal is the
        eigenvector corresponding to the smallest eigenvalue of the local
        covariance matrix — i.e. the axis of least variance, which is the
        surface normal for a locally-planar cloud.

        Must be called AFTER rebuild_tree() so neighbor queries use the
        post-stroke geometry.

        After this call, self.normals is up-to-date and the caller should
        sync it back:
            app.normals = sculptor.normals
        then trigger a VBO re-upload via app.vbo_dirty = True.
        """
        if not self._dirty_indices or self.tree is None:
            return

        dirty_arr = np.fromiter(self._dirty_indices, dtype=np.int64)
        self._dirty_indices.clear()

        n_pts = len(self.points)
        k2 = min(k, n_pts)

        # Expand dirty set: include k-nearest neighbors of every dirty point
        # so the normal boundary blends into untouched geometry.
        _, nbr_of_dirty = self.tree.query(self.points[dirty_arr], k=k2)  # (D, k2)
        all_affected = np.unique(nbr_of_dirty.ravel())                   # (M,)

        # Ensure normals array is writable
        if not self.normals.flags.writeable:
            self.normals = self.normals.copy()

        # Vectorized PCA over all affected points in one batch.
        # For each point: gather its k nearest neighbors, center them,
        # compute 3x3 covariance, take the eigenvector of the smallest eigenvalue.
        _, nbr_idxs = self.tree.query(self.points[all_affected], k=k2)   # (M, k2)
        nbr_pts  = self.points[nbr_idxs]                                  # (M, k2, 3)
        centered = nbr_pts - nbr_pts.mean(axis=1, keepdims=True)          # (M, k2, 3)
        cov      = np.einsum('mki,mkj->mij', centered, centered)          # (M, 3, 3)

        # eigh returns eigenvalues sorted ascending; col 0 of eigvecs = normal
        _, eigvecs = np.linalg.eigh(cov)                                  # (M,3), (M,3,3)
        new_normals = eigvecs[:, :, 0].astype(np.float32)                 # (M, 3)

        # Flip any normals that would reverse relative to the existing orientation
        # (PCA gives an ambiguous sign; we preserve the hemisphere of the original).
        dots = np.einsum('mi,mi->m', new_normals, self.normals[all_affected])
        new_normals[dots < 0] *= -1

        self.normals[all_affected] = new_normals
