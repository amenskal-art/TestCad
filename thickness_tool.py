# tools/thickness_tool.py

from tool_system import BaseTool
import imgui
import glfw
import numpy as np
from OpenGL.GL import *
import threading


class ThicknessTool(BaseTool):
    """
    Thickness (Solidify) Tool.

    Like Blender's Solidify modifier: offsets the surface inward and outward
    into two shells and bridges their open boundary with a rim, producing a
    closed solid.  This is a pure-numpy extrude — it is fast and single-purpose.

    It does NOT clean, re-orient, or validate the mesh.  Mesh cleanup (smoothing,
    hole filling, smooth cuts) is handled by the other dedicated tools, so the
    thickness operation stays instant.  Two things it DOES guard against, because
    they are cheap and specific to extrusion:
      * rim winding is derived from each boundary edge's own orientation, so the
        rim faces are consistent rather than a fixed guessed pattern, and
      * a vertex with a zero/degenerate normal is given a fallback direction so
        the two shells don't collapse onto it (a zero-thickness pinch point).
    """
    def __init__(self, app):
        super().__init__(app)
        self.name = "Thickness"
        self.icon_key = 'thickness'
        self.hotkey = 'T'

        self.thickness = 1.0
        self.offset = 0.0
        self.fill_rim = True
        self.even_thickness = False

        self.status_msg = "Ready"
        self._working = False
        self._pending_update = None

    def on_activate(self):
        self.app.status_text = "Thickness (Solidify) Mode"

    def draw_options(self):
        imgui.text("Thickness (Solidify) Options")
        imgui.separator()

        _, self.thickness = imgui.slider_float("Thickness", self.thickness, 0.01, 10.0, "%.3f")
        _, self.offset = imgui.slider_float("Offset", self.offset, -1.0, 1.0, "%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("-1 = Inwards, 0 = Centered, 1 = Outwards")

        _, self.fill_rim = imgui.checkbox("Fill Rim", self.fill_rim)
        _, self.even_thickness = imgui.checkbox("Even Thickness", self.even_thickness)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Adjusts thickness at sharp corners to maintain uniform wall thickness.")

        imgui.dummy(0, 5)
        if self._working:
            imgui.text_colored("Processing...", 1.0, 0.8, 0.2)
        else:
            if imgui.button("Apply Thickness", width=-1, height=30):
                self._apply_thickness()

        imgui.dummy(0, 5)
        imgui.text_disabled(self.status_msg)

    def _apply_thickness(self):
        if not self.app.has_faces or getattr(self.app, 'faces', None) is None or len(self.app.faces) == 0:
            self.status_msg = "Error: Mesh has no faces."
            return

        self._working = True
        self.status_msg = "Calculating..."
        self.app.save_state("Apply Thickness")

        # Run in background thread to avoid freezing the UI
        threading.Thread(target=self._process_mesh, daemon=True).start()

    def _process_mesh(self):
        try:
            verts = self.app.points.copy()
            faces = self.app.faces.copy().reshape(-1, 3)
            normals = getattr(self.app, 'normals', None)
            if normals is not None:
                normals = normals.copy()
            colors = self.app.base_colors.copy()

            # Compute vertex normals if missing or stale
            if normals is None or len(normals) != len(verts):
                normals = self._compute_vertex_normals(verts, faces)

            n_verts = len(verts)

            # Guard: replace any zero/degenerate normal so the vertex still
            # gets offset (otherwise both shells coincide -> zero-thickness pinch).
            normals = normals.astype(np.float32)
            nlen = np.linalg.norm(normals, axis=1)
            bad = nlen < 1e-8
            if np.any(bad):
                centroid = verts.mean(axis=0)
                fallback = verts[bad] - centroid
                flen = np.linalg.norm(fallback, axis=1, keepdims=True)
                flen[flen < 1e-8] = 1.0
                normals[bad] = (fallback / flen).astype(np.float32)

            # Offsets for the two shells
            ofs_orig = (self.offset - 1.0) * 0.5 * self.thickness
            ofs_new  = (self.offset + 1.0) * 0.5 * self.thickness

            vert_scalars = np.ones(n_verts, dtype=np.float32)
            if self.even_thickness:
                vert_scalars = self._compute_even_thickness_scalars(verts, faces, normals)

            # Shell 1 (inner) - reversed winding so its normals face opposite
            # shell 2 (together they enclose a volume).
            shell1_verts = verts + normals * (ofs_orig * vert_scalars)[:, None]
            shell1_faces = faces[:, [0, 2, 1]]

            # Shell 2 (outer) - original winding, index offset by n_verts.
            shell2_verts = verts + normals * (ofs_new * vert_scalars)[:, None]
            shell2_faces = faces.copy() + n_verts

            new_verts = np.vstack([shell1_verts, shell2_verts])
            new_faces = np.vstack([shell1_faces, shell2_faces])
            new_colors = np.vstack([colors, colors])

            # --- Rim filling -----------------------------------------------
            # Bridge each open boundary edge of the surface between the two
            # shells.  Rim winding follows the boundary edge's own direction so
            # all rim quads are consistent (vectorised - no Python loop).
            if self.fill_rim:
                bnd = self._find_boundary_edges_oriented(faces)
                if len(bnd) > 0:
                    a = bnd[:, 0]
                    b = bnd[:, 1]
                    c = bnd[:, 1] + n_verts
                    d = bnd[:, 0] + n_verts
                    tri1 = np.stack([a, b, c], axis=1)
                    tri2 = np.stack([a, c, d], axis=1)
                    rim = np.vstack([tri1, tri2]).astype(np.int64)
                    new_faces = np.vstack([new_faces, rim])

            # Stash results; GL work happens on the main thread in update().
            self._result_verts = new_verts.astype(np.float32)
            self._result_faces = new_faces.reshape(-1, 3).astype(np.uint32)
            self._result_colors = new_colors.astype(np.float32)

            def _update_gl():
                self.app.points = self._result_verts
                self.app.faces = self._result_faces
                self.app.base_colors = self._result_colors

                self.app.normals = self._compute_vertex_normals(self.app.points, self.app.faces)
                self.app.has_normals = True
                self.app.has_faces = True
                self.app.render_mode_mesh = True

                # Reset per-vertex auxiliary arrays to the NEW vertex count.
                count = len(self.app.points)
                if hasattr(self.app, 'point_object_ids'):
                    self.app.point_object_ids = np.zeros(count, dtype=np.int32)
                if hasattr(self.app, 'selected'):
                    self.app.selected = np.zeros(count, dtype=bool)

                self.app.points_colors = np.hstack(
                    [self.app.points, self.app.base_colors]).astype(np.float32)

                self.app.rebuild_ids(count)

                # Rebuild EBO since face count changed
                if hasattr(self.app, 'ebo'):
                    if self.app.ebo is not None:
                        try:
                            glDeleteBuffers(1, [self.app.ebo])
                        except Exception:
                            pass
                    self.app.ebo = glGenBuffers(1)
                else:
                    self.app.ebo = glGenBuffers(1)

                self.app.update_vbo_full()
                if hasattr(self.app, '_invalidate_lod_cache'):
                    self.app._invalidate_lod_cache()
                if hasattr(self.app, 'sync_sculptor'):
                    self.app.sync_sculptor()
                if hasattr(self.app, '_rebuild_kdtree'):
                    self.app._rebuild_kdtree()

                self.status_msg = "Thickness applied successfully."
                self._working = False

            self._pending_update = _update_gl

        except Exception as e:
            self.status_msg = f"Error: {str(e)}"
            self._working = False

    def update(self):
        if self._pending_update:
            fn = self._pending_update
            self._pending_update = None
            fn()

    def _compute_vertex_normals(self, verts, faces):
        f = faces.reshape(-1, 3).astype(np.int32)
        v0, v1, v2 = verts[f[:,0]], verts[f[:,1]], verts[f[:,2]]
        fn = np.cross(v1-v0, v2-v0)
        nv = len(verts)
        nx = np.bincount(f[:,0],fn[:,0],minlength=nv) + np.bincount(f[:,1],fn[:,0],minlength=nv) + np.bincount(f[:,2],fn[:,0],minlength=nv)
        ny = np.bincount(f[:,0],fn[:,1],minlength=nv) + np.bincount(f[:,1],fn[:,1],minlength=nv) + np.bincount(f[:,2],fn[:,1],minlength=nv)
        nz = np.bincount(f[:,0],fn[:,2],minlength=nv) + np.bincount(f[:,1],fn[:,2],minlength=nv) + np.bincount(f[:,2],fn[:,2],minlength=nv)
        n = np.stack([nx,ny,nz], axis=1)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n /= norms
        return n.astype(np.float32)

    def _find_boundary_edges_oriented(self, faces):
        """
        Return boundary edges as DIRECTED edges in their original triangle
        orientation, used exactly once each.

        A boundary edge belongs to only one triangle; keeping that triangle's
        direction lets the rim be wound consistently relative to the surface.
        """
        f = faces.astype(np.int64).reshape(-1, 3)
        if len(f) == 0:
            return np.empty((0, 2), dtype=np.int64)
        directed = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        keyed = np.sort(directed, axis=1)
        MAXV = int(np.max(f)) + 1
        h = keyed[:, 0] * MAXV + keyed[:, 1]
        uniq, counts = np.unique(h, return_counts=True)
        boundary_keys = set(uniq[counts == 1].tolist())
        mask = np.array([k in boundary_keys for k in h.tolist()], dtype=bool)
        return directed[mask]

    def _compute_even_thickness_scalars(self, verts, faces, normals):
        f = faces.astype(np.int32).reshape(-1, 3)
        v0, v1, v2 = verts[f[:,0]], verts[f[:,1]], verts[f[:,2]]
        fn = np.cross(v1-v0, v2-v0)
        fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_norms[fn_norms == 0] = 1.0
        fn /= fn_norms

        scalars = np.zeros(len(verts), dtype=np.float32)
        counts = np.zeros(len(verts), dtype=np.int32)

        for i in range(3):
            v_idx = f[:, i]
            v_normals = normals[v_idx]
            dot_prods = np.sum(v_normals * fn, axis=1)
            dot_prods = np.clip(dot_prods, 0.1, 1.0)
            s = 1.0 / dot_prods
            np.add.at(scalars, v_idx, s)
            np.add.at(counts, v_idx, 1)

        counts[counts == 0] = 1
        scalars /= counts

        return scalars
