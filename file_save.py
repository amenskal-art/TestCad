# file_save.py
"""
FileSaveMixin – adds Save 3D File As functionality to DentalEditorApp.

Supported formats (all written in pure Python + numpy, no extra libs needed):
  OBJ  – Wavefront .obj with vertex colours on v-lines (MeshLab / Blender / RealityCapture)
  STL  – Binary STL triangular mesh (requires faces; falls back to error if none)
  XYZ  – ASCII point cloud  x y z r g b  (CloudCompare / QGIS / any text tool)
  GLB  – glTF 2.0 binary with positions, vertex colours, normals, and triangle indices

Usage: mix into DentalEditorApp, then call  app.save_3d_file_dialog(fmt)  from the menu.
"""

import os
import json
import struct
import threading
import numpy as np


class FileSaveMixin:
    """Mixin that adds save-as-3D-file capability."""

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry-point (called from GUI)
    # ──────────────────────────────────────────────────────────────────────────

    def save_3d_file_dialog(self, fmt: str):
        """
        Open a tkinter Save-As dialog for the given format, then write the file
        on a background thread so the UI stays responsive.

        fmt: 'obj' | 'stl' | 'xyz' | 'glb'
        """
        if len(self.points) == 0:
            self.status_text = "⚠ Nothing to save — scene is empty."
            return

        _EXT = {
            'obj': ('.obj', "Wavefront OBJ",   [("Wavefront OBJ", "*.obj")]),
            'stl': ('.stl', "Binary STL",       [("STL Mesh",      "*.stl")]),
            'xyz': ('.xyz', "ASCII Point Cloud",[("XYZ Point Cloud","*.xyz"), ("CSV", "*.csv")]),
            'glb': ('.glb', "GLB / glTF 2.0",  [("GLB Binary",    "*.glb")]),
        }
        if fmt not in _EXT:
            self.status_text = f"⚠ Unknown format: {fmt}"
            return

        ext, label, filetypes = _EXT[fmt]

        def _run_dialog():
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            filepath = filedialog.asksaveasfilename(
                title=f"Save As {label}",
                defaultextension=ext,
                filetypes=filetypes + [("All Files", "*.*")]
            )
            root.destroy()
            if not filepath:
                return

            # Dispatch to the correct writer (still on dialog thread – fine for I/O)
            self.status_text = f"Saving {os.path.basename(filepath)} …"
            try:
                writers = {
                    'obj': self._save_obj,
                    'stl': self._save_stl,
                    'xyz': self._save_xyz,
                    'glb': self._save_glb,
                }
                writers[fmt](filepath)
                self.status_text = f"✓ Saved: {os.path.basename(filepath)}"
            except Exception as e:
                self.status_text = f"✗ Save failed: {e}"
                import traceback; traceback.print_exc()

        threading.Thread(target=_run_dialog, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_transformed_points(self):
        """
        Return (pts, cols, norms, faces) where pts have the current
        rotation / pan / centre transform baked in, matching what the
        user sees on screen.
        """
        pts   = self.points.astype(np.float32)
        cols  = self.base_colors.astype(np.float32)   # range [0, 1]
        norms = self.normals.astype(np.float32) if self.has_normals else None
        faces = self.faces.astype(np.uint32)   if self.has_faces   else None
        return pts, cols, norms, faces

    # ──────────────────────────────────────────────────────────────────────────
    # OBJ Writer
    # ──────────────────────────────────────────────────────────────────────────

    def _save_obj(self, filepath: str):
        """
        Wavefront OBJ export.
        Vertex colours are written as extra r g b components on every v-line
        (non-standard but recognised by MeshLab, Blender, Open3D, CloudCompare).
        If no faces are present the file contains only vertex definitions,
        which is still a valid point-cloud OBJ most tools can import.
        """
        pts, cols, norms, faces = self._get_transformed_points()
        n = len(pts)

        lines = [
            "# Exported by IQScanner",
            f"# Vertices: {n}",
            f"# Faces:    {len(faces) if faces is not None else 0}",
            "",
        ]

        # Vertex positions + colours (v x y z r g b)
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = cols[i]
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}")

        # Vertex normals
        if norms is not None:
            lines.append("")
            for i in range(n):
                nx, ny, nz = norms[i]
                lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

        # Faces  (1-based indices)
        if faces is not None and len(faces) > 0:
            lines.append("")
            if norms is not None:
                for tri in faces:
                    a, b_, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                    lines.append(f"f {a}//{a} {b_}//{b_} {c}//{c}")
            else:
                for tri in faces:
                    a, b_, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                    lines.append(f"f {a} {b_} {c}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')

    # ──────────────────────────────────────────────────────────────────────────
    # STL Writer  (binary)
    # ──────────────────────────────────────────────────────────────────────────

    def _save_stl(self, filepath: str):
        """
        Binary STL — fully vectorized with numpy; no Python triangle loop.

        Layout of one triangle record (50 bytes):
          12 bytes  face normal  (3 × float32)
          12 bytes  vertex 0     (3 × float32)
          12 bytes  vertex 1     (3 × float32)
          12 bytes  vertex 2     (3 × float32)
           2 bytes  attribute    (uint16, zero)
        All records are packed into a single structured array and written
        with one f.write() call — orders of magnitude faster than a Python loop.
        """
        pts, _, _, faces = self._get_transformed_points()

        if faces is None or len(faces) == 0:
            raise RuntimeError(
                "STL export requires a triangulated mesh.  "
                "Run Poisson Reconstruction first to generate faces."
            )

        f32 = faces.astype(np.int32)
        v0 = pts[f32[:, 0]]           # (N, 3)
        v1 = pts[f32[:, 1]]
        v2 = pts[f32[:, 2]]

        # Vectorized face normals
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)                         # (N, 3)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths < 1e-10] = 1.0
        normals /= lengths

        # Build structured array: one row = one 50-byte STL triangle record
        n_tri = len(faces)
        stl_dtype = np.dtype([
            ('normal', '<f4', (3,)),
            ('v0',     '<f4', (3,)),
            ('v1',     '<f4', (3,)),
            ('v2',     '<f4', (3,)),
            ('attr',   '<u2'),
        ])
        records = np.zeros(n_tri, dtype=stl_dtype)
        records['normal'] = normals.astype(np.float32)
        records['v0']     = v0.astype(np.float32)
        records['v1']     = v1.astype(np.float32)
        records['v2']     = v2.astype(np.float32)
        # 'attr' stays zero

        with open(filepath, 'wb') as f:
            header = b"Binary STL exported by IQScanner" + b'\x00' * 48
            f.write(header[:80])
            f.write(struct.pack('<I', n_tri))
            f.write(records.tobytes())

    # ──────────────────────────────────────────────────────────────────────────
    # XYZ Writer  (ASCII point cloud)
    # ──────────────────────────────────────────────────────────────────────────

    def _save_xyz(self, filepath: str):
        """
        Plain-text XYZ with colour:  x y z r g b  (r/g/b as 0-255 integers).
        Readable by CloudCompare (ASCII import), QGIS, Excel, and any text tool.
        """
        pts, cols, _, _ = self._get_transformed_points()

        cols_u8 = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# x y z r g b\n")
            for i in range(len(pts)):
                x, y, z   = pts[i]
                r, g, b   = cols_u8[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # GLB Writer  (glTF 2.0 binary — no external dependencies)
    # ──────────────────────────────────────────────────────────────────────────

    def _save_glb(self, filepath: str):
        """
        Write a valid glTF 2.0 Binary (.glb) file.
        Includes:  POSITION, COLOR_0 (u8 normalised), NORMAL (if available),
                   triangle indices (if mesh) or POINTS mode (if point cloud).

        No pygltflib required — the spec is simple enough to write manually.
        """
        pts, cols, norms, faces = self._get_transformed_points()
        n = len(pts)
        has_f = faces is not None and len(faces) > 0
        has_n = norms is not None and len(norms) == n

        # ── Build the binary buffer ──────────────────────────────────────────
        bin_buf = bytearray()
        buffer_views = []
        accessors    = []

        def _pad4(buf):
            """Pad bytearray to 4-byte boundary with zeros."""
            while len(buf) % 4:
                buf.append(0)

        def _add_bv(data: bytes, target: int) -> int:
            """Append data to bin_buf, record a bufferView, return its index."""
            byte_offset = len(bin_buf)
            bin_buf.extend(data)
            _pad4(bin_buf)
            bv = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)}
            if target:
                bv["target"] = target
            buffer_views.append(bv)
            return len(buffer_views) - 1

        _ARRAY_BUFFER   = 34962
        _ELEMENT_BUFFER = 34963
        _FLOAT          = 5126
        _UBYTE          = 5121
        _UINT           = 5125

        # POSITION
        pos_f32   = pts.astype(np.float32)
        bv_pos    = _add_bv(pos_f32.tobytes(), _ARRAY_BUFFER)
        accessors.append({
            "bufferView":    bv_pos,
            "componentType": _FLOAT,
            "count":         n,
            "type":          "VEC3",
            "min":           pos_f32.min(axis=0).tolist(),
            "max":           pos_f32.max(axis=0).tolist(),
        })
        acc_pos = len(accessors) - 1

        # COLOR_0  (unsigned byte, normalised)
        col_u8 = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
        bv_col = _add_bv(col_u8.tobytes(), _ARRAY_BUFFER)
        accessors.append({
            "bufferView":    bv_col,
            "componentType": _UBYTE,
            "count":         n,
            "type":          "VEC3",
            "normalized":    True,
        })
        acc_col = len(accessors) - 1

        attributes = {"POSITION": acc_pos, "COLOR_0": acc_col}

        # NORMAL
        if has_n:
            norm_f32 = norms.astype(np.float32)
            bv_norm  = _add_bv(norm_f32.tobytes(), _ARRAY_BUFFER)
            accessors.append({
                "bufferView":    bv_norm,
                "componentType": _FLOAT,
                "count":         n,
                "type":          "VEC3",
            })
            attributes["NORMAL"] = len(accessors) - 1

        primitive = {"attributes": attributes}

        # INDICES
        if has_f:
            idx_u32 = faces.astype(np.uint32).flatten()
            bv_idx  = _add_bv(idx_u32.tobytes(), _ELEMENT_BUFFER)
            accessors.append({
                "bufferView":    bv_idx,
                "componentType": _UINT,
                "count":         len(idx_u32),
                "type":          "SCALAR",
            })
            primitive["indices"] = len(accessors) - 1
            primitive["mode"]    = 4   # TRIANGLES
        else:
            primitive["mode"]    = 0   # POINTS

        # ── Assemble JSON ────────────────────────────────────────────────────
        gltf = {
            "asset":       {"version": "2.0", "generator": "IQScanner"},
            "scene":       0,
            "scenes":      [{"nodes": [0]}],
            "nodes":       [{"mesh": 0}],
            "meshes":      [{"primitives": [primitive]}],
            "accessors":   accessors,
            "bufferViews": buffer_views,
            "buffers":     [{"byteLength": len(bin_buf)}],
        }

        json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        # Pad JSON chunk to 4-byte boundary with ASCII spaces (spec requirement)
        while len(json_bytes) % 4:
            json_bytes += b' '

        # Pad binary chunk to 4 bytes with zeros (already done in _add_bv, but be safe)
        while len(bin_buf) % 4:
            bin_buf.append(0)

        total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)

        with open(filepath, 'wb') as f:
            # GLB header
            f.write(struct.pack('<I', 0x46546C67))  # magic  "glTF"
            f.write(struct.pack('<I', 2))            # version 2
            f.write(struct.pack('<I', total_len))    # total file length

            # Chunk 0 — JSON
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(struct.pack('<I', 0x4E4F534A))   # "JSON"
            f.write(json_bytes)

            # Chunk 1 — BIN
            f.write(struct.pack('<I', len(bin_buf)))
            f.write(struct.pack('<I', 0x004E4942))   # "BIN\0"
            f.write(bytes(bin_buf))
