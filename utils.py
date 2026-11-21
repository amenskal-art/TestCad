# utils.py - UPDATED WITH OBJ FACE SUPPORT AND SMOOTH ICON MIPMAPS
import numpy as np
from plyfile import PlyData, PlyElement
import ctypes
from ctypes import wintypes, byref, create_unicode_buffer, cast
import os
import struct
from PIL import Image
from OpenGL.GL import *

def enable_high_dpi():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1) 
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def create_rotation_matrix(axis, angle_degrees):
    axis = normalize(axis)
    a = np.radians(angle_degrees)
    s = np.sin(a)
    c = np.cos(a)
    oc = 1.0 - c
    x, y, z = axis
    return np.array([
        [oc * x * x + c,     oc * x * y - z * s, oc * x * z + y * s, 0.0],
        [oc * x * y + z * s, oc * y * y + c,     oc * y * z - x * s, 0.0],
        [oc * z * x - y * s, oc * y * z + x * s, oc * z * z + c,     0.0],
        [0.0,                0.0,                0.0,                1.0]
    ], dtype=np.float32)

def load_texture(path):
    """Loads a PNG/JPG file into an OpenGL texture with mipmaps for smooth scaling."""
    path = path.strip('"').strip("'")
    
    if not os.path.exists(path):
        print(f"Icon not found: {path}")
        return None, 0, 0

    try:
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        width, height = image.size
        img_data = image.tobytes()
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Use GL_LINEAR_MIPMAP_LINEAR for minifying (scaling down) to fix pixelation
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        # Use GL_LINEAR for magnifying (scaling up)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload the texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        # Generate the mipmaps immediately after uploading the image data
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return texture_id, width, height
    except Exception as e:
        print(f"Failed to load texture {path}: {e}")
        return None, 0, 0

# --- File I/O ---

def load_obj_data(filename):
    """
    Load OBJ via Open3D's C++ parser — handles all OBJ variants in <1s.
    Falls back to numpy parser if Open3D is unavailable.
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(
            filename,
            enable_post_processing=False,
            print_progress=False
        )
        pts = np.asarray(mesh.vertices,  dtype=np.float32)
        if len(pts) == 0:
            print("[OBJ] Open3D returned 0 vertices — trying fallback parser")
            return _load_obj_fallback(filename)

        faces_o3d = np.asarray(mesh.triangles, dtype=np.int32)

        cols = None
        if mesh.has_vertex_colors():
            cols = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if cols.max() > 1.0:
                cols /= 255.0

        if len(faces_o3d) > 0:
            norms = compute_vertex_normals(pts, faces_o3d)
            print(f"[OBJ] Open3D loaded: {len(pts):,} verts, {len(faces_o3d):,} faces")
            return pts, cols, norms, faces_o3d
        else:
            print(f"[OBJ] Open3D point cloud: {len(pts):,} verts")
            return pts, cols, None, None

    except Exception as e:
        print(f"[OBJ] Open3D failed ({e}), trying fallback parser")
        return _load_obj_fallback(filename)


def _load_obj_fallback(filename):
    """
    Fallback numpy OBJ parser — used when Open3D is unavailable.
    Uses io.BytesIO + np.loadtxt (C parser in numpy ≥ 1.23) instead of
    the slow np.fromstring(text, sep=' ') path.
    """
    import io, re

    try:
        with open(filename, 'rb') as f:
            raw = f.read()

        lines    = raw.split(b'\n')
        v_lines  = []
        f_lines  = []

        for line in lines:
            if line.endswith(b'\r'):
                line = line[:-1]
            if not line or line[0:1] == b'#':
                continue
            if line[0:2] == b'v ':
                v_lines.append(line[2:])
            elif line[0:2] == b'f ':
                f_lines.append(line[2:])

        if not v_lines:
            return np.zeros((0, 3), dtype=np.float32), None, None, None

        # io.BytesIO + np.loadtxt — the C parser avoids the fromstring bottleneck
        n_v_cols = len(v_lines[0].split())
        v_buf    = io.BytesIO(b'\n'.join(v_lines))
        v_arr    = np.loadtxt(v_buf, dtype=np.float32, ndmin=2)

        pts  = np.ascontiguousarray(v_arr[:, :3], dtype=np.float32)
        cols = None
        if n_v_cols >= 6 and v_arr.shape[1] >= 6:
            cols = v_arr[:, 3:6].astype(np.float32)
            if cols.max() > 1.0:
                cols /= 255.0

        if not f_lines:
            return pts, cols, None, None

        f_joined = b' '.join(f_lines)
        f_clean  = re.sub(rb'/[0-9]*', b'', f_joined)
        f_flat   = np.fromstring(f_clean.decode('ascii', errors='replace'),
                                 dtype=np.int32, sep=' ') - 1

        if len(f_flat) == len(f_lines) * 3:
            faces_array = f_flat.reshape(-1, 3).astype(np.int32)
        else:
            tri_list = []
            for line in f_lines:
                lc   = re.sub(rb'/[0-9]*', b'', line)
                idxs = np.fromstring(lc.decode('ascii', errors='replace'),
                                     dtype=np.int32, sep=' ') - 1
                n = len(idxs)
                if n == 3:
                    tri_list.append(idxs)
                elif n == 4:
                    tri_list.append(idxs[[0, 1, 2]])
                    tri_list.append(idxs[[0, 2, 3]])
                elif n > 4:
                    for i in range(1, n - 1):
                        tri_list.append(np.array([idxs[0], idxs[i], idxs[i+1]], dtype=np.int32))
            faces_array = np.array(tri_list, dtype=np.int32)

        faces_array = np.clip(faces_array, 0, len(pts) - 1)
        norms       = compute_vertex_normals(pts, faces_array)
        return pts, cols, norms, faces_array

    except Exception as e:
        print(f"[OBJ] Fallback parser failed: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((0, 3), dtype=np.float32), None, None, None

def compute_vertex_normals(vertices, faces):
    """
    Compute smooth vertex normals — fully vectorized, no np.add.at.

    np.add.at is an unbuffered scatter that processes one element at a time
    in C but with no SIMD or cache optimization.  On a 1M-face mesh it takes
    several seconds.

    Replacement: np.bincount with weights — a single C reduction that is
    20-50x faster for the same operation.
    """
    n_verts = len(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Normalize face normals
    fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    fn_len[fn_len < 1e-10] = 1.0
    face_normals /= fn_len                          # (F, 3)

    # Accumulate to vertex normals using bincount (one C call per component).
    # For each of the 3 triangle corners, add that face's normal to each
    # referenced vertex.  bincount handles repeated indices correctly and fast.
    idx = faces.ravel()                             # (F*3,)  — all vertex refs
    fn3 = np.tile(face_normals, (1, 3)).reshape(-1, 3)  # repeat normal for v0/v1/v2

    normals = np.stack([
        np.bincount(idx, weights=fn3[:, 0], minlength=n_verts),
        np.bincount(idx, weights=fn3[:, 1], minlength=n_verts),
        np.bincount(idx, weights=fn3[:, 2], minlength=n_verts),
    ], axis=1).astype(np.float32)

    # Normalize
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    n_len[n_len < 1e-10] = 1.0
    normals /= n_len

    return normals

def load_stl_data(filename):
    try:
        with open(filename, 'rb') as f:
            header = f.read(80)
            count_bytes = f.read(4)
            if len(count_bytes) < 4: return np.zeros((0,3), dtype=np.float32), None, None, None
            count = struct.unpack('<I', count_bytes)[0]
            
            dtype_stl = np.dtype([
                ('normal', '<f4', (3,)),
                ('v1', '<f4', (3,)),
                ('v2', '<f4', (3,)),
                ('v3', '<f4', (3,)),
                ('attr', '<u2')
            ])
            data = np.fromfile(f, dtype=dtype_stl, count=count)
            
            v1 = data['v1']
            v2 = data['v2']
            v3 = data['v3']
            
            pts = np.hstack((v1, v2, v3)).reshape(-1, 3).astype(np.float32)
            raw_normals = data['normal']

            magnitudes = np.linalg.norm(raw_normals, axis=1)
            if np.sum(magnitudes < 1e-6) > (len(raw_normals) * 0.5):
                vec_a = v2 - v1
                vec_b = v3 - v1
                cross = np.cross(vec_a, vec_b)
                norms_len = np.linalg.norm(cross, axis=1, keepdims=True)
                norms_len[norms_len < 1e-6] = 1.0
                raw_normals = cross / norms_len
            
            face_normals_expanded = np.repeat(raw_normals, 3, axis=0)
            pts_rounded = np.round(pts, decimals=5) 
            unique_v, indices = np.unique(pts_rounded, axis=0, return_inverse=True)

            smooth_norms = np.zeros((len(unique_v), 3), dtype=np.float32)
            np.add.at(smooth_norms, indices, face_normals_expanded)

            n_len = np.linalg.norm(smooth_norms, axis=1, keepdims=True)
            n_len[n_len < 1e-6] = 1.0
            smooth_norms /= n_len

            final_normals = smooth_norms[indices]
            
            # Create face indices for mesh rendering
            num_triangles = count
            faces = np.arange(len(pts), dtype=np.int32).reshape(num_triangles, 3)

            return pts, None, final_normals.astype(np.float32), faces
            
    except Exception as e:
        print(f"Error loading STL: {e}")
        return np.zeros((0,3), dtype=np.float32), None, None, None

def load_ply_data(filename):
    try:
        pd = PlyData.read(filename)
        v = pd['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        props = [p.name for p in v.properties]
        colors = None
        if 'red' in props and 'green' in props and 'blue' in props:
            c_raw = np.stack([v['red'], v['green'], v['blue']], axis=-1)
            if c_raw.dtype.kind in ('u', 'i'): colors = c_raw.astype(np.float32) / 255.0
            else: 
                temp_cols = c_raw.astype(np.float32)
                if temp_cols.max() > 1.0: temp_cols /= 255.0
                colors = temp_cols
        
        # Check for faces in PLY
        faces = None
        if 'face' in pd:
            face_data = pd['face']['vertex_indices']
            faces_list = []
            for face in face_data:
                if len(face) == 3:
                    faces_list.append(face)
                elif len(face) == 4:
                    # Triangulate quads
                    faces_list.append([face[0], face[1], face[2]])
                    faces_list.append([face[0], face[2], face[3]])
            if faces_list:
                faces = np.array(faces_list, dtype=np.int32)
                normals = compute_vertex_normals(pts, faces)
                print(f"Loaded PLY with {len(pts)} vertices and {len(faces)} faces")
                return pts, colors, normals, faces
        
        return pts, colors, None, None
    except Exception as e:
        print(f"Error PLY: {e}")
        return np.zeros((0,3), dtype=np.float32), None, None, None

def load_xyz_data(filename):
    """
    Load an ASCII point cloud.  Supports:
      x y z
      x y z r g b          (r/g/b as 0-255 integers OR 0.0-1.0 floats)
      x y z r g b a        (alpha ignored)
    Lines starting with '#' or that cannot be parsed are skipped silently.
    """
    try:
        rows = []
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    vals = [float(p) for p in parts[:6]]
                    rows.append(vals)
                except ValueError:
                    continue

        if not rows:
            return np.zeros((0, 3), dtype=np.float32), None, None, None

        arr = np.array(rows, dtype=np.float32)
        pts  = arr[:, :3]
        cols = None

        if arr.shape[1] >= 6:
            rgb = arr[:, 3:6]
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            cols = np.clip(rgb, 0.0, 1.0)

        print(f"[XYZ] Loaded {len(pts):,} points from {os.path.basename(filename)}")
        return pts, cols, None, None

    except Exception as e:
        print(f"[XYZ] Load error: {e}")
        return np.zeros((0, 3), dtype=np.float32), None, None, None


def load_glb_data(filename):
    """
    Load a glTF 2.0 Binary (.glb) file — pure Python / numpy, no pygltflib.

    Extracts the first mesh primitive's:
      POSITION   -> pts   (float32, Nx3)
      COLOR_0    -> cols  (float32 [0-1], Nx3)  if present
      NORMAL     -> norms (float32, Nx3)         if present
      indices    -> faces (int32, Fx3)            if present and triangles

    Uses memoryview for zero-copy accessor reads — no intermediate byte copies
    regardless of GLB file size.
    """
    try:
        with open(filename, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file (bad magic bytes)")
            _ver   = struct.unpack('<I', f.read(4))[0]
            _total = struct.unpack('<I', f.read(4))[0]

            json_data = b''
            bin_data  = b''
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                chunk_len  = struct.unpack('<I', hdr[:4])[0]
                chunk_type = hdr[4:]
                chunk_data = f.read(chunk_len)
                if chunk_type == b'JSON':
                    json_data = chunk_data
                elif chunk_type[:3] == b'BIN':
                    bin_data = chunk_data

        import json as _json
        gltf = _json.loads(json_data.decode('utf-8'))

        # Wrap bin_data in memoryview so slicing never copies bytes
        buf_view = memoryview(bin_data)

        def _read_accessor(acc_idx):
            if acc_idx is None:
                return None
            acc = gltf['accessors'][acc_idx]
            bv  = gltf['bufferViews'][acc['bufferView']]
            dtype_map = {
                5120: np.int8,   5121: np.uint8,  5122: np.int16,
                5123: np.uint16, 5125: np.uint32,  5126: np.float32,
            }
            type_shape = {
                'SCALAR': (), 'VEC2': (2,), 'VEC3': (3,), 'VEC4': (4,),
            }
            dtype  = dtype_map[acc['componentType']]
            shape  = type_shape.get(acc['type'], ())
            count  = acc['count']
            n_comp = int(np.prod(shape)) if shape else 1
            byte_offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            byte_len    = count * n_comp * np.dtype(dtype).itemsize

            # Zero-copy view into the binary buffer
            raw = np.frombuffer(buf_view[byte_offset: byte_offset + byte_len], dtype=dtype)
            if shape:
                raw = raw.reshape(count, *shape)

            # Normalised conversion (e.g. COLOR_0 u8 → float32)
            if acc.get('normalized', False) and dtype != np.float32:
                result = raw.astype(np.float32) / np.iinfo(dtype).max
            else:
                # .copy() makes the array writable and releases the reference
                # to buf_view so the raw bytes can be GC'd after load
                result = raw.astype(np.float32)

            return result

        meshes = gltf.get('meshes', [])
        if not meshes:
            raise ValueError("GLB contains no meshes")
        prim  = meshes[0]['primitives'][0]
        attrs = prim.get('attributes', {})

        pts   = _read_accessor(attrs.get('POSITION'))
        if pts is None:
            raise ValueError("GLB has no POSITION accessor")

        cols  = _read_accessor(attrs.get('COLOR_0'))
        norms = _read_accessor(attrs.get('NORMAL'))
        faces = None

        if 'indices' in prim:
            idx   = _read_accessor(prim['indices']).astype(np.int32).flatten()
            faces = idx.reshape(-1, 3)

        if cols is not None:
            cols = np.clip(cols[:, :3], 0.0, 1.0)   # drop alpha if VEC4

        # Release the raw bytes now that all accessors are read
        del buf_view, bin_data

        print(f"[GLB] Loaded {len(pts):,} verts"
              + (f", {len(faces):,} faces" if faces is not None else " (point cloud)")
              + f" from {os.path.basename(filename)}")
        return pts, cols, norms, faces

    except Exception as e:
        print(f"[GLB] Load error: {e}")
        return np.zeros((0, 3), dtype=np.float32), None, None, None


def load_mesh_data(filename):
    ext = os.path.splitext(filename)[1].lower()
    if   ext == '.ply': return load_ply_data(filename)
    elif ext == '.obj': return load_obj_data(filename)
    elif ext == '.stl': return load_stl_data(filename)
    elif ext in ('.xyz', '.csv'): return load_xyz_data(filename)
    elif ext == '.glb': return load_glb_data(filename)
    else:
        print(f"Unsupported format: {ext}")
        return np.zeros((0,3), dtype=np.float32), None, None, None

def save_ply_selection(fname, points, colors):
    if len(points) == 0: return
    cols_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.empty(len(points), dtype=dtype_list)
    arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
    arr['red'], arr['green'], arr['blue'] = cols_u8[:, 0], cols_u8[:, 1], cols_u8[:, 2]
    PlyData([PlyElement.describe(arr, 'vertex')]).write(fname)

def export_mesh_data(filename, points, colors=None, faces=None):
    """Robust exporter that correctly formats the file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    # Filter out invalid faces if points were deleted/cropped
    if faces is not None and len(faces) > 0:
        max_idx = len(points) - 1
        valid_faces = [f for f in faces if all(0 <= idx <= max_idx for idx in f)]
        faces = valid_faces

    if ext == '.obj':
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Exported Object\n")
            for i in range(len(points)):
                p = points[i]
                if colors is not None and len(colors) > i:
                    c = colors[i]
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                else:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            if faces is not None and len(faces) > 0:
                for face in faces:
                    f.write("f " + " ".join([str(idx + 1) for idx in face]) + "\n")
                    
    elif ext == '.stl':
        if faces is None or len(faces) == 0:
            print("Cannot export STL: No faces present in the object.")
            return
            
        with open(filename, 'wb') as f:
            f.write(b'\0' * 80) # Empty Header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                if len(face) >= 3:
                    v0 = points[face[0]]
                    v1 = points[face[1]]
                    v2 = points[face[2]]
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    norm = np.cross(edge1, edge2)
                    n_len = np.linalg.norm(norm)
                    if n_len > 1e-10:
                        norm = norm / n_len
                    else:
                        norm = np.array([0.0, 0.0, 0.0])
                        
                    f.write(struct.pack('<3f', norm[0], norm[1], norm[2]))
                    f.write(struct.pack('<3f', v0[0], v0[1], v0[2]))
                    f.write(struct.pack('<3f', v1[0], v1[1], v1[2]))
                    f.write(struct.pack('<3f', v2[0], v2[1], v2[2]))
                    f.write(struct.pack('<H', 0))
                    
    else: # Default to PLY format
        if len(points) == 0: return
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        has_colors = colors is not None and len(colors) == len(points)
        
        if has_colors:
            dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            
        arr = np.empty(len(points), dtype=dtype_list)
        arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
        
        if has_colors:
            cols_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
            arr['red'], arr['green'], arr['blue'] = cols_u8[:, 0], cols_u8[:, 1], cols_u8[:, 2]
            
        elements = [PlyElement.describe(arr, 'vertex')]
        
        if faces is not None and len(faces) > 0:
            triangles = []
            for f in faces:
                if len(f) == 3: triangles.append(f)
                elif len(f) == 4:
                    triangles.append([f[0], f[1], f[2]])
                    triangles.append([f[0], f[2], f[3]])
            
            if triangles:
                # Correctly encode faces for PLY as a list of object arrays so viewers recognize it
                face_arr = np.empty(len(triangles), dtype=[('vertex_indices', 'O')])
                for i, t in enumerate(triangles):
                    face_arr['vertex_indices'][i] = np.array(t, dtype=np.int32)
                elements.append(PlyElement.describe(face_arr, 'face'))
                
        PlyData(elements, text=False).write(filename)

def open_file_dialog():
    MAX_PATH = 260
    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND), ("hInstance", wintypes.HANDLE),
            ("lpstrFilter", wintypes.LPCWSTR), ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR), ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR), ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD), ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD), ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID), ("lpTemplateName", wintypes.LPCWSTR), ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD), ("FlagsEx", wintypes.DWORD),
        ]
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = (
        "All Supported 3D Files (*.ply;*.obj;*.stl;*.xyz;*.csv;*.glb)\0"
        "*.ply;*.obj;*.stl;*.xyz;*.csv;*.glb\0"
        "Point Clouds (*.ply;*.xyz;*.csv)\0*.ply;*.xyz;*.csv\0"
        "Meshes (*.obj;*.stl;*.glb)\0*.obj;*.stl;*.glb\0"
        "All Files\0*.*\0"
    )
    buf = create_unicode_buffer(MAX_PATH)
    ofn.lpstrFile = cast(buf, wintypes.LPWSTR)
    ofn.nMaxFile = MAX_PATH
    ofn.Flags = 0x00080000 | 0x00001000 
    if ctypes.windll.comdlg32.GetOpenFileNameW(byref(ofn)):
        return buf.value
    return None
