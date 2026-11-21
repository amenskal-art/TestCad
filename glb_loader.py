# tools/glb_loader.py
# GLB -> numpy loader for the iQScanner viewer.
#
# v2: now also returns per-vertex NORMALS. The app's triangle pipeline is
# gated on has_normals (render_mode_mesh, lighting, vbo_normals binding),
# so a GLB loaded without normals always falls back to point-cloud mode —
# which is exactly the "it's made of points" symptom.
#
# Requires:  pip install trimesh

import numpy as np
import trimesh
import os


def _extract(geoms):
    """Walk geometry list -> (verts, colors, normals, faces, any_faces)."""
    v_list, c_list, n_list, f_list = [], [], [], []
    offset = 0
    any_faces = False

    for g in geoms:
        v = np.asarray(g.vertices, dtype=np.float32)
        if v.size == 0:
            continue

        c = None
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            any_faces = True
            f_list.append(np.asarray(g.faces, dtype=np.int64) + offset)
            # trimesh computes area-weighted vertex normals on demand if the
            # file doesn't carry a NORMAL attribute (the cloud meshes don't).
            n = np.asarray(g.vertex_normals, dtype=np.float32)
            visual = g.visual.to_color() if hasattr(g.visual, "to_color") else g.visual
            vc = getattr(visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(v):
                c = np.asarray(vc)
        else:  # trimesh.PointCloud or faceless mesh
            n = np.zeros((len(v), 3), dtype=np.float32)
            pc_colors = getattr(g, "colors", None)
            if pc_colors is not None and len(pc_colors) == len(v):
                c = np.asarray(pc_colors)

        if c is None:
            c = np.full((len(v), 4), 200, dtype=np.uint8)

        v_list.append(v)
        c_list.append(c[:, :3].astype(np.float32) / 255.0)
        n_list.append(n)
        offset += len(v)

    return v_list, c_list, n_list, f_list, any_faces


def _load_geoms(path):
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump()  # geometry copies with scene transforms baked in
    return [loaded]


def load_glb_mesh_data(path):
    """Drop-in for the app's loader contract: returns (pts, cols, norms, faces).

      pts   : (N, 3) float32
      cols  : (N, 3) float32 in 0..1
      norms : (N, 3) float32 unit vectors, or None if the GLB has no triangles
      faces : (M, 3) int64 vertex indices, or None for point-primitive GLBs

    When faces is None the cloud delivered the point-cloud FALLBACK, meaning
    the native mesh export failed server-side — check the Modal dashboard
    logs for the run.
    """
    
    # --- BULLETPROOF PLY DETECTION ---
    # Even if app_ops.py saves the file as ".glb", check the actual file header bytes.
    # PLY files always start with the text "ply".
    is_ply = False
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                header = f.read(3)
                if header == b'ply':
                    is_ply = True
        except Exception:
            pass

    # --- PLY BYPASS FOR PI3X NATIVE POINT CLOUDS ---
    if is_ply or path.lower().endswith('.ply'):
        print(f"[PLY] Loading native point cloud (Header detected PLY data): {path}")
        
        # We MUST pass file_type='ply' here to force trimesh to ignore the wrong .glb extension
        cloud = trimesh.load(path, file_type='ply')
        
        pts = np.asarray(cloud.vertices, dtype=np.float32)
        
        if hasattr(cloud, 'colors') and cloud.colors is not None:
            cols = np.asarray(cloud.colors)
            if cols.shape[1] == 4:
                cols = cols[:, :3]
            # Normalize colors to 0..1 if they are 0..255
            cols = cols.astype(np.float32) / 255.0 if cols.max() > 1.0 else cols.astype(np.float32)
        else:
            cols = np.ones_like(pts, dtype=np.float32) * 0.5  # Fallback gray
            
        faces = None  # PLY from Pi3X has no faces
        
        if hasattr(cloud, 'vertex_normals') and len(cloud.vertex_normals) > 0:
            norms = np.asarray(cloud.vertex_normals, dtype=np.float32)
        else:
            norms = None
            
        print(f"[PLY] {len(pts):,} verts, NO faces — native point cloud mode")
        return pts, cols, norms, faces

    # --- STANDARD GLB PIPELINE FOR HUNYUAN ---
    v_list, c_list, n_list, f_list, any_faces = _extract(_load_geoms(path))

    if not v_list:
        empty = np.zeros((0, 3), np.float32)
        return empty, empty.copy(), None, None

    pts = np.concatenate(v_list)
    cols = np.concatenate(c_list)

    if any_faces:
        norms = np.concatenate(n_list)
        faces = np.concatenate(f_list)
        print(f"[GLB] {len(pts):,} verts, {len(faces):,} faces — mesh mode")
    else:
        norms = None
        faces = None
        print(f"[GLB] {len(pts):,} verts, NO faces — point-primitive GLB. "
              f"The cloud used the PLY fallback; check the Modal dashboard "
              f"to see why the native mesh export failed.")

    return pts, cols, norms, faces


def load_glb(path):
    """Original v1 API kept for compatibility: (vertices, colors, faces)."""
    pts, cols, _norms, faces = load_glb_mesh_data(path)
    return pts, cols, faces


def load_glb_as_open3d(path):
    """Same data as an Open3D geometry (TriangleMesh if faces exist,
    PointCloud otherwise) for code paths that want o3d objects."""
    import open3d as o3d

    v, c, n, f = load_glb_mesh_data(path)
    if f is not None and len(f):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v.astype(np.float64)),
            o3d.utility.Vector3iVector(f.astype(np.int32)),
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(c.astype(np.float64))
        mesh.vertex_normals = o3d.utility.Vector3dVector(n.astype(np.float64))
        return mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd
