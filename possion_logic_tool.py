from tool_system import BaseTool
import threading
import numpy as np
import imgui
from OpenGL.GL import *
import ctypes
import psutil
import traceback

# --- Dependencies ---
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[DEBUG] Scipy not found. Color blending disabled.")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
    print(f"[DEBUG] Open3D Version: {o3d.__version__}")
except ImportError:
    O3D_AVAILABLE = False
    print("Warning: Open3D not found.")

class PoissonReconTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Reconstruction"
        self.icon_key = "mesh"
        self.hotkey = None
        self.tool_id = 100

        # --- Mode Selection ---
        self.recon_method = 0       # 0 = Poisson, 1 = Ball Pivoting

        # --- Poisson Settings ---
        self.depth = 9
        self.scale = 1.1
        self.linear_fit = True
        self.n_threads = max(1, psutil.cpu_count(logical=True))
        self.density_threshold = 0.05  # Fix: Threshold to cut off Poisson's watertight bubbles

        # --- Ball Pivoting Settings ---
        self.bpa_radii_scale = 2.0

        # --- Common Settings ---
        self.force_normal_recalc = False
        self.blend_colors = True
        self.flat_shading = False
        self.opacity = 1.0
        self.show_wireframe = False

        # --- State ---
        self.is_processing = False
        self.progress_text = ""
        self.result_mesh = None
        self.preview_vbo = None
        self.preview_ibo = None
        self.preview_count = 0

        self.thread_lock = threading.Lock()
        self._pending_upload = None

    def on_activate(self):
        self.app.status_text = "Reconstruction Mode"

    def on_deactivate(self):
        pass

    def update(self):
        if self._pending_upload:
            self._upload_to_gpu(*self._pending_upload)
            self._pending_upload = None

    def draw_options(self):
        if not O3D_AVAILABLE:
            imgui.text_colored("Open3D Missing", 1.0, 0.0, 0.0, 1.0)
            return

        cpu_p = psutil.cpu_percent()
        imgui.text_colored(f"System: {cpu_p}% CPU", 0.0, 1.0, 0.0, 1.0)
        imgui.separator()

        imgui.text("Method")
        methods = ["Poisson (Smooth/Watertight)", "Ball Pivoting (Sharp/Exact)"]
        changed, self.recon_method = imgui.combo("##Method", self.recon_method, methods)

        imgui.separator()

        if self.recon_method == 0:
            imgui.text("Poisson Settings")
            _, self.depth = imgui.slider_int("Depth", self.depth, 6, 12)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Higher = More Detail, Slower")
            _, self.scale = imgui.slider_float("Scale", self.scale, 1.0, 1.5)

            # Fix: Added UI control for density trimming
            _, self.density_threshold = imgui.slider_float("Clean Bubble (Density)", self.density_threshold, 0.0, 0.5)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Removes the random folds/bubbles Poisson creates to close the mesh.")

        else:
            imgui.text("Ball Pivoting Settings")
            _, self.bpa_radii_scale = imgui.slider_float("Ball Size Multiplier", self.bpa_radii_scale, 0.5, 5.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("1.0 = Average point distance.\nIncrease if mesh has too many holes.\nDecrease for finer detail.")
            imgui.text_disabled("Note: BPA preserves original\npoint positions exactly.")

        imgui.dummy(0, 5)
        imgui.text("Data Prep")

        has_normals = hasattr(self.app, 'has_normals') and self.app.has_normals
        if has_normals:
            imgui.text_colored("Normals: Ready", 0.0, 1.0, 0.0, 1.0)
            _, self.force_normal_recalc = imgui.checkbox("Force Recalculate", self.force_normal_recalc)
        else:
            imgui.text_colored("Normals: Missing", 1.0, 0.8, 0.0, 1.0)
            self.force_normal_recalc = True

        imgui.dummy(0, 5)
        imgui.text("Visuals")
        if SCIPY_AVAILABLE:
            _, self.blend_colors = imgui.checkbox("Transfer Colors", self.blend_colors)

        _, self.flat_shading = imgui.checkbox("Flat Shading", self.flat_shading)

        imgui.dummy(0, 10)
        imgui.separator()

        if self.is_processing:
            imgui.text_colored(f"BUSY: {self.progress_text}", 0.0, 1.0, 1.0, 1.0)
        else:
            label = "RUN POISSON" if self.recon_method == 0 else "RUN BALL PIVOT"
            if imgui.button(label, width=-1):
                self.start_reconstruction()

        if self.result_mesh is not None and not self.is_processing:
            imgui.dummy(0, 10)
            imgui.text_colored(f"Mesh: {len(self.result_mesh['tris']):,} tris", 0.0, 1.0, 0.0, 1.0)
            imgui.separator()

            _, self.opacity = imgui.slider_float("Opacity", self.opacity, 0.1, 1.0)
            _, self.show_wireframe = imgui.checkbox("Wireframe", self.show_wireframe)

            if imgui.button("Flip Normals", width=-1):
                self.flip_normals()

            imgui.dummy(0, 5)
            if imgui.button("Bake to Scene", width=-1):
                self.bake_to_scene()
            if imgui.button("Export PLY", width=-1):
                self.export_mesh()

    def start_reconstruction(self):
        if self.app.points is None or len(self.app.points) == 0:
            return

        self.is_processing = True
        thread = threading.Thread(target=self._worker_process)
        thread.start()

    def _worker_process(self):
        try:
            print("[DEBUG] Worker Thread Started")
            self.progress_text = "Processing..."

            points = np.ascontiguousarray(self.app.points, dtype=np.float64)

            center = np.mean(points, axis=0)
            points = points - center
            print(f"[DEBUG] Data centered at {center}")

            raw_points = None
            raw_colors = None
            if self.blend_colors and SCIPY_AVAILABLE and self.app.base_colors is not None:
                raw_points = points
                raw_colors = self.app.base_colors[:, :3]

            normals = None
            if hasattr(self.app, 'has_normals') and self.app.has_normals and not self.force_normal_recalc:
                if len(self.app.normals) == len(points):
                    normals = np.ascontiguousarray(self.app.normals, dtype=np.float64)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
                del normals
            else:
                self.progress_text = "Normals..."
                print("[DEBUG] Estimating Normals")
                pcd.estimate_normals()

                print("[DEBUG] Orienting Normals")
                # Fix: Replaced forced +Z orientation with proper tangent plane smoothing.
                # This stops Ball Pivoting from failing or creating backwards/dim triangles.
                try:
                    pcd.orient_normals_consistent_tangent_plane(15)
                    print("[DEBUG] Orientation Complete (Consistent Tangent Plane)")
                except Exception as e:
                    print(f"[DEBUG] Tangent plane failed, falling back to Z-axis: {e}")
                    np_normals = np.asarray(pcd.normals)
                    view_vec = np.array([0.0, 0.0, 1.0])
                    dots = np_normals.dot(view_vec)
                    flip_indices = dots < 0
                    np_normals[flip_indices] *= -1.0
                    pcd.normals = o3d.utility.Vector3dVector(np_normals)

            self.progress_text = "Solving..."

            if self.recon_method == 0:
                print(f"[DEBUG] Running Poisson (Depth={self.depth})")
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd,
                        depth=self.depth,
                        width=0,
                        scale=self.scale,
                        linear_fit=self.linear_fit,
                        n_threads=self.n_threads
                    )

                    # Fix: Cut away the "folds" (low density points Poisson invented to close the mesh)
                    if self.density_threshold > 0.0:
                        try:
                            print(f"[DEBUG] Filtering Poisson mesh by density threshold: {self.density_threshold}")
                            densities_np = np.asarray(densities)
                            threshold = np.quantile(densities_np, self.density_threshold)
                            vertices_to_remove = densities_np < threshold
                            mesh.remove_vertices_by_mask(vertices_to_remove)
                        except Exception as e:
                            print(f"[DEBUG] Could not filter by density: {e}")

            else:
                print("[DEBUG] Running Ball Pivoting")
                self.progress_text = "Calculating Radii..."

                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                print(f"[DEBUG] Avg point distance: {avg_dist:.4f}")

                base_r = avg_dist * self.bpa_radii_scale
                radii = o3d.utility.DoubleVector([base_r, base_r * 2, base_r * 4])

                self.progress_text = "Pivoting Balls..."
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, radii
                )

            del pcd

            print("[DEBUG] Cleaning Mesh")
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            # Drop any vertices left unreferenced after density trimming / cleaning
            # so the welded mesh stays compact and its indices stay valid.
            mesh.remove_unreferenced_vertices()

            mesh_colors = None
            if self.blend_colors and SCIPY_AVAILABLE and raw_points is not None:
                print("[DEBUG] Blending Colors...")
                self.progress_text = "Blending..."
                try:
                    tree = cKDTree(raw_points)
                    mesh_verts = np.asarray(mesh.vertices)
                    _, indices = tree.query(mesh_verts, k=1, workers=-1)
                    rgb = raw_colors[indices]
                    alpha = np.ones((len(rgb), 1), dtype=np.float32)
                    mesh_colors = np.hstack([rgb, alpha]).astype(np.float32)
                except Exception as e:
                    print(f"[DEBUG] Blend Error: {e}")

            print("[DEBUG] Finalizing")
            self.progress_text = "Finalizing..."

            verts = np.array(mesh.vertices, dtype=np.float32)
            tris = np.array(mesh.triangles, dtype=np.uint32)

            print("[DEBUG] Calculating Normals via NumPy")
            v_norms = self._compute_normals_numpy(verts, tris)

            verts += center.astype(np.float32)

            if mesh_colors is None:
                mesh_colors = np.tile(np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float32), (len(verts), 1))

            with self.thread_lock:
                self.result_mesh = {
                    'verts': verts, 'tris': tris, 'norms': v_norms, 'colors': mesh_colors
                }
                self._pending_upload = (verts, tris, v_norms, mesh_colors)

            print("[DEBUG] Success")
            self.app.status_text = "Reconstruction Complete."

        except Exception as e:
            print(f"[DEBUG] FAILURE: {e}")
            traceback.print_exc()
            self.app.status_text = "Error (Check Console)"
        finally:
            self.is_processing = False

    def _compute_normals_numpy(self, verts, tris):
        try:
            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)

            vertex_normals = np.zeros_like(verts)
            np.add.at(vertex_normals, tris[:, 0], face_normals)
            np.add.at(vertex_normals, tris[:, 1], face_normals)
            np.add.at(vertex_normals, tris[:, 2], face_normals)

            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vertex_normals /= norms
            return vertex_normals.astype(np.float32)
        except:
            return np.tile(np.array([0, 0, 1], dtype=np.float32), (len(verts), 1))

    def flip_normals(self):
        if self.result_mesh is None: return
        self.result_mesh['norms'] = -self.result_mesh['norms']
        tris = self.result_mesh['tris']
        tris[:, [1, 2]] = tris[:, [2, 1]]
        self._upload_to_gpu(
            self.result_mesh['verts'], self.result_mesh['tris'],
            self.result_mesh['norms'], self.result_mesh['colors']
        )

    def _upload_to_gpu(self, verts, tris, norms, colors):
        data = np.empty((len(verts), 10), dtype=np.float32)
        data[:, 0:3] = verts
        data[:, 3:6] = norms
        data[:, 6:10] = colors

        if self.preview_vbo is None:
            self.preview_vbo = glGenBuffers(1)
            self.preview_ibo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.preview_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.preview_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tris.nbytes, tris, GL_STATIC_DRAW)

        self.preview_count = len(tris) * 3
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def render_3d(self):
        if self.preview_vbo is None or self.preview_count == 0: return

        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_POLYGON_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.flat_shading:
            glDisable(GL_LIGHTING)
        else:
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)
            glDisable(GL_CULL_FACE)
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.4, 0.4, 0.4, self.opacity])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10.0)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glBindBuffer(GL_ARRAY_BUFFER, self.preview_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.preview_ibo)

        stride = 40
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))

        if not self.flat_shading:
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, stride, ctypes.c_void_p(24))

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.show_wireframe else GL_FILL)
        glDrawElements(GL_TRIANGLES, self.preview_count, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glPopAttrib()

    def export_mesh(self):
        from utils import save_file_dialog
        if self.result_mesh is None: return
        filepath = save_file_dialog("Save Mesh", "ply")
        if filepath:
            try:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(self.result_mesh['verts'].astype(np.float64))
                # Fix: triangle indices must use Vector3iVector, not Vector3dVector.
                mesh.triangles = o3d.utility.Vector3iVector(self.result_mesh['tris'].astype(np.int32))
                mesh.vertex_normals = o3d.utility.Vector3dVector(self.result_mesh['norms'].astype(np.float64))
                if 'colors' in self.result_mesh:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(self.result_mesh['colors'][:, :3].astype(np.float64))
                o3d.io.write_triangle_mesh(filepath, mesh)
                self.app.status_text = f"Saved: {filepath}"
            except Exception as e:
                self.app.status_text = f"Export Error: {e}"

    def bake_to_scene(self):
        """
        Bake the reconstructed mesh into the main application.

        IMPORTANT: We bake a *welded indexed mesh* (shared vertices + a real
        triangle index buffer / EBO), NOT a flattened "triangle soup".

        The rest of the app (smoothing, selection-delete, volume measurement,
        surface paths, Laplacian, KD-tree) all assume a welded indexed mesh —
        exactly what you get when you export and reload a file. Baking a soup
        with ebo=None caused render_mesh_vbo() to fall back to glDrawArrays and
        gave the tools disconnected topology, which is what produced the
        "smooth deletes vertices", "filaments on delete", and "volume lag" bugs.
        """
        if self.result_mesh is None:
            return

        self.app.save_state("Bake Reconstruction")

        # --- Welded indexed mesh (do NOT flatten into a soup) ---
        verts = np.ascontiguousarray(self.result_mesh['verts'], dtype=np.float32)            # (N, 3)
        tris  = np.ascontiguousarray(self.result_mesh['tris'],  dtype=np.uint32).reshape(-1, 3)  # (T, 3)
        norms = np.ascontiguousarray(self.result_mesh['norms'], dtype=np.float32)            # (N, 3)
        cols  = np.ascontiguousarray(self.result_mesh['colors'][:, :3], dtype=np.float32)    # (N, 3) drop alpha

        print(f"[DEBUG] Baking welded mesh: {len(verts):,} verts, {len(tris):,} tris")

        # Replace main app data with the welded indexed mesh
        self.app.points      = verts
        self.app.normals     = norms
        self.app.base_colors = cols
        self.app.faces       = tris
        self.app.has_faces   = True
        self.app.has_normals = True
        self.app.render_mode_mesh = True   # switch to indexed mesh rendering

        # --- CRITICAL: ensure a valid EBO exists so render_mesh_vbo() and
        #     update_vbo_full() take the indexed glDrawElements path. ---
        if not hasattr(self.app, 'ebo') or self.app.ebo is None:
            self.app.ebo = glGenBuffers(1)

        # Update selection, object ids, and pick-ids for the new vertex count
        count = len(self.app.points)
        self.app.selected = np.zeros(count, dtype=bool)
        self.app.point_object_ids = np.zeros(count, dtype=np.int32)
        self.app.rebuild_ids(count)
        self.app.points_colors = np.hstack([self.app.points, self.app.base_colors]).astype(np.float32)  # (N, 6)

        # Upload to GPU (uploads VBO + EBO because has_faces and ebo is not None)
        self.app.update_vbo_full()

        # Discard stale point-cloud LOD indices and rebuild spatial structures
        if hasattr(self.app, '_invalidate_lod_cache'):
            self.app._invalidate_lod_cache()
        if hasattr(self.app, '_rebuild_kdtree'):
            self.app._rebuild_kdtree()
        if hasattr(self.app, 'sync_sculptor'):
            self.app.sync_sculptor()

        self.app.vbo_dirty = True   # ensure redraw on next frame

        # Clean up tool's preview
        self.result_mesh = None
        self.preview_count = 0
        self.app.status_text = "Mesh Baked (welded indexed mesh)."
        print("[DEBUG] Bake complete. render_mode_mesh:", self.app.render_mode_mesh,
              "has_normals:", self.app.has_normals, "has_faces:", self.app.has_faces)
