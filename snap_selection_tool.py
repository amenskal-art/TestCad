from tool_system import BaseTool
import imgui
import glfw
import numpy as np
from OpenGL.GL import *
from scipy.spatial import KDTree
import queue

class SnapSelectionTool(BaseTool):
    def __init__(self, app):
        super().__init__(app)
        self.name = "Prep Selection"
        self.icon_key = 'auto_fill'
        
        # --- Data Context ---
        self.mesh_vertices = None
        self.mesh_indices = None
        self.kd_tree = None
        
        # --- Input Data (From Margin Tool) ---
        self.margin_loop = []  # The float32 points defining the curve
        
        # --- Selection State ---
        self.selected_mask = None   # Boolean array of selected vertices
        self.vertex_neighbors = None # Adjacency list
        
        # --- Settings ---
        self.fence_width = 0.4 # Thickness of the boundary wall (mm) - critical for preventing leaks
        
        # --- Visualization ---
        self.select_color = [0.2, 1.0, 0.2, 0.5]  # Green Prep Overlay
        self.margin_color = [1.0, 0.8, 0.0, 1.0]  # Gold Margin Line
        self.fence_debug_color = [1.0, 0.0, 0.0, 0.3] # Red (for debugging boundary)

    def on_enter(self, context):
        """
        Called when entering. Expects 'margin_points' in context.
        """
        print("--- Entering Auto-Prep Selection ---")
        
        # 1. Load Geometry
        self.mesh_vertices = context.get('mesh_vertices')
        if self.mesh_vertices is None:
            print("Error: No vertices provided.")
            return

        # 2. Load Topology (Indices) - Critical for Flood Fill
        # We try to find indices from standard locations in the app structure
        if hasattr(self.app, 'sculptor') and hasattr(self.app.sculptor, 'indices'):
            self.mesh_indices = self.app.sculptor.indices
        elif hasattr(self.app, 'indices'):
            self.mesh_indices = self.app.indices
        else:
            print("Error: No Mesh Topology (Indices) found. Flood fill impossible.")
            self.mesh_indices = []

        # 3. Setup KDTree (for spatial queries)
        if hasattr(self.app, 'kd_tree') and self.app.kd_tree:
             self.kd_tree = self.app.kd_tree
        elif self.mesh_vertices is not None:
             print("Building KDTree...")
             self.kd_tree = KDTree(self.mesh_vertices)

        # 4. Initialize Selection Mask
        self.selected_mask = np.zeros(len(self.mesh_vertices), dtype=bool)

        # 5. Build Adjacency Graph (if not already built)
        if self.vertex_neighbors is None:
            self._build_adjacency_graph()

        # 6. AUTO-EXECUTE: Get Margin & Fill
        raw_points = context.get('margin_points', [])
        
        if raw_points and len(raw_points) > 2:
            # Ensure format is list of numpy arrays
            # Handle both dictionary points {'pos': ...} and raw arrays
            self.margin_loop = []
            for p in raw_points:
                if isinstance(p, dict):
                    self.margin_loop.append(np.array(p['pos']))
                else:
                    self.margin_loop.append(np.array(p))
            
            # Run the logic immediately
            self.select_area_inside_margin()
        else:
            print("No margin points received to auto-select from.")

    def _build_adjacency_graph(self):
        """Creates a lookup list where vertex_neighbors[i] = {neighbor_indices}"""
        if not len(self.mesh_indices): return
        
        print("Building Mesh Graph...")
        num_verts = len(self.mesh_vertices)
        self.vertex_neighbors = [set() for _ in range(num_verts)]
        
        indices = np.array(self.mesh_indices)
        if len(indices.shape) == 1: indices = indices.reshape(-1, 3)
            
        for v1, v2, v3 in indices:
            self.vertex_neighbors[v1].update([v2, v3])
            self.vertex_neighbors[v2].update([v1, v3])
            self.vertex_neighbors[v3].update([v1, v2])
        print("Graph Built.")

    def select_area_inside_margin(self):
        """
        The Core Algorithm:
        1. "Rasterize" the margin line onto the mesh vertices to create a 'Fence'.
        2. Find the center of the margin loop to use as a 'Seed'.
        3. Flood fill from the seed until hitting the fence.
        """
        if not self.margin_loop or self.kd_tree is None or self.vertex_neighbors is None: 
            return

        print("Auto-Selecting Prep inside Margin...")
        
        # --- STEP 1: CREATE THE FENCE ---
        # We sample points heavily along the margin line and find all vertices
        # within 'fence_width'. These act as a wall.
        boundary_indices = set()
        
        # Sampling density: ensure we don't skip over any triangles
        sample_step = self.fence_width * 0.4 
        sampled_points = []
        
        # Resample the loop
        loop_len = len(self.margin_loop)
        for i in range(loop_len):
            p1 = self.margin_loop[i]
            p2 = self.margin_loop[(i+1) % loop_len]
            
            dist = np.linalg.norm(p2 - p1)
            steps = max(2, int(dist / sample_step))
            
            for s in range(steps):
                t = s / steps
                sampled_points.append(p1 + (p2 - p1) * t)
                
        # Bulk query for speed
        indices_list = self.kd_tree.query_ball_point(sampled_points, self.fence_width)
        
        for idx_group in indices_list:
            boundary_indices.update(idx_group)
            
        if not boundary_indices:
            print("Error: Margin line does not touch mesh.")
            return

        # --- STEP 2: FIND THE SEED ---
        # Geometric centroid of the margin loop
        margin_array = np.array(self.margin_loop)
        centroid = np.mean(margin_array, axis=0)
        
        # Find nearest vertex to centroid
        # Note: For very curved preparations ("U" shape), simple centroid might be in air.
        # But usually KDTree query snaps to the occlusal surface correctly.
        _, seed_idx = self.kd_tree.query(centroid)
        
        # Correction: If seed is ON the boundary, check neighbors for a free spot
        if seed_idx in boundary_indices:
            # BFS search for a non-boundary neighbor
            found_better_seed = False
            local_q = queue.Queue()
            local_q.put(seed_idx)
            checked = {seed_idx}
            
            while not local_q.empty():
                curr = local_q.get()
                # Stop looking if we go too far
                if len(checked) > 100: break 
                
                for n in self.vertex_neighbors[curr]:
                    if n not in boundary_indices:
                        seed_idx = n
                        found_better_seed = True
                        break
                    if n not in checked:
                        checked.add(n)
                        local_q.put(n)
                if found_better_seed: break
            
            if not found_better_seed:
                print("Failed: Could not find a starting point inside the margin.")
                return

        # --- STEP 3: FLOOD FILL ---
        prep_region = set()
        q = queue.Queue()
        q.put(seed_idx)
        prep_region.add(seed_idx)
        
        visited = set()
        visited.add(seed_idx)
        # Add boundary to visited so the fill bounces off it
        visited.update(boundary_indices)
        
        # Safety limit: Prep should rarely exceed 30% of a full dental scan
        max_limit = len(self.mesh_vertices) * 0.3
        is_leaking = False
        
        while not q.empty():
            curr = q.get()
            
            if len(prep_region) > max_limit:
                is_leaking = True
                break
                
            for n in self.vertex_neighbors[curr]:
                if n not in visited:
                    visited.add(n)
                    prep_region.add(n)
                    q.put(n)
                    
        # --- STEP 4: FINALIZE ---
        if is_leaking:
            print("Warning: Selection leaked. Try increasing Fence Width.")
            # We still show what we got, but user might need to adjust
        else:
            # Include boundary in the final selection for a tight fit
            prep_region.update(boundary_indices)
            
            # Update Mask
            self.selected_mask[:] = False
            self.selected_mask[list(prep_region)] = True
            print(f"Success: Selected {len(prep_region)} vertices.")

    def apply_selection(self):
        """Pass the selection to the next tool"""
        selected_indices = np.where(self.selected_mask)[0]
        if len(selected_indices) == 0:
            print("No selection to apply.")
            return

        # 1. Create Data Packet
        context_data = {
            "prep_indices": selected_indices,
            "margin_points": self.margin_loop,
            "mesh_vertices": self.mesh_vertices,
            "mesh_normals": context.get('mesh_normals') if hasattr(self, 'context') else None
        }
        
        # 2. Transition (Change 'INSERTION' to your actual next step key)
        # Usually -> Insertion Path or Cement Gap
        if "INSERTION" in self.app.current_tool.workflow_registry:
            self.app.current_tool.switch_step("INSERTION")
            # You would inject context_data here typically
        else:
            print("Selection Applied. Ready for next step.")

    # =========================================================================
    # GUI & RENDERING
    # =========================================================================
    def draw_options(self):
        imgui.text_colored("Auto Prep Select", 0.0, 1.0, 0.0)
        imgui.text_wrapped("Based on your margin line, the system has detected the prep surface.")
        imgui.separator()
        
        # Stats
        if self.selected_mask is not None:
            count = np.sum(self.selected_mask)
            imgui.text(f"Selected Vertices: {count}")
        
        imgui.dummy(0, 5)
        
        # Settings
        changed, self.fence_width = imgui.slider_float("Fence Thickness", self.fence_width, 0.1, 1.5, "%.2f mm")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Make this thicker if the selection leaks out of the margin line.")
        
        if changed:
            # If user drags slider, re-calculate instantly (or on release if slow)
            self.select_area_inside_margin()
            
        imgui.dummy(0, 5)
        
        if imgui.button("Re-Calculate Selection", width=-1):
            self.select_area_inside_margin()
            
        imgui.separator()
        
        # Navigation
        if imgui.button("NEXT: Insertion Path >", width=-1, height=40):
            self.apply_selection()

    def render_3d(self):
        if self.mesh_vertices is None: return

        # 1. Draw Selection (Green Points)
        if self.selected_mask is not None and np.any(self.selected_mask):
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            glPointSize(3.0)
            glColor4f(*self.select_color)
            
            indices = np.where(self.selected_mask)[0]
            
            # Optimization: Stride for performance if selection is huge
            step = 1 if len(indices) < 15000 else 4
            
            glBegin(GL_POINTS)
            for i in range(0, len(indices), step):
                glVertex3f(*self.mesh_vertices[indices[i]])
            glEnd()
            glPopAttrib()

        # 2. Draw Margin Loop (Gold Line)
        if self.margin_loop:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST) # Draw on top
            glLineWidth(3.0)
            glColor4f(*self.margin_color)
            
            glBegin(GL_LINE_LOOP)
            for p in self.margin_loop:
                glVertex3f(*p)
            glEnd()
            glPopAttrib()

    # Stub input handlers to prevent crashes, as this is an automated step
    def on_mouse_button(self, button, action, mods, x, y): return False
    def on_mouse_move(self, x, y): return False
    def on_key(self, key, scancode, action, mods): return False
