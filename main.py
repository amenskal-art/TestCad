# main.py
import sys
import numpy as np
from core import DentalEditorApp
from utils import load_mesh_data

def main():
    if len(sys.argv) > 1: 
        pts, cols, norms, faces = load_mesh_data(sys.argv[1])
    else:
        # Initialize with an empty point cloud
        pts = np.zeros((0, 3), dtype=np.float32)
        cols = None
        norms = None
        faces = None
    app = DentalEditorApp(pts, cols, norms, faces)
    app.run()

if __name__ == "__main__":
    main()
