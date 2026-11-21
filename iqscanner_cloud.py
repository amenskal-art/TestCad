# tools/iqscanner_cloud.py
# Pure Cloud Backend for Multi-Mode Scanner
# Mode 0: HunyuanWorld-Mirror (Interiors/Large Scale)        -> GLB mesh
# Mode 1: Pi3X (Objects/Reverse Engineering Surface smoothing) -> PLY point cloud
# Mode 2: LingBot-Map (Scene Mapping / long sequences)        -> GLB
# Mode 3: HunyuanWorld-Mirror (Real Estate)                   -> PLY Gaussian splat
#         (same repo/function as Mode 0, but exported with --save_gs)

import os
import subprocess
import zipfile
import shutil
from pathlib import Path
import modal

app = modal.App("iqscanner-omega")

# ==========================================
# 1. HUNYUAN CONFIGURATION (INTERIORS + REAL ESTATE)
# ==========================================
REPO_DIR_HUNYUAN = "/workspace/HunyuanWorld-Mirror"

hunyuan_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu124")
    .env({"CACHE_BUST": "mirror_v1"})
    .run_commands(f"git clone https://github.com/amenskal-art/HunyuanWorld-Mirror-modal.git {REPO_DIR_HUNYUAN}")
    .run_commands(f"pip install -r {REPO_DIR_HUNYUAN}/requirements.txt")
    .run_commands("pip install gsplat onnxruntime --extra-index-url https://docs.gsplat.studio/whl/pt24cu124")
    .run_commands(f"curl -L -o {REPO_DIR_HUNYUAN}/skyseg.onnx https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx")
)
hunyuan_cache = modal.Volume.from_name("mirror-model-cache", create_if_missing=True)


# ==========================================
# 2. PI3X CONFIGURATION (OBJECTS)
# ==========================================
REPO_DIR_PI3X = "/workspace/Pi3"

pi3x_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("open3d", "scipy", "safetensors", "trimesh", "einops", "huggingface_hub", "opencv-python-headless", "plyfile")
    .env({"CACHE_BUST": "pi3x_v1"})
    .run_commands(f"git clone https://github.com/amenskal-art/Pi3-modal.git {REPO_DIR_PI3X}")
)
pi3x_cache = modal.Volume.from_name("pi3x-model-cache", create_if_missing=True)


# ==========================================
# 3. LINGBOT-MAP CONFIGURATION (SCENE MAPPING)
# ==========================================
# Runs the repo's own demo.py (recommended `lingbot-map-long` checkpoint
# from HuggingFace robbyant/lingbot-map). demo.py accepts an image folder
# OR a video (sampled at --fps) and supports --mask_sky (ONNX skyseg).
REPO_DIR_LINGBOT = "/workspace/lingbot-map"

lingbot_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    # Upstream-recommended torch 2.8.0 + CUDA 12.8
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .env({"CACHE_BUST": "lingbot_v1"})
    .run_commands(f"git clone https://github.com/amenskal-art/lingbot-map-modal.git {REPO_DIR_LINGBOT}")
    # The repo's own deps + its [vis] extra (demo.py imports the vis package)
    .run_commands(f"pip install 'numpy<2' && pip install -e '{REPO_DIR_LINGBOT}[vis]'")
    .run_commands("pip install opencv-python-headless onnxruntime")
    # Pre-bake the sky segmentation model so --mask_sky never downloads at runtime
    .run_commands(f"curl -L -o {REPO_DIR_LINGBOT}/skyseg.onnx https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx")
)
# Dedicated volume so the 4.6 GB lingbot-map-long.pt checkpoint downloads once
lingbot_cache = modal.Volume.from_name("lingbot-model-cache", create_if_missing=True)


# Virtual hard drive shared across modes
nfs = modal.NetworkFileSystem.from_name("iqscanner-nfs", create_if_missing=True)


@app.function(network_file_systems={"/data": nfs})
def upload_chunk(chunk: bytes, is_first: bool):
    mode = "wb" if is_first else "ab"
    with open("/data/input.zip", mode) as f:
        f.write(chunk)


def _patch_native_glb_export(repo_dir: str):
    infer_path = os.path.join(repo_dir, "infer.py")
    with open(infer_path, "r") as f:
        src = f.read()
    fixes = {
        '"world_points": all_pts[final_mask_torch].unsqueeze(0).numpy() if final_mask_torch.dim() > 0 else filtered_pts,':
            '"world_points": predictions["pts3d"][0].detach().cpu().numpy(),',
        '"images": imgs.detach().cpu().numpy(),':
            '"images": imgs[0].detach().cpu().numpy(),',
    }
    patched = src
    for old, new in fixes.items():
        patched = patched.replace(old, new)
    if patched != src:
        with open(infer_path, "w") as f:
            f.write(patched)


def _patch_lingbot_glb_export(repo_dir: str):
    """Same trick as _patch_native_glb_export, but for lingbot-map's demo.py.

    Stock demo.py ends by launching an interactive viser browser viewer and
    never writes a file. This swaps that final viewer block for a headless
    GLB export using the repo's OWN exporter (lingbot_map.vis.predictions_to_glb)
    and its OWN sky masking (apply_sky_segmentation) — no pipeline logic lives
    here, demo.py keeps doing all the work.
    """
    demo_path = os.path.join(repo_dir, "demo.py")
    with open(demo_path, "r") as f:
        src = f.read()

    old_block = '''    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")'''

    new_block = '''    # Headless GLB export (patched in by iqscanner cloud runner)
    import math as _math
    from lingbot_map.vis.glb_export import predictions_to_glb
    from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

    out_path = os.environ.get("IQSCANNER_GLB_OUT", "result.glb")
    vis_pred = prepare_for_visualization(predictions, images_cpu)

    # This checkpoint may not emit a pointmap in streaming mode. Do exactly
    # what the repo's PointCloudViewer does by default: unproject the depth
    # maps with the predicted cameras (repo util), and use depth_conf.
    if vis_pred.get("world_points") is not None:
        pts_key, conf_key = "world_points", "world_points_conf"
    else:
        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
        vis_pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            vis_pred["depth"], vis_pred["extrinsic"], vis_pred["intrinsic"]
        )
        pts_key, conf_key = "world_points_from_depth", "depth_conf"
        print("world_points absent -> unprojected depth maps (viewer fallback)")

    if args.mask_sky and vis_pred.get(conf_key) is not None:
        vis_pred[conf_key] = apply_sky_segmentation(
            vis_pred[conf_key],
            image_folder=resolved_image_folder,
            images=vis_pred.get("images"),
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )

    # Spatial downsample so the exported GLB stays a downloadable size
    wp = vis_pred[pts_key]
    budget = float(os.environ.get("IQSCANNER_MAX_POINTS", "4000000"))
    if wp is not None and wp.size:
        total = wp.shape[0] * wp.shape[1] * wp.shape[2]
        ds = max(1, int(_math.ceil(_math.sqrt(total / budget))))
        if ds > 1:
            vis_pred[pts_key] = vis_pred[pts_key][:, ::ds, ::ds]
            if vis_pred.get(conf_key) is not None:
                vis_pred[conf_key] = vis_pred[conf_key][:, ::ds, ::ds]
            imgs = vis_pred.get("images")
            if imgs is not None and imgs.ndim == 4:
                if imgs.shape[1] == 3:
                    vis_pred["images"] = imgs[:, :, ::ds, ::ds]
                else:
                    vis_pred["images"] = imgs[:, ::ds, ::ds]
            print(f"Downsampled point grid {ds}x for export")

    scene = predictions_to_glb(
        vis_pred,
        conf_thres=float(os.environ.get("IQSCANNER_CONF_PERCENTILE", "25")),
        show_cam=False,
        mask_sky=False,
    )
    scene.export(out_path)
    print(f"Saved GLB to {out_path}")'''

    if old_block in src:
        with open(demo_path, "w") as f:
            f.write(src.replace(old_block, new_block))


# ==========================================
# HUNYUAN EXECUTION PIPELINE
# Mode 0 (Metrology): save_gs=False -> GLB mesh (original behavior)
# Mode 3 (Real Estate): save_gs=True -> gaussians.ply (3D Gaussian splat)
# ==========================================
@app.function(image=hunyuan_image, gpu="A100", timeout=3600, volumes={"/cache": hunyuan_cache}, network_file_systems={"/data": nfs})
def process_mesh_hunyuan(fps: int = 1, save_gs: bool = False):
    os.environ["HF_HOME"] = "/cache/huggingface"

    data_root = "/tmp/data"
    out_root = "/tmp/output"
    zip_path = "/data/input.zip"
    # Gaussian-splat mode produces a PLY; mesh mode produces a GLB.
    save_path = "/data/result.ply" if save_gs else "/data/result.glb"

    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)
    # Clear BOTH possible result files so a stale one is never downloaded.
    for stale in ("/data/result.glb", "/data/result.ply"):
        if os.path.exists(stale):
            os.remove(stale)
    os.makedirs(data_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    data_dir = data_root
    entries = [e for e in os.listdir(data_dir) if not e.startswith(".") and e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(os.path.join(data_dir, entries[0])):
        data_dir = os.path.join(data_dir, entries[0])

    for name in os.listdir(data_dir):
        stem, ext = os.path.splitext(name)
        if ext and ext != ext.lower():
            os.rename(os.path.join(data_dir, name), os.path.join(data_dir, stem + ext.lower()))

    input_arg = data_dir
    is_video = False
    valid_files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    if len(valid_files) == 1 and os.path.isfile(os.path.join(data_dir, valid_files[0])):
        single_file = valid_files[0]
        if single_file.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".gif")):
            input_arg = os.path.join(data_dir, single_file)
            is_video = True
        elif single_file.lower().endswith(".mkv"):
            src_v = os.path.join(data_dir, single_file)
            remuxed = os.path.join(data_dir, os.path.splitext(single_file)[0] + ".mp4")
            subprocess.run(["ffmpeg", "-y", "-i", src_v, "-c", "copy", remuxed], check=True)
            os.remove(src_v)
            input_arg = remuxed
            is_video = True

    _patch_native_glb_export(REPO_DIR_HUNYUAN)
    os.chdir(REPO_DIR_HUNYUAN)

    cmd = ["python", "infer.py", "--input_path", input_arg, "--output_path", out_root]
    if is_video: cmd += ["--fps", str(fps)]
    if save_gs: cmd += ["--save_gs"]   # export 3D Gaussians (gaussians.ply)

    crashed = False
    try: subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError: crashed = True

    # ----- Gaussian-splat result (Real Estate mode) -----
    if save_gs:
        plys = sorted(Path(out_root).rglob("gaussians.ply"))
        if not plys:
            plys = sorted(Path(out_root).rglob("*.ply"))  # any PLY as fallback
        if not plys:
            shutil.rmtree(data_root, ignore_errors=True)
            raise Exception("Engine Crashed before producing any 3D output.")
        shutil.copyfile(str(plys[0]), save_path)
        shutil.rmtree(data_root, ignore_errors=True)
        shutil.rmtree(out_root, ignore_errors=True)
        return

    # ----- Mesh / GLB result (Metrology mode) — original logic, untouched -----
    glbs = sorted(Path(out_root).rglob("*.glb"))
    result_glb = None
    for preferred in ("mesh.glb", "pointcloud.glb"):
        match = [p for p in glbs if p.name == preferred]
        if match:
            result_glb = match[0]
            break
    if result_glb is None and glbs:
        result_glb = glbs[0]

    if result_glb is None:
        plys = sorted(Path(out_root).rglob("pts_from_pointmap.ply"))
        if plys:
            import trimesh
            cloud = trimesh.load(str(plys[0]))
            fallback = Path(out_root) / "fallback_pointcloud.glb"
            trimesh.Scene(cloud).export(str(fallback))
            result_glb = fallback

    if result_glb is None:
        shutil.rmtree(data_root, ignore_errors=True)
        raise Exception("Engine Crashed before producing any 3D output.")

    shutil.copyfile(result_glb, save_path)
    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)


# ==========================================
# PI3X EXECUTION PIPELINE (Reverse Engineering)
# ==========================================
@app.function(image=pi3x_image, gpu="A100", timeout=3600, volumes={"/cache": pi3x_cache}, network_file_systems={"/data": nfs})
def process_mesh_pi3x(fps: int = 1):
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    import sys
    sys.path.append(REPO_DIR_PI3X)
    
    import torch
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    import trimesh
    
    from pi3.utils.basic import load_multimodal_data
    from pi3.utils.geometry import depth_edge
    from pi3.models.pi3x import Pi3X

    data_root = "/tmp/data"
    out_root = "/tmp/output"
    zip_path = "/data/input.zip"
    save_path = "/data/result.ply"

    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)
    if os.path.exists(save_path): os.remove(save_path)
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    data_dir = data_root
    entries = [e for e in os.listdir(data_dir) if not e.startswith(".") and e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(os.path.join(data_dir, entries[0])):
        data_dir = os.path.join(data_dir, entries[0])

    input_arg = data_dir
    is_video = False
    valid_files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    if len(valid_files) == 1 and os.path.isfile(os.path.join(data_dir, valid_files[0])):
        single_file = valid_files[0]
        if single_file.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
            input_arg = os.path.join(data_dir, single_file)
            is_video = True

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    interval = max(int(30/fps), 1) if is_video else 1
    conditions = dict(intrinsics=None, poses=None, depths=None)

    imgs, conditions = load_multimodal_data(input_arg, conditions, interval=interval, device=device) 
    use_multimodal = any(v is not None for v in conditions.values())

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    if not use_multimodal: model.disable_multimodal()
    model = model.to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs=imgs, **conditions)

    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    num_views = res['points'][0].shape[0]
    pcds = []

    for v in range(num_views):
        view_mask = masks[v]
        v_points = res['points'][0, v][view_mask].cpu().numpy()
        v_colors = imgs[0, v].permute(1, 2, 0)[view_mask].cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcds.append(pcd)

    target_pcd = pcds[0]
    aligned_pcds = [target_pcd]

    for v in range(1, num_views):
        source_pcd = pcds[v]
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        source_pcd.transform(reg_p2l.transformation)
        aligned_pcds.append(source_pcd)

    final_pcd = o3d.geometry.PointCloud()
    for p in aligned_pcds: final_pcd += p

    final_pcd = final_pcd.voxel_down_sample(voxel_size=0.015)
    final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)

    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=40))
    final_pcd.orient_normals_consistent_tangent_plane(k=40)

    # Save point cloud natively into a PLY file
    o3d.io.write_point_cloud(save_path, final_pcd)
    
    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)


# ==========================================
# LINGBOT-MAP EXECUTION PIPELINE (Scene Mapping)
# Thin runner: prep data -> run the repo's demo.py -> grab the GLB.
# ==========================================
@app.function(image=lingbot_image, gpu="A100", timeout=3600, volumes={"/cache": lingbot_cache}, network_file_systems={"/data": nfs})
def process_mesh_lingbot(fps: int = 10):
    os.environ["HF_HOME"] = "/cache/huggingface"

    data_root = "/tmp/data"
    out_root = "/tmp/output"
    zip_path = "/data/input.zip"
    save_path = "/data/result.glb"

    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)
    if os.path.exists(save_path): os.remove(save_path)
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)

    data_dir = data_root
    entries = [e for e in os.listdir(data_dir) if not e.startswith(".") and e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(os.path.join(data_dir, entries[0])):
        data_dir = os.path.join(data_dir, entries[0])

    # demo.py only globs .jpg/.png/.JPG from folders -> normalize extensions
    for name in os.listdir(data_dir):
        stem, ext = os.path.splitext(name)
        ext_l = ext.lower()
        if ext_l == ".jpeg":
            ext_l = ".jpg"
        if ext and ext != ext_l:
            os.rename(os.path.join(data_dir, name), os.path.join(data_dir, stem + ext_l))

    input_args = ["--image_folder", data_dir]
    valid_files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    if len(valid_files) == 1 and os.path.isfile(os.path.join(data_dir, valid_files[0])):
        single_file = valid_files[0]
        if single_file.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".gif")):
            input_args = ["--video_path", os.path.join(data_dir, single_file), "--fps", str(fps)]
        elif single_file.lower().endswith(".mkv"):
            src_v = os.path.join(data_dir, single_file)
            remuxed = os.path.join(data_dir, os.path.splitext(single_file)[0] + ".mp4")
            subprocess.run(["ffmpeg", "-y", "-i", src_v, "-c", "copy", remuxed], check=True)
            os.remove(src_v)
            input_args = ["--video_path", remuxed, "--fps", str(fps)]

    # Recommended checkpoint, cached on the dedicated volume after first run
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="robbyant/lingbot-map", filename="lingbot-map-long.pt")
    try:
        lingbot_cache.commit()
    except Exception:
        pass

    _patch_lingbot_glb_export(REPO_DIR_LINGBOT)
    os.chdir(REPO_DIR_LINGBOT)

    result_glb = os.path.join(out_root, "result.glb")
    env = os.environ.copy()
    env["IQSCANNER_GLB_OUT"] = result_glb

    cmd = [
        "python", "demo.py",
        "--model_path", ckpt_path,
        "--use_sdpa",          # SDPA fallback (no FlashInfer JIT in container)
        "--offload_to_cpu",    # keep GPU peak memory low on long sequences
        "--mask_sky",          # filter sky points with the baked-in skyseg.onnx
    ] + input_args

    crashed = False
    try: subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError: crashed = True

    if not os.path.exists(result_glb):
        shutil.rmtree(data_root, ignore_errors=True)
        raise Exception("Engine Crashed before producing any 3D output.")

    shutil.copyfile(result_glb, save_path)
    shutil.rmtree(data_root, ignore_errors=True)
    shutil.rmtree(out_root, ignore_errors=True)


# ==========================================
# SHARED DOWNLOAD FUNCTIONS
# ==========================================
@app.function(network_file_systems={"/data": nfs}, timeout=600)
def get_result_size():
    if os.path.exists("/data/result.glb"): return os.path.getsize("/data/result.glb")
    if os.path.exists("/data/result.ply"): return os.path.getsize("/data/result.ply")
    return 0

@app.function(network_file_systems={"/data": nfs}, timeout=600)
def download_chunk(offset: int, size: int = 1024 * 1024):
    path = "/data/result.glb" if os.path.exists("/data/result.glb") else "/data/result.ply"
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)

@app.function(network_file_systems={"/data": nfs}, timeout=120)
def cleanup_result():
    for p in ["/data/result.glb", "/data/result.ply"]:
        if os.path.exists(p): os.remove(p)
