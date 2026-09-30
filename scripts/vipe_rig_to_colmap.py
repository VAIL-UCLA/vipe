#!/usr/bin/env python3
"""Convert ViPE multi-camera rig output to standard COLMAP format.

Usage:
    python vipe_rig_to_colmap.py <vipe_output_dir> [--output <colmap_dir>]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    read_depth_artifacts,
    read_instance_artifacts,
    read_instance_phrases,
    read_intrinsics_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to COLMAP quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world 4x4 to COLMAP w2c quaternion + translation."""
    w2c = np.linalg.inv(c2w_matrix)
    quaternion = quaternion_from_matrix(w2c)
    translation = w2c[:3, 3]
    return quaternion, translation


def discover_cameras(vipe_path: Path) -> list[str]:
    """Find camera names by scanning pose/*.npz files."""
    pose_dir = vipe_path / "pose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")
    cameras = sorted([p.stem for p in pose_dir.glob("*.npz")])
    if not cameras:
        raise FileNotFoundError(f"No pose .npz files found in {pose_dir}")
    return cameras


def get_frame_size(vipe_path: Path, camera_name: str) -> tuple[int, int]:
    """Read frame dimensions from the first RGB frame."""
    rgb_dir = vipe_path / "rgb"
    mp4_path = rgb_dir / f"{camera_name}.mp4"
    if mp4_path.exists():
        import imageio
        reader = imageio.get_reader(str(mp4_path), "ffmpeg")
        frame = reader.get_data(0)
        reader.close()
        return frame.shape[1], frame.shape[0]

    intrinsics_path = vipe_path / "intrinsics" / f"{camera_name}.npz"
    if intrinsics_path.exists():
        data = np.load(intrinsics_path)
        intrinsics = data["data"]
        cx, cy = intrinsics[0, 2], intrinsics[0, 3]
        return int(cx * 2), int(cy * 2)

    return 1920, 1080


def write_cameras_txt(output_dir: Path, vipe_path: Path, cameras: list[str]):
    """Write COLMAP cameras.txt with per-camera intrinsics."""
    cameras_file = output_dir / "cameras.txt"
    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")

        for cam_id, cam_name in enumerate(cameras, start=1):
            intr_path = vipe_path / "intrinsics" / f"{cam_name}.npz"
            _, intrinsics, _ = read_intrinsics_artifacts(intr_path)
            fx, fy, cx, cy = intrinsics[0].cpu().numpy()
            width, height = get_frame_size(vipe_path, cam_name)
            f.write(f"{cam_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
            logger.info(f"  Camera {cam_id} ({cam_name}): {fx:.2f}x{fy:.2f} [{width}x{height}]")


def symlink_frames(src_dir: Path, images_dir: Path, cameras: list[str], num_frames: int):
    """Symlink all camera frames into images/.

    Frame files may start at any index (e.g. 000221.jpg).  We map ViPE's
    0-based sequential frame indices to the actual files in sorted order.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    for cam_name in cameras:
        cam_dir = src_dir / cam_name
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Source camera directory not found: {cam_dir}")

        # Discover actual frame files and map ViPE index → file path
        frame_files = sorted(
            [p for p in cam_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")]
        )
        if not frame_files:
            logger.warning(f"No frame files found in {cam_dir}")
            continue

        n_map = min(num_frames, len(frame_files))
        logger.info(f"  {cam_name}: {len(frame_files)} frames on disk, using {n_map}")

        for frame_idx in range(n_map):
            src = frame_files[frame_idx]
            dst = images_dir / f"{cam_name}_{frame_idx:06d}.jpg"
            if not dst.exists() and src.is_file():
                dst.symlink_to(src.resolve())


def extract_frames(vipe_path: Path, images_dir: Path, cameras: list[str], num_frames: int):
    """Extract frames from MP4 videos into images/."""
    import imageio

    images_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = vipe_path / "rgb"

    for cam_name in cameras:
        mp4_path = rgb_dir / f"{cam_name}.mp4"
        if not mp4_path.is_file():
            raise FileNotFoundError(f"RGB video not found: {mp4_path}")
        reader = imageio.get_reader(str(mp4_path), "ffmpeg")
        for frame_idx, frame in enumerate(reader):
            if frame_idx >= num_frames:
                break
            dst = images_dir / f"{cam_name}_{frame_idx:06d}.jpg"
            if not dst.exists():
                imageio.imwrite(str(dst), frame)
            if frame_idx % 30 == 0:
                logger.info(f"  Extracted {cam_name} frame {frame_idx}/{num_frames}")
        reader.close()


def export_depth_pngs(
    vipe_path: Path,
    images_dir: Path,
    cameras: list[str],
    num_frames: int,
    depth_step: int = 1,
):
    """Export ViPE EXR depth maps as uint16 millimeter PNGs for 3DGRUT training.

    Writes ``{cam_name}_{frame_idx:06d}_depth.png`` alongside the RGB frames
    in ``images_dir``.  Values are stored as uint16 millimeters so that
    ``ColmapDataset`` can read them via ``cv2.imread(..., IMREAD_UNCHANGED) / 1000.0``.

    Pixels that are masked out by ViPE instance segmentation are set to 0 so
    they are naturally excluded by the ``gt_depth > 0.1`` filter in the trainer.
    """
    import cv2 as cv

    for cam_name in cameras:
        depth_path = vipe_path / "depth" / f"{cam_name}.zip"
        if not depth_path.exists():
            logger.warning(f"Depth file not found: {depth_path}, skipping {cam_name}")
            continue

        mask_path = vipe_path / "mask" / f"{cam_name}.zip"
        masks: dict[int, torch.Tensor] = {}
        if mask_path.exists():
            try:
                masks = {
                    frame_idx: mask
                    for frame_idx, mask in read_instance_artifacts(mask_path)
                    if frame_idx < num_frames
                }
                logger.info(f"  Loaded {len(masks)} masks for {cam_name}")
            except Exception as e:
                logger.warning(f"Failed to read masks for {cam_name}: {e}")

        depth_iter = read_depth_artifacts(depth_path)
        for frame_idx, depth in depth_iter:
            if frame_idx >= num_frames:
                break
            if frame_idx % depth_step != 0:
                continue

            if depth is None or depth.numel() == 0:
                continue

            depth_np = depth.numpy().astype(np.float64)

            if frame_idx in masks:
                instance_mask = masks[frame_idx]
                if instance_mask.shape == depth_np.shape:
                    depth_np[instance_mask.numpy() > 0] = 0.0

            depth_mm = (depth_np * 1000.0).clip(0, 65535).astype(np.uint16)
            out_path = images_dir / f"{cam_name}_{frame_idx:06d}_depth.png"
            cv.imwrite(str(out_path), depth_mm)

            if frame_idx % 30 == 0:
                logger.info(f"  Exported depth {cam_name} frame {frame_idx}/{num_frames}")

        logger.info(f"  Finished depth export for {cam_name}")


def export_instance_mask_pngs(
    vipe_path: Path,
    images_dir: Path,
    cameras: list[str],
    num_frames: int,
    mask_phrases: set[str],
):
    """Export instance-region masks as PNGs for 3DGRUT training.

    Reads ViPE instance masks and phrases, writing a binary mask where
    pixels matching any phrase in *mask_phrases* are excluded (0 = excluded,
    255 = keep).  If masks already exist (e.g. from a previous phrase),
    the new mask is combined with the existing one via element-wise min.

    Writes ``{cam_name}_{frame_idx:06d}_mask.png`` alongside the RGB frames.
    """
    import cv2 as cv

    for cam_name in cameras:
        mask_path = vipe_path / "mask" / f"{cam_name}.zip"
        phrases_path = vipe_path / "mask" / f"{cam_name}.txt"

        exclude_ids: set[int] = set()
        if phrases_path.exists():
            phrases = read_instance_phrases(phrases_path)
            exclude_ids = {iid for iid, phrase in phrases.items() if phrase.strip().lower() in mask_phrases}
            logger.info(f"  {cam_name}: {len(exclude_ids)} matching instance IDs for {mask_phrases}: {sorted(exclude_ids)}")
        else:
            logger.warning(f"No phrases file for {cam_name}, skipping")
            continue

        if not exclude_ids:
            logger.info(f"  No matching instances found for {cam_name}, skipping")
            continue

        if not mask_path.exists():
            logger.warning(f"No mask zip for {cam_name}")
            continue

        masks = {
            frame_idx: mask
            for frame_idx, mask in read_instance_artifacts(mask_path)
            if frame_idx < num_frames
        }
        logger.info(f"  Loaded {len(masks)} instance masks for {cam_name}")

        for frame_idx, instance in masks.items():
            instance_np = instance.numpy()
            new_mask = np.where(np.isin(instance_np, list(exclude_ids)), 0, 255).astype(np.uint8)
            out_path = images_dir / f"{cam_name}_{frame_idx:06d}_mask.png"
            if out_path.exists():
                existing = cv.imread(str(out_path), cv.IMREAD_UNCHANGED)
                if existing is not None:
                    new_mask = np.minimum(existing, new_mask)
            cv.imwrite(str(out_path), new_mask)

        logger.info(f"  Finished mask export for {cam_name}")


def write_images_txt(output_dir: Path, vipe_path: Path, cameras: list[str], num_frames: int):
    """Write COLMAP images.txt with interleaved per-camera poses."""
    images_file = output_dir / "images.txt"

    cam_poses: dict[str, np.ndarray] = {}
    for cam_name in cameras:
        pose_path = vipe_path / "pose" / f"{cam_name}.npz"
        pose_data = np.load(pose_path)
        cam_poses[cam_name] = pose_data["data"]

    total_images = num_frames * len(cameras)

    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {total_images}\n")

        image_id = 0
        for frame_idx in range(num_frames):
            for cam_id, cam_name in enumerate(cameras, start=1):
                image_id += 1
                c2w = cam_poses[cam_name][frame_idx]
                quaternion, translation = matrix_to_colmap_pose(c2w)
                qw, qx, qy, qz = quaternion
                tx, ty, tz = translation
                image_name = f"{cam_name}_{frame_idx:06d}.jpg"
                f.write(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {image_name}\n")
                f.write("\n")

    logger.info(f"Written images.txt with {total_images} images")


def write_points3d_txt_from_slam_maps(output_dir: Path, vipe_path: Path, cameras: list[str]):
    """Write COLMAP points3D.txt from all per-camera SLAM maps combined."""
    vipe_dir = vipe_path / "vipe"
    if not vipe_dir.is_dir():
        logger.warning("No vipe/ directory, skipping points3D.txt")
        return

    points3d_file = output_dir / "points3D.txt"

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    for cam_name in cameras:
        slam_path = vipe_dir / f"{cam_name}_slam_map.pt"
        if not slam_path.is_file():
            logger.warning(f"SLAM map not found: {slam_path}, skipping")
            continue
        slam_map = SLAMMap.load(slam_path, device=torch.device("cpu"))
        for kf_idx in range(len(slam_map.dense_disp_frame_inds)):
            xyz, rgb = slam_map.get_dense_disp_pcd(kf_idx)
            all_xyz.append(xyz.cpu().numpy())
            all_rgb.append(rgb.cpu().numpy())
        logger.info(f"  Loaded SLAM map from {cam_name}: {len(all_xyz)} keyframes so far")

    if not all_xyz:
        logger.warning("No SLAM map data loaded, skipping points3D.txt")
        return

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(xyz)}\n")

        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = int(rgb_uint8[i, 0]), int(rgb_uint8[i, 1]), int(rgb_uint8[i, 2])
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0 1 {i + 1}\n")

    logger.info(f"Written points3D.txt with {len(xyz)} points from {len(cameras)} cameras")


def _image_id_for(frame_idx: int, cam_idx: int, n_cameras: int) -> int:
    """Return 1-indexed IMAGE_ID matching the interleaved ordering in images.txt."""
    return frame_idx * n_cameras + cam_idx + 1


def write_points3d_txt_from_depth(
    output_dir: Path,
    vipe_path: Path,
    images_dir: Path,
    cameras: list[str],
    num_frames: int,
    depth_step: int = 1,
    spatial_subsample: int = 4,
    no_depth_filter: bool = False,
):
    """Write COLMAP points3D.txt by unprojecting per-frame depth maps to world space.

    Depth is read from ``depth/<camera>.zip`` (multi-view depth model output),
    then back-projected through each camera's intrinsics and composed with the
    estimated camera-to-world pose.  RGB colours are sampled from the frame
    images under ``images_dir``.

    When *no_depth_filter* is ``False`` (default), only pixels that pass the
    ``reliable_depth_mask_range`` local-consistency check are kept.  When
    ``True``, all non-zero depth pixels are used.
    """
    import cv2 as cv

    points3d_file = output_dir / "points3D.txt"

    n_cameras = len(cameras)

    cam_intrinsics: dict[str, np.ndarray] = {}
    cam_intrinsics_raw: dict[str, torch.Tensor] = {}
    for cam_name in cameras:
        intr_path = vipe_path / "intrinsics" / f"{cam_name}.npz"
        _, intrinsics, _ = read_intrinsics_artifacts(intr_path)
        cam_intrinsics_raw[cam_name] = intrinsics
        cam_intrinsics[cam_name] = intrinsics[0].cpu().numpy()

    cam_poses: dict[str, np.ndarray] = {}
    for cam_name in cameras:
        pose_path = vipe_path / "pose" / f"{cam_name}.npz"
        cam_poses[cam_name] = np.load(pose_path)["data"]

    cam_masks: dict[str, dict[int, torch.Tensor]] = {}
    for cam_name in cameras:
        mask_path = vipe_path / "mask" / f"{cam_name}.zip"
        if mask_path.exists():
            try:
                cam_masks[cam_name] = {
                    frame_idx: mask
                    for frame_idx, mask in read_instance_artifacts(mask_path)
                    if frame_idx < num_frames
                }
            except Exception as e:
                logger.warning(f"Failed to read masks for {cam_name}: {e}")

    all_points: list[tuple[int, float, float, float, int, int, int, float, int, int]] = []

    camera_rays_cache: dict[str, torch.Tensor] = {}

    for cam_idx, cam_name in enumerate(cameras):
        logger.info(f"  Processing depth maps for {cam_name}...")
        depth_path = vipe_path / "depth" / f"{cam_name}.zip"
        if not depth_path.exists():
            logger.warning(f"Depth file not found: {depth_path}, skipping {cam_name}")
            continue

        depth_iter = read_depth_artifacts(depth_path)
        frame_count = 0

        for frame_idx, depth in depth_iter:
            if frame_idx >= num_frames:
                break

            if frame_idx % depth_step != 0:
                continue

            if frame_count % 30 == 0:
                logger.info(f"    {cam_name} frame {frame_idx}/{num_frames}")

            if depth is None or depth.numel() == 0:
                continue

            frame_height, frame_width = depth.shape

            if cam_name not in camera_rays_cache:
                camera_model = CameraType.PINHOLE.build_camera_model(cam_intrinsics_raw[cam_name][0])
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                rays /= rays[..., 2:3]
                camera_rays_cache[cam_name] = torch.from_numpy(rays)

            rays = camera_rays_cache[cam_name].numpy()
            depth_sampled = depth[::spatial_subsample, ::spatial_subsample].numpy()

            pcd = rays * depth_sampled[..., None]
            if no_depth_filter:
                depth_mask = (depth_sampled > 0)
            else:
                depth_mask = (
                    reliable_depth_mask_range(torch.from_numpy(depth_sampled))
                    .numpy()
                )

            if cam_name in cam_masks and frame_idx in cam_masks[cam_name]:
                instance_mask = cam_masks[cam_name][frame_idx]
                instance_sampled = instance_mask[::spatial_subsample, ::spatial_subsample].numpy()
                valid_h, valid_w = depth_sampled.shape
                if instance_sampled.shape[:2] == (valid_h, valid_w):
                    depth_mask = depth_mask & (instance_sampled == 0)
            pcd_valid = pcd[depth_mask]

            if pcd_valid.size == 0:
                frame_count += 1
                continue

            c2w = cam_poses[cam_name][frame_idx]
            pcd_world = pcd_valid @ c2w[:3, :3].T + c2w[:3, 3][None]

            image_path = images_dir / f"{cam_name}_{frame_idx:06d}.jpg"
            if image_path.exists():
                img = cv.imread(str(image_path), cv.IMREAD_COLOR)
                if img is not None:
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img_sampled = img[::spatial_subsample, ::spatial_subsample]
                    rgb_sampled = img_sampled[depth_mask]
                    image_id = _image_id_for(frame_idx, cam_idx, n_cameras)
                    for i in range(len(pcd_world)):
                        x, y, z = pcd_world[i]
                        r, g, b = int(rgb_sampled[i, 0]), int(rgb_sampled[i, 1]), int(rgb_sampled[i, 2])
                        all_points.append((0, x, y, z, r, g, b, 0.0, image_id, 0))
                    frame_count += 1
                    continue

            image_id = _image_id_for(frame_idx, cam_idx, n_cameras)
            for i in range(len(pcd_world)):
                x, y, z = pcd_world[i]
                all_points.append((0, x, y, z, 255, 255, 255, 0.0, image_id, 0))

            frame_count += 1

        logger.info(f"    {cam_name}: processed {frame_count} frames ({len(all_points)} total points so far)")

    if not all_points:
        logger.warning("No depth points generated, skipping points3D.txt")
        return

    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_points)}\n")

        for point_id, (_, x, y, z, r, g, b, error, image_id, _) in enumerate(all_points, start=1):
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {image_id} {point_id}\n")

    logger.info(f"Written points3D.txt with {len(all_points)} points from depth unprojection")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViPE multi-camera rig output to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vipe_path", type=Path, help="Path to ViPE rig output directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: <vipe_path>_colmap)")
    parser.add_argument("--src-dir", type=Path, default=None,
                        help="Source frame directory for symlinking (default: auto-detect)")
    parser.add_argument("--no-symlink", action="store_true",
                        help="Extract frames from MP4 instead of symlinking from source")
    parser.add_argument("--camera-names", type=str, nargs="+", default=None,
                        help="Camera names in order (default: auto-discover)")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="Number of frames to include (-1 = all)")
    parser.add_argument("--no-points", action="store_true",
                        help="Skip points3D.txt generation")
    parser.add_argument("--use-depth", action="store_true",
                        help="Generate points3D.txt from depth unprojection instead of SLAM map")
    parser.add_argument("--depth-step", type=int, default=1,
                        help="Process every N-th depth frame (1 = all frames)")
    parser.add_argument("--spatial-subsample", type=int, default=4,
                        help="Spatial subsampling factor for depth unprojection (1 = full resolution)")
    parser.add_argument("--no-depth-filter", action="store_true",
                        help="Disable reliable_depth_mask_range filtering (keep all non-zero depth pixels)")
    parser.add_argument("--export-depth-png", action="store_true",
                        help="Export depth maps as uint16 mm PNGs in images/ for 3DGRUT training")
    parser.add_argument("--export-human-mask", action="store_true",
                        help="Export human-region masks as PNGs in images/ for 3DGRUT training")
    parser.add_argument("--export-car-mask", action="store_true",
                        help="Export car/vehicle-region masks as PNGs in images/ (combines with --export-human-mask)")
    args = parser.parse_args()

    vipe_path = args.vipe_path.resolve()
    if not vipe_path.is_dir():
        print(f"Error: ViPE path '{vipe_path}' does not exist.")
        return 1

    cameras = args.camera_names if args.camera_names else discover_cameras(vipe_path)
    logger.info(f"Cameras: {cameras}")

    pose_path = vipe_path / "pose" / f"{cameras[0]}.npz"
    pose_data = np.load(pose_path)
    max_frames = len(pose_data["data"])
    num_frames = max_frames if args.num_frames < 0 else min(args.num_frames, max_frames)
    logger.info(f"Frames per camera: {num_frames} (of {max_frames} total)")

    output_dir = args.output if args.output else (vipe_path.parent / f"{vipe_path.name}_colmap")
    colmap_dir = output_dir / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    src_dir: Path | None = None
    if not args.no_symlink:
        if args.src_dir:
            src_dir = args.src_dir.resolve()
        else:
            for candidate in [vipe_path.parent, vipe_path.parent.parent]:
                test_dir = candidate / cameras[0]
                if test_dir.is_dir():
                    src_dir = candidate
                    break
        if src_dir:
            logger.info(f"Symlinking frames from {src_dir}")
            symlink_frames(src_dir, images_dir, cameras, num_frames)
        else:
            logger.info("Source directory not found, falling back to MP4 extraction")
            args.no_symlink = True

    if args.no_symlink:
        extract_frames(vipe_path, images_dir, cameras, num_frames)

    if args.export_depth_png:
        logger.info("Exporting depth maps as PNGs...")
        export_depth_pngs(vipe_path, images_dir, cameras, num_frames)

    if args.export_human_mask or args.export_car_mask:
        mask_phrases: set[str] = set()
        if args.export_human_mask:
            mask_phrases.add("person")
        if args.export_car_mask:
            mask_phrases.update({"car", "vehicle", "truck", "bus"})
        logger.info(f"Exporting instance masks for phrases: {mask_phrases}")
        export_instance_mask_pngs(vipe_path, images_dir, cameras, num_frames, mask_phrases)

    write_cameras_txt(colmap_dir, vipe_path, cameras)
    write_images_txt(colmap_dir, vipe_path, cameras, num_frames)

    if not args.no_points:
        if args.use_depth:
            write_points3d_txt_from_depth(
                colmap_dir, vipe_path, images_dir, cameras, num_frames,
                depth_step=args.depth_step,
                spatial_subsample=args.spatial_subsample,
                no_depth_filter=args.no_depth_filter,
            )
        else:
            write_points3d_txt_from_slam_maps(colmap_dir, vipe_path, cameras)

    logger.info("COLMAP conversion completed!")
    logger.info(f"Output directory: {colmap_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
