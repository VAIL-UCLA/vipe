# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import yaml

from vipe.ext.lietorch import SE3
from vipe.utils.geometry import se3_matrix_inverse, se3_matrix_to_se3


def parse_cameras_yaml(yaml_path: str | Path) -> dict[str, torch.Tensor]:
    """Parse a cameras.yaml calibration file.

    Returns a dict mapping camera name to intrinsics tensor [fx, fy, cx, cy]
    extracted from the ``undistK`` (undistorted pinhole) field.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    intrinsics: dict[str, torch.Tensor] = {}
    for cam_name, cam_data in data.items():
        K = cam_data["undistK"]
        intrinsics[cam_name] = torch.tensor(
            [K[0], K[4], K[2], K[5]], dtype=torch.float32
        )
    return intrinsics


def parse_transforms_yaml(yaml_path: str | Path) -> dict[str, np.ndarray]:
    """Parse a transforms.yaml calibration file.

    Returns a dict mapping camera name to its 4x4 ``cam_T_lidar`` matrix
    (camera-from-lidar transform, row-major storage).
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    transforms: dict[str, np.ndarray] = {}
    for cam_name, flat16 in data["cam_T_lidar"].items():
        transforms[cam_name] = np.array(flat16, dtype=np.float64).reshape(4, 4)
    return transforms


def compute_rig_poses(
    cam_transforms: dict[str, np.ndarray],
    reference_camera: str,
    ordered_cam_names: list[str] | None = None,
) -> tuple[SE3, list[str]]:
    """Compute rig SE3 poses from per-camera ``cam_T_lidar`` matrices.

    The rig maps **from** each camera **to** the reference camera frame:
        rig[i] = cam_T_lidar[ref] @ inv(cam_T_lidar[i])

    Args:
        cam_transforms: {cam_name: 4x4 cam_T_lidar matrix}.
        reference_camera: Name of the reference camera.
        ordered_cam_names: Optional fixed camera order.  If ``None``, cameras
            are sorted alphabetically.

    Returns:
        Rig poses as an SE3 object of shape (V,) and the ordered camera names.
    """
    if ordered_cam_names is None:
        ordered_cam_names = sorted(cam_transforms.keys())
    else:
        ordered_cam_names = list(ordered_cam_names)

    assert reference_camera in ordered_cam_names, (
        f"Reference camera {reference_camera!r} not found among {ordered_cam_names}"
    )

    T_ref = torch.from_numpy(cam_transforms[reference_camera]).float()

    rig_mats: list[torch.Tensor] = []
    for cam_name in ordered_cam_names:
        T_i = torch.from_numpy(cam_transforms[cam_name]).float()
        T_cam_i_to_ref = T_ref @ se3_matrix_inverse(T_i, unbatch=True)
        rig_mats.append(T_cam_i_to_ref)

    rig_stack = torch.stack(rig_mats)
    rig_se3 = se3_matrix_to_se3(rig_stack, unbatch=False)
    return rig_se3, ordered_cam_names
