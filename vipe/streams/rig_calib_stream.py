# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import torch
from typing import cast, Any

from vipe.ext.lietorch import SE3
from vipe.streams.base import (
    AssignAttributesProcessor,
    CameraType,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamList,
    VideoStream,
)
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.rig_calib import (
    compute_rig_poses,
    parse_cameras_yaml,
    parse_transforms_yaml,
)

logger = logging.getLogger(__name__)


class RigCalibStreamList(StreamList):
    """A stream list that loads a multi-camera rig from a calibration directory.

    Expects a directory layout like::

        base_path/
        ├── calib/
        │   ├── cameras.yaml
        │   └── transforms.yaml
        ├── camera_front_undist/   # per-camera frame directories
        ├── camera_left_undist/
        └── camera_right_undist/

    Camera directory names are matched to calibration entries by stripping
    ``camera_dir_suffix`` (default ``"_undist"``).
    """

    def __init__(
        self,
        base_path: str,
        frame_start: int,
        frame_end: int,
        frame_skip: int,
        cached: bool = False,
        calib_dir: str = "calib",
        reference_camera: str = "camera_front",
        camera_dir_suffix: str = "_undist",
    ) -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached
        self.reference_camera = reference_camera
        self.camera_dir_suffix = camera_dir_suffix

        calib_path = self.base_path / calib_dir
        cameras_yaml = calib_path / "cameras.yaml"
        transforms_yaml = calib_path / "transforms.yaml"

        if not cameras_yaml.exists() or not transforms_yaml.exists():
            raise FileNotFoundError(
                f"Calibration files not found in {calib_path}. "
                f"Expected cameras.yaml and transforms.yaml."
            )

        self.cam_intrinsics = parse_cameras_yaml(str(cameras_yaml))
        cam_transforms = parse_transforms_yaml(str(transforms_yaml))

        self.rig, self.cam_names = compute_rig_poses(
            cam_transforms,
            reference_camera=self.reference_camera,
        )

        self._multiview: MultiviewVideoList | None = None

    def _build_multiview(self) -> MultiviewVideoList:
        video_streams: list[VideoStream] = []
        for cam_name in self.cam_names:
            cam_dir = self._find_cam_dir(cam_name)
            raw_stream: VideoStream = FrameDirStream(cam_dir, seek_range=self.frame_range)

            intrinsics_tensor = self.cam_intrinsics[cam_name]
            n_frames = len(raw_stream)
            processed = ProcessedVideoStream(
                raw_stream,
                [
                    AssignAttributesProcessor(
                        {
                            FrameAttribute.INTRINSICS: [intrinsics_tensor.clone()] * n_frames,
                            FrameAttribute.CAMERA_TYPE: [CameraType.PINHOLE] * n_frames,
                        }
                    )
                ],
            )
            video_streams.append(processed)

        return MultiviewVideoList(
            name=self.base_path.name,
            video_streams=video_streams,
            rig=self.rig,
        )

    def _find_cam_dir(self, cam_name: str) -> Path:
        for suffix in ("", self.camera_dir_suffix):
            candidate = self.base_path / f"{cam_name}{suffix}"
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Camera directory not found for {cam_name!r} under {self.base_path}. "
            f"Tried with suffix {self.camera_dir_suffix!r} and without."
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> MultiviewVideoList:
        if index != 0:
            raise IndexError(f"Index {index} out of range (only 1 multiview sequence available).")
        if self._multiview is None:
            self._multiview = self._build_multiview()
        return self._multiview

    def stream_name(self, index: int) -> str:
        return self.base_path.name
