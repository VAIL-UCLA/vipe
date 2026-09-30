# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import field_validator

from vipe.config.base_schema import BaseConfigSchema, Field


class BaseStreamListConfig(BaseConfigSchema):
    """Shared options for stream lists backed by videos or image-frame folders."""

    base_path: str = Field(
        description="Input path. For MP4 streams this can be one video or a directory of videos; for frame-dir streams "
        "this can be one frame directory or a directory containing multiple frame directories."
    )
    frame_start: int = Field(ge=0, description="First frame index to include.")
    frame_end: int = Field(
        ge=-1,
        description="Exclusive end frame index. Use -1 to process through the end of each stream.",
    )
    frame_skip: int = Field(ge=1, description="Frame stride. A value of 1 processes every frame.")
    cached: bool = Field(
        description="Cache each stream before processing. This helps with malformed videos whose frame counts are "
        "unreliable when decoded lazily."
    )

    @field_validator("frame_end")
    @classmethod
    def validate_frame_end(cls, value: int) -> int:
        if value == -1 or value >= 0:
            return value
        raise ValueError("frame_end must be -1 or a non-negative frame index")


class RawMP4StreamListConfig(BaseStreamListConfig):
    """Stream list that reads raw MP4 files."""

    instance: Literal["vipe.streams.raw_mp4_stream.RawMP4StreamList"] = Field(
        description="Implementation class for MP4 video input streams."
    )


class FrameDirStreamListConfig(BaseStreamListConfig):
    """Stream list that reads directories of image frames."""

    instance: Literal["vipe.streams.frame_dir_stream.FrameDirStreamList"] = Field(
        description="Implementation class for image-frame directory input streams."
    )


class RigCalibStreamListConfig(BaseStreamListConfig):
    """Stream list that reads a multi-camera rig with known calibration."""

    instance: Literal["vipe.streams.rig_calib_stream.RigCalibStreamList"] = Field(
        description="Implementation class for rig-calibration stream lists."
    )
    calib_dir: str = Field(
        default="calib",
        description="Subdirectory under base_path containing cameras.yaml and transforms.yaml.",
    )
    reference_camera: str = Field(
        default="camera_front",
        description="Camera name used as the rig reference (rig center).",
    )
    camera_dir_suffix: str = Field(
        default="_undist",
        description="Suffix to strip from calibration camera names to locate frame directories.",
    )


StreamsConfig = Annotated[
    RawMP4StreamListConfig | FrameDirStreamListConfig | RigCalibStreamListConfig,
    Field(discriminator="instance"),
]
