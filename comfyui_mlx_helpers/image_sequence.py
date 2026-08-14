"""Model-independent image-sequence loading helpers for ComfyUI MLX ports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np


ProgressCallback = Callable[[int, int], None]


def resolve_frame_sequence(
    sequence_path: str,
    start_frame: int,
    end_frame: int,
    frame_step: int = 1,
) -> list[Path]:
    """Resolve one Nuke-style ``###`` frame pattern with exact padding."""
    matches = list(re.finditer(r"#+", sequence_path))
    if len(matches) != 1:
        raise ValueError("sequence_path must contain exactly one contiguous # frame pattern")
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if frame_step < 1:
        raise ValueError("frame_step must be positive")

    match = matches[0]
    prefix = sequence_path[: match.start()]
    suffix = sequence_path[match.end() :]
    padding = match.end() - match.start()
    paths = [
        Path(f"{prefix}{frame:0{padding}d}{suffix}").expanduser()
        for frame in range(start_frame, end_frame + 1, frame_step)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        sample = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing {len(missing)} image sequence frame(s): {sample}")
    return paths


def linear_srgb_to_display(values: np.ndarray) -> np.ndarray:
    """Convert scene-linear sRGB values to the display sRGB transfer curve."""
    values = np.maximum(values, 0.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def load_exr_sequence(
    sequence_path: str,
    start_frame: int,
    end_frame: int,
    frame_step: int = 1,
    *,
    input_encoding: str = "linear_srgb",
    width: int = 0,
    height: int = 0,
    progress: ProgressCallback | None = None,
):
    """Load and optionally resize an EXR sequence into a Comfy ``IMAGE`` batch.

    OpenImageIO and Torch are imported lazily from the ComfyUI environment.
    Resizing happens per frame before stacking so large source plates do not
    create a full-resolution intermediate batch.
    """
    if input_encoding not in {"linear_srgb", "display_srgb"}:
        raise ValueError("input_encoding must be 'linear_srgb' or 'display_srgb'")
    if (width > 0) != (height > 0):
        raise ValueError("width and height must both be zero or both be positive")

    try:
        import OpenImageIO as oiio
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise RuntimeError(
            "EXR sequence loading requires OpenImageIO and torch from the ComfyUI environment"
        ) from exc

    paths = resolve_frame_sequence(sequence_path, start_frame, end_frame, frame_step)
    frames = []
    for index, path in enumerate(paths, start=1):
        image = oiio.ImageBuf(str(path))
        spec = image.spec()
        if spec.nchannels < 3:
            raise ValueError(f"EXR frame has fewer than three channels: {path}")
        pixels = np.asarray(image.get_pixels(oiio.FLOAT), dtype=np.float32)[..., :3]
        if input_encoding == "linear_srgb":
            pixels = linear_srgb_to_display(pixels)
        pixels = np.clip(pixels, 0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
        if width > 0 and height > 0 and (tensor.shape[1] != width or tensor.shape[0] != height):
            tensor = torch_functional.interpolate(
                tensor.permute(2, 0, 1).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0).permute(1, 2, 0)
        frames.append(tensor)
        if progress is not None:
            progress(index, len(paths))
    return torch.stack(frames, dim=0)
