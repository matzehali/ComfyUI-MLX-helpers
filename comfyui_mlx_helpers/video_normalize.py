"""Shared video dimension/frame normalization for the MLX port projects.

Different DiT/VAE families need the input video padded to model-valid spatial and
temporal sizes before encoding. The math is identical across projects — only the
spatial multiple and temporal period differ:

    LTX    : spatial_multiple = 32 or 64,  temporal_period = 8   (8n+1)
    Wan2.2 : spatial_multiple = 16,        temporal_period = 4   (4n+1)

Keeping the implementation here means an optimization (e.g. a smarter pad or a
resize tweak) lands in every port at once. Each port keeps only a thin node that
picks the right factors and calls :func:`normalize_video`.
"""
from __future__ import annotations

import math

import numpy as np


def next_multiple(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``value`` (>= ``multiple``)."""
    multiple = max(1, int(multiple))
    return int(math.ceil(max(1, int(value)) / multiple) * multiple)


def next_frame_count(frame_count: int, period: int, offset: int = 1) -> int:
    """Smallest valid frame count of the form ``period * n + offset`` that is
    >= ``frame_count``. Already-valid counts are preserved (not bumped a block)."""
    frame_count = max(1, int(frame_count))
    period = max(1, int(period))
    return int(math.ceil(max(0, frame_count - offset) / period) * period + offset)


def resample_video_frames(frames, source_fps: float, target_fps: float):
    """Duration-preserving nearest-frame rate conversion for IMAGE batches.

    The mapping uses half-up frame-boundary rounding, matching the MiniMax H3
    reference normalization path. It supports NumPy arrays and torch tensors
    without moving torch data off its current device.

    ``N`` source frames become ``round_half_up(N * target/source)`` frames. For
    example, 111 frames at 25 fps become 107 frames at 24 fps while retaining
    the original 4.44-second duration to within one output frame.
    """
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must both be positive")
    if not hasattr(frames, "shape") or len(frames.shape) < 1:
        raise ValueError("frames must be an array or tensor with a frame axis")
    frame_count = int(frames.shape[0])
    if frame_count < 1:
        raise ValueError("frames must contain at least one frame")
    if source_fps == target_fps:
        return frames

    scale = target_fps / source_fps
    slots = np.floor(np.arange(frame_count, dtype=np.float64) * scale + 0.5).astype(np.int64)
    output_count = int(math.floor(frame_count * scale + 0.5))
    repeats = np.diff(slots, append=output_count)
    indices = np.repeat(np.arange(frame_count, dtype=np.int64), repeats)
    if int(indices.shape[0]) != output_count:
        raise RuntimeError("frame-rate conversion produced an inconsistent output count")

    try:
        import torch

        if torch.is_tensor(frames):
            return frames.index_select(0, torch.as_tensor(indices, dtype=torch.long, device=frames.device))
    except ImportError:
        pass
    return np.asarray(frames)[indices]


def normalize_video(
    image,
    spatial_multiple: int,
    temporal_period: int,
    temporal_offset: int = 1,
    rescale_factor: float = 1.0,
    resize_method: str = "area",
):
    """Pad a ComfyUI IMAGE batch to model-valid dimensions/timing.

    Args:
        image: ComfyUI IMAGE tensor, ``[F, H, W, C]`` (or ``[H, W, C]``), float 0..1.
        spatial_multiple: width/height are ceil-rounded up to this multiple.
        temporal_period: frame count is rounded up to ``period * n + offset``.
        temporal_offset: the ``+offset`` in the frame-count rule (usually 1).
        rescale_factor: optional content scale applied before padding.
        resize_method: ``area`` | ``bilinear`` | ``nearest-exact`` (used only when
            ``rescale_factor`` changes the content size).

    Returns:
        ``(image, width, height, num_frames)`` — the video black-padded spatially
        (content centered) and temporally extended by repeating the last frame,
        plus the normalized dimensions to feed a sampler.
    """
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"normalize_video expects IMAGE rank 3 or 4, got {tuple(image.shape)}")

    fc, h, w, c = image.shape
    if c <= 0:
        raise ValueError(f"normalize_video expects >=1 channel, got {tuple(image.shape)}")

    scale = max(0.05, float(rescale_factor))
    content_w = max(1, int(round(w * scale)))
    content_h = max(1, int(round(h * scale)))
    target_w = next_multiple(content_w, spatial_multiple)
    target_h = next_multiple(content_h, spatial_multiple)
    target_f = next_frame_count(fc, temporal_period, temporal_offset)

    work = image if torch.is_floating_point(image) else image.float()
    if (content_h, content_w) != (h, w):
        x = work.permute(0, 3, 1, 2)
        if resize_method == "bilinear":
            x = F.interpolate(x, size=(content_h, content_w), mode="bilinear", align_corners=False)
        elif resize_method == "nearest-exact":
            try:
                x = F.interpolate(x, size=(content_h, content_w), mode="nearest-exact")
            except (ValueError, TypeError):
                x = F.interpolate(x, size=(content_h, content_w), mode="nearest")
        else:
            x = F.interpolate(x, size=(content_h, content_w), mode="area")
        work = x.permute(0, 2, 3, 1)

    out = torch.zeros((target_f, target_h, target_w, c), dtype=work.dtype, device=work.device)
    top, left = (target_h - content_h) // 2, (target_w - content_w) // 2
    out[:fc, top:top + content_h, left:left + content_w, :].copy_(work)
    if target_f > fc:  # repeat the last valid frame to satisfy the period*n+offset timing
        out[fc:].copy_(out[fc - 1:fc].expand(target_f - fc, -1, -1, -1))
    return out, target_w, target_h, target_f
