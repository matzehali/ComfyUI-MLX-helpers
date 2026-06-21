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
