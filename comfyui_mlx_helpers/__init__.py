"""Shared infrastructure library for the ComfyUI MLX port projects.

This is a plain importable library — not a ComfyUI node pack. Each MLX port keeps
its own (necessarily model-specific) loader/inference nodes and imports the
shared helpers here instead of carrying near-identical copies of the
version/logo/cleanup/model-resolution boilerplate::

    from comfyui_mlx_helpers import node_meta, resolve_weight_file, load_safetensors
    meta = node_meta.for_repo(__file__, fallback="v0.1", log_prefix="Foo-MLX")
"""

from __future__ import annotations

from . import model_resolve, node_meta
from .model_resolve import (
    configured_models_dir,
    list_safetensors,
    resolve_model_dir,
    resolve_weight_file,
)
from .mlx_runtime import (
    ANY_TYPE,
    PRECISIONS,
    AnyType,
    aggressive_cleanup,
    get_compiled_callable,
    load_safetensors,
    mx_dtype,
    mx_to_torch,
    torch_image_to_mx,
    torch_image_to_pil,
)
from .node_meta import (
    LOGO,
    RepoMeta,
    for_repo,
    resolve_version,
    with_mlx_metadata,
)
from .video_normalize import (
    next_frame_count,
    next_multiple,
    normalize_video,
)

__version__ = node_meta.VERSION

__all__ = [
    "node_meta",
    "model_resolve",
    # naming / versioning
    "LOGO",
    "RepoMeta",
    "for_repo",
    "resolve_version",
    "with_mlx_metadata",
    # model resolution
    "configured_models_dir",
    "resolve_model_dir",
    "resolve_weight_file",
    "list_safetensors",
    # mlx runtime
    "PRECISIONS",
    "mx_dtype",
    "aggressive_cleanup",
    "get_compiled_callable",
    "load_safetensors",
    "torch_image_to_mx",
    "mx_to_torch",
    "torch_image_to_pil",
    "AnyType",
    "ANY_TYPE",
    # video normalization (shared across MLX ports)
    "normalize_video",
    "next_multiple",
    "next_frame_count",
    "__version__",
]
