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

# Absolute path to the shared ComfyUI web extension directory (brand node colors).
# A node pack exposes it as its own ``WEB_DIRECTORY`` so ComfyUI serves/loads it;
# ComfyUI accepts an absolute WEB_DIRECTORY (os.path.join drops its relative base).
import os as _os

WEB_DIRECTORY = _os.path.join(_os.path.dirname(__file__), "web")


def install_node_colors(web_dir: str) -> None:
    """Copy the shared node-colors ``.js`` into a pack's own ``web`` dir.

    For packs that already define their own ``WEB_DIRECTORY`` (so they cannot
    point it at the helper); call from ``__init__`` with that web dir. The copied
    file is served per pack and self-scopes to that pack's nodes. No-op on error.
    """
    import shutil

    try:
        _os.makedirs(web_dir, exist_ok=True)
        shutil.copyfile(
            _os.path.join(WEB_DIRECTORY, "mlx_node_colors.js"),
            _os.path.join(web_dir, "mlx_node_colors.js"),
        )
    except Exception:
        pass


__all__ = [
    "node_meta",
    "model_resolve",
    "WEB_DIRECTORY",
    "install_node_colors",
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
