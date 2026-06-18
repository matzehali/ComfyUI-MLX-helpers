"""Shared infrastructure for the ComfyUI MLX port projects.

Two entry points, one codebase:

* **Importable library** — projects add this repo to their requirements and
  ``from comfyui_mlx_helpers import node_meta, resolve_weight_file, ...`` instead
  of carrying near-identical copies of the version/logo/cleanup boilerplate.
* **Node pack** — when loaded as a ComfyUI custom node it registers a few
  model-agnostic nodes (see :mod:`comfyui_mlx_helpers.nodes`).
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
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

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
    # node pack
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
