"""Shared infrastructure library for the ComfyUI MLX port projects.

This is a plain importable library — not a ComfyUI node pack. Each MLX port keeps
its own (necessarily model-specific) loader/inference nodes and imports the
shared helpers here instead of carrying near-identical copies of the
version/logo/cleanup/model-resolution boilerplate::

    from comfyui_mlx_helpers import node_meta, resolve_weight_file, load_safetensors
    meta = node_meta.for_repo(__file__, fallback="v0.1", log_prefix="Foo-MLX")
"""

from __future__ import annotations

from . import model_resolve, node_meta, output_tracing, v3_nodes, vlm_models
from .model_resolve import (
    CUSTOM_MODEL_CHOICE,
    configured_models_dir,
    configured_model_roots,
    discover_model_dirs,
    list_safetensors,
    model_dropdown_choices,
    resolve_choice_or_custom,
    resolve_model_dir,
    resolve_repo_file,
    resolve_weight_file,
)
from .mlx_runtime import (
    ANY_TYPE,
    PRECISIONS,
    AnyType,
    aggressive_cleanup,
    clear_compiled_callables,
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
from .output_tracing import (
    PARTIAL_EXECUTION_TARGETS_INPUT,
    mark_traced_inputs_lazy,
    parse_partial_execution_targets,
    partial_execution_targets_from_extra_pnginfo,
    requested_outputs_for_node,
    required_inputs_for_node,
    trace_requested_outputs,
    validate_output_dependencies,
)
from .video_normalize import (
    next_frame_count,
    next_multiple,
    normalize_video,
)
from .v3_nodes import adapt_v1_node, adapt_v1_nodes, v3_nodes_available
from .vlm_models import discover_vlm_models, is_vlm_config, resolve_vlm_choice, vlm_model_dropdown

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


def install_widget_input_sync(web_dir: str) -> None:
    """Copy live widget sync plus output-tracing transport into a pack web dir."""
    import shutil

    try:
        _os.makedirs(web_dir, exist_ok=True)
        for filename in (
            "mlx_widget_input_sync.js",
            "widget_input_sync_core.js",
            "mlx_partial_execution_targets.js",
            "partial_execution_targets_core.js",
        ):
            shutil.copyfile(
                _os.path.join(WEB_DIRECTORY, filename),
                _os.path.join(web_dir, filename),
            )
    except Exception:
        pass


def install_output_tracing(web_dir: str) -> None:
    """Copy the partial-execution target transport into a pack web dir."""
    import shutil

    try:
        _os.makedirs(web_dir, exist_ok=True)
        for filename in (
            "mlx_partial_execution_targets.js",
            "partial_execution_targets_core.js",
        ):
            shutil.copyfile(
                _os.path.join(WEB_DIRECTORY, filename),
                _os.path.join(web_dir, filename),
            )
    except Exception:
        pass


__all__ = [
    "node_meta",
    "model_resolve",
    "v3_nodes",
    "vlm_models",
    "output_tracing",
    "WEB_DIRECTORY",
    "install_node_colors",
    "install_widget_input_sync",
    "install_output_tracing",
    # naming / versioning
    "LOGO",
    "RepoMeta",
    "for_repo",
    "resolve_version",
    "with_mlx_metadata",
    # model resolution
    "CUSTOM_MODEL_CHOICE",
    "configured_models_dir",
    "configured_model_roots",
    "discover_model_dirs",
    "model_dropdown_choices",
    "resolve_choice_or_custom",
    "resolve_model_dir",
    "resolve_repo_file",
    "resolve_weight_file",
    "list_safetensors",
    "discover_vlm_models",
    "is_vlm_config",
    "resolve_vlm_choice",
    "vlm_model_dropdown",
    # mlx runtime
    "PRECISIONS",
    "mx_dtype",
    "aggressive_cleanup",
    "clear_compiled_callables",
    "get_compiled_callable",
    "load_safetensors",
    "torch_image_to_mx",
    "mx_to_torch",
    "torch_image_to_pil",
    "AnyType",
    "ANY_TYPE",
    # ComfyUI V3 node migration
    "adapt_v1_node",
    "adapt_v1_nodes",
    "v3_nodes_available",
    # output-aware lazy execution
    "PARTIAL_EXECUTION_TARGETS_INPUT",
    "mark_traced_inputs_lazy",
    "parse_partial_execution_targets",
    "partial_execution_targets_from_extra_pnginfo",
    "requested_outputs_for_node",
    "required_inputs_for_node",
    "trace_requested_outputs",
    "validate_output_dependencies",
    # video normalization (shared across MLX ports)
    "normalize_video",
    "next_multiple",
    "next_frame_count",
    "__version__",
]
