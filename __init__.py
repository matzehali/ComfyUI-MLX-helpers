"""ComfyUI entry point for the MLX helpers node pack.

Works both as a custom-node folder (relative import of the bundled package) and
when the package is pip-installed (absolute import).
"""

try:
    from comfyui_mlx_helpers.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # loaded as a custom_nodes folder, package is a subpackage
    from .comfyui_mlx_helpers.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
