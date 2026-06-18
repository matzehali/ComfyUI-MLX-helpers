"""Generic, model-agnostic ComfyUI nodes shared by the MLX port projects.

These register in the ComfyUI graph so a workflow can drop in a shared loader or
memory node. Anything model-specific (building an architecture from the loaded
weights) stays in the individual port projects, which import the helpers in
``comfyui_mlx_helpers`` to do so.
"""

from __future__ import annotations

from . import model_resolve, node_meta
from .mlx_runtime import PRECISIONS, aggressive_cleanup, load_safetensors

meta = node_meta._self  # helpers package's own version + log prefix

LOADER_HELP = """
source: Local path, HuggingFace repo id, or bare filename of the MLX weights.
models_subdir: Optional category folder under the ComfyUI/menubar models dir (e.g. depthanything).
hf_repo: When source is a bare filename, the HuggingFace repo to fetch it from.
precision: keep = as stored; otherwise cast loaded tensors to fp16/fp32/bf16.
mlx_weights output: {weights, path, files, precision} handle for a model-specific build step.
"""


class MLXModelLoader:
    """Resolve + load MLX safetensors weights (the boilerplate every port shares).

    Resolves a local path / HF repo / filename (honoring the menubar app's models
    folder, downloading when missing), loads the safetensors into MLX arrays at
    the requested precision, and hands back a weights handle. Each port project
    turns that handle into its own model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return node_meta.with_mlx_metadata({
            "required": {
                "source": ("STRING", {
                    "default": "",
                    "tooltip": "Local path, HuggingFace repo id, or bare filename.",
                }),
                "precision": (PRECISIONS, {"default": "keep"}),
            },
            "optional": {
                "models_subdir": ("STRING", {"default": ""}),
                "hf_repo": ("STRING", {"default": ""}),
            },
        }, LOADER_HELP)

    RETURN_TYPES = ("MLX_WEIGHTS",)
    RETURN_NAMES = ("mlx_weights",)
    FUNCTION = "load"
    CATEGORY = "MLX/Helpers"
    DESCRIPTION = ("Resolve (local path / HF repo / filename) and load MLX "
                   "safetensors weights at a chosen precision. Outputs a weights "
                   "handle for a model-specific build step.")

    def load(self, source, precision, models_subdir="", hf_repo=""):
        meta.banner(f"{meta.LOGO} MLX Model Loader {meta.VERSION}",
                    f"source:    {source}",
                    f"subdir:    {models_subdir or '-'}",
                    f"hf_repo:   {hf_repo or '-'}",
                    f"precision: {precision}")
        path = model_resolve.resolve_weight_file(
            source, subdir=models_subdir, hf_repo=hf_repo, status=meta.log,
        )
        dtype = None if precision == "keep" else precision
        weights = load_safetensors(path, dtype=dtype, status=meta.log)
        meta.log(f"loaded {len(weights)} tensors from {path.name}")
        handle = {
            "weights": weights,
            "path": str(path),
            "files": [path.name],
            "precision": precision,
        }
        return (handle,)


class MLXFreeMemory:
    """Free the Metal GPU cache mid-graph; passes a trigger value through.

    Wire any output into ``anything`` to force cleanup to run at that point in a
    low-memory workflow; the same value is returned so the chain continues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .mlx_runtime import ANY_TYPE

        return {
            "required": {"anything": (ANY_TYPE, {})},
            "optional": {"label": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "free"
    CATEGORY = "MLX/Helpers"
    OUTPUT_NODE = False
    DESCRIPTION = "Run aggressive MLX/Metal memory cleanup at this point in the graph."

    def free(self, anything, label=""):
        aggressive_cleanup()
        meta.log(f"freed Metal cache{f' ({label})' if label else ''}")
        return (anything,)


NODE_CLASS_MAPPINGS = {
    "MLXModelLoader": MLXModelLoader,
    "MLXFreeMemory": MLXFreeMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXModelLoader": meta.versioned(f"{meta.LOGO} MLX Model Loader"),
    "MLXFreeMemory": meta.versioned(f"{meta.LOGO} MLX Free Memory"),
}
