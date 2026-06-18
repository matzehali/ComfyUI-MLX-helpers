# ComfyUI-MLX-helpers

Shared infrastructure for the ComfyUI **MLX port** projects on Apple Silicon
(DepthAnythingV2-MLX, SpatialTrackerV2-MLX, SAM3, Gemma3, LTXVideo-CustomMLX, …).

Every MLX port was re-implementing the same boilerplate: a `node_meta.py` that
reads the git tag for the node title, an Apple-logo naming convention, model-dir
resolution that honors the menubar app's models folder and downloads from
HuggingFace, MLX/Metal memory cleanup, and torch↔MLX tensor conversion. This
package centralizes all of that so each port keeps **only** its model-specific
code.

It is two things from one codebase:

1. **An importable library** (`comfyui_mlx_helpers`) the other projects depend on.
2. **A ComfyUI node pack** that registers a few model-agnostic nodes.

## Install

It's a normal ComfyUI custom node — clone into `custom_nodes/` — **and** a
pip-installable package the other ports list as a dependency.

```bash
# As a dependency of another MLX port (private repo; uses your gh/git creds):
pip install "git+https://github.com/matzehali/ComfyUI-MLX-helpers.git"
```

Add the same line to a port's `requirements.txt` to pull the shared helpers in.

## Nodes

| Node | What it does |
| --- | --- |
|  MLX Model Loader | Resolve a local path / HF repo / filename (honoring the menubar models dir, downloading if missing), load the safetensors into MLX arrays at a chosen precision, and output a `MLX_WEIGHTS` handle for a model-specific build step. |
|  MLX Free Memory | Run aggressive MLX/Metal memory cleanup at a chosen point in the graph; passes its input through so it can be chained. |

The node title carries the Apple logo and the version from the repo's git tag.

## Using the library in a port project

Replace a project's hand-rolled `node_meta.py` with the shared helper, resolving
*that project's* own git tag for the node title:

```python
from comfyui_mlx_helpers import node_meta, resolve_weight_file, load_safetensors

meta = node_meta.for_repo(__file__, fallback="v0.4", log_prefix="DA2-MLX")

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthAnythingV2MLX": meta.versioned(f"{meta.LOGO} MLX Model Loader DepthAnythingV2"),
}

def load(source):
    meta.banner(f"{meta.LOGO} MLX DepthAnythingV2 {meta.VERSION} — Load")
    path = resolve_weight_file(source, subdir="depthanything",
                               hf_repo="Kijai/DepthAnythingV2-safetensors", status=meta.log)
    weights = load_safetensors(path, dtype="fp16", status=meta.log)
    # ... model-specific architecture build from `weights` ...
```

### Public API

- **Naming / versioning** — `for_repo(anchor, fallback, log_prefix)` →
  `RepoMeta` with `.VERSION`, `.LOGO`, `.versioned(name)`, `.banner()`, `.log()`,
  `.done()`, `.with_metadata()`. Also `resolve_version`, `with_mlx_metadata`,
  `LOGO`.
- **Model resolution** — `resolve_weight_file`, `resolve_model_dir`,
  `configured_models_dir`, `list_safetensors`.
- **MLX runtime** — `load_safetensors`, `aggressive_cleanup`,
  `get_compiled_callable`, `mx_dtype`, `PRECISIONS`, `torch_image_to_mx`,
  `mx_to_torch`, `torch_image_to_pil`, `AnyType` / `ANY_TYPE`.

## Versioning

The version shown in node titles is read from the newest semver git tag without
spawning a subprocess (so it works when ComfyUI launches from a GUI app with a
minimal PATH). Annotated and lightweight tags both work; a `-dirty` suffix is
appended when HEAD is past the newest tag. Cut a release with:

```bash
./local_release_push.sh v0.1
```

## License

Apache-2.0.
