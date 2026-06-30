> Agents: read `AGENTS.md` before working in this repo.

# ComfyUI-MLX-helpers

Shared **infrastructure library** for ComfyUI **MLX port** projects on Apple
Silicon.

Every MLX port was re-implementing the same boilerplate: a `node_meta.py` that
reads the git tag for the node title, an Apple-logo naming convention, model-dir
resolution that honors custom models folder and downloads from
HuggingFace, MLX/Metal memory cleanup, and torch↔MLX tensor conversion. This
package centralizes all of that so each port keeps **only** its model-specific
code.

> **Library only — not a node pack.** It registers no ComfyUI nodes and has no
> `__init__.py` entry point at the repo root. MLX model loaders turned out to be
> too model-specific to share as a single generic node (conv-kernel layout for
> DepthAnythingV2, `mlx-vlm`/`mlx-lm` directory loading for SAM3/Gemma3/Bernini,
> split/quantized weights for LTX), so each port keeps its own loader node and
> imports these helpers to do the shared work.

## Install

It's a pip-installable package the MLX ports list as a dependency:

```bash
# private repo; uses your gh/git credentials
pip install "git+https://github.com/matzehali/ComfyUI-MLX-helpers.git"
```

Add the same line to a port's `requirements.txt`/`pyproject.toml`.

## Using it in a port project

Resolve *that project's own* git tag for the node title, and reuse the shared
model-resolution / weight-loading helpers inside the project's own loader:

```python
from comfyui_mlx_helpers import node_meta, resolve_weight_file, load_safetensors

meta = node_meta.for_repo(__file__, fallback="v0.4", log_prefix="DA2-MLX")

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthAnythingV2MLX": meta.versioned(f"{meta.LOGO} MLX DepthAnythingV2 Loader"),
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
  `get_compiled_callable`, `clear_compiled_callables`, `mx_dtype`, `PRECISIONS`,
  `torch_image_to_mx`, `mx_to_torch`, `torch_image_to_pil`, `AnyType` /
  `ANY_TYPE`. Retain the compiled wrapper on the loaded component for
  cross-prompt reuse, and clear it whenever that component's weights change.

## Versioning

The version helper reads the newest semver git tag of the *consumer's* repo
without spawning a subprocess (so it works when ComfyUI launches from a GUI app
with a minimal PATH). Annotated and lightweight tags both work; a `-dirty`
suffix is appended when HEAD is past the newest tag. Cut a release of this repo
with:

```bash
./local_release_push.sh v0.2
```

## License

Apache-2.0.
