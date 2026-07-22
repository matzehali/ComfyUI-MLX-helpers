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
- **Model resolution/discovery** — `resolve_weight_file`, `resolve_repo_file`,
  `resolve_model_dir`, `configured_models_dir`, `discover_model_dirs`,
  `model_dropdown_choices`, `resolve_choice_or_custom`, and `list_safetensors`,
  including optional Hub revisions/pattern filters and validation-driven
  re-downloads. `resolve_repo_file` also handles nested tokenizer/config files
  and binds them to an already resolved local snapshot.
- **VLM model selectors** — `discover_vlm_models`, `vlm_model_dropdown`, and
  `resolve_vlm_choice` provide one compatible, local-first selector shared by
  VLM-MLX and nodes which embed VLM analysis. Model loading remains in the
  consumer so the helper stays architecture-neutral.
- **MLX runtime** — `load_safetensors`, `aggressive_cleanup`,
  `get_compiled_callable`, `clear_compiled_callables`, `mx_dtype`, `PRECISIONS`,
  `torch_image_to_mx`, `mx_to_torch`, `torch_image_to_pil`, `AnyType` /
  `ANY_TYPE`. Retain the compiled wrapper on the loaded component for
  cross-prompt reuse, and clear it whenever that component's weights change.
- **ComfyUI V3 migration** — `adapt_v1_node` / `adapt_v1_nodes` produce genuine
  schema-backed V3 classes while retaining serialized node IDs, socket order,
  implementation math, and the V1 mapping fallback used by mixed node packs.
  The wrapped nodes must be stateless; resident models and compiled callables
  stay on loader outputs/components. `v3_nodes_available()` allows a pack to
  retain its previous registration shim on older ComfyUI builds. Pass
  `sync_widget_inputs=True` to emit authoritative scalar input and output
  values as intermediate UI data for the connected-widget display helper.
- **Live connected-widget display** — the shared web extension keeps a widget's
  saved value as its disconnected fallback while displaying the effective
  upstream value whenever its input is linked. Primitive upstream edits refresh
  live; execution supplies authoritative values for computed links and caches
  scalar source outputs by socket. Disconnecting restores the saved fallback.
  Packs using the helper `WEB_DIRECTORY` receive it automatically; packs with
  their own web directory call
  `install_widget_input_sync(web_dir)`.
- **Output-aware lazy tracing** — `validate_output_dependencies`,
  `mark_traced_inputs_lazy`, `required_inputs_for_node`, and
  `requested_outputs_for_node` let a multi-output node declare the exact input
  names required by every output. The tracer walks backward from executable
  output nodes. The shared frontend forwards ComfyUI's selected partial-
  execution roots through an opt-in hidden input, so previewing a cheap
  scalar/path output does not schedule an unrelated image/model branch even
  when other savers remain in the submitted prompt. Undeclared third-party
  nodes conservatively retain all dependencies.

Declare every output and delegate the normal ComfyUI lazy hook:

```python
from comfyui_mlx_helpers import (
    PARTIAL_EXECUTION_TARGETS_INPUT,
    mark_traced_inputs_lazy,
    parse_partial_execution_targets,
    required_inputs_for_node,
)

class ExampleNode:
    OUTPUT_INPUT_DEPENDENCIES = {
        0: ("images",),
        1: ("path",),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return mark_traced_inputs_lazy({
            "required": {"images": ("IMAGE",), "path": ("STRING",)},
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                PARTIAL_EXECUTION_TARGETS_INPUT: "STRING",
            },
        }, cls.OUTPUT_INPUT_DEPENDENCIES)

    def check_lazy_status(self, prompt=None, unique_id=None, **kwargs):
        return required_inputs_for_node(
            prompt,
            unique_id,
            type(self),
            output_node_ids=parse_partial_execution_targets(
                kwargs.get(PARTIAL_EXECUTION_TARGETS_INPUT),
            ),
        )
```

The node execution method must accept its hidden prompt/ID arguments and tolerate
`None` for lazy inputs that are not part of the requested output path. Expose
the helper `WEB_DIRECTORY`, or call `install_output_tracing(web_dir)` when the
pack owns its web directory. If the frontend transport is unavailable or its
payload is invalid, the runtime-safe fallback traces all output nodes in the
submitted prompt.

For a pack that already owns its web directory:

```python
from comfyui_mlx_helpers import (
    install_node_colors,
    install_output_tracing,
    install_widget_input_sync,
)

install_node_colors(WEB_DIR)
install_widget_input_sync(WEB_DIR)
# install_widget_input_sync already includes the tracing transport. Packs that
# do not need widget sync may call install_output_tracing(WEB_DIR) instead.
```

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
