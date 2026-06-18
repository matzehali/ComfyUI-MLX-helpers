"""Model-directory / weight-file resolution shared by the MLX loader nodes.

Every MLX port reimplements the same boilerplate: take a string that is either a
local path, a bare filename, or a HuggingFace repo id; honor the models folder
the menubar app is configured with; fall back to ComfyUI's own models dir; and
download from the hub when nothing is present locally. This module centralizes
that so each project only keeps its model-specific "build architecture from
these weights" code.

All MLX/torch/hub imports are lazy so this module is importable without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

_SAFE_EXT = ".safetensors"


def configured_models_dir() -> Path | None:
    """Return the models folder selected in the ComfyUI menubar app, if any.

    Resolution order: ``$COMFYUI_MODELS_DIR`` env var, then the menubar app's
    ``config.json`` (``models_path``, or legacy ``hf_models_path``). Returns
    ``None`` when nothing is configured.
    """
    env_path = os.environ.get("COMFYUI_MODELS_DIR")
    if env_path:
        return Path(env_path).expanduser()

    config_path = Path.home() / "Library" / "Application Support" / "ComfyUI_MenuBar" / "config.json"
    try:
        with config_path.open("r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    configured = data.get("models_path") or data.get("hf_models_path") or ""
    return Path(configured).expanduser() if configured else None


def _base_models_dir() -> Path:
    """Menubar-configured models dir if set, else ComfyUI's default models dir."""
    configured = configured_models_dir()
    if configured:
        return configured
    try:
        import folder_paths  # provided by ComfyUI at runtime

        return Path(folder_paths.models_dir)
    except Exception:
        return Path.cwd() / "models"


def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").exists() or any(path.glob(f"*{_SAFE_EXT}"))
    )


def resolve_model_dir(
    source: str,
    *,
    subdir: str = "",
    status: Callable[[str], None] = print,
) -> Path:
    """Resolve *source* to a local model directory, downloading from HF if needed.

    *source* may be an absolute path, a name relative to the (menubar-configured
    or ComfyUI default) models dir, or a HuggingFace repo id. *subdir* is an
    optional category folder under the models dir (e.g. ``"depthanything"``).
    """
    path = Path(source).expanduser()
    if path.is_absolute() and path.exists():
        return path

    base = _base_models_dir()
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    local = base / source

    if _looks_like_model_dir(local):
        status(f"local model found at {local}")
        return local

    status(f"downloading {source} -> {local}")
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=source,
        local_dir=local,
        local_dir_use_symlinks=False,  # real files, no blob symlinks
        cache_dir=local / ".cache",
        resume_download=True,
    )
    return Path(downloaded)


def resolve_weight_file(
    source: str,
    *,
    subdir: str = "",
    hf_repo: str = "",
    status: Callable[[str], None] = print,
) -> Path:
    """Resolve *source* to a single weights file, downloading from HF if needed.

    Handles three shapes:

    * absolute path to an existing file -> returned as-is;
    * a bare filename together with ``hf_repo`` -> looked up under the models dir
      (optionally ``subdir``) and fetched via ``hf_hub_download`` when missing;
    * anything else -> treated as a directory via :func:`resolve_model_dir`, and
      the first ``*.safetensors`` inside is returned.
    """
    path = Path(source).expanduser()
    if path.is_absolute() and path.is_file():
        return path

    if hf_repo and (source.endswith(_SAFE_EXT) or "/" not in source):
        base = _base_models_dir()
        if subdir:
            base = base / subdir
        target = base / source
        if target.is_file():
            status(f"local weights found at {target}")
            return target
        base.mkdir(parents=True, exist_ok=True)
        status(f"downloading {source} from {hf_repo}")
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(hf_repo, source, local_dir=str(base)))

    model_dir = resolve_model_dir(source, subdir=subdir, status=status)
    if model_dir.is_file():
        return model_dir
    files = sorted(model_dir.glob(f"*{_SAFE_EXT}"))
    if not files:
        raise FileNotFoundError(f"No {_SAFE_EXT} file found in {model_dir}")
    return files[0]


def list_safetensors(directory) -> list[Path]:
    """Sorted ``*.safetensors`` files directly inside *directory*."""
    directory = Path(directory)
    return sorted(directory.glob(f"*{_SAFE_EXT}")) if directory.is_dir() else []
