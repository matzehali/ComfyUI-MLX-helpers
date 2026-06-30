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
import time
from pathlib import Path
from typing import Callable

_SAFE_EXT = ".safetensors"
_PROGRESS_WIDTH = 24
_PROGRESS_MIN_SECONDS = 2.0
_PROGRESS_MIN_FRACTION = 0.05


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


def _format_size(value: int | float | None) -> str:
    if value is None:
        return "?"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(amount) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{amount:.0f} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TB"


def _make_log_tqdm(status: Callable[[str], None], label: str):
    """Build a tiny tqdm-compatible progress class backed by the node logger.

    ``huggingface_hub`` accepts ``tqdm_class`` for both whole-repo snapshots and
    single-file downloads. ComfyUI logs do not render terminal progress bars, so
    this class emits rate-limited plain text progress lines instead.
    """

    class _LogTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            del args
            self.iterable = iterable
            self.desc = kwargs.get("desc") or label
            self.total = kwargs.get("total")
            if self.total is None and iterable is not None:
                try:
                    self.total = len(iterable)
                except TypeError:
                    self.total = None
            self.n = kwargs.get("initial") or 0
            self.unit = kwargs.get("unit") or "it"
            self.unit_scale = bool(kwargs.get("unit_scale"))
            self._started = time.monotonic()
            self._last_log = 0.0
            self._last_fraction = -1.0
            self._last_logged_n: int | float | None = None
            self._last_logged_total: int | float | None = None
            self._closed = False
            self.refresh(force=True)

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            self.close()

        def _line(self) -> str:
            total = self.total if self.total not in (None, 0) else None
            if total:
                fraction = max(0.0, min(1.0, float(self.n) / float(total)))
                filled = int(round(fraction * _PROGRESS_WIDTH))
                bar = "#" * filled + "-" * (_PROGRESS_WIDTH - filled)
                percent = f"{fraction * 100:5.1f}%"
            else:
                bar = "?" * _PROGRESS_WIDTH
                percent = "  ?.?%"

            if self.unit == "B" or self.unit_scale:
                current = _format_size(self.n)
                ending = _format_size(total) if total else "?"
            else:
                current = f"{int(self.n)} {self.unit}"
                ending = f"{int(total)} {self.unit}" if total else f"? {self.unit}"

            elapsed = max(time.monotonic() - self._started, 0.001)
            rate = ""
            if self.unit == "B" or self.unit_scale:
                rate = f" {_format_size(float(self.n) / elapsed)}/s"

            return f"{label}: [{bar}] {percent} {current}/{ending}{rate}"

        def _should_log(self, force: bool) -> bool:
            if force:
                return True
            now = time.monotonic()
            total = self.total if self.total not in (None, 0) else None
            fraction = float(self.n) / float(total) if total else None
            advanced_fraction = (
                fraction is not None
                and (self._last_fraction < 0 or fraction - self._last_fraction >= _PROGRESS_MIN_FRACTION)
            )
            return advanced_fraction or (now - self._last_log >= _PROGRESS_MIN_SECONDS)

        def refresh(self, *args, force: bool = False, **kwargs):
            del args, kwargs
            if self._closed or not self._should_log(force):
                return
            status(self._line())
            self._last_logged_n = self.n
            self._last_logged_total = self.total
            self._last_log = time.monotonic()
            total = self.total if self.total not in (None, 0) else None
            if total:
                self._last_fraction = float(self.n) / float(total)

        def update(self, n: int | float | None = 1) -> None:
            self.n += n or 0
            self.refresh()

        def set_description(self, desc: str | None = None, *args, **kwargs) -> None:
            del args, kwargs
            if desc:
                self.desc = desc
            self.refresh(force=True)

        def close(self) -> None:
            if not self._closed:
                if self.n != self._last_logged_n or self.total != self._last_logged_total:
                    self.refresh(force=True)
                self._closed = True

    return _LogTqdm


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
        tqdm_class=_make_log_tqdm(status, f"download {source}"),
    )
    status(f"download complete: {downloaded}")
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

        downloaded = hf_hub_download(
            hf_repo,
            source,
            local_dir=str(base),
            tqdm_class=_make_log_tqdm(status, f"download {source}"),
        )
        status(f"download complete: {downloaded}")
        return Path(downloaded)

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
