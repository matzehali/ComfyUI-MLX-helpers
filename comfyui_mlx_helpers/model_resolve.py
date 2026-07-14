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
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

_SAFE_EXT = ".safetensors"
CUSTOM_MODEL_CHOICE = "custom — use custom_model_id"
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


def configured_model_roots() -> list[Path]:
    """Candidate roots for local model discovery, ordered by runtime priority."""
    roots: list[Path] = []
    configured = configured_models_dir()
    if configured:
        roots.append(configured)
    try:
        import folder_paths  # provided by ComfyUI at runtime

        roots.append(Path(folder_paths.models_dir))
    except Exception:
        pass

    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.expanduser().resolve())
        except Exception:
            key = str(root.expanduser())
        if key not in seen:
            result.append(root)
            seen.add(key)
    return result


def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").exists()
        or (path / "model_index.json").exists()
        or any(path.glob(f"*{_SAFE_EXT}"))
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
        _lock = threading.RLock()

        @classmethod
        def get_lock(cls):
            return cls._lock

        @classmethod
        def set_lock(cls, lock) -> None:
            cls._lock = lock

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
    revision: str | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    ignore_patterns: str | Sequence[str] | None = None,
    force_download: bool = False,
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

    if _looks_like_model_dir(local) and not force_download:
        status(f"local model found at {local}")
        return local

    status(f"downloading {source} -> {local}")
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=source,
        local_dir=local,
        cache_dir=local / ".cache",
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force_download,
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
    revision: str | None = None,
    validator: Callable[[Path], bool] | None = None,
    force_download: bool = False,
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
        if validator is not None and not validator(path):
            raise OSError(f"Local weights failed validation: {path}")
        return path

    if hf_repo and (source.endswith(_SAFE_EXT) or "/" not in source):
        base = _base_models_dir()
        if subdir:
            base = base / subdir
        target = base / source
        if target.is_file() and not force_download and (validator is None or validator(target)):
            status(f"local weights found at {target}")
            return target
        if target.is_file() and validator is not None and not validator(target):
            status(f"local weights failed validation; re-downloading {target}")
            force_download = True
        base.mkdir(parents=True, exist_ok=True)
        status(f"downloading {source} from {hf_repo}")
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            hf_repo,
            source,
            local_dir=str(base),
            revision=revision,
            force_download=force_download,
            tqdm_class=_make_log_tqdm(status, f"download {source}"),
        )
        downloaded_path = Path(downloaded)
        if validator is not None and not validator(downloaded_path):
            raise OSError(f"Downloaded weights failed validation: {downloaded_path}")
        status(f"download complete: {downloaded}")
        return downloaded_path

    model_dir = resolve_model_dir(
        source,
        subdir=subdir,
        status=status,
        revision=revision,
        force_download=force_download,
    )
    if model_dir.is_file():
        return model_dir
    files = sorted(model_dir.glob(f"*{_SAFE_EXT}"))
    if not files:
        raise FileNotFoundError(f"No {_SAFE_EXT} file found in {model_dir}")
    if validator is not None and not validator(files[0]):
        raise OSError(f"Resolved weights failed validation: {files[0]}")
    return files[0]


def resolve_repo_file(
    hf_repo: str,
    filename: str,
    *,
    subdir: str = "",
    local_repo_dir: str | Path | None = None,
    status: Callable[[str], None] = print,
    revision: str | None = None,
    validator: Callable[[Path], bool] | None = None,
    force_download: bool = False,
) -> Path:
    """Resolve one arbitrary file from a Hugging Face repository.

    Unlike :func:`resolve_weight_file`, *filename* may be a nested JSON,
    tokenizer, or other non-safetensors asset. Files retain their repository
    layout below ``<models>/<subdir>/<org>/<repo>`` so upstream libraries can
    consume a normal local snapshot while still honoring the shared model root.
    ``local_repo_dir`` lets a caller reuse an already-resolved snapshot for a
    canonical upstream repo id.
    """

    if not hf_repo or not filename:
        raise ValueError("hf_repo and filename must be non-empty")

    if local_repo_dir is not None:
        repo_dir = Path(local_repo_dir).expanduser()
    else:
        source_path = Path(hf_repo).expanduser()
        if source_path.is_absolute() and source_path.is_dir():
            repo_dir = source_path
        else:
            repo_dir = _base_models_dir()
            if subdir:
                repo_dir = repo_dir / subdir
            repo_dir = repo_dir / hf_repo

    target = repo_dir / filename
    if target.is_file() and not force_download and (validator is None or validator(target)):
        status(f"local repo file found at {target}")
        return target
    if target.is_file() and validator is not None and not validator(target):
        status(f"local repo file failed validation; re-downloading {target}")
        force_download = True

    source_path = Path(hf_repo).expanduser()
    if source_path.is_absolute() and source_path.is_dir():
        raise FileNotFoundError(f"Missing {filename} in local repository {source_path}")

    repo_dir.mkdir(parents=True, exist_ok=True)
    status(f"downloading {filename} from {hf_repo}")
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            hf_repo,
            filename,
            local_dir=str(repo_dir),
            revision=revision,
            force_download=force_download,
            tqdm_class=_make_log_tqdm(status, f"download {filename}"),
        )
    )
    if validator is not None and not validator(downloaded):
        raise OSError(f"Downloaded repository file failed validation: {downloaded}")
    status(f"download complete: {downloaded}")
    return downloaded


def list_safetensors(directory) -> list[Path]:
    """Sorted ``*.safetensors`` files directly inside *directory*."""
    directory = Path(directory)
    return sorted(directory.glob(f"*{_SAFE_EXT}")) if directory.is_dir() else []


def discover_model_dirs(
    *,
    marker_file: str,
    patterns: Sequence[str],
    predicate: Callable[[dict], bool],
    roots: Sequence[Path] | None = None,
    exclude_parts: Sequence[str] = (),
) -> list[str]:
    """Return portable model ids whose marker JSON matches *predicate*.

    The helper intentionally knows nothing about VLM, Flux, Krea, or other model
    families. Callers supply the marker filename, bounded glob patterns, and the
    JSON predicate that defines compatibility for their own loader.
    """
    search_roots = list(roots) if roots is not None else configured_model_roots()
    if not search_roots:
        return []

    ids: set[str] = set()
    excluded = set(exclude_parts)
    seen_markers: set[str] = set()
    for root in search_roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for pattern in patterns:
            for marker in root.glob(pattern):
                if marker.name != marker_file or not marker.is_file():
                    continue
                if excluded.intersection(marker.parts):
                    continue
                try:
                    key = str(marker.resolve())
                except Exception:
                    key = str(marker)
                if key in seen_markers:
                    continue
                seen_markers.add(key)
                try:
                    data = json.loads(marker.read_text())
                except Exception:
                    continue
                if not predicate(data):
                    continue
                try:
                    ids.add(marker.parent.relative_to(root).as_posix())
                except ValueError:
                    ids.add(str(marker.parent))
    return sorted(ids)


def model_dropdown_choices(
    local_choices: Sequence[str],
    remote_choices: Sequence[str] = (),
    *,
    custom_choice: str = CUSTOM_MODEL_CHOICE,
    default_contains: str | None = None,
) -> tuple[list[str], str]:
    """Merge discovered local ids, curated remotes, and a custom sentinel."""
    local = list(dict.fromkeys(str(choice) for choice in local_choices if str(choice)))
    localset = set(local)
    remote = [str(choice) for choice in remote_choices if str(choice) and str(choice) not in localset]
    choices = local + remote
    if custom_choice not in choices:
        choices.append(custom_choice)

    default = choices[0]
    if default_contains:
        needle = default_contains.lower()
        default = next((choice for choice in choices if needle in choice.lower()), default)
    return choices, default


def resolve_choice_or_custom(choice: str, custom_value: str, *, custom_choice: str = CUSTOM_MODEL_CHOICE) -> str:
    """Resolve a dropdown choice plus a custom text field into one model id."""
    selected = (choice or "").strip()
    custom = (custom_value or "").strip()
    if selected == custom_choice:
        return custom
    return selected or custom
