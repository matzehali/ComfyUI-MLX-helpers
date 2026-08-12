"""MLX runtime helpers: memory cleanup, compile caching, weight loading, and
ComfyUI <-> MLX tensor conversion.

Everything imports ``mlx`` / ``torch`` lazily so the package stays importable on
non-Apple machines (e.g. for `python -m py_compile` checks in CI).
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# ComfyUI dtype label -> resolver. Kept as strings so the table is importable
# without MLX; resolved against ``mlx.core`` on demand.
_DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
PRECISIONS = ["keep", "fp16", "fp32", "bf16"]


def materialize(*arrays) -> None:
    """Force MLX to evaluate one or more lazy arrays."""
    import mlx.core as mx

    mx.eval(*arrays)  # noqa: S307 - MLX graph evaluation, not Python eval


def mx_dtype(precision: str):
    """Map a precision label (``fp16``/``fp32``/``bf16``) to an mlx dtype, or None."""
    import mlx.core as mx

    name = _DTYPES.get(precision)
    return getattr(mx, name) if name else None


def aggressive_cleanup() -> None:
    """Free the Metal GPU cache and run a Python GC pass."""
    try:
        import mlx.core as mx

        # mx.metal.clear_cache() is deprecated in MLX >= 0.31.
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def get_compiled_callable(owner, cache_name: str, fn, label: str = ""):
    """Return a cached ``mx.compile`` wrapper for a loaded component method.

    Falls back transparently to the uncompiled callable if compilation isn't
    available or fails, and remembers that decision so it isn't retried.
    """
    try:
        import mlx.core as mx
    except Exception:
        return fn
    if not hasattr(mx, "compile"):
        return fn

    cache = getattr(owner, "_mlxhelpers_compiled", None)
    if cache is None:
        cache = {}
        setattr(owner, "_mlxhelpers_compiled", cache)
    # Bound methods are recreated on access; key on the instance id so the cache
    # follows the currently loaded component.
    target = getattr(fn, "__self__", fn)
    key = (cache_name, id(target))
    compiled = cache.get(key)
    if compiled is None:
        print(f"[MLX Compile] compiling {label or cache_name} ...")
        try:
            compiled_fn = mx.compile(fn)
        except Exception as exc:
            print(f"[MLX Compile] disabled {label or cache_name}: {exc}")
            cache[key] = fn
            return fn

        disabled = False

        def compiled(*args, **kwargs):
            nonlocal disabled
            if disabled:
                return fn(*args, **kwargs)
            try:
                return compiled_fn(*args, **kwargs)
            except Exception as exc:
                disabled = True
                print(f"[MLX Compile] disabled {label or cache_name} after failure: {exc}")
                cache[key] = fn
                return fn(*args, **kwargs)

        cache[key] = compiled
    return compiled


def clear_compiled_callables(owner) -> None:
    """Discard wrappers retained by :func:`get_compiled_callable` on *owner*.

    Call this before or immediately after mutating/replacing the owner's weights.
    Existing external references to a wrapper cannot be revoked, so model code
    must also stop using those references across the mutation boundary.
    """
    cache = getattr(owner, "_mlxhelpers_compiled", None)
    if cache is not None:
        cache.clear()


def load_safetensors(path, *, dtype=None, status: Callable[[str], None] = print) -> dict:
    """Load one ``.safetensors`` file (or every one in a directory) into a dict
    of MLX arrays, optionally cast to *dtype* (an mlx dtype or precision label).
    """
    import mlx.core as mx

    if isinstance(dtype, str):
        dtype = mx_dtype(dtype)

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors found at {path}")

    weights: dict = {}
    for f in files:
        status(f"loading weights {f.name}")
        part = mx.load(str(f))
        if dtype is not None:
            part = {k: v.astype(dtype) for k, v in part.items()}
        weights.update(part)
    return weights


@dataclass(frozen=True)
class ShardedSafetensorIndex:
    """A Hugging Face safetensor component index without loaded tensor data."""

    root: Path
    weight_map: dict[str, str]
    total_size: int | None = None

    @classmethod
    def discover(cls, path) -> "ShardedSafetensorIndex":
        """Read the single ``*.safetensors.index.json`` below *path*.

        Passing the index file itself is supported. A component with one or
        more unindexed safetensor files is also accepted; their keys are read
        one file at a time to construct the in-memory key map.
        """
        import mlx.core as mx

        path = Path(path)
        if path.is_file() and path.name.endswith(".safetensors.index.json"):
            index_files = [path]
            root = path.parent
        else:
            root = path
            index_files = sorted(root.glob("*.safetensors.index.json"))

        if len(index_files) > 1:
            names = ", ".join(item.name for item in index_files)
            raise ValueError(f"Expected one safetensor index in {root}, found: {names}")
        if index_files:
            payload = json.loads(index_files[0].read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid or empty weight_map in {index_files[0]}")
            total_size = payload.get("metadata", {}).get("total_size")
            return cls(root=root, weight_map=dict(weight_map), total_size=total_size)

        files = sorted(root.glob("*.safetensors")) if root.is_dir() else []
        if path.is_file() and path.suffix == ".safetensors":
            root, files = path.parent, [path]
        if not files:
            raise FileNotFoundError(f"No .safetensors or index found at {path}")

        weight_map: dict[str, str] = {}
        for file in files:
            part = mx.load(str(file))
            duplicate = set(weight_map).intersection(part)
            if duplicate:
                raise ValueError(f"Duplicate tensor keys in {file.name}: {sorted(duplicate)[:5]}")
            weight_map.update({key: file.name for key in part})
            del part
        return cls(root=root, weight_map=weight_map)

    @property
    def shard_names(self) -> tuple[str, ...]:
        """Shard filenames in deterministic index order."""
        return tuple(dict.fromkeys(self.weight_map.values()))

    def iter_shards(self) -> Iterator[tuple[Path, tuple[str, ...]]]:
        """Yield each shard path and the keys assigned to it."""
        for name in self.shard_names:
            keys = tuple(key for key, shard in self.weight_map.items() if shard == name)
            yield self.root / name, keys

    def validate_files(self) -> None:
        """Raise if any shard referenced by the index is missing."""
        missing = [str(path) for path, _keys in self.iter_shards() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing safetensor shards:\n" + "\n".join(missing))


@dataclass(frozen=True)
class ShardedLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shard_count: int


def load_module_sharded(
    module,
    path,
    *,
    dtype=None,
    key_transform: Callable[[str], str | None] | None = None,
    tensor_transform: Callable[[str, object], object] | None = None,
    strict: bool = True,
    materialize_weights: bool = True,
    status: Callable[[str], None] = print,
) -> ShardedLoadReport:
    """Load indexed safetensors into an ``mlx.nn.Module`` one shard at a time.

    ``key_transform`` may rename a checkpoint key or return ``None`` to skip it.
    ``tensor_transform`` runs after optional dtype conversion. The function
    never retains a second full-model weight dictionary in Python memory.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if isinstance(dtype, str):
        dtype = mx_dtype(dtype)

    index = ShardedSafetensorIndex.discover(path)
    index.validate_files()
    expected = {key for key, _value in tree_flatten(module.parameters())}
    loaded: set[str] = set()
    unexpected: set[str] = set()

    for shard_number, (shard_path, indexed_keys) in enumerate(index.iter_shards(), start=1):
        status(f"loading shard {shard_number}/{len(index.shard_names)}: {shard_path.name}")
        part = mx.load(str(shard_path))
        absent = set(indexed_keys).difference(part)
        if absent:
            raise ValueError(f"Index keys missing from {shard_path.name}: {sorted(absent)[:5]}")

        transformed: list[tuple[str, object]] = []
        for source_key in indexed_keys:
            target_key = key_transform(source_key) if key_transform else source_key
            if target_key is None:
                continue
            value = part[source_key]
            if dtype is not None and getattr(value, "dtype", None) != dtype:
                value = value.astype(dtype)
            if tensor_transform is not None:
                value = tensor_transform(target_key, value)
            if target_key not in expected:
                unexpected.add(target_key)
                continue
            transformed.append((target_key, value))
            loaded.add(target_key)

        if transformed:
            module.load_weights(transformed, strict=False)
            if materialize_weights:
                mx.eval(*(value for _key, value in transformed))
        del transformed, part
        aggressive_cleanup()

    missing = expected.difference(loaded)
    report = ShardedLoadReport(
        loaded_keys=tuple(sorted(loaded)),
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=tuple(sorted(unexpected)),
        shard_count=len(index.shard_names),
    )
    if strict and (missing or unexpected):
        raise ValueError(
            "Safetensor/module key mismatch: "
            f"{len(missing)} missing, {len(unexpected)} unexpected; "
            f"missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
        )
    return report


def torch_image_to_mx(image, batch_idx: int | None = None):
    """ComfyUI IMAGE tensor (B,H,W,C float32 [0,1]) -> mlx array.

    With *batch_idx* returns a single ``[H,W,C]`` frame; otherwise the full batch.
    """
    import mlx.core as mx
    import numpy as np

    arr = image.detach().cpu().numpy().astype(np.float32)
    if batch_idx is not None:
        arr = arr[batch_idx]
    return mx.array(arr)


def mx_to_torch(array):
    """mlx array -> torch tensor (via numpy, no copy where possible)."""
    import numpy as np
    import torch

    return torch.from_numpy(np.array(array, copy=False))


def mx_video_frames_to_torch(frames):
    """Convert ``[B,C,F,H,W]`` MLX video in ``[-1,1]`` to Comfy IMAGE."""
    import mlx.core as mx
    import numpy as np
    import torch

    value = frames if frames.dtype == mx.float32 else frames.astype(mx.float32)
    value = (mx.clip(value, -1.0, 1.0) + 1.0) * 0.5
    frames_np = np.array(value)[0]
    frames_np = np.transpose(frames_np, (1, 2, 3, 0))
    return torch.from_numpy(np.ascontiguousarray(frames_np))


def mx_audio_to_torch(waveform, sample_rate: int = 48000) -> dict:
    """Convert an MLX ``[B,C,T]`` waveform to a Comfy AUDIO dictionary."""
    import mlx.core as mx
    import numpy as np
    import torch

    value = waveform if waveform.dtype == mx.float32 else waveform.astype(mx.float32)
    return {
        "waveform": torch.from_numpy(np.array(value)),
        "sample_rate": sample_rate,
    }


def torch_image_to_pil(image, batch_idx: int = 0):
    """ComfyUI IMAGE tensor (B,H,W,C float32 [0,1]) -> PIL.Image at *batch_idx*."""
    import numpy as np
    from PIL import Image

    arr = image[batch_idx].detach().cpu().numpy()
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8"))


class AnyType(str):
    """ComfyUI wildcard socket type for dependency-only / pass-through inputs."""

    def __ne__(self, _other: object) -> bool:
        return False


ANY_TYPE = AnyType("*")
