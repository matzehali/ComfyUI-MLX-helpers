"""MLX runtime helpers: memory cleanup, compile caching, weight loading, and
ComfyUI <-> MLX tensor conversion.

Everything imports ``mlx`` / ``torch`` lazily so the package stays importable on
non-Apple machines (e.g. for `python -m py_compile` checks in CI).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

# ComfyUI dtype label -> resolver. Kept as strings so the table is importable
# without MLX; resolved against ``mlx.core`` on demand.
_DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
PRECISIONS = ["keep", "fp16", "fp32", "bf16"]


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
        try:
            compiled_fn = mx.compile(fn)
        except Exception as exc:
            print(f"[MLX Compile] disabled {label or cache_name}: {exc}")
            cache[key] = fn
            return fn

        def compiled(*args, **kwargs):
            try:
                return compiled_fn(*args, **kwargs)
            except Exception as exc:
                print(f"[MLX Compile] disabled {label or cache_name} after failure: {exc}")
                cache[key] = fn
                return fn(*args, **kwargs)

        cache[key] = compiled
    return compiled


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
