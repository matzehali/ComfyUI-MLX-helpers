from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from comfyui_mlx_helpers.mlx_runtime import (
    ShardedSafetensorIndex,
    clear_compiled_callables,
    get_compiled_callable,
    load_module_sharded,
    materialize,
    mx_video_frames_to_torch,
)


class _Owner:
    pass


class TensorConversionTests(unittest.TestCase):
    def test_video_conversion_supports_both_common_vae_ranges(self):
        minus_one_one = mx_video_frames_to_torch(mx.zeros((1, 3, 1, 2, 2)))
        zero_one = mx_video_frames_to_torch(
            mx.full((1, 3, 1, 2, 2), 0.25), value_range="zero_one"
        )
        self.assertEqual(tuple(minus_one_one.shape), (1, 2, 2, 3))
        self.assertAlmostEqual(float(minus_one_one.mean()), 0.5)
        self.assertAlmostEqual(float(zero_one.mean()), 0.25)

    def test_video_conversion_rejects_unknown_range(self):
        with self.assertRaisesRegex(ValueError, "value_range"):
            mx_video_frames_to_torch(mx.zeros((1, 3, 1, 2, 2)), value_range="raw")


class CompiledCallableTests(unittest.TestCase):
    def test_same_owner_and_method_reuses_wrapper(self):
        owner = _Owner()

        def forward(value):
            return value + 1

        with patch("mlx.core.compile", wraps=mx.compile) as compile_mock:
            first = get_compiled_callable(owner, "forward", forward, "test forward")
            second = get_compiled_callable(owner, "forward", forward, "test forward")
            result = second(mx.array([2.0]))
            mx.eval(result)

        self.assertIs(first, second)
        self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(float(result.item()), 3.0)

    def test_execution_failure_disables_the_retained_wrapper(self):
        owner = _Owner()
        calls = {"compiled": 0, "raw": 0}

        def raw(value):
            calls["raw"] += 1
            return value + 1

        def fake_compile(_fn):
            def failing(value):
                del value
                calls["compiled"] += 1
                raise RuntimeError("trace failed")

            return failing

        with patch("mlx.core.compile", side_effect=fake_compile):
            wrapped = get_compiled_callable(owner, "forward", raw, "test forward")
            first = wrapped(mx.array([1.0]))
            second = wrapped(mx.array([2.0]))
            mx.eval(first, second)

        self.assertEqual(calls["compiled"], 1)
        self.assertEqual(calls["raw"], 2)
        self.assertEqual(float(first.item()), 2.0)
        self.assertEqual(float(second.item()), 3.0)

    def test_clear_discards_owner_cache_before_weight_mutation(self):
        owner = _Owner()

        def forward(value):
            return value

        first = get_compiled_callable(owner, "forward", forward)
        clear_compiled_callables(owner)
        second = get_compiled_callable(owner, "forward", forward)

        self.assertIsNot(first, second)


class ShardedSafetensorTests(unittest.TestCase):
    def test_index_discovers_and_loads_each_shard(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(2, 2, bias=False)
                self.second = nn.Linear(2, 1, bias=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mx.save_safetensors(str(root / "part-1.safetensors"), {"first.weight": mx.ones((2, 2))})
            mx.save_safetensors(str(root / "part-2.safetensors"), {"second.weight": mx.full((1, 2), 2.0)})
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 24},
                        "weight_map": {
                            "first.weight": "part-1.safetensors",
                            "second.weight": "part-2.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            index = ShardedSafetensorIndex.discover(root)
            self.assertEqual(index.shard_names, ("part-1.safetensors", "part-2.safetensors"))
            model = Tiny()
            progress = []
            report = load_module_sharded(
                model,
                root,
                status=lambda _message: None,
                progress=lambda done, total: progress.append((done, total)),
            )
            materialize(model.first.weight, model.second.weight)

        self.assertEqual(report.shard_count, 2)
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertEqual(report.missing_keys, ())
        self.assertTrue(mx.array_equal(model.first.weight, mx.ones((2, 2))).item())
        self.assertTrue(mx.array_equal(model.second.weight, mx.full((1, 2), 2.0)).item())

    def test_strict_loader_rejects_missing_module_key(self):
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mx.save_safetensors(str(root / "model.safetensors"), {"weight": mx.ones((2, 2))})
            with self.assertRaisesRegex(ValueError, "1 missing"):
                load_module_sharded(model, root, status=lambda _message: None)


if __name__ == "__main__":
    unittest.main()
