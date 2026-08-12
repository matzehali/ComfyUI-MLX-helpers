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
)


class _Owner:
    pass


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
            report = load_module_sharded(model, root, status=lambda _message: None)
            materialize(model.first.weight, model.second.weight)

        self.assertEqual(report.shard_count, 2)
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
