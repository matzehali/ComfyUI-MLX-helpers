from __future__ import annotations

import unittest
from unittest.mock import patch

import mlx.core as mx

from comfyui_mlx_helpers.mlx_runtime import (
    clear_compiled_callables,
    get_compiled_callable,
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


if __name__ == "__main__":
    unittest.main()
