from __future__ import annotations

import unittest

from comfyui_mlx_helpers.output_tracing import (
    mark_traced_inputs_lazy,
    parse_partial_execution_targets,
    required_inputs_for_node,
    trace_requested_outputs,
    validate_output_dependencies,
)


class _ImageSource:
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _PathSource:
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _TracedPassThrough:
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", ["exr", "png"])
    OUTPUT_INPUT_DEPENDENCIES = {
        0: ("images", "full_path"),
        1: ("full_path",),
        2: ("full_path",),
        3: ("full_path",),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "full_path": ("STRING", {"default": "render.exr"}),
            }
        }


class _Preview:
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}


class _UnknownMultiOutput:
    RETURN_TYPES = ("IMAGE", "STRING")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "text": ("STRING",)}}


MAPPINGS = {
    "ImageSource": _ImageSource,
    "PathSource": _PathSource,
    "TracedPassThrough": _TracedPassThrough,
    "Preview": _Preview,
    "UnknownMultiOutput": _UnknownMultiOutput,
}


class OutputTracingTests(unittest.TestCase):
    def test_partial_execution_targets_are_decoded_conservatively(self):
        self.assertEqual(parse_partial_execution_targets('["preview", 12]'), ("preview", 12))
        self.assertEqual(parse_partial_execution_targets([]), ())
        self.assertIsNone(parse_partial_execution_targets(None))
        self.assertIsNone(parse_partial_execution_targets("not json"))
        self.assertIsNone(parse_partial_execution_targets('{"preview": true}'))
        self.assertIsNone(parse_partial_execution_targets('["preview", null]'))

    def test_metadata_output_prunes_unrelated_image_branch(self):
        prompt = {
            "image": {"class_type": "ImageSource", "inputs": {}},
            "path": {"class_type": "PathSource", "inputs": {}},
            "sidecar": {
                "class_type": "TracedPassThrough",
                "inputs": {"images": ["image", 0], "full_path": ["path", 0]},
            },
            "preview": {"class_type": "Preview", "inputs": {"value": ["sidecar", 3]}},
        }

        traced = trace_requested_outputs(prompt, class_mappings=MAPPINGS)

        self.assertEqual(traced["sidecar"], frozenset({3}))
        self.assertEqual(traced["path"], frozenset({0}))
        self.assertNotIn("image", traced)
        self.assertEqual(
            required_inputs_for_node(
                prompt,
                "sidecar",
                _TracedPassThrough,
                class_mappings=MAPPINGS,
            ),
            ["full_path"],
        )

    def test_image_output_requests_image_and_path(self):
        prompt = {
            "image": {"class_type": "ImageSource", "inputs": {}},
            "path": {"class_type": "PathSource", "inputs": {}},
            "sidecar": {
                "class_type": "TracedPassThrough",
                "inputs": {"images": ["image", 0], "full_path": ["path", 0]},
            },
            "preview": {"class_type": "Preview", "inputs": {"value": ["sidecar", 0]}},
        }

        traced = trace_requested_outputs(prompt, class_mappings=MAPPINGS)

        self.assertEqual(traced["sidecar"], frozenset({0}))
        self.assertEqual(traced["image"], frozenset({0}))
        self.assertEqual(traced["path"], frozenset({0}))

    def test_partial_preview_ignores_other_output_roots_in_submitted_prompt(self):
        prompt = {
            "image": {"class_type": "ImageSource", "inputs": {}},
            "path": {"class_type": "PathSource", "inputs": {}},
            "sidecar": {
                "class_type": "TracedPassThrough",
                "inputs": {"images": ["image", 0], "full_path": ["path", 0]},
            },
            "preview": {"class_type": "Preview", "inputs": {"value": ["sidecar", 3]}},
            "saver": {"class_type": "Preview", "inputs": {"value": ["sidecar", 0]}},
        }

        conservative = trace_requested_outputs(prompt, class_mappings=MAPPINGS)
        preview_only = trace_requested_outputs(
            prompt,
            class_mappings=MAPPINGS,
            output_node_ids=parse_partial_execution_targets('["preview"]'),
        )

        self.assertEqual(conservative["sidecar"], frozenset({0, 3}))
        self.assertIn("image", conservative)
        self.assertEqual(preview_only["sidecar"], frozenset({3}))
        self.assertNotIn("image", preview_only)
        self.assertEqual(
            required_inputs_for_node(
                prompt,
                "sidecar",
                _TracedPassThrough,
                class_mappings=MAPPINGS,
                output_node_ids=parse_partial_execution_targets('["preview"]'),
            ),
            ["full_path"],
        )

    def test_unknown_nodes_conservatively_trace_every_input(self):
        prompt = {
            "image": {"class_type": "ImageSource", "inputs": {}},
            "path": {"class_type": "PathSource", "inputs": {}},
            "unknown": {
                "class_type": "UnknownMultiOutput",
                "inputs": {"images": ["image", 0], "text": ["path", 0]},
            },
            "preview": {"class_type": "Preview", "inputs": {"value": ["unknown", 1]}},
        }

        traced = trace_requested_outputs(prompt, class_mappings=MAPPINGS)

        self.assertIn("image", traced)
        self.assertIn("path", traced)

    def test_dependency_declaration_must_cover_every_output(self):
        with self.assertRaisesRegex(ValueError, r"missing=\[2, 3\]"):
            validate_output_dependencies(
                _TracedPassThrough,
                {0: ("images",), 1: ("full_path",)},
            )

    def test_lazy_marking_preserves_specs_and_marks_declared_inputs(self):
        result = mark_traced_inputs_lazy(
            _TracedPassThrough.INPUT_TYPES(),
            _TracedPassThrough.OUTPUT_INPUT_DEPENDENCIES,
        )

        self.assertTrue(result["required"]["images"][1]["lazy"])
        self.assertTrue(result["required"]["full_path"][1]["lazy"])
        self.assertEqual(result["required"]["full_path"][1]["default"], "render.exr")
        self.assertNotIn("lazy", _TracedPassThrough.INPUT_TYPES()["required"]["full_path"][1])


if __name__ == "__main__":
    unittest.main()
