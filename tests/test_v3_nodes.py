from __future__ import annotations

import unittest
from types import SimpleNamespace

from comfyui_mlx_helpers.v3_nodes import adapt_v1_node, adapt_v1_nodes, v3_nodes_available
from comfyui_mlx_helpers.output_tracing import PARTIAL_EXECUTION_TARGETS_INPUT


class _LegacyNode:
    CATEGORY = "MLX/Test"
    DESCRIPTION = "Legacy implementation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "CUSTOM_RESULT")
    RETURN_NAMES = ("image", "custom")
    OUTPUT_IS_LIST = (False, True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 1}),
                "mode": (("fast", "exact"), {"default": "exact"}),
            },
            "optional": {
                "mlx_node_help": (
                    "STRING",
                    {"default": "Help", "multiline": True, "readonly": True},
                ),
            },
        }

    def run(self, count, mode, mlx_node_help="Help"):
        return (f"{mode}:{count}", [count])


class _LegacyListNode:
    CATEGORY = "MLX/Test"
    FUNCTION = "run"
    RETURN_TYPES = ("INT",)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"values": ("INT", {"default": 1})}}

    def run(self, values):
        return (sum(values),)


class _LegacyScalarNode:
    CATEGORY = "MLX/Test"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT", "IMAGE")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": ""})}}

    def run(self, text):
        return (text, len(text), "image-placeholder")


class _LegacyComboOutputNode:
    CATEGORY = "MLX/Test"
    FUNCTION = "run"
    RETURN_TYPES = (["exr", "png", "jpg"],)
    RETURN_NAMES = ("file_type",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_type": (("exr", "png", "jpg"),)}}

    def run(self, file_type):
        return (file_type,)


class _LegacyLazyNode:
    CATEGORY = "MLX/Test"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": ("STRING", {"lazy": True})},
            "hidden": {
                "prompt_data": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def check_lazy_status(
        self,
        value=None,
        prompt_data=None,
        unique_id=None,
        **kwargs,
    ):
        if prompt_data == {"node": unique_id}:
            return ["value"]
        return []

    def run(self, value, prompt_data=None, unique_id=None):
        return (value,)


class _LegacyPartialTargetNode:
    CATEGORY = "MLX/Test"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": ("STRING", {"lazy": True})},
            "hidden": {
                "prompt_data": "PROMPT",
                "unique_id": "UNIQUE_ID",
                PARTIAL_EXECUTION_TARGETS_INPUT: "STRING",
            },
        }

    def check_lazy_status(self, _mlx_partial_execution_targets=None, **kwargs):
        return list(_mlx_partial_execution_targets or ())

    def run(self, value, **kwargs):
        return (value,)


@unittest.skipUnless(v3_nodes_available(), "ComfyUI V3 API is not importable")
class V3NodeAdapterTests(unittest.TestCase):
    def test_schema_keeps_serialized_contract_and_v3_identity(self):
        from comfy_api.latest import io

        node = adapt_v1_node(
            "ExampleMLXNode",
            _LegacyNode,
            display_name="Example MLX",
            sync_widget_inputs=True,
        )
        self.assertTrue(issubclass(node, io.ComfyNode))
        self.assertTrue(node.V3_ADAPTER)

        schema = node.GET_SCHEMA()
        self.assertEqual(schema.node_id, "ExampleMLXNode")
        self.assertEqual(schema.display_name, "Example MLX")
        self.assertEqual(schema.category, _LegacyNode.CATEGORY)
        self.assertEqual([item.id for item in schema.inputs], ["count", "mode", "mlx_node_help"])
        self.assertEqual(node.RETURN_TYPES, ["IMAGE", "CUSTOM_RESULT"])
        self.assertEqual(node.RETURN_NAMES, ["image", "custom"])
        self.assertEqual(node.OUTPUT_IS_LIST, [False, True])
        self.assertFalse(node.OUTPUT_NODE)
        self.assertTrue(node.HAS_INTERMEDIATE_OUTPUT)

    def test_execution_delegates_and_preserves_widget_ui_payload(self):
        node = adapt_v1_node("ExampleMLXNode", _LegacyNode, sync_widget_inputs=True)
        output = node.execute(count=3, mode="fast")
        self.assertEqual(output.result, ("fast:3", [3]))
        self.assertEqual(
            output.ui["mlx_resolved_inputs"],
            [{"count": [3], "mode": ["fast"]}],
        )
        self.assertEqual(output.ui["mlx_resolved_outputs"], [[None, None]])

    def test_input_is_list_is_kept(self):
        node = adapt_v1_node("ListMLXNode", _LegacyListNode)
        self.assertTrue(node.INPUT_IS_LIST)
        self.assertEqual(node.execute(values=[1, 2, 3]).result, (6,))

    def test_scalar_outputs_are_available_to_downstream_widget_sync(self):
        node = adapt_v1_node("ScalarMLXNode", _LegacyScalarNode, sync_widget_inputs=True)
        output = node.execute(text="hello")
        self.assertEqual(output.ui["mlx_resolved_outputs"], [["hello", 5, None]])

    def test_legacy_combo_output_becomes_a_v3_combo(self):
        node = adapt_v1_node("ComboOutputNode", _LegacyComboOutputNode)
        schema = node.GET_SCHEMA()
        self.assertEqual(node.RETURN_TYPES, ["COMBO"])
        self.assertEqual(schema.outputs[0].id, "output_0")
        self.assertEqual(schema.outputs[0].display_name, "file_type")

    def test_lazy_hook_receives_v3_hidden_prompt_and_unique_id(self):
        node = adapt_v1_node("LazyMLXNode", _LegacyLazyNode)
        node.hidden = SimpleNamespace(
            prompt={"node": "lazy-7"},
            unique_id="lazy-7",
        )
        self.assertEqual(node.check_lazy_status(value=None), ["value"])

    def test_partial_targets_use_supported_v3_workflow_metadata(self):
        from comfy_api.latest import io

        node = adapt_v1_node("PartialTargetMLXNode", _LegacyPartialTargetNode)
        schema = node.GET_SCHEMA()
        self.assertIn(io.Hidden.extra_pnginfo, schema.hidden)
        node.hidden = SimpleNamespace(
            prompt={"node": "partial-1"},
            unique_id="partial-1",
            extra_pnginfo={
                "workflow": {
                    "extra": {PARTIAL_EXECUTION_TARGETS_INPUT: '["preview"]'}
                }
            },
        )
        self.assertEqual(node.check_lazy_status(value=None), ["preview"])

    def test_mapping_keeps_ids_and_old_comfy_fallback_shape(self):
        result = adapt_v1_nodes(
            {"ExampleMLXNode": _LegacyNode},
            {"ExampleMLXNode": "Example MLX"},
        )
        self.assertEqual(list(result), ["ExampleMLXNode"])
        self.assertEqual(result["ExampleMLXNode"].GET_SCHEMA().display_name, "Example MLX")


if __name__ == "__main__":
    unittest.main()
