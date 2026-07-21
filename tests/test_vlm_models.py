from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from comfyui_mlx_helpers.model_resolve import CUSTOM_MODEL_CHOICE
from comfyui_mlx_helpers.vlm_models import discover_vlm_models, resolve_vlm_choice, vlm_model_dropdown


class VLMModelDiscoveryTests(unittest.TestCase):
    def test_discovery_filters_incompatible_and_nested_component_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good = root / "mlx-community" / "gemma-4-e4b-it-mxfp8"
            excluded_type = root / "mlx-community" / "old-llava"
            nested = root / "pipeline" / "text_encoder"
            fast = root / "FastVLM" / "model"
            for path in (good, excluded_type, nested, fast):
                path.mkdir(parents=True)
            (good / "config.json").write_text('{"vision_config": {}}')
            (excluded_type / "config.json").write_text(
                '{"model_type": "llava_qwen2", "vision_config": {}}'
            )
            (nested / "config.json").write_text('{"vision_config": {}}')
            (fast / "config.json").write_text('{"vision_config": {}}')

            found = discover_vlm_models(roots=[root])

        self.assertEqual(found, ["mlx-community/gemma-4-e4b-it-mxfp8"])

    def test_dropdown_keeps_local_choices_first_and_selects_requested_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local" / "gemma-4"
            local.mkdir(parents=True)
            (local / "config.json").write_text('{"vision_config": {}}')

            choices, default = vlm_model_dropdown(
                roots=[root],
                remote_choices=("remote/gemma-4", "remote/other"),
                default_contains="local/gemma-4",
            )

        self.assertEqual(choices[0], "local/gemma-4")
        self.assertEqual(default, "local/gemma-4")
        self.assertEqual(choices[-1], CUSTOM_MODEL_CHOICE)

    def test_empty_custom_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Select a VLM"):
            resolve_vlm_choice(CUSTOM_MODEL_CHOICE, "")


if __name__ == "__main__":
    unittest.main()
