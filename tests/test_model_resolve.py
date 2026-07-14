from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from comfyui_mlx_helpers.model_resolve import (
    CUSTOM_MODEL_CHOICE,
    discover_model_dirs,
    model_dropdown_choices,
    resolve_choice_or_custom,
    resolve_model_dir,
    resolve_repo_file,
    resolve_weight_file,
)


class ModelResolveDownloadProgressTests(unittest.TestCase):
    def test_snapshot_download_gets_log_progress_class(self):
        lines: list[str] = []

        def fake_snapshot_download(*args, **kwargs):
            self.assertEqual(kwargs["repo_id"], "mlx-community/test-vlm")
            progress_cls = kwargs.get("tqdm_class")
            self.assertIsNotNone(progress_cls)
            lock = threading.RLock()
            progress_cls.set_lock(lock)
            self.assertIs(progress_cls.get_lock(), lock)
            progress = progress_cls(total=100, initial=0, unit="B", unit_scale=True)
            progress.update(50)
            progress.update(50)
            progress.close()
            return str(Path(kwargs["local_dir"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"COMFYUI_MODELS_DIR": tmpdir}, clear=False):
                with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
                    resolved = resolve_model_dir("mlx-community/test-vlm", status=lines.append)

        self.assertEqual(resolved.name, "test-vlm")
        self.assertTrue(any("downloading mlx-community/test-vlm" in line for line in lines))
        self.assertTrue(any("[############" in line for line in lines))
        self.assertTrue(any("100.0%" in line for line in lines))
        self.assertTrue(any("download complete:" in line for line in lines))

    def test_diffusers_model_index_counts_as_installed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "org" / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "model_index.json").write_text("{}")
            with patch.dict("os.environ", {"COMFYUI_MODELS_DIR": tmpdir}, clear=False):
                with patch("huggingface_hub.snapshot_download") as download:
                    resolved = resolve_model_dir("org/model")

        self.assertEqual(resolved, model_dir)
        download.assert_not_called()

    def test_snapshot_options_are_forwarded_without_removed_hub_arguments(self):
        seen: dict = {}

        def fake_snapshot_download(*args, **kwargs):
            del args
            seen.update(kwargs)
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"COMFYUI_MODELS_DIR": tmpdir}, clear=False):
                with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
                    resolve_model_dir(
                        "org/model",
                        revision="abc123",
                        allow_patterns=["*.json", "weights/**"],
                        ignore_patterns="*.bin",
                    )

        self.assertEqual(seen["revision"], "abc123")
        self.assertEqual(seen["allow_patterns"], ["*.json", "weights/**"])
        self.assertEqual(seen["ignore_patterns"], "*.bin")
        self.assertNotIn("resume_download", seen)
        self.assertNotIn("local_dir_use_symlinks", seen)

    def test_invalid_weight_is_force_redownloaded_and_revalidated(self):
        lines: list[str] = []

        def valid(path: Path) -> bool:
            return path.read_bytes() == b"valid"

        def fake_hf_hub_download(*args, **kwargs):
            del args
            target = Path(kwargs["local_dir"]) / "nested" / "model.safetensors"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"valid")
            return str(target)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo" / "nested" / "model.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"broken")
            with patch.dict("os.environ", {"COMFYUI_MODELS_DIR": tmpdir}, clear=False):
                with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download) as download:
                    resolved = resolve_weight_file(
                        "nested/model.safetensors",
                        subdir="repo",
                        hf_repo="org/model",
                        validator=valid,
                        status=lines.append,
                    )

            self.assertEqual(resolved.read_bytes(), b"valid")

        self.assertTrue(download.call_args.kwargs["force_download"])
        self.assertTrue(any("failed validation" in line for line in lines))

    def test_repo_file_preserves_nested_layout_and_reuses_local_copy(self):
        calls = []

        def fake_hf_hub_download(repo_id, filename, **kwargs):
            calls.append((repo_id, filename, kwargs))
            target = Path(kwargs["local_dir"]) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
            return str(target)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"COMFYUI_MODELS_DIR": tmpdir}, clear=False):
                with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
                    first = resolve_repo_file(
                        "org/model",
                        "tokenizer/config.json",
                        status=lambda _line: None,
                    )
                    second = resolve_repo_file(
                        "org/model",
                        "tokenizer/config.json",
                        status=lambda _line: None,
                    )

        self.assertEqual(first, second)
        self.assertEqual(first.relative_to(tmpdir).as_posix(), "org/model/tokenizer/config.json")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ("org/model", "tokenizer/config.json"))

    def test_repo_file_can_bind_canonical_id_to_resolved_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "custom-snapshot"
            target = repo / "weights" / "model.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"model")
            with patch("huggingface_hub.hf_hub_download") as download:
                resolved = resolve_repo_file(
                    "canonical/repo",
                    "weights/model.bin",
                    local_repo_dir=repo,
                )

        self.assertEqual(resolved, target)
        download.assert_not_called()

    def test_discover_model_dirs_filters_marker_json_with_portable_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vlm = root / "mlx-community" / "gemma-3-12b-it-4bit"
            text_encoder = root / "some-model" / "text_encoder"
            ordinary = root / "ordinary" / "model"
            vlm.mkdir(parents=True)
            text_encoder.mkdir(parents=True)
            ordinary.mkdir(parents=True)
            (vlm / "config.json").write_text('{"vision_config": {}}')
            (text_encoder / "config.json").write_text('{"vision_config": {}}')
            (ordinary / "config.json").write_text('{"model_type": "llama"}')

            found = discover_model_dirs(
                marker_file="config.json",
                patterns=("*/config.json", "*/*/config.json"),
                predicate=lambda data: "vision_config" in data,
                roots=[root],
                exclude_parts=("text_encoder",),
            )

        self.assertEqual(found, ["mlx-community/gemma-3-12b-it-4bit"])

    def test_model_dropdown_choices_dedupes_remote_and_keeps_custom_last(self):
        choices, default = model_dropdown_choices(
            ["local/vlm"],
            ["local/vlm", "remote/vlm", "remote/gemma-3-12b-it-4bit"],
            default_contains="gemma-3-12b",
        )

        self.assertEqual(choices, ["local/vlm", "remote/vlm", "remote/gemma-3-12b-it-4bit", CUSTOM_MODEL_CHOICE])
        self.assertEqual(default, "remote/gemma-3-12b-it-4bit")

    def test_resolve_choice_or_custom_prefers_custom_only_for_sentinel(self):
        self.assertEqual(resolve_choice_or_custom("remote/vlm", "/tmp/local"), "remote/vlm")
        self.assertEqual(resolve_choice_or_custom(CUSTOM_MODEL_CHOICE, "/tmp/local"), "/tmp/local")
        self.assertEqual(resolve_choice_or_custom("", "/tmp/local"), "/tmp/local")


if __name__ == "__main__":
    unittest.main()
