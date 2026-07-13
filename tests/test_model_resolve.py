from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from comfyui_mlx_helpers.model_resolve import resolve_model_dir, resolve_weight_file


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


if __name__ == "__main__":
    unittest.main()
