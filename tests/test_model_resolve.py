from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comfyui_mlx_helpers.model_resolve import resolve_model_dir


class ModelResolveDownloadProgressTests(unittest.TestCase):
    def test_snapshot_download_gets_log_progress_class(self):
        lines: list[str] = []

        def fake_snapshot_download(*args, **kwargs):
            self.assertEqual(kwargs["repo_id"], "mlx-community/test-vlm")
            progress_cls = kwargs.get("tqdm_class")
            self.assertIsNotNone(progress_cls)
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


if __name__ == "__main__":
    unittest.main()
