from pathlib import Path
import tempfile
import unittest

from comfyui_mlx_helpers import install_output_tracing, install_widget_input_sync


class WebExtensionInstallTests(unittest.TestCase):
    def test_widget_sync_installer_copies_runtime_and_core(self):
        with tempfile.TemporaryDirectory() as directory:
            install_widget_input_sync(directory)
            copied = {path.name for path in Path(directory).iterdir()}
            self.assertEqual(
                copied,
                {
                    "mlx_widget_input_sync.js",
                    "widget_input_sync_core.js",
                    "mlx_partial_execution_targets.js",
                    "partial_execution_targets_core.js",
                },
            )

    def test_output_tracing_installer_copies_runtime_and_core(self):
        with tempfile.TemporaryDirectory() as directory:
            install_output_tracing(directory)
            copied = {path.name for path in Path(directory).iterdir()}
            self.assertEqual(
                copied,
                {
                    "mlx_partial_execution_targets.js",
                    "partial_execution_targets_core.js",
                },
            )


if __name__ == "__main__":
    unittest.main()
