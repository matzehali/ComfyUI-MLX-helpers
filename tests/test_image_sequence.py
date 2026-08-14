from __future__ import annotations

import numpy as np
import pytest

from comfyui_mlx_helpers.image_sequence import linear_srgb_to_display, resolve_frame_sequence


def test_resolve_frame_sequence_preserves_arbitrary_padding(tmp_path):
    for frame in (1, 3, 5):
        (tmp_path / f"plate.{frame:06d}.exr").touch()

    paths = resolve_frame_sequence(str(tmp_path / "plate.######.exr"), 1, 5, 2)

    assert [path.name for path in paths] == [
        "plate.000001.exr",
        "plate.000003.exr",
        "plate.000005.exr",
    ]


def test_resolve_frame_sequence_reports_missing_frames(tmp_path):
    (tmp_path / "plate.0001.exr").touch()

    with pytest.raises(FileNotFoundError, match="Missing 1 image sequence frame"):
        resolve_frame_sequence(str(tmp_path / "plate.####.exr"), 1, 2)


def test_resolve_frame_sequence_validates_pattern_and_range(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_frame_sequence(str(tmp_path / "plate.exr"), 1, 1)
    with pytest.raises(ValueError, match="greater than or equal"):
        resolve_frame_sequence(str(tmp_path / "plate.####.exr"), 2, 1)
    with pytest.raises(ValueError, match="positive"):
        resolve_frame_sequence(str(tmp_path / "plate.####.exr"), 1, 2, 0)


def test_linear_srgb_to_display_boundary_values():
    values = np.array([0.0, 0.0031308, 1.0], dtype=np.float32)
    converted = linear_srgb_to_display(values)

    np.testing.assert_allclose(converted, [0.0, 0.04044994, 1.0], rtol=0, atol=2e-7)
