import numpy as np

from comfyui_mlx_helpers.video_normalize import resample_video_frames


def test_resample_25_to_24_preserves_plate_duration():
    frames = np.arange(111, dtype=np.int64)[:, None]
    output = resample_video_frames(frames, 25.0, 24.0)
    assert output.shape == (107, 1)
    assert output[0, 0] == 0
    assert output[-1, 0] == 110
    assert abs(output.shape[0] / 24.0 - frames.shape[0] / 25.0) < 1 / 24.0


def test_resample_same_rate_preserves_object():
    frames = np.zeros((5, 2, 2, 3), dtype=np.float32)
    assert resample_video_frames(frames, 24.0, 24.0) is frames


def test_resample_rejects_invalid_rates_and_empty_batches():
    import pytest

    with pytest.raises(ValueError, match="positive"):
        resample_video_frames(np.zeros((1, 1)), 0, 24)
    with pytest.raises(ValueError, match="at least one"):
        resample_video_frames(np.zeros((0, 1)), 25, 24)
