import numpy as np
import pytest


@pytest.fixture
def sine_wav(tmp_path):
    """A 1-second 110 Hz (open A string) mono wav."""
    import soundfile as sf

    sr = 22050
    t = np.arange(sr) / sr
    y = 0.5 * np.sin(2 * np.pi * 110 * t).astype("float32")
    path = tmp_path / "a110.wav"
    sf.write(path, y, sr)
    return path
