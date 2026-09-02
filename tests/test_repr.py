import numpy as np

from autotab.param import CON_WIN_SIZE, CQT_N_BINS, NUM_CLASSES
from autotab.TabDataReprGen import TabDataReprGen


def test_windows_shape_and_dtype(sine_wav):
    x = TabDataReprGen().load_rep_from_raw_file(sine_wav)
    assert x.ndim == 4
    assert x.shape[1:] == (CQT_N_BINS, CON_WIN_SIZE, 1)
    assert x.dtype == np.float32
    assert x.shape[0] == 1 + 22050 // 512


def test_stereo_is_downmixed(tmp_path):
    import soundfile as sf

    sr = 44100
    y = np.zeros((sr, 2), dtype="float32")
    y[:, 0] = 0.3
    path = tmp_path / "stereo.wav"
    sf.write(path, y, sr)
    x = TabDataReprGen().load_rep_from_raw_file(path)
    assert x.shape[0] == 1 + 22050 // 512  # resampled to 22050


def test_clean_labels_one_hot():
    g = TabDataReprGen()
    labels = g.clean_labels(np.array([[-1, 0, 3, 19, 20, -5]]))
    assert labels.shape == (1, 6, NUM_CLASSES)
    assert labels[0].argmax(-1).tolist() == [0, 1, 4, 0, 0, 0]


def test_invalid_mode():
    import pytest

    with pytest.raises(ValueError):
        TabDataReprGen(mode="x")
