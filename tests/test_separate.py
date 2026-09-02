import os

import numpy as np
import pytest

demucs = pytest.importorskip("demucs", reason="separate extra not installed")

from autotab import separate as sep  # noqa: E402


def test_pick_device_respects_env(monkeypatch):
    monkeypatch.setenv("AUTOTAB_DEVICE", "cpu")
    assert sep.pick_device() == "cpu"
    assert sep.pick_device("mps") == "mps"


@pytest.mark.skipif(not os.environ.get("AUTOTAB_SLOW_TESTS"), reason="set AUTOTAB_SLOW_TESTS=1")
def test_separate_returns_guitar_stem():
    sr = 44100
    t = np.arange(sr * 3) / sr
    mix = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.random.default_rng(0).normal(size=len(t))
    stems = sep.separate(mix.astype("float32"), sr, stems=("guitar", "other"))
    assert set(stems) == {"guitar", "other"}
    assert stems["guitar"].shape == (len(t), 2)
    assert np.isfinite(stems["guitar"]).all()
