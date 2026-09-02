import numpy as np
import pytest

from autotab.evaluate import apply_silence_weight, best_weight, metrics, sweep
from autotab.param import NUM_CLASSES, NUM_STRINGS


def _fake_predictions(n=50, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, NUM_STRINGS, NUM_CLASSES))
    logits[..., 0] += 2.0  # a model biased towards "not played"
    y = np.exp(logits)
    y /= y.sum(-1, keepdims=True)
    gt = np.zeros_like(y)
    truth = rng.integers(0, NUM_CLASSES, size=(n, NUM_STRINGS))
    np.put_along_axis(gt, truth[..., None], 1, axis=-1)
    return y, gt


def test_weight_one_is_identity():
    y, _ = _fake_predictions()
    assert apply_silence_weight(y, 1.0) is y


def test_lower_weight_never_reduces_played_predictions():
    y, _ = _fake_predictions()
    played = [(apply_silence_weight(y, w).argmax(-1) > 0).sum() for w in (1.0, 0.5, 0.1, 0.01)]
    assert played == sorted(played)
    np.testing.assert_allclose(apply_silence_weight(y, 0.1).sum(-1), 1.0)


def test_invalid_weight():
    y, _ = _fake_predictions()
    with pytest.raises(ValueError):
        apply_silence_weight(y, 0)


def test_metrics_and_sweep():
    y, gt = _fake_predictions()
    m = metrics(gt, gt)
    assert m["accuracy"] == 1.0 and m["tab_f"] == 1.0 and m["pitch_f"] == 1.0
    table = sweep(y, gt, [1.0, 0.1])
    assert list(table.index) == [1.0, 0.1]
    assert table.loc[0.1, "tab_recall"] >= table.loc[1.0, "tab_recall"]
    assert best_weight(table, "tab_f") in (1.0, 0.1)
