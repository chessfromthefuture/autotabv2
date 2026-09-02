import numpy as np
import pytest
from keras import ops

from autotab.param import CON_WIN_SIZE, CQT_N_BINS, MODEL_PATH, NUM_CLASSES, NUM_STRINGS
from autotab.TabCNN import avg_acc, build_model, catcross_by_string, load_pretrained


def test_build_model_output_is_per_string_softmax():
    model = build_model()
    x = np.random.rand(2, CQT_N_BINS, CON_WIN_SIZE, 1).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (2, NUM_STRINGS, NUM_CLASSES)
    np.testing.assert_allclose(y.sum(-1), 1.0, atol=1e-5)


def test_loss_and_metric_values():
    y_true = np.zeros((1, NUM_STRINGS, NUM_CLASSES), dtype="float32")
    y_true[0, :, 3] = 1
    assert float(ops.mean(catcross_by_string(y_true, y_true))) == pytest.approx(0.0, abs=1e-5)
    assert float(avg_acc(y_true, y_true)) == pytest.approx(1.0)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="pretrained weights not present")
def test_pretrained_weights_load(sine_wav):
    from autotab.TabDataReprGen import TabDataReprGen

    model = load_pretrained(MODEL_PATH)
    x = TabDataReprGen().load_rep_from_raw_file(sine_wav)
    y = model.predict(x, verbose=0)
    assert y.shape == (x.shape[0], NUM_STRINGS, NUM_CLASSES)
    # a pure 110 Hz tone is an open A string: class 1 on string index 1 should dominate mid-file
    mid = y[len(y) // 2]
    assert mid[1].argmax() == 1
