"""Evaluate the model on GuitarSet npz files and calibrate the silence class.

The network is trained on frames where 67 % of the string slots are "not
played", so it learns a strong prior for class 0 and misses many notes.
Multiplying the class-0 probability by a weight < 1 before the argmax trades
a little precision for a lot of recall without retraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autotab.Metrics import (
    pitch_f_measure,
    pitch_precision,
    pitch_recall,
    tab_disamb,
    tab_f_measure,
    tab_precision,
    tab_recall,
)
from autotab.param import SPEC_REPR_DIR

# Chosen by `autotab calibrate` on the GuitarSet sample files (see README).
DEFAULT_SILENCE_WEIGHT = 0.05

SWEEP_WEIGHTS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]


def apply_silence_weight(y_pred: np.ndarray, weight: float = DEFAULT_SILENCE_WEIGHT) -> np.ndarray:
    """Scale the "string not played" probability and renormalise per string."""
    if weight == 1.0:
        return y_pred
    if weight <= 0:
        raise ValueError("silence weight must be > 0")
    y = np.array(y_pred, dtype="float64", copy=True)
    y[..., 0] *= weight
    y /= y.sum(axis=-1, keepdims=True)
    return y


def metrics(y_pred: np.ndarray, y_gt: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float((y_pred.argmax(-1) == y_gt.argmax(-1)).mean()),
        "pitch_precision": pitch_precision(y_pred, y_gt),
        "pitch_recall": pitch_recall(y_pred, y_gt),
        "pitch_f": pitch_f_measure(y_pred, y_gt),
        "tab_precision": tab_precision(y_pred, y_gt),
        "tab_recall": tab_recall(y_pred, y_gt),
        "tab_f": tab_f_measure(y_pred, y_gt),
        "tab_disamb": tab_disamb(y_pred, y_gt),
    }


def load_predictions(model, npz_dir=None, names=None, spec_repr="c"):
    """Run the model over every npz (or the named ones) and return the stacked
    raw predictions and ground-truth labels."""
    from autotab.TabDataReprGen import TabDataReprGen

    npz_dir = Path(npz_dir) if npz_dir else SPEC_REPR_DIR / spec_repr
    paths = sorted(npz_dir.glob("*.npz"))
    if names:
        paths = [p for p in paths if p.stem in set(names)]
    if not paths:
        raise FileNotFoundError(f"no npz files in {npz_dir}; run `autotab preprocess` first")
    gen = TabDataReprGen(mode=spec_repr)
    preds, gts = [], []
    for path in paths:
        loaded = np.load(path)
        preds.append(model.predict(gen.windows_from_repr(loaded["repr"]), verbose=0))
        gts.append(loaded["labels"])
    return np.concatenate(preds), np.concatenate(gts), [p.stem for p in paths]


def sweep(y_pred, y_gt, weights=SWEEP_WEIGHTS) -> pd.DataFrame:
    rows = []
    for w in weights:
        row = {"silence_weight": w}
        row.update(metrics(apply_silence_weight(y_pred, w), y_gt))
        rows.append(row)
    return pd.DataFrame(rows).set_index("silence_weight")


def best_weight(table: pd.DataFrame, metric: str = "tab_f") -> float:
    return float(table[metric].idxmax())
