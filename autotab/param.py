"""Project-wide paths and settings.

Everything can be overridden with environment variables so the same code
runs locally, in Docker, or on a Hugging Face Space without edits:

    AUTOTAB_DATA_DIR   directory holding spec_repr/ (default: ./data)
    AUTOTAB_GUITARSET  directory holding GuitarSet audio/ and annotation/
                       (default: ./data/GuitarSet)
    AUTOTAB_MODEL      path to the pretrained weights .h5
                       (default: ./models/full_val0_75acc_weights.h5)
    AUTOTAB_SAVE_DIR   where training runs are written (default: ./saved)
"""

from __future__ import annotations

import os
from pathlib import Path

# The repository root is two levels above this file (autotab/param.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("AUTOTAB_DATA_DIR", PROJECT_ROOT / "data"))
GUITARSET_DIR = Path(os.environ.get("AUTOTAB_GUITARSET", DATA_DIR / "GuitarSet"))
SPEC_REPR_DIR = DATA_DIR / "spec_repr"
MODEL_PATH = Path(
    os.environ.get("AUTOTAB_MODEL", PROJECT_ROOT / "models" / "full_val0_75acc_weights.h5")
)
SAVE_DIR = Path(os.environ.get("AUTOTAB_SAVE_DIR", PROJECT_ROOT / "saved"))

# Backwards-compatible names used by the 2021 notebooks.
LOCAL_DATA = str(SPEC_REPR_DIR) + "/"
LOCAL_MODEL = str(MODEL_PATH)

# Guitar / label constants shared by every module.
STRING_MIDI_PITCHES = [40, 45, 50, 55, 59, 64]  # E A D G B e
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]
NUM_STRINGS = 6
HIGHEST_FRET = 19
NUM_CLASSES = HIGHEST_FRET + 2  # frets 0..19 plus "string not played"

# Audio front-end constants (must match the pretrained weights).
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
CQT_N_BINS = 192
CQT_BINS_PER_OCTAVE = 24
CON_WIN_SIZE = 9

INPUT_BINS = {"c": CQT_N_BINS, "m": 128, "cm": CQT_N_BINS + 128, "s": N_FFT // 2 + 1}


if __name__ == "__main__":
    print(f"DATA_DIR      {DATA_DIR}")
    print(f"GUITARSET_DIR {GUITARSET_DIR}")
    print(f"MODEL_PATH    {MODEL_PATH}")
    print(f"SAVE_DIR      {SAVE_DIR}")
