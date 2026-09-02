import numpy as np
import pandas as pd

from autotab import TabErgonomics as te
from autotab import TabPrediction as tp
from autotab.param import NUM_CLASSES, STRING_NAMES


def _labels(frames):
    """frames: list of 6-int fret lists (-1 not played, 0 open) -> one-hot array."""
    out = np.zeros((len(frames), 6, NUM_CLASSES), dtype="float32")
    for i, frame in enumerate(frames):
        for s, fret in enumerate(frame):
            out[i, s, fret + 1] = 1
    return out


def test_make_full_tab_roundtrip():
    frames = [[-1, 0, 2, 2, 0, -1], [3, -1, -1, -1, -1, -1]]
    tab = tp.make_full_tab(_labels(frames))
    assert list(tab.index) == STRING_NAMES
    assert tab[0].tolist() == frames[0]
    assert tab[1].tolist() == frames[1]


def test_make_smart_tab_prefers_ergonomic_fingering():
    frames = [[-1, 0, 2, 2, 0, -1]] * 3
    tab = tp.make_smart_tab(_labels(frames))
    assert tab.shape == (6, 3)
    assert tab[0].tolist() == frames[0]


def test_squeezed_and_dynamic_tabs():
    frames = [[-1, 0, 2, 2, 0, -1]] * 9 + [[3, -1, -1, -1, -1, -1]] * 9
    full = tp.make_full_tab(_labels(frames))
    squeezed = tp.make_squeezed_tab(full, n=9)
    assert squeezed.shape == (6, 2)
    assert squeezed.iloc[:, 0].tolist() == frames[0]
    dynamic = tp.make_dynamic_tab(full, n=5)
    assert dynamic.shape[1] < full.shape[1]


def test_web_tabs_format():
    frames = [[-1, 0, 2, 2, 0, -1]] * 4
    text = tp.web_tabs(tp.make_full_tab(_labels(frames)), num_div=2, len_div=8)
    lines = text.strip().split("\n")
    assert lines[0].startswith("e|")
    assert lines[5].startswith("E|")
    assert all(len(line) == len(lines[0]) for line in lines[:6])
    assert "|" in lines[0][2:]


def test_web_tabs_all_silence():
    text = tp.web_tabs(pd.DataFrame(-1, index=STRING_NAMES, columns=range(5)))
    assert text.startswith("e|-")


def test_best_frame_keeps_every_pitch_once():
    curr, prev = [10, -1, 0, 7, 3, 3], [3, 5, 2, 7, 0, -1]
    best = te.best_frame(curr, prev)
    # every string is either silent or unchanged from the input
    assert all(b == -1 or b == c for b, c in zip(best, curr, strict=True))
    # each midi pitch of the input survives exactly once (duplicates resolved)
    pitches = [p for p in te.frame_to_midi(best) if p != -1]
    assert sorted(set(pitches)) == sorted({p for p in te.frame_to_midi(curr) if p != -1})
    assert len(pitches) == len(set(pitches))
