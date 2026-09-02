"""Turn model output (frames x 6 strings x 21 classes) into readable tablature."""

from __future__ import annotations

import numpy as np
import pandas as pd

import autotab.TabErgonomics as te
from autotab.evaluate import DEFAULT_SILENCE_WEIGHT, apply_silence_weight
from autotab.param import MODEL_PATH, STRING_NAMES
from autotab.TabCNN import load_pretrained
from autotab.TabDataReprGen import TabDataReprGen

DISPLAY_ORDER = ["e", "B", "G", "D", "A", "E"]  # high string on top, as in printed tabs


def make_empty_tab() -> pd.DataFrame:
    return pd.DataFrame(index=STRING_NAMES, dtype="int64")


def make_full_tab(labels, num_frames=None) -> pd.DataFrame:
    """One column per frame with the argmax fret of every string.
    -1 = string not played, 0 = open string, n = fret n."""
    num_frames = len(labels) if num_frames is None else num_frames
    frets = np.argmax(np.asarray(labels)[:num_frames], axis=-1).T  # (6, frames)
    return pd.DataFrame(frets, index=STRING_NAMES) - 1


def make_smart_tab(labels, num_frames=None) -> pd.DataFrame:
    """Like make_full_tab but resolves duplicate pitches across strings to the
    most ergonomic fingering given the previous frame."""
    num_frames = len(labels) if num_frames is None else num_frames
    columns = []
    prev_fret = [-1] * 6
    for frame_idx in range(num_frames):
        fret = [int(np.argmax(s)) for s in labels[frame_idx]]
        smart_fret = te.best_frame(fret, prev_fret)
        columns.append(smart_fret)
        prev_fret = smart_fret
    return pd.DataFrame(np.array(columns).T, index=STRING_NAMES) - 1


def make_squeezed_tab(tablature: pd.DataFrame, n=9) -> pd.DataFrame:
    """Collapse every n frames into one column holding the per-string mode."""
    cols = {}
    for batch_idx in range(0, tablature.shape[1], n):
        frame_batch = tablature.iloc[:, batch_idx : batch_idx + n]
        cols[batch_idx] = frame_batch.mode(axis="columns")[0]
    return pd.DataFrame(cols).astype(int)


def make_dynamic_tab(full_tab: pd.DataFrame, n=5) -> pd.DataFrame:
    """Keep a column whenever the fretting changes, otherwise keep one column
    per n unchanged frames so held notes still occupy space."""
    cols = {}
    col_read, col_write = 0, 0
    total_frames = full_tab.shape[1]
    while col_read < total_frames - 1:
        changed = False
        max_col = col_read + n
        while (not changed) and col_read < max_col and col_read < total_frames - 1:
            if not full_tab.iloc[:, col_read].equals(full_tab.iloc[:, col_read + 1]):
                cols[col_write] = full_tab.iloc[:, col_read]
                col_write += 1
                changed = True
            col_read += 1
        if not changed:
            cols[col_write] = full_tab.iloc[:, col_read - 1]
            col_write += 1
    if not cols:
        return full_tab.iloc[:, :1].copy()
    return pd.DataFrame(cols).astype(int)


def _trim_trailing_silence(tabs: pd.DataFrame) -> pd.DataFrame:
    playing = (tabs != -1).any(axis=0).to_numpy()
    if not playing.any():
        return tabs.iloc[:, :1]
    last = int(np.nonzero(playing)[0][-1])
    return tabs.iloc[:, : last + 1]


def str_row(row, len_div=16) -> str:
    """Render one string as ``3-5-7|---...`` with a bar every len_div chars."""
    values = row.values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    joined = "".join("-" if v == -1 else str(v) for v in values)
    return "|".join(joined[i : i + len_div] for i in range(0, len(joined), len_div))


def web_tabs(tabs: pd.DataFrame, num_div=4, len_div=16) -> str:
    """Multi-line ASCII tablature, num_div bars of len_div characters per line."""
    tabs = _trim_trailing_silence(tabs)
    rows = {s: str_row(tabs.loc[s], len_div) for s in DISPLAY_ORDER}
    len_plus = len_div + 1
    line_width = len_plus * num_div
    longest = max(len(r) for r in rows.values())
    num_lines = longest // line_width + 1

    out = []
    for line in range(num_lines):
        for s in DISPLAY_ORDER:
            chunk = rows[s][line * line_width : (line + 1) * line_width]
            if line == num_lines - 1:  # pad the last line to a full width
                len_last = len(chunk)
                finish = len_div - (len_last - len_plus * (len_last // len_plus))
                full_divs = (line_width - len_last) // len_plus
                chunk += "-" * finish + "|" + ("-" * len_div + "|") * full_divs
            out.append(f"{s}|{chunk}")
        out.append("")
    return "\n".join(out)


def print_tabs(tabs, num_div=4, len_div=16):
    print(web_tabs(tabs, num_div=num_div, len_div=len_div))


def load_model_and_weights(weights_path=None):
    return load_pretrained(weights_path or MODEL_PATH)


def load_x_new(filename):
    return TabDataReprGen().load_rep_from_raw_file(filename)


def predict_tab(
    source,
    model=None,
    mode="simple",
    num_div=4,
    len_div=16,
    silence_weight=DEFAULT_SILENCE_WEIGHT,
) -> str:
    """End-to-end: audio (path or file-like) -> ASCII tablature.

    mode: "simple" (mode over 9-frame windows), "rhythm" (keep fret changes),
    or "frames" (every frame).
    silence_weight: multiplier on the "not played" class; 1.0 is the raw
    model, smaller values make it write more notes (see autotab.evaluate)."""
    model = model or load_model_and_weights()
    x_new = load_x_new(source)
    y_pred = apply_silence_weight(model.predict(x_new, verbose=0), silence_weight)
    all_frames = make_smart_tab(y_pred)
    if mode == "simple":
        tab = make_squeezed_tab(all_frames)
    elif mode == "rhythm":
        tab = make_dynamic_tab(all_frames)
    elif mode == "frames":
        tab = all_frames
    else:
        raise ValueError(f"unknown mode {mode!r}; use simple, rhythm or frames")
    return web_tabs(tab, num_div=num_div, len_div=len_div)
