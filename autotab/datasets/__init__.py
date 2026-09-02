"""Dataset adapters. Each module exposes ``NAME`` and ``tracks(root, **opts)``
returning :class:`autotab.datasets.base.Track` objects."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from autotab.datasets.base import Track, notes_to_frame_frets  # noqa: F401

REGISTRY = {
    "guitarset": "autotab.datasets.guitarset",
    "synthtab": "autotab.datasets.synthtab",
    "egdb": "autotab.datasets.egdb",
    "goat": "autotab.datasets.goat",
    "idmt": "autotab.datasets.idmt",
    "guitar-techs": "autotab.datasets.guitar_techs",
}

DESCRIPTIONS = {
    "guitarset": "GuitarSet 2018: 360 solo acoustic excerpts, hexaphonic labels (JAMS)",
    "synthtab": "SynthTab 2024: 60k tracks synthesised from Guitar Pro tabs (JAMS, ticks)",
    "egdb": "EGDB 2022: 240 electric DI pieces x 6 amp tones (per-channel MIDI)",
    "goat": "GOAT 2025: 5.9 h real electric DI + amp renders (tokens + MIDI); access on request",
    "idmt": "IDMT-SMT-Guitar 2014: electric notes, chords and licks (XML with string/fret)",
    "guitar-techs": "Guitar-TECHS 2025: electric techniques, excerpts, chords, scales (MIDI)",
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {', '.join(REGISTRY)}")
    return import_module(REGISTRY[name])


def tracks(name: str, root: Path, **opts) -> list[Track]:
    return get(name).tracks(Path(root), **opts)
