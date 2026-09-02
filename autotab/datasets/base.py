"""Common track abstraction shared by every dataset adapter.

A ``Track`` is one audio file plus a callable returning its notes as six
per-string lists of ``(onset_s, offset_s, fret)`` tuples, low E first. That is
everything ``TabDataReprGen`` needs to build frame labels, so adding a dataset
means writing one small module that yields Tracks."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autotab.param import HIGHEST_FRET, STRING_MIDI_PITCHES

StringNotes = list[list[tuple[float, float, int]]]  # 6 x [(onset, offset, fret)]

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


@dataclass
class Track:
    dataset: str
    name: str
    audio: Path
    notes: Callable[[], StringNotes]
    player: str | None = None
    tags: dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        """npz file stem: ``<dataset>-<name>`` with path separators and
        whitespace replaced, unique within the dataset."""
        return f"{self.dataset}-{sanitize(self.name)}"


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", name).strip("-")


def empty_notes() -> StringNotes:
    return [[] for _ in range(6)]


def notes_to_frame_frets(notes: StringNotes, times: np.ndarray) -> np.ndarray:
    """(frames, 6) array of fret numbers, -1 where the string is silent.
    A note covers frames with onset <= t < offset; when notes overlap on one
    string the later onset wins, matching jams' ``to_samples`` behaviour."""
    frets = np.full((len(times), 6), -1, dtype=np.int16)
    for s, string_notes in enumerate(notes):
        for onset, offset, fret in sorted(string_notes):
            if offset <= onset:
                continue
            lo, hi = np.searchsorted(times, [onset, offset])
            frets[lo:hi, s] = int(fret)
    return frets


def clip_fret(fret: int) -> int | None:
    """Frets outside the model's range (0..HIGHEST_FRET) are dropped."""
    return fret if 0 <= fret <= HIGHEST_FRET else None


def pitch_to_fret(pitch: float, string_index: int, tuning=STRING_MIDI_PITCHES) -> int:
    return round(pitch) - tuning[string_index]


def find_audio(directory: Path, stem: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def limit(tracks: Iterable[Track], n: int | None) -> list[Track]:
    tracks = list(tracks)
    return tracks[:n] if n else tracks
