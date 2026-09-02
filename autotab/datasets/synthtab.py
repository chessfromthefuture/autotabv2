"""SynthTab (Zang et al., ICASSP 2024): tablature rendered from Guitar Pro
files with commercial guitar plugins, ~60 000 tracks.

Annotations are JAMS files with the custom ``note_tab`` namespace: one
annotation per string carrying ``sandbox.open_tuning`` (MIDI pitch of the open
string) and ``sandbox.string_index``; observation values hold ``fret``; times
and durations are Guitar Pro *ticks* (960 per quarter note) that the ``tempo``
annotation converts to seconds.

Two layouts are supported:

    <root>/<split>/<song>/ground_truth.jams          (demo / dev layout)
    <root>/<split>/<song>/<guitar>/<mic>.flac|mp3

    <root>/<timbre>/<guitar>/<song>/<mic>.flac       (full release)
    <root>/jams/<song>/ground_truth.jams
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from autotab.datasets.base import AUDIO_SUFFIXES, Track, clip_fret, empty_notes

NAME = "synthtab"
QUARTER_TICKS = 960  # guitarpro.Duration.quarterTime
SCHEMA = Path(__file__).with_name("schemas") / "note_tab.json"


@lru_cache(maxsize=1)
def _register_namespace():
    import jams

    jams.schema.add_namespace(str(SCHEMA))


def ticks_to_seconds(ticks: np.ndarray, tempo_changes) -> np.ndarray:
    """Piece-wise conversion using (onset_tick, tempo_bpm, duration_ticks)."""
    seconds = np.zeros(np.shape(ticks), dtype=float)
    for onset, tempo, duration in tempo_changes:
        covered = np.clip(np.asarray(ticks, dtype=float) - onset, 0, duration)
        seconds += (60.0 / tempo) * covered / QUARTER_TICKS
    return seconds


def load_notes(jams_path: Path):
    import jams

    _register_namespace()
    jam = jams.load(str(jams_path), validate=False)
    tempo_changes = [(t.time, t.value, t.duration) for t in jam.annotations["tempo"][0].data]
    per_string = []
    for anno in jam.annotations["note_tab"]:
        open_pitch = int(anno.sandbox["open_tuning"])
        ticks = np.array([[obs.time, obs.time + obs.duration] for obs in anno.data], dtype=float)
        frets = [int(obs.value["fret"]) for obs in anno.data]
        secs = ticks_to_seconds(ticks, tempo_changes) if len(ticks) else np.zeros((0, 2))
        per_string.append(
            (open_pitch, [(float(a), float(b), f) for (a, b), f in zip(secs, frets, strict=True)])
        )
    per_string.sort(key=lambda x: x[0])  # low E first
    notes = empty_notes()
    for s, (_, string_notes) in enumerate(per_string[:6]):
        notes[s] = [(a, b, f) for a, b, f in string_notes if clip_fret(f) is not None]
    return notes


def _find_jams(root: Path, audio_dir: Path) -> Path | None:
    candidates = [
        audio_dir / "ground_truth.jams",
        audio_dir.parent / "ground_truth.jams",
        root / "jams" / audio_dir.name / "ground_truth.jams",
        root / "jams" / audio_dir.parent.name / "ground_truth.jams",
    ]
    return next((c for c in candidates if c.exists()), None)


def tracks(root: Path, mics: str = "first", timbres=None, **_) -> list[Track]:
    """``mics``: "first" keeps one microphone per guitar, "all" keeps every one.
    ``timbres``: optional list of top-level folders (acoustic, electric_clean, …)."""
    root = Path(root)
    out = []
    for audio_dir in sorted(
        {p.parent for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES}
    ):
        rel = audio_dir.relative_to(root)
        if rel.parts and rel.parts[0] == "jams":
            continue
        if timbres and not any(t in rel.parts for t in timbres):
            continue
        jams_path = _find_jams(root, audio_dir)
        if jams_path is None:
            continue
        audio_files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
        if mics == "first":
            audio_files = audio_files[:1]
        for audio in audio_files:
            out.append(
                Track(
                    dataset=NAME,
                    name="/".join([*rel.parts, audio.stem]),
                    audio=audio,
                    notes=lambda p=jams_path: load_notes(p),
                    tags={"timbre": rel.parts[0] if rel.parts else ""},
                )
            )
    return out
