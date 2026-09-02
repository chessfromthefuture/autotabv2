"""Guitar-TECHS (Pedroza et al., ICASSP 2025): electric guitar techniques,
musical excerpts, chords and scales from three players, recorded as direct
input, mic'd amp and two head/room microphones.

Verified layout (P3_music.zip, the others follow the same pattern):

    <root>/P3_music/midi/midi_NN.mid                   one track per string,
                                                       track names e B G D A E
    <root>/P3_music/audio/directinput/directinput_NN.wav
    <root>/P3_music/audio/micamp/micamp_NN.wav

Natural harmonics can be annotated above fret 19; those notes are dropped
because the model only has classes up to fret 19."""

from __future__ import annotations

import re
from pathlib import Path

from autotab.datasets.base import AUDIO_SUFFIXES, Track, clip_fret, empty_notes, pitch_to_fret
from autotab.param import STRING_NAMES

NAME = "guitar-techs"
INDEX_RE = re.compile(r"_(\d+)$")


def load_notes(midi_path: Path):
    import mido

    mid = mido.MidiFile(str(midi_path))
    tempo = next((m.tempo for tr in mid.tracks for m in tr if m.type == "set_tempo"), 500000)
    notes = empty_notes()
    for track in mid.tracks:
        if track.name not in STRING_NAMES:
            continue
        s = STRING_NAMES.index(track.name)
        tick = 0
        active: dict[int, int] = {}
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = tick
            elif msg.type in ("note_off", "note_on") and msg.note in active:
                start = active.pop(msg.note)
                fret = clip_fret(pitch_to_fret(msg.note, s))
                if fret is not None and tick > start:
                    to_s = tempo / 1e6 / mid.ticks_per_beat
                    notes[s].append((start * to_s, tick * to_s, fret))
    return notes


def tracks(root: Path, parts=None, kinds=None, **_) -> list[Track]:
    """``parts``: e.g. ["P3_music"]; ``kinds``: e.g. ["directinput", "micamp"]."""
    root = Path(root)
    out = []
    for midi_path in sorted(root.rglob("midi_*.mid*")):
        part_dir = midi_path.parent.parent
        if parts and part_dir.name not in parts:
            continue
        m = INDEX_RE.search(midi_path.stem)
        if not m:
            continue
        index = m.group(1)
        for kind_dir in sorted(p for p in (part_dir / "audio").glob("*") if p.is_dir()):
            if kinds and kind_dir.name not in kinds:
                continue
            audio = next(
                (
                    p
                    for p in kind_dir.glob(f"{kind_dir.name}_{index}.*")
                    if p.suffix.lower() in AUDIO_SUFFIXES
                ),
                None,
            )
            if audio is None:
                continue
            out.append(
                Track(
                    dataset=NAME,
                    name=f"{part_dir.name}/{kind_dir.name}/{index}",
                    audio=audio,
                    notes=lambda p=midi_path: load_notes(p),
                    player=part_dir.name.split("_")[0],
                    tags={"part": part_dir.name, "kind": kind_dir.name},
                )
            )
    return out
