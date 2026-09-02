"""EGDB (Chen et al., ICASSP 2022): 240 electric guitar pieces recorded DI
through a hexaphonic pickup and re-amped with five amp simulations.

    <root>/audio_<amp>/<n>.wav      amp in DI, Ftwin, JCjazz, Marshall, Mesa, Plexi
    <root>/audio_label/<n>.midi     one MIDI channel per string, channel 0 = high e
"""

from __future__ import annotations

from pathlib import Path

from autotab.datasets.base import Track, clip_fret, empty_notes, pitch_to_fret

NAME = "egdb"
AMPS = ["DI", "Ftwin", "JCjazz", "Marshall", "Mesa", "Plexi"]


def load_notes(midi_path: Path):
    import mido

    events = []  # (time, string_index, pitch, is_onset)
    time = 0.0
    for msg in mido.MidiFile(str(midi_path)):
        time += msg.time
        if msg.type in ("note_on", "note_off"):
            onset = msg.type == "note_on" and msg.velocity > 0
            events.append((time, 5 - msg.channel, msg.note, onset))

    notes = empty_notes()
    open_notes: dict[int, tuple[float, int]] = {}  # string -> (onset, pitch)
    for time, string, pitch, is_onset in events:
        if string in open_notes:
            prev_onset, prev_pitch = open_notes.pop(string)
            fret = clip_fret(pitch_to_fret(prev_pitch, string))
            if fret is not None and time > prev_onset:
                notes[string].append((prev_onset, time, fret))
        if is_onset:
            open_notes[string] = (time, pitch)
    end = events[-1][0] if events else 0.0
    for string, (onset, pitch) in open_notes.items():  # notes never switched off
        fret = clip_fret(pitch_to_fret(pitch, string))
        if fret is not None and end > onset:
            notes[string].append((onset, end, fret))
    return notes


def tracks(root: Path, amps=None, **_) -> list[Track]:
    root = Path(root)
    amps = list(amps) if amps else AMPS
    out = []
    for midi_path in sorted(
        (root / "audio_label").glob("*.mid*"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    ):
        for amp in amps:
            audio = root / f"audio_{amp}" / f"{midi_path.stem}.wav"
            if audio.exists():
                out.append(
                    Track(
                        dataset=NAME,
                        name=f"{amp}/{midi_path.stem}",
                        audio=audio,
                        notes=lambda p=midi_path: load_notes(p),
                        tags={"amp": amp},
                    )
                )
    return out
