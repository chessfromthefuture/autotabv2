"""IDMT-SMT-Guitar (Kehling et al., 2014): electric guitar notes, chords and
licks with XML annotations that carry string and fret per note.

    <root>/dataset{1,2,3}/audio/<name>.wav
    <root>/dataset{1,2,3}/annotation/<name>.xml
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from autotab.datasets.base import Track, clip_fret, empty_notes, find_audio
from autotab.param import STRING_MIDI_PITCHES

NAME = "idmt"


def load_notes(xml_path: Path):
    tree = ET.parse(xml_path)
    tuning_text = tree.findtext("globalParameter/instrumentTuning")
    tuning = (
        [int(p) for p in tuning_text.replace(",", " ").split()]
        if tuning_text
        else list(STRING_MIDI_PITCHES)
    )
    order = sorted(range(len(tuning)), key=lambda i: tuning[i])  # file string index -> low-to-high
    position = {file_idx: pos for pos, file_idx in enumerate(order)}
    notes = empty_notes()
    for event in tree.iterfind("transcription/event"):
        string_idx = int(event.findtext("stringNumber")) - 1
        if string_idx not in position:
            continue
        fret = clip_fret(int(event.findtext("fretNumber")))
        onset = float(event.findtext("onsetSec"))
        offset = float(event.findtext("offsetSec"))
        if fret is not None and offset > onset:
            notes[position[string_idx]].append((onset, offset, fret))
    return notes


def tracks(root: Path, subsets=("dataset1", "dataset2", "dataset3"), **_) -> list[Track]:
    root = Path(root)
    out = []
    for subset in subsets:
        for xml_path in sorted((root / subset / "annotation").glob("*.xml")):
            audio = find_audio(root / subset / "audio", xml_path.stem)
            if audio is None:
                continue
            out.append(
                Track(
                    dataset=NAME,
                    name=f"{subset}/{xml_path.stem}",
                    audio=audio,
                    notes=lambda p=xml_path: load_notes(p),
                    tags={"subset": subset},
                )
            )
    return out
