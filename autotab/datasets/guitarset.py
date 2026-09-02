"""GuitarSet (Xi et al., ISMIR 2018): ``audio/audio_mic/<name>_mic.wav`` and
``annotation/<name>.jams`` with one ``note_midi`` annotation per string."""

from __future__ import annotations

from pathlib import Path

from autotab.datasets.base import Track, clip_fret, empty_notes, pitch_to_fret

NAME = "guitarset"


def load_notes(jams_path: Path):
    import jams

    jam = jams.load(str(jams_path))
    notes = empty_notes()
    for s, anno in enumerate(jam.annotations["note_midi"][:6]):
        for obs in anno.data:
            fret = clip_fret(pitch_to_fret(obs.value, s))
            if fret is not None:
                notes[s].append((float(obs.time), float(obs.time + obs.duration), fret))
    return notes


def tracks(root: Path, audio_kind: str = "mic", **_) -> list[Track]:
    root = Path(root)
    anno_dir = root / "annotation"
    audio_dir = root / "audio" / f"audio_{audio_kind}"
    out = []
    for jams_path in sorted(anno_dir.glob("*.jams")):
        name = jams_path.stem
        audio = audio_dir / f"{name}_{audio_kind}.wav"
        if not audio.exists():
            continue
        out.append(
            Track(
                dataset=NAME,
                name=name,
                audio=audio,
                notes=lambda p=jams_path: load_notes(p),
                player=name.split("_")[0],
            )
        )
    return out
