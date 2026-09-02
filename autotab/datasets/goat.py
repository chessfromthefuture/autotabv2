"""GOAT (Loth et al., ISMIR 2025): real electric guitar DI recordings with
tablature. Access is granted on request via Zenodo; this adapter follows the
public examples in the GitHub repository:

    <root>/**/<name>.txt        DadaGP-style tokens with string and fret
    <root>/**/<name>.mid        MIDI aligned to the recording (no string info)
    <root>/**/<name>_di.wav, <name>_amp1.wav, …

Strings come from the tokens, timing from the aligned MIDI: MIDI notes are
matched to token notes in order of onset by pitch, so a note keeps its string
and fret while taking the performed onset and offset. Notes that cannot be
matched fall back to the lowest fret position that produces the pitch."""

from __future__ import annotations

import re
from pathlib import Path

from autotab.datasets.base import Track, clip_fret, empty_notes
from autotab.param import STRING_MIDI_PITCHES

NAME = "goat"
QUARTER_TICKS = 960
NOTE_RE = re.compile(r"^(?P<inst>[a-z]+\d*):note:s(?P<string>\d):f(?P<fret>\d+)$")


def parse_tokens(text: str):
    """Yield token notes as dicts with nominal time (s), string (0 = low E),
    fret, pitch and whether the note is tied to the previous one."""
    tempo = 120.0
    tick = 0
    downtune = 0
    notes = []
    last = None
    for token in text.split():
        if token.startswith("tempo:"):
            tempo = float(token.split(":")[1])
        elif token.startswith("downtune:"):
            downtune = int(token.split(":")[1])
        elif token.startswith("wait:"):
            tick += int(token.split(":")[1])
        elif token == "nfx:tie" and last is not None:
            last["tied"] = True
        else:
            m = NOTE_RE.match(token)
            if m and m.group("inst").startswith(("clean", "distorted")):
                string = 6 - int(m.group("string"))  # tokens count s1 = high e
                fret = int(m.group("fret"))
                if 0 <= string < 6:
                    last = {
                        "time": (60.0 / tempo) * tick / QUARTER_TICKS,
                        "string": string,
                        "fret": fret,
                        "pitch": STRING_MIDI_PITCHES[string] + fret - downtune,
                        "tied": False,
                    }
                    notes.append(last)
    return notes


def _midi_notes(midi_path: Path):
    import mido

    time = 0.0
    active: dict[int, float] = {}
    out = []
    for msg in mido.MidiFile(str(midi_path)):
        time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = time
        elif msg.type in ("note_off", "note_on") and msg.note in active:
            out.append((active.pop(msg.note), time, msg.note))
    return sorted(out)


def _fallback_position(pitch: int):
    options = [(pitch - open_pitch, s) for s, open_pitch in enumerate(STRING_MIDI_PITCHES)]
    options = [(f, s) for f, s in options if clip_fret(f) is not None]
    if not options:
        return None
    return min(options)[1], min(options)[0]


def align(token_notes, midi_notes, window: int = 8):
    """Greedy in-order matching of MIDI notes to token notes by pitch."""
    tokens = [t for t in token_notes if not t["tied"]]
    pointer = 0
    notes = empty_notes()
    matched = 0
    for onset, offset, pitch in midi_notes:
        hit = next(
            (
                i
                for i in range(pointer, min(pointer + window, len(tokens)))
                if tokens[i]["pitch"] == pitch
            ),
            None,
        )
        if hit is not None:
            string, fret = tokens[hit]["string"], tokens[hit]["fret"]
            pointer = hit + 1
            matched += 1
        else:
            fallback = _fallback_position(pitch)
            if fallback is None:
                continue
            string, fret = fallback
        if clip_fret(fret) is not None and offset > onset:
            notes[string].append((onset, offset, fret))
    return notes, matched, len(midi_notes)


def load_notes(txt_path: Path, midi_path: Path | None):
    token_notes = parse_tokens(txt_path.read_text())
    if midi_path is None or not midi_path.exists():
        # no aligned MIDI: nominal timing, duration until the next note on the string
        notes = empty_notes()
        by_string = [[] for _ in range(6)]
        for n in token_notes:
            by_string[n["string"]].append(n)
        for s, seq in enumerate(by_string):
            for i, n in enumerate(seq):
                if n["tied"] and notes[s]:
                    onset, _, fret = notes[s].pop()
                else:
                    onset, fret = n["time"], n["fret"]
                offset = seq[i + 1]["time"] if i + 1 < len(seq) else n["time"] + 1.0
                if clip_fret(fret) is not None and offset > onset:
                    notes[s].append((onset, offset, fret))
        return notes
    notes, _, _ = align(token_notes, _midi_notes(midi_path))
    return notes


def tracks(root: Path, variants=None, **_) -> list[Track]:
    root = Path(root)
    out = []
    for txt in sorted(root.rglob("*.txt")):
        midi = next(
            (p for p in (txt.with_suffix(".mid"), txt.with_suffix(".midi")) if p.exists()), None
        )
        audios = sorted(txt.parent.glob(f"{txt.stem}_*.wav"))
        for audio in audios:
            variant = audio.stem[len(txt.stem) + 1 :]
            if variant.startswith("gp"):  # Guitar Pro renders, not real recordings
                continue
            if variants and variant not in variants:
                continue
            out.append(
                Track(
                    dataset=NAME,
                    name=f"{txt.relative_to(root).with_suffix('')}_{variant}",
                    audio=audio,
                    notes=lambda t=txt, m=midi: load_notes(t, m),
                    tags={"variant": variant},
                )
            )
    return out
