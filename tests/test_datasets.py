"""Adapter tests on tiny synthetic fixtures in each dataset's native format."""

import json

import numpy as np
import pytest

from autotab.datasets import base, egdb, goat, guitarset, idmt, synthtab
from autotab.param import STRING_MIDI_PITCHES

TIMES = np.arange(0, 4, 0.1)


def _frets_at(notes, t):
    return base.notes_to_frame_frets(notes, np.array([t]))[0].tolist()


def test_notes_to_frame_frets_basic():
    notes = base.empty_notes()
    notes[0].append((0.5, 1.0, 3))
    notes[5].append((0.0, 4.0, 0))
    frets = base.notes_to_frame_frets(notes, TIMES)
    assert frets.shape == (len(TIMES), 6)
    assert frets[6].tolist() == [3, -1, -1, -1, -1, 0]  # t = 0.6
    assert frets[12].tolist() == [-1, -1, -1, -1, -1, 0]  # t = 1.2


def test_track_stem_is_filesystem_safe(tmp_path):
    t = base.Track(
        "synthtab", "acoustic/gibson thumb/song (1)/mic", tmp_path / "a.wav", base.empty_notes
    )
    assert t.stem == "synthtab-acoustic-gibson-thumb-song-1-mic"


# ----------------------------------------------------------------- GuitarSet
def test_guitarset_adapter(tmp_path):
    import jams

    root = tmp_path / "gs"
    (root / "annotation").mkdir(parents=True)
    (root / "audio" / "audio_mic").mkdir(parents=True)
    jam = jams.JAMS()
    jam.file_metadata.duration = 4.0
    for s, open_pitch in enumerate(STRING_MIDI_PITCHES):
        ann = jams.Annotation(namespace="note_midi", duration=4.0)
        if s == 2:
            ann.append(time=1.0, duration=1.0, value=open_pitch + 5)
        jam.annotations.append(ann)
    jam.save(str(root / "annotation" / "00_test.jams"))
    (root / "audio" / "audio_mic" / "00_test_mic.wav").write_bytes(b"")
    tracks = guitarset.tracks(root)
    assert len(tracks) == 1 and tracks[0].player == "00"
    assert _frets_at(tracks[0].notes(), 1.5) == [-1, -1, 5, -1, -1, -1]


# ------------------------------------------------------------------ SynthTab
def _synthtab_jams(path, tempo=120.0):
    import jams

    synthtab._register_namespace()
    jam = jams.JAMS()
    jam.file_metadata.duration = 10.0
    tempo_ann = jams.Annotation(namespace="tempo")
    tempo_ann.append(time=0, duration=100000, value=tempo, confidence=1.0)
    jam.annotations.append(tempo_ann)
    for s, open_pitch in enumerate(STRING_MIDI_PITCHES):
        ann = jams.Annotation(namespace="note_tab")
        ann.sandbox.update(string_index=6 - s, open_tuning=open_pitch)
        if s == 1:  # A string, fret 2, from beat 2 for one beat (ticks: 960/quarter)
            ann.append(time=1920, duration=960, value={"fret": 2}, confidence=1.0)
        jam.annotations.append(ann)
    jam.save(str(path))


def test_synthtab_ticks_and_layouts(tmp_path):
    root = tmp_path / "st"
    song = root / "train" / "song1"
    (song / "gibson").mkdir(parents=True)
    _synthtab_jams(song / "ground_truth.jams", tempo=120.0)
    (song / "gibson" / "mic1.flac").write_bytes(b"")
    (song / "gibson" / "mic2.flac").write_bytes(b"")
    assert len(synthtab.tracks(root)) == 1
    assert len(synthtab.tracks(root, mics="all")) == 2
    notes = synthtab.tracks(root)[0].notes()
    # 1920 ticks at 120 bpm = 1.0 s onset, 0.5 s long
    assert notes[1] == [(1.0, 1.5, 2)]
    assert _frets_at(notes, 1.2) == [-1, 2, -1, -1, -1, -1]

    # full-release layout: <root>/<timbre>/<guitar>/<song>/mic.flac + <root>/jams/<song>/
    root2 = tmp_path / "st2"
    (root2 / "acoustic" / "gibson" / "song2").mkdir(parents=True)
    (root2 / "jams" / "song2").mkdir(parents=True)
    _synthtab_jams(root2 / "jams" / "song2" / "ground_truth.jams", tempo=60.0)
    (root2 / "acoustic" / "gibson" / "song2" / "mic.flac").write_bytes(b"")
    tracks = synthtab.tracks(root2)
    assert len(tracks) == 1 and tracks[0].tags["timbre"] == "acoustic"
    assert tracks[0].notes()[1] == [(2.0, 3.0, 2)]  # 60 bpm doubles the times
    assert synthtab.tracks(root2, timbres=["electric_clean"]) == []


def test_synthtab_tempo_changes():
    secs = synthtab.ticks_to_seconds(
        np.array([0, 960, 1920, 2880]), [(0, 120, 960), (960, 60, 100000)]
    )
    np.testing.assert_allclose(secs, [0.0, 0.5, 1.5, 2.5])


# ---------------------------------------------------------------------- EGDB
def _egdb_midi(path):
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))  # 120 bpm
    # channel 2 -> string index 3 (G, open 55): fret 4 = B3 (59), one beat from beat 1
    track.append(mido.Message("note_on", channel=2, note=59, velocity=70, time=480))
    track.append(mido.Message("note_on", channel=2, note=59, velocity=0, time=480))
    # channel 5 -> string 0 (low E, 40): open string, switched off by note_off
    track.append(mido.Message("note_on", channel=5, note=40, velocity=70, time=0))
    track.append(mido.Message("note_off", channel=5, note=40, velocity=0, time=960))
    mid.save(str(path))


def test_egdb_adapter(tmp_path):
    root = tmp_path / "egdb"
    (root / "audio_label").mkdir(parents=True)
    (root / "audio_DI").mkdir()
    (root / "audio_Marshall").mkdir()
    _egdb_midi(root / "audio_label" / "7.midi")
    (root / "audio_DI" / "7.wav").write_bytes(b"")
    (root / "audio_Marshall" / "7.wav").write_bytes(b"")
    tracks = egdb.tracks(root)
    assert [t.name for t in tracks] == ["DI/7", "Marshall/7"]
    assert [t.name for t in egdb.tracks(root, amps=["Marshall"])] == ["Marshall/7"]
    notes = tracks[0].notes()
    assert notes[3] == [(0.5, 1.0, 4)]
    assert notes[0] == [(1.0, 2.0, 0)]


# ---------------------------------------------------------------------- IDMT
IDMT_XML = """<?xml version="1.0"?>
<instrumentRecording>
  <globalParameter><instrumentTuning>64, 59, 55, 50, 45, 40</instrumentTuning></globalParameter>
  <transcription>
    <event><pitch>64</pitch><onsetSec>0.10</onsetSec><offsetSec>0.60</offsetSec>
      <fretNumber>0</fretNumber><stringNumber>1</stringNumber></event>
    <event><pitch>47</pitch><onsetSec>1.00</onsetSec><offsetSec>1.50</offsetSec>
      <fretNumber>2</fretNumber><stringNumber>5</stringNumber></event>
  </transcription>
</instrumentRecording>
"""


def test_idmt_adapter(tmp_path):
    root = tmp_path / "idmt"
    (root / "dataset1" / "annotation").mkdir(parents=True)
    (root / "dataset1" / "audio").mkdir()
    (root / "dataset1" / "annotation" / "lick.xml").write_text(IDMT_XML)
    (root / "dataset1" / "audio" / "lick.wav").write_bytes(b"")
    tracks = idmt.tracks(root)
    assert len(tracks) == 1
    notes = tracks[0].notes()
    assert notes[5] == [(0.1, 0.6, 0)]  # tuning 64 = high e -> index 5
    assert notes[1] == [(1.0, 1.5, 2)]  # tuning 45 = A -> index 1


# ---------------------------------------------------------------------- GOAT
GOAT_TOKENS = (
    "unknown downtune:0 tempo:120 start new_measure clean0:note:s2:f11 wait:960 "
    "clean0:note:s6:f0 wait:960 clean0:note:s6:f0 nfx:tie wait:960 end"
)


def test_goat_tokens_and_alignment(tmp_path):
    tokens = goat.parse_tokens(GOAT_TOKENS)
    assert [(t["time"], t["string"], t["fret"], t["pitch"], t["tied"]) for t in tokens] == [
        (0.0, 4, 11, 70, False),
        (0.5, 0, 0, 40, False),
        (1.0, 0, 0, 40, True),
    ]
    # aligned MIDI: same pitches, performed slightly late and the tied note merged
    midi = [(0.05, 0.55, 70), (0.55, 1.6, 40)]
    notes, matched, total = goat.align(tokens, midi)
    assert (matched, total) == (2, 2)
    assert notes[4] == [(0.05, 0.55, 11)]
    assert notes[0] == [(0.55, 1.6, 0)]
    # unmatched pitch falls back to the lowest fret that produces it
    notes, matched, _ = goat.align(tokens, [(0.0, 0.5, 45)])
    assert matched == 0 and notes[1] == [(0.0, 0.5, 0)]

    # nominal timing when no MIDI is present
    root = tmp_path / "goat"
    root.mkdir()
    (root / "ex.txt").write_text(GOAT_TOKENS)
    (root / "ex_di.wav").write_bytes(b"")
    (root / "ex_gp.wav").write_bytes(b"")  # Guitar Pro render, skipped
    tracks = goat.tracks(root)
    assert [t.name for t in tracks] == ["ex_di"]
    notes = tracks[0].notes()
    assert notes[0] == [(0.5, 2.0, 0)]  # tie extends the open E to the end + 1 s
    assert notes[4] == [(0.0, 1.0, 11)]


def test_registry():
    from autotab import datasets

    assert set(datasets.REGISTRY) >= {"guitarset", "synthtab", "egdb", "goat", "idmt"}
    with pytest.raises(KeyError):
        datasets.get("nope")


def test_schema_is_packaged():
    assert json.loads(synthtab.SCHEMA.read_text())["note_tab"]["value"]["required"] == ["fret"]


# -------------------------------------------------------------- Guitar-TECHS
def test_guitar_techs_adapter(tmp_path):
    import mido

    from autotab.datasets import guitar_techs

    root = tmp_path / "gt"
    part = root / "P3_music"
    (part / "midi").mkdir(parents=True)
    (part / "audio" / "directinput").mkdir(parents=True)
    (part / "audio" / "micamp").mkdir(parents=True)
    mid = mido.MidiFile(ticks_per_beat=960)
    head = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000)])  # 120 bpm
    g = mido.MidiTrack([mido.MetaMessage("track_name", name="G")])
    g.append(mido.Message("note_on", channel=2, note=57, velocity=80, time=960))  # G string fret 2
    g.append(mido.Message("note_off", channel=2, note=57, velocity=0, time=480))
    g.append(mido.Message("note_on", channel=2, note=90, velocity=80, time=0))  # harmonic > fret 19
    g.append(mido.Message("note_off", channel=2, note=90, velocity=0, time=480))
    e = mido.MidiTrack([mido.MetaMessage("track_name", name="E")])
    e.append(mido.Message("note_on", channel=5, note=40, velocity=80, time=0))
    e.append(mido.Message("note_off", channel=5, note=40, velocity=0, time=1920))
    mid.tracks.extend([head, g, e])
    mid.save(str(part / "midi" / "midi_03.mid"))
    (part / "audio" / "directinput" / "directinput_03.wav").write_bytes(b"")
    (part / "audio" / "micamp" / "micamp_03.wav").write_bytes(b"")
    tracks = guitar_techs.tracks(root)
    assert [t.name for t in tracks] == ["P3_music/directinput/03", "P3_music/micamp/03"]
    assert [t.name for t in guitar_techs.tracks(root, kinds=["micamp"])] == ["P3_music/micamp/03"]
    notes = tracks[0].notes()
    assert notes[3] == [(0.5, 0.75, 2)]  # harmonic dropped
    assert notes[0] == [(0.0, 1.0, 0)]
