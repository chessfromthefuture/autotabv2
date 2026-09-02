# Training data beyond GuitarSet

AutoTab was trained on [GuitarSet](https://zenodo.org/record/1422265) (2018): 360 solo acoustic
excerpts, 3 hours, 6 players, per-string annotations from a hexaphonic pickup. It is still the
reference benchmark, but it is small and acoustic-only, which is why the model struggles on
electric, distorted or mixed recordings. Surveyed September 2026; sizes and licences from the
linked pages.

| Dataset | Year | Audio | Size | String/fret labels | Format | Licence / access |
|---|---|---|---|---|---|---|
| [GuitarSet](https://zenodo.org/record/1422265) | 2018 | acoustic, mic + hex pickup | 3 h, 360 files | yes (hexaphonic) | JAMS, one `note_midi` annotation per string | CC BY 4.0, open |
| [EGDB](https://ss12f32v.github.io/Guitar-Transcription/) | 2022 | electric DI + 3 amp renders | 2 h × 4 tones, 240 files | yes (hexaphonic) | MIDI per string | open (Google Drive) |
| [SynthTab](https://github.com/yongyizang/SynthTab) | 2024 | synthesised from Guitar Pro tabs; acoustic, clean, distorted, muted | 60 000 tracks, ~2 TB (JAMS only ≈1 GB, zips < 50 GB) | yes (from the tab) | JAMS per string, MIDI | CC BY-NC 4.0, open |
| [GAPS](https://arxiv.org/abs/2408.08653) | 2024 | classical nylon, 200+ performers, real recordings | 14 h | no (note MIDI + score only) | MIDI, MusicXML | CC BY-NC-SA 4.0 |
| [Guitar-TECHS](https://zenodo.org/records/14963133) | 2025 | electric, 3 players, DI + amp + 2 mics | techniques, excerpts, chords, scales | yes (MIDI, synchronised) | MIDI, WAV | Zenodo, open |
| [GOAT](https://zenodo.org/records/15690894) | 2025 | electric DI, many guitars and players, + amp augmentation | 5.9 h unique, 29.5 h augmented, 4.9 TB | yes (tablature) | Guitar Pro, tokens, aligned MIDI | CC BY-NC 4.0, **request access** |
| [IDMT-SMT-Guitar](https://zenodo.org/records/7544110) | 2014 | electric, 7 guitars | 1.3 GB, ~4 700 notes | partly (subsets 1–2) | XML | CC BY-NC-ND 4.0 |

## Recommendation

1. **SynthTab, one timbre at a time.** It is the only large dataset that already ships per-string
   JAMS like GuitarSet, so it drops into `autotab preprocess` with minimal glue. Start with the
   `acoustic` zip to stay in-domain for the shipped weights, then add `electric_clean` and
   `electric_distortion` to make the model usable on rock recordings. Being synthetic, it is best
   used for pre-training followed by fine-tuning on GuitarSet + EGDB.
2. **EGDB** for real electric guitar with per-string MIDI. Convert the six per-string MIDI tracks
   to frame labels with `pretty_midi` (string = track index, fret = pitch − open-string pitch),
   which is a ~40-line adapter next to `TabDataReprGen.load_rep_and_labels_from_raw_file`.
3. **GOAT** once access is granted: the largest real electric set with tablature, and the natural
   evaluation set for the guitar-isolation path, because its amp renders resemble mixed music.
4. **Skip GAPS** for tablature: it has no string assignment. It is still useful if you later add a
   pitch-only front-end (for example Basic Pitch) and only learn string assignment.

## Fitting new data into the pipeline

`TabDataReprGen` expects, per file, `<name>_mic.wav` under `audio/audio_mic/` and `<name>.jams`
under `annotation/` with six `note_midi` annotations ordered low E to high e. Anything that can be
brought into that shape works unchanged. Set `AUTOTAB_GUITARSET=/path/to/dataset` to point the
preprocessing at another folder, and give each dataset its own `AUTOTAB_DATA_DIR` so the npz
files stay separate. The cross-validation split keys on the two-digit player prefix of the file
name, so prefix new files with `06_`, `07_`, … to keep them out of the GuitarSet folds.

Data augmentation that is valid for tablature: time-stretch, added noise or room reverb, EQ and
amp simulation, and mixing with backing stems (see `docs/isolation_benchmark.csv`). Pitch shifting
is **not** valid unless the fret labels are shifted with it.
