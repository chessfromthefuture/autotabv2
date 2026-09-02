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
| [Guitar-TECHS](https://zenodo.org/records/14963133) | 2025 | electric, 3 players, DI + amp + 2 mics | 4.1 GB in 9 zips: techniques, excerpts, chords, scales | yes (one MIDI track per string) | MIDI, WAV 48 kHz | CC BY 4.0, Zenodo |
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

## Built-in adapters

Every dataset above except GAPS has an adapter in `autotab/datasets/` that turns its native
annotation format into the frame labels the CNN trains on. One command prepares any of them:

```bash
autotab datasets                                             # list adapters
autotab download egdb --root data/EGDB --amps DI Marshall    # Google Drive, ~1 GB
autotab download idmt --root data/IDMT                       # Zenodo, 1.3 GB
autotab download guitar-techs --root data/GuitarTECHS --parts P3_music
autotab download synthtab --root data/SynthTab               # prints the Box links
autotab download goat --root data/GOAT                       # prints the access-request steps

autotab preprocess --dataset egdb --root data/EGDB -j 8
autotab preprocess --dataset synthtab --root data/SynthTab --timbres acoustic --mics first -j 8
autotab preprocess --dataset goat --root data/GOAT --variants di amp1
autotab preprocess --dataset idmt --root data/IDMT
autotab train --id-file ''                                   # trains on every npz found
```

| adapter | native format | how the labels are derived |
|---|---|---|
| `guitarset` | JAMS, one `note_midi` annotation per string | fret = pitch − open-string pitch |
| `synthtab` | JAMS `note_tab` namespace, times in Guitar Pro ticks | tempo annotation converts ticks to seconds (960 per quarter); strings ordered by `open_tuning`; both the demo layout and the full-release `jams/` layout are detected |
| `egdb` | one MIDI channel per string (channel 0 = high e), no tempo issues | note-on/off pairs per channel; fret from pitch |
| `goat` | DadaGP tokens (`clean0:note:s2:f11`, `wait:`, `nfx:tie`) plus MIDI aligned to the recording | strings and frets from the tokens, onsets and offsets from the MIDI by in-order pitch matching (98 / 98 notes on the public example, 96 % frame agreement) |
| `idmt` | XML events with `stringNumber`, `fretNumber`, `onsetSec`, `offsetSec` | direct; `instrumentTuning` fixes the string order |
| `guitar-techs` | `midi/midi_NN.mid` with one track per string (named e B G D A E) and `audio/<kind>/<kind>_NN.wav` | note-on/off per track; harmonics above fret 19 are dropped |

The npz files are named `<dataset>-<track>.npz`. The 6-fold cross-validation still holds out one
GuitarSet player per fold; files from every other dataset always go to the training split, which
is the "pre-train on more data, evaluate on GuitarSet" recipe used in the SynthTab paper. Keep
each dataset in its own `AUTOTAB_DATA_DIR` if you want to train on subsets.

Sanity check of the adapters with the shipped acoustic weights (pitch F-measure, silence weight
0.05): EGDB DI 0.42, GOAT DI 0.50, Guitar-TECHS DI 0.42 versus 0.59 on GuitarSet itself. Labels that
were misaligned would score near zero, so the conversions are sound; the gap is the acoustic-only
training. Note that on electric material the raw model is much less silence-biased (pitch precision
0.75 at weight 1.0), so rerun `autotab calibrate --files <dataset>-*` per dataset instead of
reusing the GuitarSet weight.

Adding another dataset means one module with a `tracks(root, **opts)` function that yields
`Track` objects whose `notes()` returns six lists of `(onset_s, offset_s, fret)`; register it in
`autotab/datasets/__init__.py`. GAPS has no string information, so it is deliberately left out.

Data augmentation that is valid for tablature: time-stretch, added noise or room reverb, EQ and
amp simulation, and mixing with backing stems (see `docs/isolation_benchmark.csv`). Pitch shifting
is **not** valid unless the fret labels are shifted with it.
