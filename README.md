# 🎸 AutoTab v2 — AI Guitar Tablature Transcription

> A CNN that listens to a guitar recording and writes out the tablature: which string, which fret, frame by frame. Trained on the GuitarSet dataset from Queen Mary University of London.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-d00000?style=flat-square&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=flat-square&logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Data-GuitarSet-orange?style=flat-square)
[![CI](https://github.com/chessfromthefuture/autotabv2/actions/workflows/ci.yml/badge.svg)](https://github.com/chessfromthefuture/autotabv2/actions/workflows/ci.yml)

Built in 2021 at the Le Wagon data science bootcamp by Bruno Biehler, Hozefa Sadriwala and Ruslan Sizer. Refreshed in 2026 for Python 3.12, Keras 3 and current librosa; the original pretrained weights still load and are shipped in `models/`.

---

## What it does

```
e|----------------|----------------|----------------|
B|----------------|---------------8|8---------------|
G|----------------|7777------------|----------------|
D|-32--3-7-27-33--|----------------|-----------5----|
A|6---------------|----------------|----------------|
E|----------------|----------------|----------------|
```

1. The audio is normalised, resampled to 22.05 kHz and turned into a 192-bin constant-Q transform (CQT).
2. Every frame is fed to the CNN together with its 4 neighbours on each side (a 9-frame context window).
3. The network outputs, for each of the 6 strings, a softmax over 21 classes: *not played*, *open*, or frets 1–19.
4. An ergonomics pass resolves the same pitch playable on several strings to the fingering closest to the previous frame.
5. Frames are collapsed into readable tab, either every 9 frames (*simple*), on fret changes (*rhythm*), or not at all (*frames*).

The architecture follows Wiggins & Kim, [*Guitar Tablature Estimation with a Convolutional Neural Network*](https://archives.ismir.net/ismir2019/paper/000033.pdf) (ISMIR 2019).

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it downloads Python 3.12 for you if needed).

```bash
git clone https://github.com/chessfromthefuture/autotabv2.git
cd autotabv2
make install          # creates .venv and installs autotab with the app + dev extras
make test             # 18 unit tests, including loading the pretrained weights
```

### Transcribe a recording

```bash
.venv/bin/autotab predict path/to/guitar.wav                 # simple tab
.venv/bin/autotab predict path/to/guitar.wav --mode rhythm   # keep fret changes
.venv/bin/autotab predict a.wav b.wav -o tabs.txt            # several files, saved to disk
```

`.wav`, `.flac`, `.ogg` and `.mp3` work; stereo is mixed down to mono. Clean recordings of a single guitar work best: the model was trained on solo acoustic GuitarSet takes. `--silence-weight 1.0` gives the raw 2021 behaviour; the default `0.05` writes about three times as many correct notes (see below).

### Full songs: isolate the guitar first

```bash
uv pip install -e ".[separate]"                                  # Demucs + PyTorch, once
.venv/bin/autotab predict song.mp3 --isolate --mode rhythm       # transcribe only the guitar
.venv/bin/autotab separate song.mp3 --stems guitar other vocals  # or just export the stems
```

`--isolate` runs Meta's [Demucs](https://github.com/facebookresearch/demucs) 6-stem model (`htdemucs_6s`) to strip vocals, drums, bass and keys before transcription. On an Apple Silicon Mac it uses the GPU and takes roughly a minute for a 4-minute song. See the benchmark below.

### Web app

```bash
make app              # Streamlit UI on http://localhost:8501
```

Or with Docker: `docker build -t autotab . && docker run -p 8501:8501 autotab`.

---

## Calibrated silence class

The network sees 67 % "string not played" slots during training and learns a strong prior for silence: the raw model names a note in only 4 % of string slots, and when it does it is almost always right. Scaling the *not played* probability by a weight before the argmax trades a little precision for a lot of recall, with no retraining:

```bash
.venv/bin/autotab calibrate            # sweep the weight on your preprocessed npz files
.venv/bin/autotab evaluate             # raw model vs. the default weight
```

Measured on the two GuitarSet sample files (1 924 frames, player 00), full table in [`docs/calibration.csv`](docs/calibration.csv):

| silence weight | frame accuracy | pitch P / R / F | tab P / R / F |
|---:|---:|---|---|
| 1.0 (raw 2021 model) | 0.70 | 0.95 / 0.13 / **0.22** | 0.81 / 0.11 / **0.19** |
| 0.1 | 0.76 | 0.76 / 0.48 / 0.59 | 0.59 / 0.43 / 0.49 |
| **0.05 (default)** | 0.74 | 0.67 / 0.58 / **0.62** | 0.51 / 0.52 / **0.51** |
| 0.02 | 0.65 | 0.49 / 0.66 / 0.56 | 0.38 / 0.58 / 0.46 |

Caveat: these two files were most likely part of the 2021 training set, so the absolute numbers are optimistic. Re-run `autotab calibrate` on held-out players once you have the full dataset and adjust `DEFAULT_SILENCE_WEIGHT` in `autotab/evaluate.py`.

---

## Guitar isolation benchmark

To measure what isolation buys, a GuitarSet sample (with ground truth) was mixed at equal peak level with real drums, bass and vocals separated from a pop song, then transcribed three ways. Numbers are pitch F-measure with the default silence weight, full table in [`docs/isolation_benchmark.csv`](docs/isolation_benchmark.csv):

| input | comp excerpt | solo excerpt |
|---|---:|---:|
| clean guitar recording | 0.64 | 0.59 |
| guitar + drums, bass, vocals | 0.35 | 0.26 |
| same mix, Demucs-isolated guitar (`--isolate`) | **0.59** | **0.58** |

The model itself was never trained on electric or mixed music, so for real songs the next lever is more diverse training data. See [`docs/datasets.md`](docs/datasets.md) for a survey of seven guitar datasets (SynthTab, EGDB, GOAT, GAPS, Guitar-TECHS, …) and a recommended order to add them.

---

## Training on GuitarSet

1. Download `GuitarSet_audio_and_annotation.zip` (7.5 GB) from [Zenodo](https://zenodo.org/record/1422265) and unpack it so you have

   ```
   data/GuitarSet/
   ├── annotation/          360 .jams files
   └── audio/audio_mic/     360 *_mic.wav files
   ```

2. Build the spectral representations (one `.npz` per file, ~1.7 MB each):

   ```bash
   .venv/bin/autotab preprocess --mode c -j 8
   ```

3. Train with 6-fold cross-validation (each fold holds out one of the 6 players):

   ```bash
   .venv/bin/autotab train --mode c --epochs 8 --id-file ''
   ```

   Weights, predictions and `results.csv` land in `saved/<mode>_<timestamp>/`. Pass `--id-file id.csv` to train on the exact sample list in `data/spec_repr/id.csv` instead of every npz found, and `--folds 0 1` to run only some folds.

Modes: `c` CQT (default, matches the shipped weights), `m` mel spectrogram, `cm` both stacked, `s` STFT.

Paths can be overridden with environment variables `AUTOTAB_DATA_DIR`, `AUTOTAB_GUITARSET`, `AUTOTAB_MODEL` and `AUTOTAB_SAVE_DIR` (see `autotab/param.py`).

---

## Project structure

```
autotabv2/
├── autotab/
│   ├── param.py            paths + constants (env-var overridable)
│   ├── TabDataReprGen.py   wav (+ jams) -> CQT frames and one-hot labels
│   ├── DataGenerator.py    Keras PyDataset serving context windows from npz
│   ├── TabCNN.py           model, loss, metric, cross-validation experiment
│   ├── Metrics.py          pitch / tab precision, recall, F-measure
│   ├── TabErgonomics.py    picks the most playable fingering per frame
│   ├── TabPrediction.py    model output -> ASCII tablature
│   ├── evaluate.py         metrics, silence-class calibration
│   ├── separate.py         Demucs guitar isolation for full songs
│   ├── interpreter.py      GuitarSet .jams helpers (plots, MIDI)
│   └── cli.py              `autotab predict | separate | evaluate | calibrate | preprocess | train`
├── streamlit/autotab_app.py
├── models/full_val0_75acc_weights.h5   pretrained weights (2021, CQT mode)
├── docs/                   dataset survey, calibration and isolation benchmarks
├── data/spec_repr/         id.csv sample lists; npz files are generated here
├── notebooks/              exploration notebooks from 2021 (not maintained)
├── tests/
├── pyproject.toml          dependencies, `autotab` console script
├── Makefile · Dockerfile · .github/workflows/ci.yml
```

---

## What changed in the 2026 update

- **Runs again**: Python 3.12, TensorFlow 2.21 / Keras 3, librosa 1.0, NumPy 2, pandas 3. The 2021 `.h5` weights load unchanged and reproduce their 77 % per-string frame accuracy on GuitarSet.
- **No cloud coupling**: Google Cloud Storage code, credentials and the Heroku/GCP deploy scripts are gone; everything is local paths.
- **Calibrated output**: a tuned weight on the silence class roughly triples pitch and tab F-measure of the unchanged 2021 weights (`autotab calibrate`, `autotab evaluate`).
- **Full songs**: optional Demucs-based guitar isolation (`--isolate`, `autotab separate`, app toggle), mp3 input.
- **One entry point**: `autotab predict|separate|evaluate|calibrate|preprocess|train` replaces the Makefile targets, the parallel script and the three Streamlit variants.
- **Faster data loading**: the training generator caches each npz instead of re-reading it for every single frame.
- **Tests and CI**: pytest suite and a GitHub Actions workflow; `uv.lock` pins the exact environment.
- **Streamlit app** rewritten for the current API, with model caching, a note-sensitivity control, audio playback and a download button.

Known limitations of the model itself (unchanged from 2021): it is trained on solo guitar only, sees just 0.2 s of context per frame, and does not detect note onsets, so long notes look like repeated frets.

---

## Roadmap

- [ ] Retrain on GuitarSet + SynthTab/EGDB with the 2026 stack and publish new weights (see docs/datasets.md)
- [ ] Onset detection so held notes are written once
- [ ] Export to Guitar Pro / MusicXML
- [ ] Try a transformer or CRNN backbone

---

## References

- Q. Xi, R. Bittner, J. Pauwels, X. Ye, J. P. Bello, [*GuitarSet*](https://archives.ismir.net/ismir2018/paper/000188.pdf), ISMIR 2018
- A. Wiggins, Y. Kim, [*Guitar Tablature Estimation with a CNN*](https://archives.ismir.net/ismir2019/paper/000033.pdf), ISMIR 2019
- Original fork: [hozefazs/autotab](https://github.com/hozefazs/autotab), earlier experiment: [autotab](https://github.com/chessfromthefuture/autotab)
