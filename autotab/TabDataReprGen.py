"""Audio front-end: turn a .wav (plus optional .jams annotation) into the
spectral representation and frame-level labels the CNN consumes."""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from keras.utils import to_categorical

from autotab.param import (
    CON_WIN_SIZE,
    CQT_BINS_PER_OCTAVE,
    CQT_N_BINS,
    GUITARSET_DIR,
    HIGHEST_FRET,
    HOP_LENGTH,
    N_FFT,
    NUM_CLASSES,
    SAMPLE_RATE,
    SPEC_REPR_DIR,
    STRING_MIDI_PITCHES,
)


def read_audio_mono(source) -> tuple[np.ndarray, int]:
    """Read a wav/flac/ogg file (path or file-like) as float32 mono."""
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


class TabDataReprGen:
    """Preprocessing modes: c = CQT, m = mel spectrogram, cm = both stacked,
    s = STFT magnitude. The shipped weights were trained with "c"."""

    def __init__(self, mode: str = "c", guitarset_dir: os.PathLike | str | None = None):
        self.guitarset_dir = Path(guitarset_dir) if guitarset_dir else GUITARSET_DIR
        self.path_audio = self.guitarset_dir / "audio" / "audio_mic"
        self.path_anno = self.guitarset_dir / "annotation"

        self.string_midi_pitches = STRING_MIDI_PITCHES
        self.highest_fret = HIGHEST_FRET
        self.num_classes = NUM_CLASSES

        self.output: dict[str, np.ndarray] = {}

        if mode not in ("c", "m", "cm", "s"):
            raise ValueError(f"invalid representation mode {mode!r}; use c, m, cm or s")
        self.preproc_mode = mode
        self.downsample = True
        self.normalize = True
        self.sr_downs = SAMPLE_RATE

        self.cqt_n_bins = CQT_N_BINS
        self.cqt_bins_per_octave = CQT_BINS_PER_OCTAVE
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH

        self.con_win_size = CON_WIN_SIZE
        self.half_win = self.con_win_size // 2

        self.save_path = SPEC_REPR_DIR / self.preproc_mode

        self.sr_original: int | None = None
        self.sr_curr: int | None = None

    # ------------------------------------------------------------------ labels
    def load_rep_and_labels_from_raw_file(self, filename: str) -> int:
        """Load <filename>_mic.wav and <filename>.jams, compute the spectral
        representation and one-hot labels into self.output. Returns #frames."""
        import jams  # heavy import; only needed for training data

        file_audio = self.path_audio / f"{filename}_mic.wav"
        file_anno = self.path_anno / f"{filename}.jams"
        jam = jams.load(str(file_anno))
        data, self.sr_original = read_audio_mono(file_audio)
        self.sr_curr = self.sr_original

        self.output["repr"] = np.swapaxes(self.preprocess_audio(data), 0, 1)

        frame_indices = range(len(self.output["repr"]))
        times = librosa.frames_to_time(frame_indices, sr=self.sr_curr, hop_length=self.hop_length)

        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            frets = []
            for sample in string_label_samples:
                if len(sample) == 0:
                    frets.append(-1)
                else:
                    frets.append(int(round(sample[0]) - self.string_midi_pitches[string_num]))
            labels.append(frets)

        labels = np.swapaxes(np.array(labels), 0, 1)  # (frames, 6)
        self.output["labels"] = self.clean_labels(labels)
        return len(labels)

    def correct_numbering(self, n: int) -> int:
        """Shift frets by one so 0 = not played, 1 = open, 2 = fret 1, ..."""
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n

    def categorical(self, label):
        return to_categorical(label, self.num_classes)

    def clean_label(self, label):
        return self.categorical([self.correct_numbering(n) for n in label])

    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])

    # ------------------------------------------------------------------- audio
    def preprocess_audio(self, data: np.ndarray) -> np.ndarray:
        """Normalise, resample and transform. Returns (n_bins, n_frames)."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample and self.sr_original != self.sr_downs:
            data = librosa.resample(data, orig_sr=self.sr_original, target_sr=self.sr_downs)
            self.sr_curr = self.sr_downs

        if self.preproc_mode == "c":
            return self._cqt(data)
        if self.preproc_mode == "m":
            return self._mel(data)
        if self.preproc_mode == "cm":
            return np.concatenate((self._cqt(data), self._mel(data)), axis=0)
        return np.abs(librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length))

    def _cqt(self, data):
        return np.abs(
            librosa.cqt(
                data,
                sr=self.sr_curr,
                hop_length=self.hop_length,
                n_bins=self.cqt_n_bins,
                bins_per_octave=self.cqt_bins_per_octave,
            )
        )

    def _mel(self, data):
        return librosa.feature.melspectrogram(
            y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length
        )

    # --------------------------------------------------------------- npz files
    def save_data(self, filename):
        np.savez(filename, **self.output)

    def list_filenames(self) -> list[str]:
        names = sorted(p.stem for p in self.path_anno.glob("*.jams"))
        if not names:
            raise FileNotFoundError(f"no .jams files found in {self.path_anno}")
        return names

    def get_nth_filename(self, n: int) -> str:
        return self.list_filenames()[n]

    def load_and_save_repr_file(self, filename: str) -> Path:
        num_frames = self.load_rep_and_labels_from_raw_file(filename)
        self.save_path.mkdir(parents=True, exist_ok=True)
        out = self.save_path / f"{filename}.npz"
        self.save_data(out)
        print(f"done: {filename}, {num_frames} frames -> {out}")
        return out

    def load_and_save_repr_nth_file(self, n: int) -> Path:
        return self.load_and_save_repr_file(self.get_nth_filename(n))

    # ------------------------------------------------------- other datasets
    def load_rep_and_labels_from_track(self, track) -> int:
        """Same as load_rep_and_labels_from_raw_file but for any
        :class:`autotab.datasets.base.Track` (SynthTab, EGDB, GOAT, …)."""
        from autotab.datasets.base import notes_to_frame_frets

        data, self.sr_original = read_audio_mono(track.audio)
        self.sr_curr = self.sr_original
        self.output["repr"] = np.swapaxes(self.preprocess_audio(data), 0, 1)
        times = librosa.frames_to_time(
            range(len(self.output["repr"])), sr=self.sr_curr, hop_length=self.hop_length
        )
        frets = notes_to_frame_frets(track.notes(), times)
        self.output["labels"] = self.clean_labels(frets)
        return len(frets)

    def save_track(self, track) -> Path:
        num_frames = self.load_rep_and_labels_from_track(track)
        self.save_path.mkdir(parents=True, exist_ok=True)
        out = self.save_path / f"{track.stem}.npz"
        self.save_data(out)
        print(f"done: {track.dataset} {track.name}, {num_frames} frames -> {out}")
        return out

    # -------------------------------------------------------------- inference
    def load_rep_from_raw_file(self, source) -> np.ndarray:
        """Turn an audio file (path or file-like object) into model input of
        shape (n_frames, n_bins, con_win_size, 1)."""
        data, sr = read_audio_mono(source)
        return self.load_rep_from_audio(data, sr)

    def load_rep_from_audio(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Same as load_rep_from_raw_file but for an in-memory mono signal."""
        self.sr_original = sr
        self.sr_curr = sr
        repr_ = np.swapaxes(self.preprocess_audio(data), 0, 1)  # (frames, bins)
        return self.windows_from_repr(repr_)

    def windows_from_repr(self, repr_: np.ndarray) -> np.ndarray:
        full_x = np.pad(repr_, [(self.half_win, self.half_win), (0, 0)], mode="constant")
        # sliding windows: (frames, con_win, bins) -> (frames, bins, con_win, 1)
        idx = np.arange(len(repr_))[:, None] + np.arange(self.con_win_size)[None, :]
        windows = full_x[idx]
        return np.expand_dims(np.swapaxes(windows, 1, 2), -1).astype("float32")


def main(args):
    """(index, mode) -> preprocess the index-th GuitarSet file into an npz."""
    n, m = args
    TabDataReprGen(mode=m).load_and_save_repr_nth_file(n)


if __name__ == "__main__":
    main([0, "c"])
