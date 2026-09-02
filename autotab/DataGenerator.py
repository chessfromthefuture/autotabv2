"""Keras dataset that serves context windows of spectral frames from npz files."""

from __future__ import annotations

import numpy as np
from keras.utils import PyDataset

from autotab.param import CON_WIN_SIZE, INPUT_BINS, NUM_CLASSES, NUM_STRINGS


class DataGenerator(PyDataset):
    """Each ID looks like ``<file>_<frame_idx>`` and maps to one training
    sample: a window of ``con_win_size`` frames centred on ``frame_idx``."""

    def __init__(
        self,
        list_IDs,
        data_path="",
        batch_size=128,
        shuffle=True,
        label_dim=(NUM_STRINGS, NUM_CLASSES),
        spec_repr="c",
        con_win_size=CON_WIN_SIZE,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.list_IDs = list(list_IDs)
        self.data_path = str(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        self.rng = np.random.default_rng(seed)

        self.X_dim = (self.batch_size, INPUT_BINS[spec_repr], self.con_win_size, 1)
        self.y_dim = (self.batch_size, *self.label_dim)

        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        return self._data_generation([self.list_IDs[k] for k in indexes])

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _load(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Load (and cache) the padded representation + labels of one file."""
        if filename not in self._cache:
            loaded = np.load(f"{self.data_path}{self.spec_repr}/{filename}")
            full_x = np.pad(loaded["repr"], [(self.halfwin, self.halfwin), (0, 0)], mode="constant")
            self._cache[filename] = (full_x, loaded["labels"])
        return self._cache[filename]

    def _data_generation(self, list_IDs_temp):
        X = np.empty(self.X_dim, dtype="float32")
        y = np.empty(self.y_dim, dtype="float32")
        for i, ID in enumerate(list_IDs_temp):
            filename = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_idx = int(ID.split("_")[-1])
            full_x, labels = self._load(filename)
            sample_x = full_x[frame_idx : frame_idx + self.con_win_size]
            X[i] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)
            y[i] = labels[frame_idx]
        return X, y
