"""A CNN that classifies the fret played on each of the 6 strings, frame by frame.

Architecture follows Wiggins & Kim, "Guitar Tablature Estimation with a
Convolutional Neural Network" (ISMIR 2019). The shipped weights in
models/full_val0_75acc_weights.h5 were trained with this exact layout on the
CQT ("c") representation, so the layer order must not change."""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers, ops

from autotab.DataGenerator import DataGenerator
from autotab.Metrics import (
    pitch_f_measure,
    pitch_precision,
    pitch_recall,
    tab_disamb,
    tab_f_measure,
    tab_precision,
    tab_recall,
)
from autotab.param import (
    CON_WIN_SIZE,
    INPUT_BINS,
    NUM_CLASSES,
    NUM_STRINGS,
    SAVE_DIR,
    SPEC_REPR_DIR,
)


@keras.saving.register_keras_serializable(package="autotab")
def catcross_by_string(y_true, y_pred):
    """Categorical cross-entropy per string, summed over the 6 strings."""
    per_string = keras.losses.categorical_crossentropy(y_true, y_pred, axis=-1)  # (batch, 6)
    return ops.sum(per_string, axis=-1)


@keras.saving.register_keras_serializable(package="autotab")
def avg_acc(y_true, y_pred):
    """Fraction of (frame, string) pairs whose predicted fret is correct."""
    match = ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1))
    return ops.mean(ops.cast(match, "float32"))


def build_model(spec_repr: str = "c", con_win_size: int = CON_WIN_SIZE) -> keras.Model:
    """Build and compile the TabCNN. Softmax is applied per string (last axis)."""
    input_shape = (INPUT_BINS[spec_repr], con_win_size, 1)
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES * NUM_STRINGS),
            layers.Reshape((NUM_STRINGS, NUM_CLASSES)),
            layers.Softmax(axis=-1),
        ],
        name="TabCNN",
    )
    model.compile(loss=catcross_by_string, optimizer=keras.optimizers.Adadelta(), metrics=[avg_acc])
    return model


def load_pretrained(weights_path, spec_repr: str = "c") -> keras.Model:
    """Build the model and load weights from a .h5 or .weights.h5 file."""
    model = build_model(spec_repr=spec_repr)
    model.load_weights(str(weights_path))
    return model


class TabCNN:
    """6-fold cross-validation experiment: GuitarSet has 6 players (00..05);
    each fold holds one player out for validation."""

    def __init__(
        self,
        batch_size=128,
        epochs=8,
        con_win_size=CON_WIN_SIZE,
        spec_repr="c",
        data_path=None,
        id_file="id.csv",
        save_path=None,
        workers=1,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.workers = workers

        self.data_path = str(Path(data_path) if data_path else SPEC_REPR_DIR) + "/"
        self.id_file = id_file
        self.save_path = Path(save_path) if save_path else SAVE_DIR

        self.load_IDs()

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_folder = self.save_path / f"{self.spec_repr}_{stamp}"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_folder / "log.txt"

        self.metrics = {k: [] for k in ("pp", "pr", "pf", "tp", "tr", "tf", "tdr")}
        self.folds_run: list[int] = []

        self.input_shape = (INPUT_BINS[self.spec_repr], self.con_win_size, 1)
        self.num_classes = NUM_CLASSES
        self.num_strings = NUM_STRINGS
        self.model = None

    # ------------------------------------------------------------------ data
    def load_IDs(self):
        """Read sample IDs (``<file>_<frame>``) from id.csv. If the csv is
        missing or ``id_file`` is None, derive the IDs from the npz files that
        actually exist under data/spec_repr/<mode>/."""
        csv_file = Path(self.data_path) / self.id_file if self.id_file else None
        if csv_file and csv_file.exists():
            print(f"getting ids from {csv_file}", flush=True)
            self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])
        else:
            npz_dir = Path(self.data_path) / self.spec_repr
            print(f"no id csv; deriving ids from {npz_dir}", flush=True)
            self.list_IDs = ids_from_npz_dir(npz_dir)
        if not self.list_IDs:
            raise FileNotFoundError(f"no training samples found under {self.data_path}")

    def partition_data(self, data_split: int):
        """Player ``data_split`` (0..5) becomes the validation set."""
        self.data_split = data_split
        self.partition = {"training": [], "validation": []}
        for ID in self.list_IDs:
            prefix = ID.split("_")[0]
            # GuitarSet files start with the player number 00..05; files from
            # other datasets (synthtab-…, egdb-…) are always used for training.
            guitarist = int(prefix) if prefix.isdigit() else -1
            key = "validation" if guitarist == data_split else "training"
            self.partition[key].append(ID)

        self.training_generator = DataGenerator(
            self.partition["training"],
            data_path=self.data_path,
            batch_size=self.batch_size,
            shuffle=True,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
            workers=self.workers,
        )
        n_val = len(self.partition["validation"])
        self.validation_generator = (
            DataGenerator(
                self.partition["validation"],
                data_path=self.data_path,
                batch_size=n_val,
                shuffle=False,
                spec_repr=self.spec_repr,
                con_win_size=self.con_win_size,
            )
            if n_val
            else None
        )
        self.split_folder = self.save_folder / str(self.data_split)
        self.split_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- model
    def build_model(self):
        self.model = build_model(spec_repr=self.spec_repr, con_win_size=self.con_win_size)
        return self.model

    def log_model(self):
        with open(self.log_file, "w") as fh:
            fh.write(f"batch_size: {self.batch_size}\n")
            fh.write(f"epochs: {self.epochs}\n")
            fh.write(f"spec_repr: {self.spec_repr}\n")
            fh.write(f"data_path: {self.data_path}\n")
            fh.write(f"con_win_size: {self.con_win_size}\n")
            fh.write(f"id_file: {self.id_file}\n")
            fh.write(f"samples: {len(self.list_IDs)}\n")
            self.model.summary(print_fn=lambda x: fh.write(x + "\n"))

    def train(self):
        if len(self.training_generator) == 0:
            raise ValueError(
                f"training set has {len(self.partition['training'])} samples, fewer than "
                f"batch_size={self.batch_size}; lower batch_size or add data"
            )
        return self.model.fit(self.training_generator, epochs=self.epochs, verbose=1)

    def save_weights(self):
        self.model.save_weights(str(self.split_folder / "model.weights.h5"))

    def test(self):
        if self.validation_generator is None:
            print(f"fold {self.data_split}: no validation files for this player, skipping test")
            self.X_test = self.y_gt = self.y_pred = None
            return
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test, verbose=0)

    def save_predictions(self):
        if self.y_pred is None:
            return
        np.savez(self.split_folder / "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        if self.y_pred is None:
            for key in ("pp", "pr", "pf", "tp", "tr", "tf", "tdr"):
                self.metrics[key].append(np.nan)
            return
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        output = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN folds
            for key, vals in self.metrics.items():
                output[key] = [*vals, np.nanmean(vals), np.nanstd(vals)]
        output["data"] = [f"g{f}" for f in self.folds_run] + ["mean", "std dev"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder / "results.csv")
        return df

    def run_cross_validation(self, folds=range(6)):
        self.build_model()
        self.log_model()
        for fold in folds:
            print(f"\nfold {fold}")
            self.partition_data(fold)
            if not self.partition["training"]:
                print("no training data for this fold, skipping")
                continue
            self.build_model()
            print("training...")
            self.train()
            self.save_weights()
            print("testing...")
            self.test()
            self.save_predictions()
            self.evaluate()
            self.folds_run.append(fold)
        print("saving results...")
        return self.save_results_csv()


def ids_from_npz_dir(npz_dir) -> list[str]:
    ids = []
    for npz in sorted(Path(npz_dir).glob("*.npz")):
        n_frames = len(np.load(npz)["labels"])
        ids.extend(f"{npz.stem}_{i}" for i in range(n_frames))
    return ids


if __name__ == "__main__":
    TabCNN().run_cross_validation()
