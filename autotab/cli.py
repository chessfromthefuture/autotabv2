"""Command line interface: ``autotab predict|preprocess|train``."""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool

from autotab.evaluate import DEFAULT_SILENCE_WEIGHT
from autotab.param import MODEL_PATH


def cmd_predict(args):
    from autotab.TabPrediction import load_model_and_weights, predict_tab

    model = load_model_and_weights(args.weights)
    for path in args.audio:
        tab = predict_tab(
            path,
            model=model,
            mode=args.mode,
            num_div=args.bars,
            len_div=args.bar_len,
            silence_weight=args.silence_weight,
        )
        if len(args.audio) > 1:
            print(f"### {path}")
        print(tab)
        if args.output:
            with open(args.output, "a" if len(args.audio) > 1 else "w") as fh:
                fh.write(tab + "\n")


def _preprocess_one(job):
    from autotab.TabDataReprGen import TabDataReprGen

    name, mode = job
    TabDataReprGen(mode=mode).load_and_save_repr_file(name)
    return name


def cmd_preprocess(args):
    from autotab.TabDataReprGen import TabDataReprGen

    names = TabDataReprGen(mode=args.mode).list_filenames()
    if args.index is not None:
        names = [names[i] for i in args.index]
    jobs = [(n, args.mode) for n in names]
    print(f"preprocessing {len(jobs)} file(s) with mode={args.mode}, {args.jobs} worker(s)")
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(_preprocess_one, jobs)
    else:
        for job in jobs:
            _preprocess_one(job)


def cmd_evaluate(args):
    from autotab.evaluate import apply_silence_weight, load_predictions, metrics
    from autotab.TabPrediction import load_model_and_weights

    model = load_model_and_weights(args.weights)
    y_pred, y_gt, names = load_predictions(model, args.npz_dir, args.files, args.mode)
    print(f"{len(names)} file(s), {len(y_gt)} frames: {', '.join(names)}\n")
    rows = {
        "raw model (weight 1.0)": metrics(y_pred, y_gt),
        f"silence weight {args.silence_weight}": metrics(
            apply_silence_weight(y_pred, args.silence_weight), y_gt
        ),
    }
    import pandas as pd

    print(pd.DataFrame(rows).round(3).to_string())


def cmd_calibrate(args):
    from autotab.evaluate import best_weight, load_predictions, sweep
    from autotab.TabPrediction import load_model_and_weights

    model = load_model_and_weights(args.weights)
    y_pred, y_gt, names = load_predictions(model, args.npz_dir, args.files, args.mode)
    print(f"{len(names)} file(s), {len(y_gt)} frames\n")
    table = sweep(y_pred, y_gt, args.weights_grid)
    print(table.round(3).to_string())
    best = best_weight(table, args.metric)
    print(f"\nbest silence weight by {args.metric}: {best}")
    if args.output:
        table.to_csv(args.output)
        print(f"table written to {args.output}")


def cmd_train(args):
    from autotab.TabCNN import TabCNN

    tabcnn = TabCNN(
        batch_size=args.batch_size,
        epochs=args.epochs,
        spec_repr=args.mode,
        id_file=args.id_file,
        workers=args.jobs,
    )
    folds = args.folds if args.folds else range(6)
    df = tabcnn.run_cross_validation(folds)
    print(df.to_string())
    print(f"\nresults written to {tabcnn.save_folder}")


def build_parser():
    p = argparse.ArgumentParser(prog="autotab", description="Guitar tablature from audio.")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("predict", help="transcribe one or more wav files to tablature")
    pr.add_argument("audio", nargs="+", help="wav/flac/ogg file(s)")
    pr.add_argument("--mode", choices=["simple", "rhythm", "frames"], default="simple")
    pr.add_argument("--weights", default=str(MODEL_PATH))
    pr.add_argument("--bars", type=int, default=4, help="bars per line")
    pr.add_argument("--bar-len", type=int, default=16, help="characters per bar")
    pr.add_argument("-o", "--output", help="also write the tab to this text file")
    pr.add_argument(
        "--silence-weight",
        type=float,
        default=DEFAULT_SILENCE_WEIGHT,
        help="multiplier on the 'not played' class; 1.0 = raw model, lower = more notes",
    )
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="metrics on preprocessed GuitarSet npz files")
    ev.add_argument("--weights", default=str(MODEL_PATH))
    ev.add_argument("--mode", choices=["c", "m", "cm", "s"], default="c")
    ev.add_argument("--npz-dir", help="defaults to data/spec_repr/<mode>")
    ev.add_argument("--files", nargs="*", help="only these file stems")
    ev.add_argument("--silence-weight", type=float, default=DEFAULT_SILENCE_WEIGHT)
    ev.set_defaults(func=cmd_evaluate)

    ca = sub.add_parser("calibrate", help="sweep the silence weight and report the best")
    ca.add_argument("--weights", default=str(MODEL_PATH))
    ca.add_argument("--mode", choices=["c", "m", "cm", "s"], default="c")
    ca.add_argument("--npz-dir")
    ca.add_argument("--files", nargs="*")
    ca.add_argument("--metric", default="tab_f", help="column to maximise, e.g. tab_f, pitch_f")
    ca.add_argument("--weights-grid", type=float, nargs="*", default=None)
    ca.add_argument("-o", "--output", help="write the sweep table to this csv")
    ca.set_defaults(func=cmd_calibrate)

    pp = sub.add_parser("preprocess", help="build npz representations from GuitarSet")
    pp.add_argument("--mode", choices=["c", "m", "cm", "s"], default="c")
    pp.add_argument("--index", type=int, nargs="*", help="only these file indices (sorted)")
    pp.add_argument("-j", "--jobs", type=int, default=1)
    pp.set_defaults(func=cmd_preprocess)

    tr = sub.add_parser("train", help="6-fold cross-validation training")
    tr.add_argument("--mode", choices=["c", "m", "cm", "s"], default="c")
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--folds", type=int, nargs="*", help="subset of folds 0..5")
    tr.add_argument(
        "--id-file",
        default="id.csv",
        help="csv of sample ids under data/spec_repr; pass '' to derive from npz files",
    )
    tr.add_argument("-j", "--jobs", type=int, default=1, help="data loader workers")
    tr.set_defaults(func=cmd_train)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "weights_grid", "x") is None:
        from autotab.evaluate import SWEEP_WEIGHTS

        args.weights_grid = SWEEP_WEIGHTS
    if getattr(args, "id_file", None) == "":
        args.id_file = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
