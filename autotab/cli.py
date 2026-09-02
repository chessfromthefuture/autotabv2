"""Command line interface: ``autotab predict|preprocess|train``."""

from __future__ import annotations

import argparse
import os
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
            isolate=args.isolate,
        )
        if len(args.audio) > 1:
            print(f"### {path}")
        print(tab)
        if args.output:
            with open(args.output, "a" if len(args.audio) > 1 else "w") as fh:
                fh.write(tab + "\n")


def cmd_separate(args):
    from autotab.separate import isolate_to_file

    for path in args.audio:
        out = args.output or f"{os.path.splitext(path)[0]}_guitar.wav"
        written = isolate_to_file(path, out, stems=args.stems, model=args.model, device=args.device)
        for stem, target in written.items():
            print(f"{stem}: {target}")


def _preprocess_one(job):
    from autotab.TabDataReprGen import TabDataReprGen

    track, mode = job
    out = TabDataReprGen(mode=mode).save_path / f"{track.stem}.npz"
    if out.exists():
        return track.stem
    TabDataReprGen(mode=mode).save_track(track)
    return track.stem


def _dataset_options(args):
    opts = {}
    if args.amps:
        opts["amps"] = args.amps
    if args.mics:
        opts["mics"] = args.mics
    if args.timbres:
        opts["timbres"] = args.timbres
    if args.variants:
        opts["variants"] = args.variants
    return opts


def cmd_preprocess(args):
    from autotab import datasets
    from autotab.param import GUITARSET_DIR

    root = args.root or (GUITARSET_DIR if args.dataset == "guitarset" else None)
    if root is None:
        raise SystemExit(f"--root is required for --dataset {args.dataset}")
    tracks = datasets.tracks(args.dataset, root, **_dataset_options(args))
    if args.index is not None:
        tracks = [tracks[i] for i in args.index]
    if args.limit:
        tracks = tracks[: args.limit]
    if not tracks:
        raise SystemExit(f"no {args.dataset} tracks found under {root}")
    jobs = [(t, args.mode) for t in tracks]
    print(
        f"preprocessing {len(jobs)} {args.dataset} track(s) from {root} "
        f"with mode={args.mode}, {args.jobs} worker(s)"
    )
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(_preprocess_one, jobs)
    else:
        for job in jobs:
            _preprocess_one(job)


def cmd_datasets(args):
    from autotab import datasets

    for name, desc in datasets.DESCRIPTIONS.items():
        print(f"{name:13s} {desc}")
    print("\nDownload: autotab download <name> --root <dir>")
    print("          (synthtab, goat and guitarset print the manual steps)")
    print("Prepare:  autotab preprocess --dataset <name> --root <dir> [--limit N] [-j 8]")


def cmd_download(args):
    from autotab.datasets.download import download

    download(args.name, args.root, parts=args.parts, amps=args.amps, keep_zip=args.keep_zip)


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
    pr.add_argument(
        "--isolate",
        action="store_true",
        help="separate the guitar from a full mix first (needs the 'separate' extra)",
    )
    pr.set_defaults(func=cmd_predict)

    se = sub.add_parser("separate", help="isolate the guitar (and other stems) from a full song")
    se.add_argument("audio", nargs="+", help="wav/flac/ogg/mp3 file(s)")
    se.add_argument("-o", "--output", help="output wav for the guitar stem (single input only)")
    se.add_argument("--stems", nargs="+", default=["guitar"], help="e.g. guitar other vocals")
    se.add_argument("--model", default="htdemucs_6s")
    se.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    se.set_defaults(func=cmd_separate)

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

    pp = sub.add_parser("preprocess", help="build npz representations from a dataset")
    pp.add_argument(
        "--dataset", default="guitarset", help="guitarset, synthtab, egdb, goat, idmt, guitar-techs"
    )
    pp.add_argument("--root", help="dataset folder (default for guitarset: data/GuitarSet)")
    pp.add_argument("--mode", choices=["c", "m", "cm", "s"], default="c")
    pp.add_argument("--index", type=int, nargs="*", help="only these track indices (sorted)")
    pp.add_argument("--limit", type=int, help="stop after this many tracks")
    pp.add_argument("--amps", nargs="*", help="egdb: amp folders, e.g. DI Marshall")
    pp.add_argument("--mics", choices=["first", "all"], help="synthtab: microphones per guitar")
    pp.add_argument("--timbres", nargs="*", help="synthtab: acoustic electric_clean …")
    pp.add_argument("--variants", nargs="*", help="goat: di amp1 …")
    pp.add_argument("-j", "--jobs", type=int, default=1)
    pp.set_defaults(func=cmd_preprocess)

    ds = sub.add_parser("datasets", help="list supported datasets")
    ds.set_defaults(func=cmd_datasets)

    dl = sub.add_parser("download", help="fetch an open dataset (or print manual steps)")
    dl.add_argument("name", help="egdb, idmt, guitar-techs, synthtab, goat, guitarset")
    dl.add_argument("--root", required=True, help="where to put it")
    dl.add_argument("--parts", nargs="*", help="guitar-techs: zip names to fetch, e.g. P3_music")
    dl.add_argument("--amps", nargs="*", help="egdb: amp folders (default DI)")
    dl.add_argument("--keep-zip", action="store_true")
    dl.set_defaults(func=cmd_download)

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
