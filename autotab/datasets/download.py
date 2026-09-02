"""Fetch the openly hosted datasets. SynthTab (Box) and GOAT (access request)
have to be downloaded by hand; this module prints the instructions."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ZENODO = {
    "idmt": {
        "record": 7544110,
        "files": ["IDMT-SMT-GUITAR_V2.zip"],
        "strip_top": True,
    },
    "guitar-techs": {
        "record": 14963133,
        "files": [
            "P1_singlenotes.zip",
            "P1_techniques.zip",
            "P1_scales.zip",
            "P1_chords.zip",
            "P2_singlenotes.zip",
            "P2_techniques.zip",
            "P2_scales.zip",
            "P2_chords.zip",
            "P3_music.zip",
        ],
        "strip_top": False,
    },
}

EGDB_FOLDER = "https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_"

MANUAL = {
    "synthtab": (
        "SynthTab is hosted on the University of Rochester Box:\n"
        "  dev set (a few GB):  https://rochester.app.box.com/v/SynthTab-Dev\n"
        "  full (about 2 TB):   https://rochester.app.box.com/v/SynthTab-Full\n"
        "Download all_jams_midi_V2_60000_tracks.zip plus one timbre zip (acoustic first),\n"
        "unpack them under the same root and run:\n"
        "  autotab preprocess --dataset synthtab --root <root>"
    ),
    "goat": (
        "GOAT files are restricted: request access at https://zenodo.org/records/15690894\n"
        "(describe your intended use). Unpack next to each other the .txt tokens, .mid and\n"
        "*_di.wav / *_amp*.wav files and run:\n"
        "  autotab preprocess --dataset goat --root <root>"
    ),
    "guitarset": (
        "GuitarSet: https://zenodo.org/record/1422265\n"
        "(GuitarSet_audio_and_annotation.zip, 7.5 GB).\n"
        "Unpack so that <root>/annotation/*.jams and <root>/audio/audio_mic/*_mic.wav exist."
    ),
}


def _fetch(url: str, target: Path, progress=print):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        progress(f"already have {target.name}")
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    progress(f"downloading {url} -> {target}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.rename(target)
    return target


def _unzip(zip_path: Path, dest: Path, strip_top: bool, progress=print):
    progress(f"unpacking {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        if strip_top:
            top = {n.split("/")[0] for n in zf.namelist() if "/" in n}
            if len(top) == 1:
                for member in zf.infolist():
                    rel = member.filename.split("/", 1)[1] if "/" in member.filename else ""
                    if not rel or member.is_dir():
                        continue
                    out = dest / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                return
        zf.extractall(dest)


def download(
    name: str, root: Path, parts=None, amps=None, keep_zip=False, workers=4, progress=print
) -> Path:
    root = Path(root)
    if name in MANUAL:
        progress(MANUAL[name])
        return root
    if name in ZENODO:
        spec = ZENODO[name]
        files = [f for f in spec["files"] if not parts or any(p in f for p in parts)]
        for fname in files:
            url = f"https://zenodo.org/api/records/{spec['record']}/files/{fname}/content"
            zip_path = _fetch(url, root / fname, progress)
            _unzip(zip_path, root, spec["strip_top"], progress)
            if not keep_zip:
                zip_path.unlink()
        return root
    if name == "egdb":
        return download_egdb(root, amps=amps, workers=workers, progress=progress)
    raise KeyError(f"no downloader for {name!r}")


def download_egdb(root: Path, amps=None, workers=4, progress=print) -> Path:
    """Fetch labels plus the requested amp folders (default: DI only, ~0.5 GB).
    Google Drive serves each file slowly, so a few parallel workers help a lot."""
    from concurrent.futures import ThreadPoolExecutor

    root = Path(root)

    try:
        import gdown
    except ImportError as exc:
        raise ImportError('EGDB download needs gdown: uv pip install -e ".[data]"') from exc

    wanted = ["audio_label", *(f"audio_{a}" for a in (amps or ["DI"]))]  # labels first
    progress(f"listing {EGDB_FOLDER}")
    files = gdown.download_folder(EGDB_FOLDER, skip_download=True, quiet=True)
    todo = [f for f in files if str(f.path).split("/")[0] in wanted]
    todo.sort(key=lambda f: (wanted.index(str(f.path).split("/")[0]), str(f.path)))
    have = sum((root / str(f.path)).exists() for f in todo)
    progress(f"{len(todo)} files ({have} already present) -> {root}; about 2 MB each")
    missing = [f for f in todo if not (root / str(f.path)).exists()]

    def fetch(f):
        target = root / str(f.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(target), quiet=True)
        return str(f.path)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, path in enumerate(pool.map(fetch, missing), 1):
            progress(f"  [{have + i}/{len(todo)}] {path}")
    progress(f"done. Next: autotab preprocess --dataset egdb --root {root} -j 8")
    return root
