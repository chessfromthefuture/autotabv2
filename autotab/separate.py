"""Isolate the guitar from a full mix before transcription.

Uses Demucs' 6-stem model (``htdemucs_6s``: drums, bass, other, vocals,
guitar, piano). Install the optional extra first::

    uv pip install -e ".[separate]"

The first run downloads the ~80 MB checkpoint to ~/.cache/torch."""

from __future__ import annotations

import functools
import os

import numpy as np

DEFAULT_MODEL = "htdemucs_6s"
GUITAR_STEM = "guitar"


def _require_demucs():
    try:
        import demucs.api  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "guitar isolation needs the optional 'separate' extra: "
            'uv pip install -e ".[separate]"'
        ) from exc


def pick_device(preferred: str | None = None) -> str:
    """Best available torch device: explicit > AUTOTAB_DEVICE > mps > cuda > cpu."""
    _require_demucs()
    import torch

    choice = preferred or os.environ.get("AUTOTAB_DEVICE")
    if choice:
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=2)
def get_separator(model: str = DEFAULT_MODEL, device: str | None = None):
    _require_demucs()
    from demucs.api import Separator

    return Separator(model=model, device=pick_device(device), progress=False)


def separate(
    audio: np.ndarray, sr: int, stems=(GUITAR_STEM,), model: str = DEFAULT_MODEL, device=None
) -> dict[str, np.ndarray]:
    """Split ``audio`` of shape (samples,) or (samples, channels) into the
    requested stems. Returns {stem: (samples, channels) float32}."""
    import torch

    audio = np.asarray(audio, dtype="float32")
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)  # Demucs expects stereo
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    separator = get_separator(model, device)
    _, out = separator.separate_tensor(torch.from_numpy(audio.T.copy()), sr)
    unknown = set(stems) - set(out)
    if unknown:
        raise ValueError(
            f"model {model} has no stem(s) {sorted(unknown)}; available: {sorted(out)}"
        )
    return {name: out[name].cpu().numpy().T for name in stems}


def isolate_guitar(source, model: str = DEFAULT_MODEL, device=None) -> tuple[np.ndarray, int]:
    """Read an audio file (path or file-like; wav/flac/ogg/mp3) and return the
    guitar stem as mono float32 plus its sample rate."""
    import soundfile as sf

    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    guitar = separate(audio, sr, stems=(GUITAR_STEM,), model=model, device=device)[GUITAR_STEM]
    return guitar.mean(axis=1), sr


def isolate_to_file(source, out_path, stems=(GUITAR_STEM,), model=DEFAULT_MODEL, device=None):
    """Write each requested stem next to ``out_path`` (the guitar stem gets
    ``out_path`` itself, other stems get ``<out_path stem>_<stem>.wav``)."""
    from pathlib import Path

    import soundfile as sf

    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    out_path = Path(out_path)
    written = {}
    for name, data in separate(audio, sr, stems=stems, model=model, device=device).items():
        target = (
            out_path if name == GUITAR_STEM else out_path.with_name(f"{out_path.stem}_{name}.wav")
        )
        sf.write(target, data, sr)
        written[name] = target
    return written
