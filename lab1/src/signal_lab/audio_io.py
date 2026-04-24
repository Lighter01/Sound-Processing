"""Audio loading and normalization helpers."""

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an already prepared audio sample and resample it if needed.

    The laboratory uses manually prepared excerpts, so this function only loads
    a complete file, converts it to mono when requested, and resamples it to the
    target sampling rate. Timecode trimming is intentionally not implemented.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
    return np.asarray(audio, dtype=np.float64), int(sr)


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize an audio signal without changing silent arrays."""

    if peak <= 0:
        raise ValueError("peak must be positive")

    signal = np.asarray(audio, dtype=np.float64)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs == 0.0:
        return signal.copy()
    return signal / max_abs * peak
