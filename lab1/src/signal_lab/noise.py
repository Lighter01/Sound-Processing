"""White and pink noise generation."""

import numpy as np

from .audio_io import normalize_audio


def generate_white_noise(duration: float, sr: int, seed: int | None = None) -> np.ndarray:
    """Generate zero-mean white noise with unit variance."""

    if duration <= 0:
        raise ValueError("duration must be positive")
    if sr <= 0:
        raise ValueError("sr must be positive")

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration * sr))
    return rng.normal(0.0, 1.0, n_samples).astype(np.float64)


def generate_pink_noise(duration: float, sr: int, seed: int | None = None) -> np.ndarray:
    """Generate pink noise by spectral shaping of white noise.

    Pink noise has power approximately proportional to 1/f. We shape the FFT of
    white noise with an amplitude weight of 1/sqrt(f), then transform it back to
    the time domain. The DC bin is set to zero to avoid division by zero and to
    remove a constant offset from the result.
    """

    white = generate_white_noise(duration, sr, seed)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(white.size, d=1.0 / sr)

    weights = np.zeros_like(freqs)
    nonzero = freqs > 0
    weights[nonzero] = 1.0 / np.sqrt(freqs[nonzero])

    pink = np.fft.irfft(spectrum * weights, n=white.size)
    pink -= np.mean(pink)
    return normalize_audio(pink)
