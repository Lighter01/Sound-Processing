"""Signal and filter analysis functions."""

from collections.abc import Mapping

import librosa
import numpy as np
from scipy import signal

from .filters import FilterCoefficients, hilbert_envelope


def _as_signal(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float64)


def _align_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(reference.size, estimate.size)
    if n == 0:
        raise ValueError("signals must be non-empty")
    return reference[:n], estimate[:n]


def amplitude_to_db(values: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """Convert non-negative amplitude-like values to dB with a finite floor."""

    safe = np.maximum(np.asarray(values, dtype=np.float64), 10 ** (floor_db / 20.0))
    return 20.0 * np.log10(safe)


def power_to_db(values: np.ndarray, floor_db: float = -160.0) -> np.ndarray:
    """Convert non-negative power-like values to dB with a finite floor."""

    safe = np.maximum(np.asarray(values, dtype=np.float64), 10 ** (floor_db / 10.0))
    return 10.0 * np.log10(safe)


def compute_magnitude_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided amplitude spectrum using a Hann window."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([])

    window = np.hanning(x.size)
    windowed = x * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)

    # The coherent gain of a Hann window is mean(window). Dividing by it keeps
    # sinusoidal amplitudes comparable between different signals.
    coherent_gain = np.mean(window) if np.any(window) else 1.0
    magnitude = np.abs(spectrum) / max(x.size * coherent_gain, 1.0)
    if magnitude.size > 2:
        magnitude[1:-1] *= 2.0
    return freqs, magnitude


def compute_power_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple one-sided power spectrum."""

    freqs, magnitude = compute_magnitude_spectrum(audio, sr)
    return freqs, magnitude**2


def compute_psd_welch(
    audio: np.ndarray,
    sr: int,
    nperseg: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD with Welch averaging."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([])
    segment = min(nperseg, x.size)
    return signal.welch(x, fs=sr, nperseg=segment, scaling="density")


def compute_stft_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an STFT magnitude spectrogram in dB."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([]), np.empty((0, 0))

    nperseg = min(n_fft, x.size)
    noverlap = min(max(nperseg - hop_length, 0), nperseg - 1)
    freqs, times, stft = signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude_db = amplitude_to_db(np.abs(stft))
    return times, freqs, magnitude_db


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a mel-scaled power spectrogram in dB."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([]), np.empty((0, 0))

    mel_power = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=min(n_fft, max(x.size, 2)),
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop_length)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2.0)
    return times, mel_freqs, mel_db


def compute_rms(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMS energy in a sliding window."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([])
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop_length)
    return times, rms


def compute_hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """Compute an amplitude envelope via the Hilbert transform."""

    return hilbert_envelope(audio)


def compute_autocorrelation(
    audio: np.ndarray,
    sr: int,
    max_lag_seconds: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized non-negative autocorrelation lags."""

    x = _as_signal(audio)
    if x.size == 0:
        return np.array([]), np.array([])
    centered = x - np.mean(x)
    corr = signal.correlate(centered, centered, mode="full", method="auto")
    corr = corr[corr.size // 2 :]
    if corr[0] != 0:
        corr = corr / corr[0]
    max_lag = min(int(round(max_lag_seconds * sr)), corr.size - 1)
    lags = np.arange(max_lag + 1) / sr
    return lags, corr[: max_lag + 1]


def compute_band_energy(
    audio: np.ndarray,
    sr: int,
    bands: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    """Estimate energy in frequency bands by integrating Welch PSD."""

    freqs, psd = compute_psd_welch(audio, sr)
    energies: dict[str, float] = {}
    for name, (low, high) in bands.items():
        clipped_high = min(high, sr / 2.0)
        mask = (freqs >= low) & (freqs <= clipped_high)
        energies[name] = float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    return energies


def compute_snr(clean: np.ndarray, processed_or_noisy: np.ndarray) -> float:
    """Compute SNR in dB against a clean reference signal."""

    reference, estimate = _align_pair(_as_signal(clean), _as_signal(processed_or_noisy))
    noise = estimate - reference
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compute_frequency_response(
    sr: int,
    b: np.ndarray | None = None,
    a: np.ndarray | None = None,
    sos: np.ndarray | None = None,
    worN: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a filter magnitude response in dB."""

    if sos is not None:
        freqs, response = signal.sosfreqz(sos, worN=worN, fs=sr)
    elif b is not None and a is not None:
        freqs, response = signal.freqz(b, a, worN=worN, fs=sr)
    else:
        raise ValueError("provide either sos or both b and a")
    return freqs, amplitude_to_db(np.abs(response))


def compute_phase_response(
    sr: int,
    b: np.ndarray | None = None,
    a: np.ndarray | None = None,
    sos: np.ndarray | None = None,
    worN: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an unwrapped filter phase response in radians."""

    if sos is not None:
        freqs, response = signal.sosfreqz(sos, worN=worN, fs=sr)
    elif b is not None and a is not None:
        freqs, response = signal.freqz(b, a, worN=worN, fs=sr)
    else:
        raise ValueError("provide either sos or both b and a")
    return freqs, np.unwrap(np.angle(response))


def compute_impulse_response(
    b: np.ndarray | None = None,
    a: np.ndarray | None = None,
    sos: np.ndarray | None = None,
    n_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the causal impulse response of a filter."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    if sos is not None:
        response = signal.sosfilt(sos, impulse)
    elif b is not None and a is not None:
        response = signal.lfilter(b, a, impulse)
    else:
        raise ValueError("provide either sos or both b and a")
    return np.arange(n_samples), response


def filter_spec_kwargs(spec: FilterCoefficients) -> dict[str, np.ndarray]:
    """Return coefficient keyword arguments for response helper functions."""

    if spec.kind == "sos":
        if spec.sos is None:
            raise ValueError(f"{spec.title} has no SOS coefficients")
        return {"sos": spec.sos}
    if spec.b is None or spec.a is None:
        raise ValueError(f"{spec.title} has no b/a coefficients")
    return {"b": spec.b, "a": spec.a}
