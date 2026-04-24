"""Audio filters and analytical transformations used in the notebook."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pywt
from scipy import signal


@dataclass(frozen=True)
class FilterStep:
    """A reusable filter operation with notebook-friendly metadata."""

    name: str
    title: str
    function: Callable[..., np.ndarray]
    params: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class FilterCoefficients:
    """Filter coefficients used for system-level response plots."""

    name: str
    title: str
    kind: Literal["ba", "sos"]
    b: np.ndarray | None = None
    a: np.ndarray | None = None
    sos: np.ndarray | None = None
    description: str = ""


def _as_signal(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float64)


def _validate_cutoff(sr: int, *cutoffs: float) -> None:
    nyquist = sr / 2.0
    for cutoff in cutoffs:
        if not 0 < cutoff < nyquist:
            raise ValueError(f"cutoff {cutoff} Hz must be between 0 and Nyquist ({nyquist} Hz)")


def _safe_sosfiltfilt(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    x = _as_signal(audio)
    if x.size < 4:
        return signal.sosfilt(sos, x)

    # filtfilt removes phase delay, which makes waveform comparisons easier in
    # the lab. For very short signals we reduce padlen to avoid SciPy errors.
    default_padlen = 3 * (2 * len(sos) + 1)
    padlen = min(default_padlen, x.size - 1)
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, audio: np.ndarray) -> np.ndarray:
    x = _as_signal(audio)
    if x.size < 4:
        return signal.lfilter(b, a, x)

    default_padlen = 3 * max(len(a), len(b))
    padlen = min(default_padlen, x.size - 1)
    return signal.filtfilt(b, a, x, padlen=padlen)


def design_high_pass(sr: int, cutoff: float, order: int = 4) -> FilterCoefficients:
    """Design a Butterworth high-pass filter."""

    _validate_cutoff(sr, cutoff)
    sos = signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return FilterCoefficients("high_pass", f"ФВЧ {cutoff:g} Гц", "sos", sos=sos)


def design_low_pass(sr: int, cutoff: float, order: int = 4) -> FilterCoefficients:
    """Design a Butterworth low-pass filter."""

    _validate_cutoff(sr, cutoff)
    sos = signal.butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return FilterCoefficients("low_pass", f"ФНЧ {cutoff:g} Гц", "sos", sos=sos)


def design_band_pass(sr: int, lowcut: float, highcut: float, order: int = 4) -> FilterCoefficients:
    """Design a Butterworth band-pass filter."""

    if not lowcut < highcut:
        raise ValueError("lowcut must be lower than highcut")
    _validate_cutoff(sr, lowcut, highcut)
    sos = signal.butter(order, (lowcut, highcut), btype="bandpass", fs=sr, output="sos")
    return FilterCoefficients(
        "band_pass",
        f"Полосовой {lowcut:g}-{highcut:g} Гц",
        "sos",
        sos=sos,
    )


def design_notch(sr: int, freq: float, quality: float = 30.0) -> FilterCoefficients:
    """Design a narrow IIR notch filter."""

    _validate_cutoff(sr, freq)
    if quality <= 0:
        raise ValueError("quality must be positive")
    b, a = signal.iirnotch(w0=freq, Q=quality, fs=sr)
    return FilterCoefficients("notch", f"Режекторный {freq:g} Гц", "ba", b=b, a=a)


def design_butterworth_low_pass(sr: int, cutoff: float, order: int = 4) -> FilterCoefficients:
    """Design a Butterworth low-pass filter with maximally flat passband."""

    _validate_cutoff(sr, cutoff)
    sos = signal.butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return FilterCoefficients("butter_low_pass", f"Баттерворт ФНЧ {cutoff:g} Гц", "sos", sos=sos)


def design_chebyshev1_low_pass(
    sr: int,
    cutoff: float,
    order: int = 4,
    ripple: float = 1.0,
) -> FilterCoefficients:
    """Design a Chebyshev Type I low-pass filter."""

    _validate_cutoff(sr, cutoff)
    if ripple <= 0:
        raise ValueError("ripple must be positive")
    sos = signal.cheby1(order, ripple, cutoff, btype="lowpass", fs=sr, output="sos")
    return FilterCoefficients(
        "cheby1_low_pass",
        f"Чебышев I ФНЧ {cutoff:g} Гц",
        "sos",
        sos=sos,
    )


def design_bessel_low_pass(
    sr: int,
    cutoff: float,
    order: int = 4,
    norm: str = "phase",
) -> FilterCoefficients:
    """Design a Bessel low-pass filter.

    Bessel filters are useful in the transient scenario because their phase
    response is smoother near the passband, so waveform shape is less distorted.
    """

    _validate_cutoff(sr, cutoff)
    sos = signal.bessel(order, cutoff, btype="lowpass", fs=sr, output="sos", norm=norm)
    return FilterCoefficients("bessel_low_pass", f"Бессель ФНЧ {cutoff:g} Гц", "sos", sos=sos)


def design_fir_low_pass(sr: int, cutoff: float, numtaps: int = 101) -> FilterCoefficients:
    """Design a windowed-sinc FIR low-pass filter."""

    _validate_cutoff(sr, cutoff)
    if numtaps < 3:
        raise ValueError("numtaps must be at least 3")
    if numtaps % 2 == 0:
        numtaps += 1
    b = signal.firwin(numtaps, cutoff, fs=sr, pass_zero="lowpass")
    a = np.array([1.0], dtype=np.float64)
    return FilterCoefficients("fir_low_pass", f"КИХ ФНЧ {cutoff:g} Гц", "ba", b=b, a=a)


def high_pass_filter(audio: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth high-pass filter."""

    spec = design_high_pass(sr, cutoff, order)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def low_pass_filter(audio: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""

    spec = design_low_pass(sr, cutoff, order)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def band_pass_filter(
    audio: np.ndarray,
    sr: int,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter."""

    spec = design_band_pass(sr, lowcut, highcut, order)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def notch_filter(audio: np.ndarray, sr: int, freq: float, quality: float = 30.0) -> np.ndarray:
    """Apply a narrow notch filter around a selected interference frequency."""

    spec = design_notch(sr, freq, quality)
    return _safe_filtfilt(spec.b, spec.a, audio)  # type: ignore[arg-type]


def butterworth_low_pass(audio: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""

    spec = design_butterworth_low_pass(sr, cutoff, order)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def chebyshev1_low_pass(
    audio: np.ndarray,
    sr: int,
    cutoff: float,
    order: int = 4,
    ripple: float = 1.0,
) -> np.ndarray:
    """Apply a Chebyshev Type I low-pass filter."""

    spec = design_chebyshev1_low_pass(sr, cutoff, order, ripple)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def bessel_low_pass(audio: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Apply a Bessel low-pass filter."""

    spec = design_bessel_low_pass(sr, cutoff, order)
    return _safe_sosfiltfilt(spec.sos, audio)  # type: ignore[arg-type]


def fir_low_pass(audio: np.ndarray, sr: int, cutoff: float, numtaps: int = 101) -> np.ndarray:
    """Apply a FIR low-pass filter.

    A zero-phase application is used for fair visual comparison with IIR
    filters. The underlying FIR coefficients remain linear-phase.
    """

    spec = design_fir_low_pass(sr, cutoff, numtaps)
    return _safe_filtfilt(spec.b, spec.a, audio)  # type: ignore[arg-type]


def wavelet_denoise(
    audio: np.ndarray,
    wavelet: str = "db8",
    level: int | None = None,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """Denoise audio using wavelet soft-thresholding.

    The threshold is estimated from the finest detail coefficients using the
    median absolute deviation. This keeps the method simple and deterministic,
    which is important for comparing two different processing orders.
    """

    if threshold_scale <= 0:
        raise ValueError("threshold_scale must be positive")

    x = _as_signal(audio)
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode="symmetric", level=level)
    if len(coeffs) < 2:
        return x.copy()

    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(x.size, 2)))

    denoised_coeffs = [coeffs[0]]
    denoised_coeffs.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    restored = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode="symmetric")
    return np.asarray(restored[: x.size], dtype=np.float64)


def hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """Return the amplitude envelope from the analytic signal."""

    x = _as_signal(audio)
    if x.size == 0:
        return x.copy()
    return np.abs(signal.hilbert(x))
