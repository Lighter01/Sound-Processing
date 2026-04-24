"""Matplotlib visualizations for signal and filter comparisons."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    amplitude_to_db,
    compute_autocorrelation,
    compute_band_energy,
    compute_frequency_response,
    compute_impulse_response,
    compute_magnitude_spectrum,
    compute_mel_spectrogram,
    compute_phase_response,
    compute_psd_welch,
    compute_rms,
    compute_stft_spectrogram,
    filter_spec_kwargs,
    power_to_db,
)
from .filters import FilterCoefficients

SignalMap = Mapping[str, np.ndarray]


def _make_axes(n_items: int, title: str, height_per_axis: float = 2.4, sharex: bool = False):
    fig, axes = plt.subplots(
        n_items,
        1,
        figsize=(12, max(3.2, height_per_axis * n_items)),
        sharex=sharex,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=14)
    return fig, axes_array


def _limit_frequency_axis(
    ax,
    sr: int,
    upper_hz: float | None = None,
    axis: str = "x",
) -> None:
    upper = min(upper_hz if upper_hz is not None else sr / 2.0, sr / 2.0)
    if axis == "y":
        ax.set_ylim(0, upper)
    else:
        ax.set_xlim(0, upper)


def _set_subplot_title(ax, label: str) -> None:
    """Put the signal label on the subplot itself to avoid y-label ambiguity."""

    ax.set_title(label, loc="left", fontsize=10, pad=4)


def _enable_dashed_grid(ax) -> None:
    """Enable a local dashed grid for line-based plots only."""

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)


def plot_waveforms_stack(signals: SignalMap, sr: int, title: str):
    """Plot stacked waveforms, one subplot per signal."""

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        x = np.asarray(audio)
        times = np.arange(x.size) / sr
        ax.plot(times, x, linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        _set_subplot_title(ax, label)
        ax.set_ylabel("Amplitude")
        _enable_dashed_grid(ax)
    axes[-1].set_xlabel("Time, s")
    return fig, axes


def plot_spectra_stack(
    signals: SignalMap,
    sr: int,
    title: str,
    upper_hz: float | None = 20_000,
):
    """Plot stacked amplitude spectra in dB."""

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        freqs, magnitude = compute_magnitude_spectrum(audio, sr)
        ax.plot(freqs, amplitude_to_db(magnitude), linewidth=0.9)
        _set_subplot_title(ax, label)
        ax.set_ylabel("Magnitude, dB")
        _enable_dashed_grid(ax)
        _limit_frequency_axis(ax, sr, upper_hz)
    axes[-1].set_xlabel("Frequency, Hz")
    return fig, axes


def plot_psd_stack(
    signals: SignalMap,
    sr: int,
    title: str,
    upper_hz: float | None = 20_000,
):
    """Plot stacked Welch PSD estimates in dB."""

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        freqs, psd = compute_psd_welch(audio, sr)
        ax.plot(freqs, power_to_db(psd), linewidth=0.9)
        _set_subplot_title(ax, label)
        ax.set_ylabel("PSD, dB")
        _enable_dashed_grid(ax)
        _limit_frequency_axis(ax, sr, upper_hz)
    axes[-1].set_xlabel("Frequency, Hz")
    return fig, axes


def plot_spectrograms_stack(
    signals: SignalMap,
    sr: int,
    title: str,
    upper_hz: float | None = 20_000,
):
    """Plot stacked STFT spectrograms."""

    fig, axes = _make_axes(len(signals), title, height_per_axis=2.8, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        times, freqs, spectrogram_db = compute_stft_spectrogram(audio, sr)
        mesh = ax.pcolormesh(times, freqs, spectrogram_db, shading="auto", cmap="magma")
        _set_subplot_title(ax, label)
        ax.set_ylabel("Frequency, Hz")
        _limit_frequency_axis(ax, sr, upper_hz, axis="y")
        fig.colorbar(mesh, ax=ax, label="dB")
    axes[-1].set_xlabel("Time, s")
    return fig, axes


def plot_rms_stack(signals: SignalMap, sr: int, title: str):
    """Plot stacked RMS energy curves."""

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        times, rms = compute_rms(audio, sr)
        ax.plot(times, rms, linewidth=1.0)
        _set_subplot_title(ax, label)
        ax.set_ylabel("RMS")
        _enable_dashed_grid(ax)
    axes[-1].set_xlabel("Time, s")
    return fig, axes


def plot_mel_spectrograms_stack(signals: SignalMap, sr: int, title: str):
    """Plot stacked mel spectrograms."""

    fig, axes = _make_axes(len(signals), title, height_per_axis=2.8, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        times, mel_freqs, mel_db = compute_mel_spectrogram(audio, sr)
        mesh = ax.pcolormesh(times, mel_freqs, mel_db, shading="auto", cmap="viridis")
        _set_subplot_title(ax, label)
        ax.set_ylabel("Mel frequency, Hz")
        fig.colorbar(mesh, ax=ax, label="dB")
    axes[-1].set_xlabel("Time, s")
    return fig, axes


def plot_envelopes_stack(signals: SignalMap, sr: int, title: str):
    """Plot waveforms together with Hilbert envelopes."""

    from .analysis import compute_hilbert_envelope

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        x = np.asarray(audio)
        times = np.arange(x.size) / sr
        envelope = compute_hilbert_envelope(x)
        ax.plot(times, x, linewidth=0.55, alpha=0.55, label="Signal")
        ax.plot(times, envelope, linewidth=1.0, label="Hilbert envelope")
        _set_subplot_title(ax, label)
        ax.set_ylabel("Amplitude")
        _enable_dashed_grid(ax)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time, s")
    return fig, axes


def plot_autocorrelation_stack(
    signals: SignalMap,
    sr: int,
    title: str,
    max_lag_seconds: float = 0.05,
):
    """Plot normalized autocorrelation for selected short fragments."""

    fig, axes = _make_axes(len(signals), title, sharex=True)
    for ax, (label, audio) in zip(axes, signals.items(), strict=True):
        lags, corr = compute_autocorrelation(audio, sr, max_lag_seconds=max_lag_seconds)
        ax.plot(lags * 1000.0, corr, linewidth=1.0)
        _set_subplot_title(ax, label)
        ax.set_ylabel("Correlation")
        _enable_dashed_grid(ax)
    axes[-1].set_xlabel("Lag, ms")
    return fig, axes


def plot_band_energy_comparison(
    signals: SignalMap,
    sr: int,
    bands: Mapping[str, tuple[float, float]],
    title: str,
):
    """Plot grouped bar chart with band energy estimates."""

    labels = list(signals.keys())
    band_names = list(bands.keys())
    values = np.array(
        [[compute_band_energy(audio, sr, bands)[band] for band in band_names] for audio in signals.values()]
    )

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.8 / max(len(band_names), 1)
    for idx, band_name in enumerate(band_names):
        ax.bar(x + idx * width - 0.4 + width / 2.0, values[:, idx], width, label=band_name)
    ax.set_title(title)
    ax.set_ylabel("Integrated PSD energy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    return fig, ax


def plot_power_comparison(signals: SignalMap, title: str):
    """Plot mean signal power for several audio versions."""

    labels = list(signals.keys())
    powers = [float(np.mean(np.asarray(audio) ** 2)) for audio in signals.values()]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(labels, powers)
    ax.set_title(title)
    ax.set_ylabel("Mean power")
    ax.tick_params(axis="x", rotation=20)
    return fig, ax


def plot_filter_response_comparison(
    filters: Sequence[FilterCoefficients],
    sr: int,
    title: str,
    upper_hz: float | None = 20_000,
):
    """Plot system magnitude responses for filter designs."""

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for spec in filters:
        freqs, response_db = compute_frequency_response(sr=sr, **filter_spec_kwargs(spec))
        ax.plot(freqs, response_db, linewidth=1.1, label=spec.title)
    ax.set_title(title)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Magnitude, dB")
    _enable_dashed_grid(ax)
    _limit_frequency_axis(ax, sr, upper_hz)
    ax.legend()
    return fig, ax


def plot_filter_phase_response_comparison(
    filters: Sequence[FilterCoefficients],
    sr: int,
    title: str,
    upper_hz: float | None = 20_000,
):
    """Plot system phase responses for filter designs."""

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for spec in filters:
        freqs, phase = compute_phase_response(sr=sr, **filter_spec_kwargs(spec))
        ax.plot(freqs, phase, linewidth=1.1, label=spec.title)
    ax.set_title(title)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Unwrapped phase, rad")
    _enable_dashed_grid(ax)
    _limit_frequency_axis(ax, sr, upper_hz)
    ax.legend()
    return fig, ax


def plot_filter_impulse_response_comparison(
    filters: Sequence[FilterCoefficients],
    sr: int,
    title: str,
    n_samples: int = 512,
):
    """Plot causal impulse responses for filter designs."""

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for spec in filters:
        samples, response = compute_impulse_response(n_samples=n_samples, **filter_spec_kwargs(spec))
        ax.plot(samples / sr * 1000.0, response, linewidth=1.0, label=spec.title)
    ax.set_title(title)
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Amplitude")
    _enable_dashed_grid(ax)
    ax.legend()
    return fig, ax
