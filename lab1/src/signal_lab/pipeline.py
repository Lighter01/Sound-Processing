"""Helpers for applying individual filter steps and chains."""

from collections import OrderedDict

import numpy as np

from .filters import FilterStep


def apply_individual_filters(
    audio: np.ndarray,
    sr: int,
    filter_steps: list[FilterStep] | tuple[FilterStep, ...],
) -> "OrderedDict[str, np.ndarray]":
    """Apply each filter step independently to the same input signal."""

    results: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for step in filter_steps:
        results[step.title] = step.function(audio, sr, **step.params)
    return results


def apply_chain(
    audio: np.ndarray,
    sr: int,
    chain_steps: list[FilterStep] | tuple[FilterStep, ...],
) -> np.ndarray:
    """Apply filter steps sequentially."""

    result = np.asarray(audio, dtype=np.float64).copy()
    for step in chain_steps:
        result = step.function(result, sr, **step.params)
    return result


def build_signal_versions(
    original: np.ndarray,
    individual_results: "OrderedDict[str, np.ndarray] | dict[str, np.ndarray]",
    chain_result: np.ndarray,
    original_label: str = "Исходный сигнал",
    chain_label: str = "После полной цепочки",
) -> "OrderedDict[str, np.ndarray]":
    """Build an ordered mapping for stacked comparison plots."""

    versions: "OrderedDict[str, np.ndarray]" = OrderedDict()
    versions[original_label] = original
    for label, signal in individual_results.items():
        versions[f"После {label}"] = signal
    versions[chain_label] = chain_result
    return versions
