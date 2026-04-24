"""Utilities for laboratory work on audio signal analysis and filtering."""

from .config import (
    FRAME_LENGTH,
    HOP_LENGTH,
    MAX_DURATION,
    N_FFT,
    NOISE_DURATION,
    NOISE_KIND,
    RANDOM_SEED,
    SUBGROUP_NUMBER,
    TARGET_SR,
)

__all__ = [
    "FRAME_LENGTH",
    "HOP_LENGTH",
    "MAX_DURATION",
    "N_FFT",
    "NOISE_DURATION",
    "NOISE_KIND",
    "RANDOM_SEED",
    "SUBGROUP_NUMBER",
    "TARGET_SR",
]
