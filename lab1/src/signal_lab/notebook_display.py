"""Notebook display helpers for Markdown notes and audio players."""

import numpy as np
from IPython.display import Audio, Markdown, display

from .filters import FilterStep


def show_markdown_header(title: str, level: int = 2) -> None:
    """Display a Markdown header with a clamped heading level."""

    safe_level = min(max(level, 1), 6)
    display(Markdown(f"{'#' * safe_level} {title}"))


def show_markdown_note(text: str) -> None:
    """Display a Markdown note or explanation."""

    display(Markdown(text))


def show_audio_player(title: str, audio: np.ndarray, sr: int) -> None:
    """Display an audio player with a Markdown title."""

    display(Markdown(f"#### {title}"))
    display(Audio(np.asarray(audio, dtype=np.float64), rate=sr))


def show_signal_block(title: str, description: str, audio: np.ndarray, sr: int) -> None:
    """Display a described source signal block."""

    display(Markdown(f"### {title}\n\n{description}"))
    show_audio_player(title, audio, sr)


def show_filter_result_block(step: FilterStep, audio: np.ndarray, sr: int) -> None:
    """Display a filter result with its explanation and audio player."""

    description = f"\n\n{step.description}" if step.description else ""
    display(Markdown(f"### {step.title}{description}"))
    show_audio_player(step.title, audio, sr)


def show_chain_result_block(title: str, description: str, audio: np.ndarray, sr: int) -> None:
    """Display the final result of a processing chain."""

    display(Markdown(f"### {title}\n\n{description}"))
    show_audio_player(title, audio, sr)
