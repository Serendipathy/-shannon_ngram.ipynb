"""Per-step tables and charts for the notebook. matplotlib is imported lazily."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence

from .config import Settings
from .generator import Paragraph, ParagraphGenerator, Step
from .model import NextWordModel


def step_table(steps: Sequence[Step]) -> list[dict]:
    """One row per sampled word, ready for IPython.display or a DataFrame."""
    return [
        {
            "i": step.index,
            "context": " ".join(step.context) or "_START_",
            "level": step.level,
            "backed_off": step.backed_off,
            "word": step.chosen,
            "p": round(step.probability, 6),
            "entropy_bits": round(step.entropy, 4),
            "candidates": step.candidates,
            "total_count": step.total_count,
        }
        for step in steps
    ]


def entropy_series(steps: Sequence[Step]) -> list[float]:
    return [step.entropy for step in steps]


def max_entropy_series(steps: Sequence[Step]) -> list[float]:
    """Entropy if every candidate were equally likely: log2(candidates)."""
    return [math.log2(step.candidates) if step.candidates > 0 else 0.0 for step in steps]


def mean_entropy(steps: Sequence[Step]) -> float:
    series = entropy_series(steps)
    return fmean(series) if series else 0.0


def plot_entropy(steps: Sequence[Step], ax=None):
    """Entropy in bits per sampled word, with backed-off steps marked.

    The dashed line is the uniform (maximum) entropy for the same candidates;
    the gap between the two lines is how much the corpus prefers some words.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))

    series = entropy_series(steps)
    positions = list(range(len(series)))
    ax.plot(positions, series, linewidth=1.4, label="entropy")
    ax.plot(
        positions,
        max_entropy_series(steps),
        linewidth=1.0,
        linestyle="--",
        label="maximum (uniform)",
    )

    backed = [i for i, s in enumerate(steps) if s.backed_off]
    if backed:
        ax.scatter(
            backed,
            [series[i] for i in backed],
            s=28,
            zorder=3,
            label="backed off",
        )

    ax.set_xlabel("step")
    ax.set_ylabel("entropy (bits)")
    ax.set_title("Next-word entropy per sampled word")
    ax.margins(x=0.01)
    ax.legend(loc="upper right", frameon=False)
    return ax


def compare_context_lengths(
    settings: Settings,
    client,
    seed: int,
    lengths: Sequence[int] = (1, 2, 3, 4),
) -> list[Paragraph]:
    """One paragraph per context length, all from the same seed."""
    from dataclasses import replace

    paragraphs = []
    for length in lengths:
        local = replace(
            settings,
            context_length=length,
            min_context_length=min(settings.min_context_length, length),
        )
        model = NextWordModel(client, local)
        paragraphs.append(ParagraphGenerator(model, local, seed=seed).paragraph())
    return paragraphs


def comparison_table(lengths: Sequence[int], paragraphs: Sequence[Paragraph]) -> list[dict]:
    rows = []
    for length, paragraph in zip(lengths, paragraphs):
        steps = paragraph.steps
        rows.append(
            {
                "context_length": length,
                "effective_level": round(fmean([s.level for s in steps]), 2) if steps else 0,
                "words": len(steps),
                "sentences": len(paragraph.sentences),
                "mean_entropy_bits": round(mean_entropy(steps), 3),
            }
        )
    return rows
