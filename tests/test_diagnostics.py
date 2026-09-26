"""Maximum (uniform) entropy series and its line on the entropy plot."""

from __future__ import annotations

import math

import pytest

from shannon_ngram import diagnostics
from shannon_ngram.generator import Step


def _step(index: int, entropy: float, candidates: int, backed_off: bool = False) -> Step:
    return Step(
        index=index,
        context=("ctx",),
        level=1,
        backed_off=backed_off,
        chosen="w",
        probability=0.5,
        entropy=entropy,
        candidates=candidates,
        total_count=10,
    )


STEPS = [_step(0, 0.9, 4), _step(1, 0.0, 1), _step(2, 2.5, 8, backed_off=True)]


def test_max_entropy_is_log2_of_candidates():
    assert diagnostics.max_entropy_series(STEPS) == [2.0, 0.0, 3.0]


def test_max_entropy_bounds_the_entropy():
    for step, top in zip(STEPS, diagnostics.max_entropy_series(STEPS)):
        assert step.entropy <= top


def test_max_entropy_with_no_candidates_is_zero():
    assert diagnostics.max_entropy_series([_step(0, 0.0, 0)]) == [0.0]


def test_plot_entropy_draws_the_maximum_line():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    ax = diagnostics.plot_entropy(STEPS)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert list(lines["maximum (uniform)"].get_ydata()) == [
        math.log2(4),
        math.log2(1),
        math.log2(8),
    ]
    assert list(lines["entropy"].get_ydata()) == [0.9, 0.0, 2.5]
