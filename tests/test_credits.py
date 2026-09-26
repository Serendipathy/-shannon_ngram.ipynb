"""AC 12 — the data credit line is present in the README and in notebook Step 12."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "shannon_ngram.ipynb"

CREDIT = "Data: Google Books Ngram v3 (CC BY 3.0) via ngrams.dev"


def test_credit_line_in_readme():
    assert CREDIT in (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _markdown_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown"
    ]


def test_credit_line_in_notebook_step_12():
    step_12 = [src for src in _markdown_cells() if "Step 12" in src]
    assert step_12, "notebook has no Step 12 markdown cell"
    assert any(CREDIT in src for src in step_12)


def test_notebook_has_every_step_from_0_to_12():
    sources = _markdown_cells()
    for step in range(13):
        assert any(f"Step {step} " in src for src in sources), f"missing Step {step}"


def test_notebook_is_committed_without_outputs():
    """Keeps the diff readable and stops API responses leaking into git."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_readme_names_the_install_and_run_commands():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install -e ." in readme
    assert "nbconvert --to notebook --execute" in readme
    assert "NGRAM_CONTEXT_LENGTH" in readme
