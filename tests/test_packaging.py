"""AC 2 — nothing under src/ hard-codes a machine-specific path."""

from __future__ import annotations

from pathlib import Path

import pytest

import shannon_ngram

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

FORBIDDEN = ("/Users/", "/opt/", "/home/", "C:\\")


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_src_has_python_files():
    assert _source_files(), "no sources found under src/"


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_no_absolute_paths(needle):
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if needle in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_package_imports_and_has_version():
    assert shannon_ngram.__version__ == "0.1.0"


def test_package_is_installed_from_src():
    installed = Path(shannon_ngram.__file__).resolve()
    assert installed == (SRC / "shannon_ngram" / "__init__.py").resolve()
