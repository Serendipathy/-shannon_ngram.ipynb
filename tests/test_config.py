"""AC 3 — every setting comes from the environment or .env."""

from __future__ import annotations

import os
import re
from dataclasses import fields
from pathlib import Path

import pytest

from shannon_ngram.config import (
    FIELD_ENV,
    Settings,
    default_cache_path,
    load_settings,
    settings_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env():
    # load_dotenv writes straight into os.environ, so monkeypatch cannot undo it.
    saved = os.environ.copy()
    for env_var, _ in FIELD_ENV.values():
        os.environ.pop(env_var, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_defaults_match_the_req():
    s = load_settings()
    assert s.api_base_url == "https://api.ngrams.dev"
    assert s.corpus == "eng"
    assert s.context_length == 4
    assert s.min_context_length == 1
    assert s.max_sentences == 5
    assert s.result_limit == 100
    assert s.max_pages == 1
    assert s.case_sensitive is False
    assert s.cache_path == default_cache_path()


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("NGRAM_CONTEXT_LENGTH", "3")
    monkeypatch.setenv("NGRAM_MAX_SENTENCES", "2")
    monkeypatch.setenv("NGRAM_RESULT_LIMIT", "25")
    monkeypatch.setenv("NGRAM_CASE_SENSITIVE", "true")
    monkeypatch.setenv("NGRAM_CACHE_PATH", "/tmp/x/ngrams.sqlite3")
    monkeypatch.setenv("NGRAM_API_BASE_URL", "https://example.invalid")

    s = load_settings()
    assert s.context_length == 3
    assert s.max_sentences == 2
    assert s.result_limit == 25
    assert s.case_sensitive is True
    assert s.cache_path == Path("/tmp/x/ngrams.sqlite3")
    assert s.api_base_url == "https://example.invalid"


def test_blank_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("NGRAM_CACHE_PATH", "")
    assert load_settings().cache_path == default_cache_path()


def test_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("NGRAM_CONTEXT_LENGTH", "3")
    assert load_settings(context_length=2).context_length == 2


def test_unknown_override_rejected():
    with pytest.raises(ValueError):
        load_settings(nonesuch=1)


@pytest.mark.parametrize(
    "env_var,value",
    [
        ("NGRAM_RESULT_LIMIT", "101"),
        ("NGRAM_RESULT_LIMIT", "0"),
        ("NGRAM_MAX_PAGES", "0"),
        ("NGRAM_MAX_SENTENCES", "0"),
        ("NGRAM_CONTEXT_LENGTH", "0"),
        ("NGRAM_MAX_RETRIES", "-1"),
        ("NGRAM_REQUEST_TIMEOUT", "0"),
    ],
)
def test_out_of_range_values_rejected(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValueError):
        load_settings()


def test_min_context_length_cannot_exceed_context_length(monkeypatch):
    monkeypatch.setenv("NGRAM_CONTEXT_LENGTH", "2")
    monkeypatch.setenv("NGRAM_MIN_CONTEXT_LENGTH", "3")
    with pytest.raises(ValueError):
        load_settings()


def test_env_file_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("NGRAM_CORPUS=ger\nNGRAM_MAX_SENTENCES=4\n", encoding="utf-8")
    s = load_settings(env_file)
    assert s.corpus == "ger"
    assert s.max_sentences == 4


def _env_example_keys() -> set[str]:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    return {
        m.group(1)
        for m in re.finditer(r"^([A-Z0-9_]+)=", text, flags=re.MULTILINE)
    }


def test_env_example_covers_every_field():
    documented = _env_example_keys()
    expected = {env_var for env_var, _ in FIELD_ENV.values()}
    assert documented == expected


def test_env_example_has_one_line_per_settings_field():
    assert len(_env_example_keys()) == len(fields(Settings))


def test_gitignore_ignores_env():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert ".env" in lines
    assert "*.sqlite3" in lines


def test_settings_table_lists_every_field():
    rows = settings_table(load_settings())
    assert len(rows) == len(fields(Settings))
    assert ("NGRAM_CONTEXT_LENGTH", "4") in rows
