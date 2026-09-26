"""Settings for shannon_ngram, read from the environment or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

API_RESULT_LIMIT_MAX = 100

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    corpus: str
    context_length: int
    min_context_length: int
    max_sentences: int
    max_sentence_words: int
    result_limit: int
    max_pages: int
    cache_path: Path
    request_timeout: float
    max_retries: int
    case_sensitive: bool
    user_agent: str


ENV_PREFIX = "NGRAM_"

#: field name -> (env var, default). The cache path default is None so that it
#: is computed from Path.home() at call time and never stored as a literal.
FIELD_ENV = {
    "api_base_url": ("NGRAM_API_BASE_URL", "https://api.ngrams.dev"),
    "corpus": ("NGRAM_CORPUS", "eng"),
    "context_length": ("NGRAM_CONTEXT_LENGTH", 4),
    "min_context_length": ("NGRAM_MIN_CONTEXT_LENGTH", 1),
    "max_sentences": ("NGRAM_MAX_SENTENCES", 5),
    "max_sentence_words": ("NGRAM_MAX_SENTENCE_WORDS", 40),
    "result_limit": ("NGRAM_RESULT_LIMIT", 100),
    "max_pages": ("NGRAM_MAX_PAGES", 1),
    "cache_path": ("NGRAM_CACHE_PATH", None),
    "request_timeout": ("NGRAM_REQUEST_TIMEOUT", 10.0),
    "max_retries": ("NGRAM_MAX_RETRIES", 3),
    "case_sensitive": ("NGRAM_CASE_SENSITIVE", False),
    "user_agent": ("NGRAM_USER_AGENT", "shannon-ngram/0.1"),
}


def default_cache_path() -> Path:
    """Computed at call time so no machine-specific path is ever written down."""
    return Path.home() / ".cache" / "shannon_ngram" / "ngrams.sqlite3"


def _as_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"cannot read {value!r} as a boolean")


def _coerce(name: str, value):
    if isinstance(value, str):
        kind = {f.name: f.type for f in fields(Settings)}[name]
        if kind is bool or kind == "bool":
            return _as_bool(value)
        if kind is int or kind == "int":
            return int(value)
        if kind is float or kind == "float":
            return float(value)
        if kind is Path or kind == "Path":
            return Path(value).expanduser()
    return value


def load_settings(env_file: str | os.PathLike | None = None, **overrides) -> Settings:
    """Build Settings from overrides, then the environment, then the defaults.

    ``env_file`` is loaded with python-dotenv without overriding variables that
    are already set in the real environment.
    """
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)

    unknown = set(overrides) - set(FIELD_ENV)
    if unknown:
        raise ValueError(f"unknown setting(s): {sorted(unknown)}")

    values = {}
    for name, (env_var, default) in FIELD_ENV.items():
        if name in overrides:
            raw = overrides[name]
        else:
            raw = os.environ.get(env_var)
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                raw = default
        if raw is None and name == "cache_path":
            raw = default_cache_path()
        values[name] = _coerce(name, raw)

    settings = Settings(**values)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if not 1 <= settings.result_limit <= API_RESULT_LIMIT_MAX:
        raise ValueError(
            f"result_limit must be 1..{API_RESULT_LIMIT_MAX}, got {settings.result_limit}"
        )
    if settings.context_length < 1:
        raise ValueError("context_length must be >= 1")
    if not 1 <= settings.min_context_length <= settings.context_length:
        raise ValueError("min_context_length must be 1..context_length")
    if settings.max_sentences < 1:
        raise ValueError("max_sentences must be >= 1")
    if settings.max_sentence_words < 1:
        raise ValueError("max_sentence_words must be >= 1")
    if settings.max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if settings.request_timeout <= 0:
        raise ValueError("request_timeout must be > 0")
    if settings.max_retries < 0:
        raise ValueError("max_retries must be >= 0")


def settings_table(settings: Settings) -> list[tuple[str, str]]:
    """(env var, value) rows for the notebook's Step 0 table."""
    return [
        (FIELD_ENV[f.name][0], str(getattr(settings, f.name)))
        for f in fields(Settings)
    ]
