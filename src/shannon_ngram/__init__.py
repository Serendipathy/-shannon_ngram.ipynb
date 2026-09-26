"""Paragraph generation from Google Books n-gram counts served by ngrams.dev."""

from __future__ import annotations

from importlib import import_module

from .config import Settings, load_settings, settings_table

__version__ = "0.1.0"

#: name -> submodule it lives in. Resolved lazily so that importing the package
#: never pulls in requests, sqlite3 or matplotlib unless the name is used.
_LAZY = {
    "NgramApiError": "client",
    "NgramToken": "client",
    "Ngram": "client",
    "SearchPage": "client",
    "NgramClient": "client",
    "cache_key": "cache",
    "CacheStats": "cache",
    "NgramCache": "cache",
    "CachedNgramClient": "cache",
    "Distribution": "model",
    "NextWordModel": "model",
    "aggregate_counts": "model",
    "counts_to_distribution": "model",
    "entropy_bits": "model",
    "GenerationError": "generator",
    "Step": "generator",
    "Sentence": "generator",
    "Paragraph": "generator",
    "ParagraphGenerator": "generator",
    "detokenise": "generator",
}

__all__ = [
    "Settings",
    "load_settings",
    "settings_table",
    "__version__",
    *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        module = import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
