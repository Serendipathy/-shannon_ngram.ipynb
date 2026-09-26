"""SQLite cache for search results, keyed on everything that changes a response."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .client import Ngram, NgramClient

SCHEMA = """
CREATE TABLE IF NOT EXISTS ngram_pages (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    fetched_at REAL NOT NULL
)
"""


def cache_key(
    query: str,
    *,
    limit: int,
    flags: Sequence[str],
    max_pages: int,
    corpus: str = "eng",
) -> str:
    """Stable digest of every input that can change the response body."""
    material = json.dumps(
        {
            "corpus": corpus,
            "query": query,
            "limit": limit,
            "flags": sorted(flags),
            "max_pages": max_pages,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    path: Path
    rows: int
    bytes: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class NgramCache:
    def __init__(self, path: Path, ttl_seconds: int | None = None) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[Ngram] | None:
        """None means "not cached"; [] means "cached, and the API returned nothing"."""
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM ngram_pages WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        payload, fetched_at = row
        if self.ttl_seconds is not None and time.time() - fetched_at >= self.ttl_seconds:
            self._conn.execute("DELETE FROM ngram_pages WHERE key = ?", (key,))
            self._conn.commit()
            self.misses += 1
            return None
        self.hits += 1
        return [Ngram.from_json(raw) for raw in json.loads(payload)]

    def put(self, key: str, ngrams: Sequence[Ngram]) -> None:
        payload = json.dumps([n.to_json() for n in ngrams], separators=(",", ":"))
        self._conn.execute(
            "INSERT OR REPLACE INTO ngram_pages (key, payload, fetched_at) VALUES (?, ?, ?)",
            (key, payload, time.time()),
        )
        self._conn.commit()

    def stats(self) -> CacheStats:
        rows = self._conn.execute("SELECT COUNT(*) FROM ngram_pages").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return CacheStats(
            path=self.path, rows=rows, bytes=size, hits=self.hits, misses=self.misses
        )

    def clear(self) -> None:
        self._conn.execute("DELETE FROM ngram_pages")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NgramCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class CachedNgramClient:
    """search_all() with a read-through cache. Empty results are cached too.

    Caching the empty list is what keeps the backoff walk off the network: a
    context that has no results is asked for on every sentence, and without a
    stored empty the cache would miss every time.
    """

    def __init__(self, client: NgramClient, cache: NgramCache) -> None:
        self.client = client
        self.cache = cache

    @property
    def settings(self):
        return self.client.settings

    @property
    def network_calls(self) -> int:
        return self.client.network_calls

    def search_all(
        self,
        query: str,
        *,
        max_pages: int | None = None,
        limit: int | None = None,
        flags: Sequence[str] = (),
    ) -> list[Ngram]:
        settings = self.client.settings
        key = cache_key(
            query,
            limit=settings.result_limit if limit is None else limit,
            flags=flags,
            max_pages=settings.max_pages if max_pages is None else max_pages,
            corpus=settings.corpus,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        ngrams = self.client.search_all(query, max_pages=max_pages, limit=limit, flags=flags)
        self.cache.put(key, ngrams)
        return ngrams

    def close(self) -> None:
        self.client.close()
        self.cache.close()
