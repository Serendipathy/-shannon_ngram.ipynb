"""AC 4 — a repeated identical query makes zero network calls."""

from __future__ import annotations

import pytest

from shannon_ngram.cache import CachedNgramClient, NgramCache, cache_key
from shannon_ngram.client import NgramClient
from shannon_ngram.config import load_settings

from test_client import FakeSession, ngram_json, page


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        cache_path=tmp_path / "ngrams.sqlite3", max_retries=0, request_timeout=1.0
    )


@pytest.fixture
def cache(settings):
    c = NgramCache(settings.cache_path)
    yield c
    c.close()


def counting_client(settings, responses):
    session = FakeSession(responses)
    return NgramClient(settings, session=session), session


def test_repeat_query_makes_zero_network_calls(settings, cache):
    client, session = counting_client(settings, [page([ngram_json(".", 10)])])
    cached = CachedNgramClient(client, cache)

    first = cached.search_all("on the table *")
    assert client.network_calls == 1

    before = client.network_calls
    second = cached.search_all("on the table *")
    third = cached.search_all("on the table *")

    assert client.network_calls - before == 0
    assert first == second == third
    assert cache.stats().hits == 2


def test_empty_result_is_cached(settings, cache):
    """The backoff path asks for the same empty context on every sentence."""
    client, _ = counting_client(settings, [page([])])
    cached = CachedNgramClient(client, cache)

    assert cached.search_all("zzz qqq wibble *") == []
    assert client.network_calls == 1
    assert cached.search_all("zzz qqq wibble *") == []
    assert client.network_calls == 1  # no second call for the known-empty context


def test_different_query_is_a_different_key(settings, cache):
    client, _ = counting_client(
        settings, [page([ngram_json(".", 10)]), page([ngram_json("of", 5)])]
    )
    cached = CachedNgramClient(client, cache)
    cached.search_all("on the table *")
    cached.search_all("the cat *")
    assert client.network_calls == 2


def test_flags_change_the_key(settings, cache):
    client, _ = counting_client(
        settings, [page([ngram_json(".", 10)]), page([ngram_json(".", 9)])]
    )
    cached = CachedNgramClient(client, cache)
    cached.search_all("the table . *")
    cached.search_all("the table . *", flags=("cs",))
    assert client.network_calls == 2


def test_limit_and_max_pages_change_the_key():
    base = dict(query="a *", limit=100, flags=(), max_pages=1, corpus="eng")
    assert cache_key(**base) != cache_key(**{**base, "limit": 50})
    assert cache_key(**base) != cache_key(**{**base, "max_pages": 2})
    assert cache_key(**base) != cache_key(**{**base, "corpus": "ger"})
    assert cache_key(**base) != cache_key(**{**base, "flags": ("cs",)})


def test_flag_order_does_not_change_the_key():
    base = dict(query="a *", limit=100, max_pages=1, corpus="eng")
    assert cache_key(flags=("cs", "rq"), **base) == cache_key(flags=("rq", "cs"), **base)


def test_cache_survives_reopen(settings):
    with NgramCache(settings.cache_path) as first:
        client, _ = counting_client(settings, [page([ngram_json(".", 10)])])
        CachedNgramClient(client, first).search_all("on the table *")

    with NgramCache(settings.cache_path) as second:
        client2, _ = counting_client(settings, [])
        result = CachedNgramClient(client2, second).search_all("on the table *")

    assert result[0].inserted_token.text == "."
    assert client2.network_calls == 0


def test_round_trip_preserves_counts_and_inserted_flag(settings, cache):
    client, _ = counting_client(settings, [page([ngram_json("of", 502754)])])
    cached = CachedNgramClient(client, cache)
    live = cached.search_all("on the table *")
    from_cache = cached.search_all("on the table *")
    assert from_cache == live
    assert from_cache[0].abs_total_match_count == 502754
    assert from_cache[0].inserted_token.inserted is True


def test_parent_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "ngrams.sqlite3"
    with NgramCache(path):
        pass
    assert path.exists()


def test_stats_report_rows_and_hit_rate(settings, cache):
    client, _ = counting_client(settings, [page([ngram_json(".", 10)])])
    cached = CachedNgramClient(client, cache)
    cached.search_all("on the table *")
    cached.search_all("on the table *")
    stats = cache.stats()
    assert stats.rows == 1
    assert (stats.hits, stats.misses) == (1, 1)
    assert stats.hit_rate == 0.5
    assert stats.bytes > 0


def test_clear_empties_the_cache(settings, cache):
    client, _ = counting_client(
        settings, [page([ngram_json(".", 10)]), page([ngram_json(".", 10)])]
    )
    cached = CachedNgramClient(client, cache)
    cached.search_all("on the table *")
    cache.clear()
    assert cache.stats().rows == 0
    cached.search_all("on the table *")
    assert client.network_calls == 2


def test_expired_entry_is_refetched(settings):
    with NgramCache(settings.cache_path, ttl_seconds=0) as cache:
        client, _ = counting_client(
            settings, [page([ngram_json(".", 10)]), page([ngram_json(".", 10)])]
        )
        cached = CachedNgramClient(client, cache)
        cached.search_all("on the table *")
        cached.search_all("on the table *")
        assert client.network_calls == 2
