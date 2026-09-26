"""AC 5 (probabilities), AC 6 (case merge), AC 7 (backoff)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from shannon_ngram.client import Ngram
from shannon_ngram.config import load_settings
from shannon_ngram.model import (
    Distribution,
    NextWordModel,
    aggregate_counts,
    build_query,
    counts_to_distribution,
    entropy_bits,
    is_usable_token,
    normalise_key,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

POPULATED = [
    "on_the_table_star.json",
    "the_table_dot_star.json",
    "start_the_star.json",
    "backoff_l1_ok.json",
]


def load_ngrams(name: str) -> list[Ngram]:
    body = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [Ngram.from_json(raw) for raw in body["ngrams"]]


class FixtureClient:
    """Replays a {query: [Ngram]} map. Anything unknown is an empty result."""

    def __init__(self, responses: dict[str, list[Ngram]]):
        self.responses = responses
        self.queries: list[str] = []

    def search_all(self, query, *, max_pages=None, limit=None, flags=()):
        self.queries.append(query)
        return list(self.responses.get(query, []))


# --- token filtering -----------------------------------------------------------


def test_sentence_start_is_not_a_candidate():
    ngrams = load_ngrams("start_the_star.json")
    counts = aggregate_counts(ngrams)
    assert "_START_" not in counts


def test_sentence_end_is_a_candidate():
    counts = aggregate_counts(load_ngrams("the_table_dot_star.json"))
    assert counts["_END_"] > 0


def test_punctuation_is_a_candidate():
    counts = aggregate_counts(load_ngrams("on_the_table_star.json"))
    assert counts["."] > 0


def test_is_usable_token_rejects_empty_and_wildcard():
    from shannon_ngram.client import NgramToken

    assert not is_usable_token(NgramToken(text="", kind="TERM"))
    assert not is_usable_token(NgramToken(text="*", kind="STAR"))
    assert not is_usable_token(NgramToken(text="_START_", kind="SENTENCE_START"))
    assert is_usable_token(NgramToken(text=".", kind="TERM"))


def test_normalise_key_honours_case_flag():
    assert normalise_key("The", case_sensitive=False) == "the"
    assert normalise_key("The", case_sensitive=True) == "The"


# --- AC 6: case variants merged ------------------------------------------------


def test_case_variants_merged():
    """`on the table *` returns `.` and `of` on more than one row each."""
    ngrams = load_ngrams("on_the_table_star.json")
    raw_counts: dict[str, list[int]] = {}
    for ngram in ngrams:
        token = ngram.inserted_token
        raw_counts.setdefault(token.text.casefold(), []).append(ngram.abs_total_match_count)

    duplicated = {k: v for k, v in raw_counts.items() if len(v) > 1}
    assert duplicated, "fixture no longer contains duplicate next tokens"

    merged = aggregate_counts(ngrams, case_sensitive=False)
    merged_by_key = {k.casefold(): v for k, v in merged.items()}
    assert len(merged_by_key) == len(merged), "two labels collapsed to one key"
    for key, counts in duplicated.items():
        assert merged_by_key[key] == sum(counts)


def test_case_sensitive_aggregation_keeps_variants_apart():
    ngrams = load_ngrams("the_table_dot_star.json")
    insensitive = aggregate_counts(ngrams, case_sensitive=False)
    sensitive = aggregate_counts(ngrams, case_sensitive=True)
    assert len(sensitive) >= len(insensitive)
    assert sum(sensitive.values()) == sum(insensitive.values())


def test_merged_label_is_the_most_frequent_surface_form():
    from shannon_ngram.client import Ngram as N, NgramToken as T

    def row(text, count):
        return N(
            id=text + str(count),
            abs_total_match_count=count,
            rel_total_match_count=0.0,
            tokens=(T(text="x", kind="TERM"), T(text=text, kind="TERM", inserted=True)),
        )

    counts = aggregate_counts([row("The", 10), row("the", 90)])
    assert counts == {"the": 100}


# --- AC 5: distributions sum to 1 ----------------------------------------------


@pytest.mark.parametrize("name", POPULATED)
def test_distribution_sums_to_one(name):
    dist = counts_to_distribution(aggregate_counts(load_ngrams(name)))
    assert dist
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_every_walk_corpus_context_sums_to_one():
    corpus = json.loads((FIXTURES / "walk_corpus.json").read_text(encoding="utf-8"))
    checked = 0
    for body in corpus.values():
        ngrams = [Ngram.from_json(raw) for raw in body["ngrams"]]
        counts = aggregate_counts(ngrams)
        if not counts:
            continue
        assert abs(sum(counts_to_distribution(counts).values()) - 1.0) < 1e-9
        checked += 1
    assert checked > 100


def test_empty_counts_returns_empty_dist():
    assert counts_to_distribution({}) == {}
    assert counts_to_distribution({"a": 0}) == {}


def test_probabilities_are_proportional_to_counts():
    dist = counts_to_distribution({"a": 30, "b": 10})
    assert dist["a"] == pytest.approx(0.75)
    assert dist["b"] == pytest.approx(0.25)


# --- entropy -------------------------------------------------------------------


def test_entropy_of_a_certain_outcome_is_zero():
    assert entropy_bits({"a": 1.0}) == 0.0


def test_entropy_of_four_equal_outcomes_is_two_bits():
    assert entropy_bits({k: 0.25 for k in "abcd"}) == pytest.approx(2.0)


def test_entropy_is_bounded_by_log2_of_candidate_count():
    dist = counts_to_distribution(aggregate_counts(load_ngrams("on_the_table_star.json")))
    assert 0 < entropy_bits(dist) <= math.log2(len(dist))


# --- query building (D-007) ----------------------------------------------------


def test_build_query_without_sentence_start():
    assert build_query(["on", "the", "table"], at_sentence_start=False) == "on the table *"


def test_build_query_keeps_five_tokens_at_most():
    q = build_query(["a", "b", "c", "d", "e", "f"], at_sentence_start=False)
    assert q.split() == ["c", "d", "e", "f", "*"]


def test_sentence_start_costs_a_word_slot():
    q = build_query(["In", "the", "beginning", "of"], at_sentence_start=True)
    assert q.split() == ["_START_", "the", "beginning", "of", "*"]
    assert len(q.split()) == 5


def test_build_query_with_no_context_at_sentence_start():
    assert build_query([], at_sentence_start=True) == "_START_ *"


def test_build_query_drops_a_stray_sentence_start_token():
    assert build_query(["_START_", "the"], at_sentence_start=True) == "_START_ the *"


# --- AC 7: backoff -------------------------------------------------------------


@pytest.fixture
def backoff_levels():
    return {level: load_ngrams(f"backoff_l{level}_empty.json") for level in (4, 3, 2)} | {
        1: load_ngrams("backoff_l1_ok.json")
    }


CONTEXT = ("a", "small", "green", "quixotic")


def queries_for(context, first_hit_level):
    """The query the model will send at each level, per D-007."""
    return {
        level: " ".join(list(context[-level:]) + ["*"])
        for level in range(1, len(context) + 1)
    }


def test_backoff_records_level(backoff_levels):
    settings = load_settings(context_length=4, min_context_length=1)
    queries = queries_for(CONTEXT, 1)
    client = FixtureClient({queries[1]: backoff_levels[1]})
    model = NextWordModel(client, settings)

    dist = model.distribution(CONTEXT)

    assert dist is not None
    assert dist.level == 1
    assert dist.context == ("quixotic",)
    assert client.queries == [queries[4], queries[3], queries[2], queries[1]]


@pytest.mark.parametrize("first_hit", [4, 3, 2, 1])
def test_backoff_stops_at_the_first_populated_level(backoff_levels, first_hit):
    settings = load_settings(context_length=4, min_context_length=1)
    queries = queries_for(CONTEXT, first_hit)
    client = FixtureClient({queries[first_hit]: backoff_levels[1]})
    model = NextWordModel(client, settings)

    dist = model.distribution(CONTEXT)

    assert dist.level == first_hit
    assert len(client.queries) == 4 - first_hit + 1
    assert client.queries[-1] == queries[first_hit]


def test_backoff_exhausted_returns_none():
    settings = load_settings(context_length=4, min_context_length=1)
    client = FixtureClient({})
    assert NextWordModel(client, settings).distribution(CONTEXT) is None
    assert len(client.queries) == 4


def test_min_context_length_floors_the_walk():
    settings = load_settings(context_length=4, min_context_length=2)
    client = FixtureClient({})
    NextWordModel(client, settings).distribution(CONTEXT)
    assert len(client.queries) == 3  # levels 4, 3, 2 only


def test_sentence_start_query_is_used_when_nothing_generated_yet():
    settings = load_settings(context_length=4)
    client = FixtureClient({"_START_ *": load_ngrams("start_the_star.json")})
    dist = NextWordModel(client, settings).distribution([], at_sentence_start=True)
    assert client.queries == ["_START_ *"]
    assert dist.level == 0
    assert dist.used_sentence_start is True


def test_sentence_start_is_dropped_once_the_context_fills_the_query():
    settings = load_settings(context_length=4)
    client = FixtureClient({})
    NextWordModel(client, settings).distribution(
        ["In", "the", "beginning", "of"], at_sentence_start=True
    )
    assert client.queries[0] == "In the beginning of *"  # level 4: no room for _START_
    assert client.queries[1] == "_START_ the beginning of *"  # level 3: it fits again


def test_no_context_and_not_at_sentence_start_yields_none():
    settings = load_settings()
    client = FixtureClient({})
    assert NextWordModel(client, settings).distribution([]) is None
    assert client.queries == []


def test_case_sensitive_setting_sends_the_cs_flag():
    class FlagRecordingClient(FixtureClient):
        def __init__(self):
            super().__init__({})
            self.flags = []

        def search_all(self, query, *, max_pages=None, limit=None, flags=()):
            self.flags.append(tuple(flags))
            return super().search_all(query, flags=flags)

    client = FlagRecordingClient()
    NextWordModel(client, load_settings(case_sensitive=True)).distribution(["the"])
    assert client.flags == [("cs",)]

    client2 = FlagRecordingClient()
    NextWordModel(client2, load_settings(case_sensitive=False)).distribution(["the"])
    assert client2.flags == [()]


def test_distribution_fields_are_consistent():
    settings = load_settings(context_length=1)
    client = FixtureClient({"table *": load_ngrams("on_the_table_star.json")})
    dist = NextWordModel(client, settings).distribution(["table"])
    assert isinstance(dist, Distribution)
    assert dist.total_count == sum(dist.counts.values())
    assert dist.candidates == len(dist.counts) == len(dist.dist)
    assert dist.entropy == pytest.approx(entropy_bits(dist.dist))
    top = dist.top(3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]
