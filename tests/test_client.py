"""Client behaviour, checked against a fake transport — never the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shannon_ngram.client import (
    MAX_QUERY_TOKENS,
    Ngram,
    NgramApiError,
    NgramClient,
    SearchPage,
)
from shannon_ngram.config import load_settings

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Records every call and replays a queue of responses."""

    def __init__(self, responses):
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []
        self._responses = list(responses)
        self.closed = False

    def request(self, method, url, *, timeout=None, params=None, json=None, **kwargs):
        self.calls.append(
            {"method": method, "url": url, "params": params, "json": json, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {method} {url}")
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def page(ngrams=(), next_page_token=None, query="on the table *"):
    return FakeResponse(
        200,
        {
            "query": query,
            "queryTokens": [
                {"text": "on", "kind": "TERM"},
                {"text": "*", "kind": "STAR"},
            ],
            "ngrams": list(ngrams),
            **({"nextPageToken": next_page_token} if next_page_token else {}),
        },
    )


def ngram_json(word, count, *, context=("on", "the", "table")):
    return {
        "id": f"id-{word}-{count}",
        "absTotalMatchCount": count,
        "relTotalMatchCount": count / 1e12,
        "tokens": [{"text": w, "kind": "TERM"} for w in context]
        + [{"text": word, "kind": "TERM", "inserted": True}],
    }


@pytest.fixture
def settings():
    return load_settings(
        api_base_url="https://api.ngrams.dev",
        corpus="eng",
        max_retries=0,
        request_timeout=1.0,
    )


def test_search_url_is_built_from_settings(settings):
    session = FakeSession([page()])
    client = NgramClient(settings, session=session)
    assert client.search_url == "https://api.ngrams.dev/eng/search"
    client.search("on the table *")
    assert session.calls[0]["url"] == "https://api.ngrams.dev/eng/search"


def test_user_agent_header_is_set(settings):
    session = FakeSession([])
    NgramClient(settings, session=session)
    assert session.headers["User-Agent"] == settings.user_agent


def test_flags_are_sent_as_single_param(settings):
    session = FakeSession([page()])
    NgramClient(settings, session=session).search("the table . *", flags=("cs",))
    params = session.calls[0]["params"]
    assert params["flags"] == "cs"
    assert "cs" not in params  # the ignored ?cs=true form must never be produced


def test_multiple_flags_are_comma_joined(settings):
    session = FakeSession([page()])
    NgramClient(settings, session=session).search("the *", flags=("cs", "rq"))
    assert session.calls[0]["params"]["flags"] == "cs,rq"


def test_no_flags_means_no_flags_param(settings):
    session = FakeSession([page()])
    NgramClient(settings, session=session).search("the *")
    assert "flags" not in session.calls[0]["params"]


def test_limit_over_100_raises_ngram_api_error(settings):
    client = NgramClient(settings, session=FakeSession([]))
    with pytest.raises(NgramApiError) as excinfo:
        client.search("the *", limit=101)
    assert excinfo.value.code == "INVALID_PARAMETER.LIMIT"


def test_server_side_limit_error_is_surfaced(settings):
    session = FakeSession(
        [FakeResponse(400, {"error": {"code": "INVALID_PARAMETER.LIMIT"}})]
    )
    with pytest.raises(NgramApiError) as excinfo:
        NgramClient(settings, session=session).search("the *")
    assert excinfo.value.code == "INVALID_PARAMETER.LIMIT"
    assert excinfo.value.status == 400


def test_too_many_tokens_raises(settings):
    session = FakeSession(
        [FakeResponse(400, {"error": {"code": "INVALID_QUERY.TOO_MANY_TOKENS"}})]
    )
    with pytest.raises(NgramApiError) as excinfo:
        NgramClient(settings, session=session).search("_START_ In the beginning of *")
    assert excinfo.value.code == "INVALID_QUERY.TOO_MANY_TOKENS"


def test_max_query_tokens_is_five():
    assert MAX_QUERY_TOKENS == 5


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr("shannon_ngram.client.time.sleep", lambda _s: None)
    settings = load_settings(max_retries=1, request_timeout=1.0)
    session = FakeSession([FakeResponse(429, {}), page([ngram_json(".", 10)])])
    client = NgramClient(settings, session=session)
    result = client.search("on the table *")
    assert len(result.ngrams) == 1
    assert client.network_calls == 2


def test_no_retry_on_400(settings):
    session = FakeSession(
        [FakeResponse(400, {"error": {"code": "BAD"}}), page()]
    )
    client = NgramClient(settings, session=session)
    with pytest.raises(NgramApiError):
        client.search("the *")
    assert client.network_calls == 1


def test_paging_follows_next_page_token(settings):
    session = FakeSession(
        [
            page([ngram_json(".", 10)], next_page_token="tok-2"),
            page([ngram_json("of", 5)]),
        ]
    )
    client = NgramClient(settings, session=session)
    ngrams = client.search_all("on the table *", max_pages=2)
    assert [n.inserted_token.text for n in ngrams] == [".", "of"]
    assert session.calls[0]["params"].get("start") is None
    assert session.calls[1]["params"]["start"] == "tok-2"


def test_search_all_stops_when_no_next_page(settings):
    session = FakeSession([page([ngram_json(".", 10)])])
    client = NgramClient(settings, session=session)
    assert len(client.search_all("on the table *", max_pages=5)) == 1
    assert client.network_calls == 1


def test_search_all_honours_max_pages_from_settings(settings):
    session = FakeSession([page([ngram_json(".", 10)], next_page_token="tok-2")])
    client = NgramClient(settings, session=session)
    client.search_all("on the table *")
    assert client.network_calls == 1  # settings.max_pages defaults to 1


def test_empty_result_is_not_an_error(settings):
    session = FakeSession([page([])])
    assert NgramClient(settings, session=session).search_all("zzz qqq *") == []


def test_inserted_token_is_found(settings):
    session = FakeSession([page([ngram_json(".", 10)])])
    ngrams = NgramClient(settings, session=session).search_all("on the table *")
    assert ngrams[0].inserted_token.text == "."
    assert ngrams[0].abs_total_match_count == 10


def test_ngram_without_inserted_token_returns_none():
    raw = {"id": "x", "absTotalMatchCount": 1, "tokens": [{"text": "a", "kind": "TERM"}]}
    assert Ngram.from_json(raw).inserted_token is None


def test_round_trip_json_keeps_counts():
    original = Ngram.from_json(ngram_json("of", 77))
    assert Ngram.from_json(original.to_json()) == original


def test_batch_posts_queries_object(settings):
    session = FakeSession(
        [FakeResponse(200, {"results": [{"query": "a *", "ngrams": []}]})]
    )
    NgramClient(settings, session=session).batch(["a *"])
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["json"] == {"queries": ["a *"]}


def test_batch_of_nothing_makes_no_call(settings):
    client = NgramClient(settings, session=FakeSession([]))
    assert client.batch([]) == []
    assert client.network_calls == 0


def test_batch_over_100_queries_raises(settings):
    client = NgramClient(settings, session=FakeSession([]))
    with pytest.raises(NgramApiError):
        client.batch([f"q{i} *" for i in range(101)])


def test_batch_does_not_expand_star(settings):
    """Pins the probed limitation: batch treats '*' as a literal TERM."""
    payload = {
        "results": [
            {
                "query": "on the table *",
                "queryTokens": [
                    {"text": "on", "kind": "TERM"},
                    {"text": "the", "kind": "TERM"},
                    {"text": "table", "kind": "TERM"},
                    {"text": "*", "kind": "TERM"},
                ],
                "ngrams": [
                    {
                        "id": "b1",
                        "absTotalMatchCount": 1642,
                        "tokens": [
                            {"text": "on", "kind": "TERM"},
                            {"text": "the", "kind": "TERM"},
                            {"text": "table", "kind": "TERM"},
                            {"text": "*", "kind": "TERM"},
                        ],
                    }
                ],
            }
        ]
    }
    session = FakeSession([FakeResponse(200, payload)])
    pages = NgramClient(settings, session=session).batch(["on the table *"])
    star = pages[0].query_tokens[-1]
    assert star.text == "*" and star.kind == "TERM"
    assert pages[0].ngrams[0].inserted_token is None


def test_close_only_closes_sessions_it_owns(settings):
    session = FakeSession([])
    NgramClient(settings, session=session).close()
    assert session.closed is False


def test_unparsable_error_body_still_raises(settings):
    session = FakeSession([FakeResponse(503, ValueError("not json"))])
    settings = load_settings(max_retries=0, request_timeout=1.0)
    with pytest.raises(NgramApiError) as excinfo:
        NgramClient(settings, session=session).search("the *")
    assert excinfo.value.status == 503


def test_search_page_from_json_reads_next_page_token():
    raw = {"query": "a *", "queryTokens": [], "ngrams": [], "nextPageToken": "tok"}
    assert SearchPage.from_json(raw, query="a *").next_page_token == "tok"


# --- captured responses: pin the real wire shape -------------------------------


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_captured_search_page_parses(settings):
    body = load_fixture("on_the_table_star.json")
    session = FakeSession([FakeResponse(200, body)])
    page_ = NgramClient(settings, session=session).search("on the table *")
    assert len(page_.ngrams) == 100
    assert page_.query_tokens[-1].kind == "STAR"
    assert all(n.inserted_token is not None for n in page_.ngrams)


def test_captured_sentence_end_is_a_sentence_end_token():
    body = load_fixture("the_table_dot_star.json")
    kinds = {
        t["kind"]
        for n in body["ngrams"]
        for t in n["tokens"]
        if t.get("inserted") and t["text"] == "_END_"
    }
    assert kinds == {"SENTENCE_END"}


def test_captured_sentence_start_query_token():
    body = load_fixture("start_the_star.json")
    assert body["queryTokens"][0]["kind"] == "SENTENCE_START"


def test_captured_limit_error_matches_the_clients_own_code():
    captured = load_fixture("error_limit_400.json")
    assert captured["status"] == 400
    assert captured["body"]["error"]["code"] == "INVALID_PARAMETER.LIMIT"


def test_captured_too_many_tokens_error_is_raised(settings):
    captured = load_fixture("error_too_many_tokens_400.json")
    session = FakeSession([FakeResponse(captured["status"], captured["body"])])
    with pytest.raises(NgramApiError) as excinfo:
        NgramClient(settings, session=session).search("_START_ In the beginning of *")
    assert excinfo.value.code == "INVALID_QUERY.TOO_MANY_TOKENS"


def test_captured_batch_response_has_no_star_expansion():
    """D-006, pinned against the real response."""
    captured = load_fixture("batch_response.json")
    assert captured["status"] == 200
    first = captured["body"]["results"][0]
    assert [t["kind"] for t in first["queryTokens"]][-1] == "TERM"
    assert not any(
        t.get("inserted") for n in first["ngrams"] for t in n["tokens"]
    )


def test_backoff_fixtures_are_empty_at_4_3_2_and_populated_at_1():
    for name in ("backoff_l4_empty.json", "backoff_l3_empty.json", "backoff_l2_empty.json"):
        body = load_fixture(name)
        assert body["ngrams"] == [], name
        assert "nextPageToken" not in body, name
    assert load_fixture("backoff_l1_ok.json")["ngrams"]


def test_walk_corpus_is_closed():
    """Every successor it offers has its own entry, so an offline walk cannot miss."""
    corpus = load_fixture("walk_corpus.json")
    assert "_START_ *" in corpus
    for query, body in corpus.items():
        for ngram in body["ngrams"]:
            word = next(t["text"] for t in ngram["tokens"] if t.get("inserted"))
            assert word == "_END_" or f"{word} *" in corpus, f"{query} -> {word}"


