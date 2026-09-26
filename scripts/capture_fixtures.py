"""Capture the ngrams.dev responses the offline test suite runs against.

Run once, online:

    python scripts/capture_fixtures.py

Everything it writes lands in ``tests/fixtures/``. The suite itself never
touches the network (see ``tests/conftest.py``), so this script is the only
place where a real request is made.

``walk_corpus.json`` is a deliberately small, *closed* bigram corpus: every
successor it offers has its own entry, so a random walk over it can never ask
for a query that was not captured. That is what lets the generator tests run
offline without mocking the generator itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

sys.path.insert(0, str(REPO_ROOT / "src"))

from shannon_ngram.client import NgramClient  # noqa: E402
from shannon_ngram.config import load_settings  # noqa: E402

#: top-K successors kept per context. Small on purpose: the corpus has to close
#: over its own successors, and the vocabulary grows with K.
WALK_LIMIT = 6

#: hard ceiling on distinct words in walk_corpus.json
WALK_MAX_WORDS = 250

SENTENCE_START = "_START_"
SENTENCE_END = "_END_"

#: candidate contexts for the backoff fixtures: 4 words that should return
#: nothing, narrowing to 1 word that should return something.
BACKOFF_CANDIDATES = [
    ("the", "table", "of", "quixotic"),
    ("a", "small", "green", "quixotic"),
    ("on", "the", "table", "quixotic"),
]


def get_json(client: NgramClient, query: str, limit: int) -> dict:
    """Raw response body, unparsed — fixtures must keep the wire shape."""
    params = {"query": query, "limit": limit}
    client._network_calls += 1
    response = client._session.get(
        client.search_url, params=params, timeout=client.settings.request_timeout
    )
    response.raise_for_status()
    return response.json()


def try_get_json(client: NgramClient, query: str, limit: int) -> dict | None:
    """None when the API rejects the query.

    Some single-token contexts are not valid queries at all — a bare ``"`` comes
    back 400 INVALID_QUERY — so such words cannot enter a closed corpus.
    """
    params = {"query": query, "limit": limit}
    client._network_calls += 1
    response = client._session.get(
        client.search_url, params=params, timeout=client.settings.request_timeout
    )
    if response.status_code == 400:
        return None
    response.raise_for_status()
    return response.json()


def get_error(client: NgramClient, query: str, limit: int) -> dict:
    params = {"query": query, "limit": limit}
    client._network_calls += 1
    response = client._session.get(
        client.search_url, params=params, timeout=client.settings.request_timeout
    )
    return {"status": response.status_code, "body": response.json()}


def write(name: str, payload) -> None:
    path = FIXTURES / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def successors(body: dict) -> list[str]:
    out = []
    for ngram in body.get("ngrams", []):
        for token in ngram.get("tokens", []):
            if token.get("inserted"):
                out.append(token["text"])
    return out


def capture_named(client: NgramClient) -> None:
    write("on_the_table_star.json", get_json(client, "on the table *", 100))
    write("the_table_dot_star.json", get_json(client, "the table . *", 100))
    write("start_the_star.json", get_json(client, "_START_ The *", 100))
    write("error_limit_400.json", get_error(client, "the cat *", 101))
    write(
        "error_too_many_tokens_400.json",
        get_error(client, "_START_ In the beginning of *", 5),
    )

    client._network_calls += 1
    batch = client._session.post(
        client.batch_url,
        json={"queries": ["on the table *", "the cat *"]},
        timeout=client.settings.request_timeout,
    )
    write("batch_response.json", {"status": batch.status_code, "body": batch.json()})


def capture_backoff(client: NgramClient) -> None:
    for context in BACKOFF_CANDIDATES:
        levels = {
            4: get_json(client, " ".join(context) + " *", 100),
            3: get_json(client, " ".join(context[1:]) + " *", 100),
            2: get_json(client, " ".join(context[2:]) + " *", 100),
            1: get_json(client, " ".join(context[3:]) + " *", 100),
        }
        empty = [level for level in (4, 3, 2) if not levels[level].get("ngrams")]
        if empty == [4, 3, 2] and levels[1].get("ngrams"):
            for level in (4, 3, 2):
                write(f"backoff_l{level}_empty.json", levels[level])
            write("backoff_l1_ok.json", levels[1])
            print(f"backoff context: {' '.join(context)}")
            return
        print(f"  candidate {' '.join(context)} rejected (empty at {empty})")
    raise SystemExit("no backoff candidate produced empty results at levels 4, 3 and 2")


def capture_walk_corpus(client: NgramClient) -> None:
    """Crawl a closed bigram corpus starting from the sentence-start context."""
    corpus: dict[str, dict] = {}

    start_query = f"{SENTENCE_START} *"
    corpus[start_query] = get_json(client, start_query, WALK_LIMIT)

    pending = [w for w in successors(corpus[start_query]) if w not in (SENTENCE_START,)]
    seen: set[str] = set()
    rejected: set[str] = set()

    while pending and len(seen) < WALK_MAX_WORDS:
        word = pending.pop(0)
        if word in seen or word in rejected or word == SENTENCE_END:
            continue
        query = f"{word} *"
        body = try_get_json(client, query, WALK_LIMIT)
        if body is None:
            rejected.add(word)
            continue
        seen.add(word)
        corpus[query] = body
        for nxt in successors(body):
            if nxt not in seen and nxt not in rejected and nxt != SENTENCE_END:
                pending.append(nxt)
        if len(seen) % 25 == 0:
            print(f"  {len(seen)} words captured")

    captured = set(seen)
    if rejected:
        print(f"  {len(rejected)} words rejected by the API as queries: {sorted(rejected)}")
    pruned_rows = 0
    for query, body in corpus.items():
        kept = []
        for ngram in body.get("ngrams", []):
            word = next(
                (t["text"] for t in ngram.get("tokens", []) if t.get("inserted")), None
            )
            if word == SENTENCE_END or word in captured:
                kept.append(ngram)
            else:
                pruned_rows += 1
        body["ngrams"] = kept
        body.pop("nextPageToken", None)
        body.pop("nextPageLink", None)

    empty = [q for q, b in corpus.items() if not b["ngrams"]]
    print(
        f"walk corpus: {len(corpus)} queries, {len(captured)} words, "
        f"{pruned_rows} rows pruned to close the corpus, {len(empty)} dead ends"
    )
    write("walk_corpus.json", corpus)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-walk", action="store_true", help="named fixtures only")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    client = NgramClient(settings)
    try:
        capture_named(client)
        capture_backoff(client)
        if not args.skip_walk:
            capture_walk_corpus(client)
    finally:
        print(f"{client.network_calls} requests made")
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
