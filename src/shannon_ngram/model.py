"""Next-word distributions: token filtering, case aggregation, backoff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .client import (
    MAX_QUERY_TOKENS,
    KIND_SENTENCE_START,
    Ngram,
    NgramApiError,
    NgramToken,
)
from .config import Settings

SENTENCE_START = "_START_"
SENTENCE_END = "_END_"

WILDCARD = "*"

#: A context the API will not parse is read as "no continuations at this level"
#: and the model backs off. Two ways it happens, both driven by what the sampler
#: emitted rather than by a bug here:
#:   BAD_TERM_GROUP    — a legal corpus token that is not a legal query term (`"`)
#:   TOO_MANY_TOKENS   — one corpus token the API re-splits into several (`don't`),
#:                       so a 4-word context can exceed the 5-token cap
#: build_query's own ≤ 5 space-separated tokens are pinned by unit test, so this
#: cannot hide a query-building bug.
UNQUERYABLE_CODE_PREFIX = "INVALID_QUERY."


def is_usable_token(token: NgramToken) -> bool:
    """A candidate next word.

    Punctuation stays in — ``.`` is the most likely successor of many contexts
    and carries the sentence boundary. ``_END_`` stays in as the terminal
    symbol. ``_START_`` is dropped: nothing may follow a sentence start.
    """
    if token is None or not token.text:
        return False
    if token.text == SENTENCE_START or token.kind == KIND_SENTENCE_START:
        return False
    if token.text == WILDCARD:
        return False
    return True


def normalise_key(text: str, *, case_sensitive: bool) -> str:
    return text if case_sensitive else text.casefold()


def aggregate_counts(
    ngrams: Sequence[Ngram], *, case_sensitive: bool = False
) -> dict[str, int]:
    """Sum counts per next word, merging case variants unless asked not to.

    The label kept for a merged group is the surface form with the highest
    single count, so ``the``/``The`` reports whichever the corpus prefers.
    """
    totals: dict[str, int] = {}
    best_surface: dict[str, tuple[int, str]] = {}
    for ngram in ngrams:
        token = ngram.inserted_token
        if not is_usable_token(token):
            continue
        key = normalise_key(token.text, case_sensitive=case_sensitive)
        count = ngram.abs_total_match_count
        totals[key] = totals.get(key, 0) + count
        current = best_surface.get(key)
        if current is None or (count, token.text) > current:
            best_surface[key] = (count, token.text)
    return {best_surface[key][1]: total for key, total in totals.items()}


def counts_to_distribution(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {word: count / total for word, count in counts.items()}


def entropy_bits(dist: Mapping[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def build_query(
    context: Sequence[str],
    *,
    at_sentence_start: bool,
    max_tokens: int = MAX_QUERY_TOKENS,
) -> str:
    """``_START_ w1 … wk *`` or ``w1 … wk *``, never longer than max_tokens.

    ``_START_`` counts against the API's 5-token cap, so it costs one of the
    context slots; the rightmost words are the ones kept.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must leave room for a context word and the wildcard")
    prefix = [SENTENCE_START] if at_sentence_start else []
    room = max_tokens - len(prefix) - 1
    words = [w for w in context if w and w != SENTENCE_START][-room:] if room > 0 else []
    return " ".join([*prefix, *words, WILDCARD])


@dataclass(frozen=True)
class Distribution:
    context: tuple[str, ...]
    level: int
    used_sentence_start: bool
    counts: dict[str, int]
    dist: dict[str, float]
    total_count: int
    entropy: float
    candidates: int

    def top(self, n: int = 15) -> list[tuple[str, int, float]]:
        ordered = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(w, c, self.dist[w]) for w, c in ordered[:n]]


class NextWordModel:
    """Turns a context into a probability distribution, backing off 4 → 1."""

    def __init__(self, client, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    @property
    def _flags(self) -> tuple[str, ...]:
        return ("cs",) if self.settings.case_sensitive else ()

    def _search(self, query: str) -> list[Ngram]:
        """A context the API refuses to parse has no continuations — back off."""
        try:
            return self.client.search_all(query, flags=self._flags)
        except NgramApiError as error:
            if (error.code or "").startswith(UNQUERYABLE_CODE_PREFIX):
                return []
            raise

    def levels(self, context: Sequence[str], *, at_sentence_start: bool) -> list[int]:
        """Context lengths to try, longest first.

        Level 0 exists only at a sentence start with nothing generated yet: the
        query is then ``_START_ *``.
        """
        words = [w for w in context if w and w != SENTENCE_START]
        if not words:
            return [0] if at_sentence_start else []
        top = min(self.settings.context_length, len(words))
        return list(range(top, self.settings.min_context_length - 1, -1))

    def distribution(
        self, context: Sequence[str], *, at_sentence_start: bool = False
    ) -> Distribution | None:
        """First level with a non-empty aggregate, or None if all are empty."""
        settings = self.settings
        words = [w for w in context if w and w != SENTENCE_START]

        for level in self.levels(words, at_sentence_start=at_sentence_start):
            used = words[-level:] if level else []
            # _START_ costs one of the five query slots, so it has to go once the
            # context alone fills the room left for words.
            use_start = at_sentence_start and level <= MAX_QUERY_TOKENS - 2
            query = build_query(used, at_sentence_start=use_start)
            ngrams = self._search(query)
            counts = aggregate_counts(ngrams, case_sensitive=settings.case_sensitive)
            if counts:
                dist = counts_to_distribution(counts)
                return Distribution(
                    context=tuple(used),
                    level=level,
                    used_sentence_start=use_start,
                    counts=counts,
                    dist=dist,
                    total_count=sum(counts.values()),
                    entropy=entropy_bits(dist),
                    candidates=len(counts),
                )
        return None
