"""HTTP client for the ngrams.dev REST API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import API_RESULT_LIMIT_MAX, Settings

MAX_QUERY_TOKENS = 5

KIND_TERM = "TERM"
KIND_SENTENCE_START = "SENTENCE_START"
KIND_SENTENCE_END = "SENTENCE_END"
KIND_STAR = "STAR"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class NgramApiError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class NgramToken:
    text: str
    kind: str
    inserted: bool = False

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "NgramToken":
        return cls(
            text=raw.get("text", ""),
            kind=raw.get("kind", KIND_TERM),
            inserted=bool(raw.get("inserted", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "inserted": self.inserted}


@dataclass(frozen=True)
class Ngram:
    id: str
    abs_total_match_count: int
    rel_total_match_count: float
    tokens: tuple[NgramToken, ...]

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Ngram":
        return cls(
            id=raw.get("id", ""),
            abs_total_match_count=int(raw.get("absTotalMatchCount", 0)),
            rel_total_match_count=float(raw.get("relTotalMatchCount", 0.0)),
            tokens=tuple(NgramToken.from_json(t) for t in raw.get("tokens", ())),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "absTotalMatchCount": self.abs_total_match_count,
            "relTotalMatchCount": self.rel_total_match_count,
            "tokens": [t.to_json() for t in self.tokens],
        }

    @property
    def inserted_token(self) -> NgramToken | None:
        for token in self.tokens:
            if token.inserted:
                return token
        return None


@dataclass(frozen=True)
class SearchPage:
    query: str
    query_tokens: tuple[NgramToken, ...]
    ngrams: tuple[Ngram, ...]
    next_page_token: str | None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any], *, query: str) -> "SearchPage":
        return cls(
            query=raw.get("query", query),
            query_tokens=tuple(NgramToken.from_json(t) for t in raw.get("queryTokens", ())),
            ngrams=tuple(Ngram.from_json(n) for n in raw.get("ngrams", ())),
            next_page_token=raw.get("nextPageToken"),
        )


class NgramClient:
    """Thin wrapper over GET /{corpus}/search and POST /{corpus}/batch.

    ``flags`` are sent as one comma-joined ``flags=`` parameter. Passing them as
    separate booleans (``cs=true``) is accepted by the API and silently ignored,
    which is why the joined form is the only one used here.
    """

    def __init__(self, settings: Settings, session: "Any | None" = None) -> None:
        self.settings = settings
        self._owns_session = session is None
        if session is None:
            import requests

            session = requests.Session()
        self._session = session
        self._session.headers.update({"User-Agent": settings.user_agent})
        self._network_calls = 0

    @property
    def network_calls(self) -> int:
        return self._network_calls

    @property
    def search_url(self) -> str:
        return f"{self.settings.api_base_url.rstrip('/')}/{self.settings.corpus}/search"

    @property
    def batch_url(self) -> str:
        return f"{self.settings.api_base_url.rstrip('/')}/{self.settings.corpus}/batch"

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "NgramClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _check_limit(self, limit: int) -> int:
        if not 1 <= limit <= API_RESULT_LIMIT_MAX:
            raise NgramApiError(
                f"limit must be 1..{API_RESULT_LIMIT_MAX}, got {limit}",
                code="INVALID_PARAMETER.LIMIT",
            )
        return limit

    def _request(self, method: str, url: str, **kwargs) -> Mapping[str, Any]:
        attempts = self.settings.max_retries + 1
        last_error: NgramApiError | None = None
        for attempt in range(attempts):
            self._network_calls += 1
            response = self._session.request(
                method, url, timeout=self.settings.request_timeout, **kwargs
            )
            status = response.status_code
            if status == 200:
                return response.json()
            code, message = _error_from_response(response)
            last_error = NgramApiError(message, code=code, status=status)
            if status not in RETRY_STATUSES or attempt == attempts - 1:
                raise last_error
            time.sleep(min(2.0**attempt, 8.0))
        raise last_error  # pragma: no cover - loop always raises or returns

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        start: str | None = None,
        flags: Sequence[str] = (),
    ) -> SearchPage:
        limit = self._check_limit(self.settings.result_limit if limit is None else limit)
        params: dict[str, Any] = {"query": query, "limit": limit}
        if start:
            params["start"] = start
        if flags:
            params["flags"] = ",".join(flags)
        payload = self._request("GET", self.search_url, params=params)
        return SearchPage.from_json(payload, query=query)

    def search_all(
        self,
        query: str,
        *,
        max_pages: int | None = None,
        limit: int | None = None,
        flags: Sequence[str] = (),
    ) -> list[Ngram]:
        max_pages = self.settings.max_pages if max_pages is None else max_pages
        ngrams: list[Ngram] = []
        start: str | None = None
        for _ in range(max_pages):
            page = self.search(query, limit=limit, start=start, flags=flags)
            ngrams.extend(page.ngrams)
            start = page.next_page_token
            if not start:
                break
        return ngrams

    def batch(self, queries: Sequence[str], *, limit: int | None = None) -> list[SearchPage]:
        """Literal n-gram lookup only.

        The batch endpoint parses ``*`` as an ordinary TERM rather than a
        wildcard, so it cannot return next-word distributions and the generator
        never calls it. Kept because the API exposes it.
        """
        if not queries:
            return []
        if len(queries) > 100:
            raise NgramApiError("batch accepts at most 100 queries", code="INVALID_REQUEST_BODY")
        limit = self._check_limit(self.settings.result_limit if limit is None else limit)
        payload = self._request(
            "POST",
            self.batch_url,
            json={"queries": list(queries)},
            params={"limit": limit},
        )
        results = payload.get("results", [])
        return [
            SearchPage.from_json(raw, query=query)
            for query, raw in zip(queries, results)
        ]


def _error_from_response(response) -> tuple[str | None, str]:
    try:
        body = response.json()
    except Exception:
        return None, f"HTTP {response.status_code}"
    error = body.get("error") if isinstance(body, Mapping) else None
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message") or code or f"HTTP {response.status_code}"
        return code, f"HTTP {response.status_code}: {message}"
    return None, f"HTTP {response.status_code}"
