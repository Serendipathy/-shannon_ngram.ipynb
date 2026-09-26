# shannon_ngram

Generate English paragraphs by sampling each next word from Google Books n-gram counts,
served by the [ngrams.dev](https://ngrams.dev) REST API. A Shannon-style word-level
approximation to English, built from a corpus rather than from a hand-made table.

All logic lives in the importable package `shannon_ngram` under `src/`. The notebook in
`notebooks/` only imports and calls it, so the same code runs headless on a server.

## Install

```sh
python3 -m venv ~/.venvs/shannon_ngram
~/.venvs/shannon_ngram/bin/pip install -e .                      # runtime only
~/.venvs/shannon_ngram/bin/pip install -e ".[dev,notebook]"      # plus pytest, jupyter, matplotlib
```

Runtime dependencies are `requests` and `python-dotenv`. `pytest`, `jupyter` and
`matplotlib` are extras, so a server install stays small.

## Run

```python
from shannon_ngram import (
    CachedNgramClient, NgramCache, NgramClient, NextWordModel,
    ParagraphGenerator, load_settings,
)

s = load_settings()
client = CachedNgramClient(NgramClient(s), NgramCache(s.cache_path))
generator = ParagraphGenerator(NextWordModel(client, s), s, seed=42)
print(generator.paragraph().text)
```

The notebook walks the same path in twelve steps, from settings to a per-word entropy chart
and a context-length-1-to-4 comparison:

```sh
~/.venvs/shannon_ngram/bin/jupyter nbconvert --to notebook --execute notebooks/shannon_ngram.ipynb
```

## Tests

```sh
~/.venvs/shannon_ngram/bin/pytest
```

The suite is offline by design: `tests/conftest.py` replaces `socket.socket` with a stub
that raises, and every test reads a recorded response from `tests/fixtures/`. Re-capture the
fixtures with `python scripts/capture_fixtures.py` — the only code in the repo that makes a
real request.

## Settings

Every setting is read from the environment or a `.env` file; nothing is hard-coded. Copy
`.env.example` to `.env` and edit. `.env` is gitignored.

| Variable | Default | Meaning |
|---|---|---|
| `NGRAM_API_BASE_URL` | `https://api.ngrams.dev` | API base URL |
| `NGRAM_CORPUS` | `eng` | Corpus code |
| `NGRAM_CONTEXT_LENGTH` | `4` | Words of context per next-word lookup |
| `NGRAM_MIN_CONTEXT_LENGTH` | `1` | Floor of the backoff walk |
| `NGRAM_MAX_SENTENCES` | `5` | Sentences per paragraph |
| `NGRAM_MAX_SENTENCE_WORDS` | `40` | Sentence length guard |
| `NGRAM_RESULT_LIMIT` | `100` | Rows per request (API maximum) |
| `NGRAM_MAX_PAGES` | `1` | Pages fetched per context |
| `NGRAM_CACHE_PATH` | `~/.cache/shannon_ngram/ngrams.sqlite3` | SQLite cache file |
| `NGRAM_REQUEST_TIMEOUT` | `10.0` | HTTP timeout, seconds |
| `NGRAM_MAX_RETRIES` | `3` | Retries on 429 and 5xx |
| `NGRAM_CASE_SENSITIVE` | `false` | `true` sends `flags=cs` instead of merging case variants |
| `NGRAM_USER_AGENT` | `shannon-ngram/0.1` | User-Agent header |

## How it works, and what the corpus does to the output

- **Case variants are merged, not filtered.** The API returns `on the table` and
  `on the Table` as separate rows. Sending `flags=cs` *drops* rows and with them real
  probability mass, so counts are summed over the casefolded next word instead.
- **Punctuation is a sampleable token.** `.` is the most likely successor of many contexts
  and carries the sentence boundary. Prose spacing is applied once, at render time, by
  `detokenise()`.
- **Backoff 4 → 1.** An unseen four-word context returns nothing, so the leftmost word is
  dropped and the query repeated, down to one word. The level actually used is recorded on
  every step. A context the API refuses to parse at all counts as "nothing at this level".
- **`_START_` costs a query slot.** Queries are capped at five tokens including the
  wildcard, so a sentence opening carries `_START_` plus at most three words; `_START_` is
  dropped once the context alone fills the query.
- **Every sentence records why it stopped**: `end_token`, `length_guard` or `dead_end`.
- **The English corpus drops n-grams seen fewer than 40 times.** The tail of every
  distribution here is truncated at the source, so the output is slightly more concentrated
  on common continuations than English really is. That is a property of the data, not a bug.
- **Batching cannot help.** The batch endpoint reads `*` as a literal character rather than
  a wildcard, so next-word lookups are one request each. The on-disk cache, not batching, is
  what makes repeated runs fast.

Data: Google Books Ngram v3 (CC BY 3.0) via ngrams.dev
