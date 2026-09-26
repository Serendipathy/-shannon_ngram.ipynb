[2026-09-26 05:55 UTC | Claude claude-opus-5-5]

# Decision log — 🟩_shannon_n-gram

One line per decision. Only decisions that constrain future work. Append-only: never delete a row; supersede it.
Status: LOCKED (not reopened unless the PM says so explicitly) · OPEN · SUPERSEDED by D-nnn.

| ID | Date | Decision | Reason | Status | Commit |
|---|---|---|---|---|---|
| D-001 | 2026-09-26 | Word-sequence counts come from the ngrams.dev REST API (`https://api.ngrams.dev`); no raw Google Books Ngram files are downloaded | Full English v3 export is ~13.9 TB compressed (github.com/Squirrelistic/sgbr) | LOCKED | — |
| D-002 | 2026-09-26 | All logic lives in the Python package `shannon_ngram` under `src/`; the notebook only imports and calls it | The tool must move to a VPS without rewriting | LOCKED | — |
| D-003 | 2026-09-26 | VPS hosting, web UI and access control are out of scope for the notebook build | Keep the first build to the notebook | LOCKED | — |
| D-004 | 2026-09-26 | This decision log uses one line per decision and records only decisions that constrain future work | Avoids duplicating commit messages while keeping decisions that produce no commit | LOCKED | — |
| D-005 | 2026-09-26 | ngrams.dev flags are sent as one `flags=` parameter (e.g. `flags=cs`); `cs=true` is silently ignored | Verified by Claude Code probe, reply 260926_0555_del_notebook_build_plan.md §8 A | LOCKED | — |
| D-006 | 2026-09-26 | The batch endpoint is not used for next-word lookup; it treats `*` as a literal character | Verified by probe, same reply §8 B; batch returned literal "on the table *" (1642) vs real "on the table ." (6556170) | LOCKED | — |
| D-007 | 2026-09-26 | At sentence start the query is `_START_` + at most 3 words + `*`; `_START_` is dropped once 4 words exist | `_START_` counts toward the 5-token cap (HTTP 400 INVALID_QUERY.TOO_MANY_TOKENS), same reply §8 C | LOCKED | — |
| D-008 | 2026-09-26 | An `INVALID_QUERY.*` response from ngrams.dev is treated as an empty result and triggers backoff; the 5-token cap applies to API tokens, not words (`don't` re-splits into 3) | Found by the live notebook run, reply 260926_0640_del_notebook_build_go.md §6.1 | LOCKED | 54ae8da, c59944b |
| D-009 | 2026-09-26 | A lone `"` stays in the generated text but is stripped from the context before querying ngrams.dev; when the previous piece is `"`, `"` is removed from the candidate next pieces | Any query containing a lone `"` returns 400 BAD_TERM_GROUP (verified 6 queries, 2026-09-26), causing `dead_end` sentences such as `".`; stripping alone would allow `" "` because `"` is the top successor of "she said ," (8,474,973) and 2nd sentence opener (3.38 bn). PM decision | LOCKED | — |
