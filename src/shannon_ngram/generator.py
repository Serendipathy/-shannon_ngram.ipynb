"""Seeded sampling of sentences and paragraphs from next-word distributions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .config import Settings
from .model import SENTENCE_END, SENTENCE_START, NextWordModel, query_words

STOP_END_TOKEN = "end_token"
STOP_LENGTH_GUARD = "length_guard"
STOP_DEAD_END = "dead_end"
STOP_REASONS = (STOP_END_TOKEN, STOP_LENGTH_GUARD, STOP_DEAD_END)

#: no space before these when rendering
CLOSING = set(".,;:!?)]}%’”'\"")
#: no space after these
OPENING = set("([{$‘“")
#: glued to both neighbours inside a word
INFIX = {"-", "--", "/", "'", "’"}

TERMINAL_PUNCTUATION = {".", "!", "?"}


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    index: int
    context: tuple[str, ...]
    level: int
    backed_off: bool
    chosen: str
    probability: float
    entropy: float
    candidates: int
    total_count: int


@dataclass(frozen=True)
class Sentence:
    tokens: tuple[str, ...]
    text: str
    steps: tuple[Step, ...]
    stop_reason: str


@dataclass(frozen=True)
class Paragraph:
    sentences: tuple[Sentence, ...]
    text: str

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(step for sentence in self.sentences for step in sentence.steps)


def detokenise(tokens: Sequence[str]) -> str:
    """Render token space as prose. Presentation only — sampling never sees this.

    A lone ``"`` is ambiguous, so quotes are counted within the tokens: odd ones
    open (glued to the next word), even ones close (glued to the previous). An
    odd count is closed after the final piece, unless the unmatched quote has no
    word after it — then there is nothing quoted and the quote is dropped.
    """
    pieces = [t for t in tokens if t and t not in (SENTENCE_START, SENTENCE_END)]
    quote_positions = [i for i, t in enumerate(pieces) if t == '"']
    if len(quote_positions) % 2 == 1:
        last = quote_positions[-1]
        if not any(c.isalnum() for t in pieces[last + 1 :] for c in t):
            del pieces[last]

    out: list[str] = []
    glue_next = False
    quotes = 0
    for token in pieces:
        opening_quote = False
        if token == '"':
            quotes += 1
            opening_quote = quotes % 2 == 1
        if not out:
            out.append(token)
        elif glue_next or (
            not opening_quote and (token in INFIX or token[0] in CLOSING)
        ):
            out[-1] += token
        else:
            out.append(token)
        glue_next = opening_quote or token[-1] in OPENING or token in INFIX
    if quotes % 2 == 1 and out:
        out[-1] += '"'
    text = " ".join(out)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


class ParagraphGenerator:
    """Samples words from the model's distributions with one seeded RNG.

    Candidates are put in a canonical ``(-count, token)`` order before the
    cumulative sum, so the same seed and the same cache always yield the same
    paragraph regardless of dict iteration order.
    """

    def __init__(self, model: NextWordModel, settings: Settings, seed: int | None = None) -> None:
        self.model = model
        self.settings = settings
        self.seed = seed
        self.random = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.random = random.Random(self.seed)

    def sample_next(
        self,
        context: Sequence[str],
        *,
        at_sentence_start: bool = False,
        index: int = 0,
    ) -> tuple[str, Step] | None:
        dist = self.model.distribution(context, at_sentence_start=at_sentence_start)
        if dist is None:
            return None

        ordered = sorted(dist.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        target = self.random.random() * dist.total_count
        cumulative = 0
        chosen = ordered[-1][0]
        for word, count in ordered:
            cumulative += count
            if target < cumulative:
                chosen = word
                break

        step = Step(
            index=index,
            context=dist.context,
            level=dist.level,
            backed_off=dist.level < _requested_level(context, self.settings),
            chosen=chosen,
            probability=dist.dist[chosen],
            entropy=dist.entropy,
            candidates=dist.candidates,
            total_count=dist.total_count,
        )
        return chosen, step

    def sentence(self, seed_words: Sequence[str] = ()) -> Sentence:
        tokens: list[str] = [w for w in seed_words if w and w != SENTENCE_START]
        steps: list[Step] = []
        generated = 0
        stop_reason = STOP_LENGTH_GUARD

        while generated < self.settings.max_sentence_words:
            # _START_ stays in the query until the sentence itself is long
            # enough to fill the context on its own (D-007).
            result = self.sample_next(
                tokens,
                at_sentence_start=len(query_words(tokens)) < self.settings.context_length,
                index=len(steps),
            )
            if result is None:
                stop_reason = STOP_DEAD_END
                if not tokens:
                    raise GenerationError(
                        "no distribution for the sentence-start context; "
                        "the corpus or the cache is empty"
                    )
                break
            word, step = result
            steps.append(step)
            generated += 1
            if word == SENTENCE_END:
                stop_reason = STOP_END_TOKEN
                break
            tokens.append(word)

        if tokens and tokens[-1] not in TERMINAL_PUNCTUATION:
            tokens.append(".")

        return Sentence(
            tokens=tuple(tokens),
            text=detokenise(tokens),
            steps=tuple(steps),
            stop_reason=stop_reason,
        )

    def paragraph(
        self, seed_words: Sequence[str] = (), max_sentences: int | None = None
    ) -> Paragraph:
        limit = self.settings.max_sentences if max_sentences is None else max_sentences
        sentences: list[Sentence] = []
        for position in range(limit):
            sentence = self.sentence(seed_words if position == 0 else ())
            sentences.append(sentence)
        text = " ".join(s.text for s in sentences if s.text).strip()
        return Paragraph(sentences=tuple(sentences), text=text)


def _requested_level(context: Sequence[str], settings: Settings) -> int:
    return min(settings.context_length, len(query_words(context)))
