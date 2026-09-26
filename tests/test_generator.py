"""AC 8 (same seed, same paragraph) and AC 9 (bounded paragraphs, recorded stops)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shannon_ngram.client import Ngram
from shannon_ngram.config import load_settings
from shannon_ngram.generator import (
    STOP_DEAD_END,
    STOP_END_TOKEN,
    STOP_LENGTH_GUARD,
    STOP_REASONS,
    GenerationError,
    Paragraph,
    ParagraphGenerator,
    detokenise,
)
from shannon_ngram.model import NextWordModel

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: walk_corpus.json is a closed *bigram* corpus, so the walk runs at context
#: length 1; longer contexts are covered by the backoff fixtures in test_model.
WALK_CONTEXT_LENGTH = 1


class CorpusClient:
    """Serves the frozen walk corpus. Unknown query -> empty result."""

    def __init__(self, corpus: dict[str, list[Ngram]]):
        self.corpus = corpus
        self.queries: list[str] = []

    def search_all(self, query, *, max_pages=None, limit=None, flags=()):
        self.queries.append(query)
        return list(self.corpus.get(query, []))


@pytest.fixture(scope="module")
def walk_corpus() -> dict[str, list[Ngram]]:
    raw = json.loads((FIXTURES / "walk_corpus.json").read_text(encoding="utf-8"))
    return {
        query: [Ngram.from_json(n) for n in body["ngrams"]]
        for query, body in raw.items()
    }


@pytest.fixture
def settings():
    return load_settings(
        context_length=WALK_CONTEXT_LENGTH,
        min_context_length=1,
        max_sentences=5,
        max_sentence_words=40,
    )


def make_generator(walk_corpus, settings, seed):
    model = NextWordModel(CorpusClient(walk_corpus), settings)
    return ParagraphGenerator(model, settings, seed=seed)


# --- AC 8: determinism ---------------------------------------------------------


def test_same_seed_same_paragraph(walk_corpus, settings):
    first = make_generator(walk_corpus, settings, seed=7).paragraph()
    second = make_generator(walk_corpus, settings, seed=7).paragraph()
    assert first.text == second.text
    assert first.steps == second.steps
    assert first.text


def test_different_seed_differs(walk_corpus, settings):
    """Guards against a constant-output bug faking the determinism pass."""
    texts = {
        make_generator(walk_corpus, settings, seed=s).paragraph().text
        for s in range(12)
    }
    assert len(texts) > 1


def test_reset_replays_the_same_paragraph(walk_corpus, settings):
    generator = make_generator(walk_corpus, settings, seed=3)
    first = generator.paragraph()
    generator.reset()
    assert generator.paragraph().text == first.text


def test_sampling_order_does_not_depend_on_dict_order(walk_corpus, settings):
    shuffled = {k: list(reversed(v)) for k, v in walk_corpus.items()}
    a = make_generator(walk_corpus, settings, seed=11).paragraph()
    b = make_generator(shuffled, settings, seed=11).paragraph()
    assert a.text == b.text


# --- AC 9: bounded paragraphs, every sentence has a recorded stop ---------------


@pytest.mark.parametrize("seed", range(50))
def test_fifty_paragraphs_bounded(walk_corpus, settings, seed):
    paragraph = make_generator(walk_corpus, settings, seed=seed).paragraph()
    assert isinstance(paragraph, Paragraph)
    assert 1 <= len(paragraph.sentences) <= settings.max_sentences
    for sentence in paragraph.sentences:
        assert sentence.stop_reason in STOP_REASONS
        assert len(sentence.steps) <= settings.max_sentence_words
        if sentence.stop_reason == STOP_END_TOKEN:
            assert sentence.steps[-1].chosen == "_END_"
        if sentence.stop_reason == STOP_LENGTH_GUARD:
            assert len(sentence.steps) == settings.max_sentence_words


def test_max_sentences_is_honoured(walk_corpus, settings):
    generator = make_generator(walk_corpus, settings, seed=1)
    assert len(generator.paragraph(max_sentences=2).sentences) == 2


def test_length_guard_closes_the_sentence(walk_corpus):
    tight = load_settings(context_length=1, max_sentence_words=3, max_sentences=1)
    paragraph = make_generator(walk_corpus, tight, seed=5).paragraph()
    sentence = paragraph.sentences[0]
    assert len(sentence.steps) <= 3
    if sentence.stop_reason == STOP_LENGTH_GUARD:
        assert sentence.tokens[-1] in {".", "!", "?"}


def test_dead_end_is_recorded_not_raised(settings):
    """A context with no successors ends the sentence and says so."""
    corpus = {
        "_START_ *": _rows({"zzz": 10}),
        "zzz *": [],
    }
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient(corpus), settings), settings, seed=0
    )
    sentence = generator.sentence()
    assert sentence.stop_reason == STOP_DEAD_END
    assert sentence.tokens[-1] == "."
    assert sentence.text == "zzz."


def test_dead_end_with_nothing_generated_raises(settings):
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient({}), settings), settings, seed=0
    )
    with pytest.raises(GenerationError):
        generator.sentence()


def test_end_token_is_not_rendered(walk_corpus, settings):
    for seed in range(20):
        paragraph = make_generator(walk_corpus, settings, seed=seed).paragraph()
        assert "_END_" not in paragraph.text
        assert "_START_" not in paragraph.text


def test_steps_property_concatenates_sentence_steps(walk_corpus, settings):
    paragraph = make_generator(walk_corpus, settings, seed=2).paragraph()
    assert len(paragraph.steps) == sum(len(s.steps) for s in paragraph.sentences)


def test_every_step_records_level_and_probability(walk_corpus, settings):
    paragraph = make_generator(walk_corpus, settings, seed=4).paragraph()
    for step in paragraph.steps:
        assert 0 <= step.level <= settings.context_length
        assert 0.0 < step.probability <= 1.0
        assert step.candidates >= 1
        assert step.total_count > 0
        assert step.entropy >= 0.0


# --- sampling mechanics --------------------------------------------------------


def _rows(counts: dict[str, int]) -> list[Ngram]:
    return [
        Ngram.from_json(
            {
                "id": f"{word}-{count}",
                "absTotalMatchCount": count,
                "tokens": [
                    {"text": "ctx", "kind": "TERM"},
                    {"text": word, "kind": "TERM", "inserted": True},
                ],
            }
        )
        for word, count in counts.items()
    ]


def test_sample_next_respects_the_distribution(settings):
    corpus = {"ctx *": _rows({"a": 900, "b": 100})}
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient(corpus), settings), settings, seed=0
    )
    draws = [generator.sample_next(["ctx"])[0] for _ in range(400)]
    share = draws.count("a") / len(draws)
    assert 0.85 < share < 0.95


def test_sample_next_returns_none_on_dead_end(settings):
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient({}), settings), settings, seed=0
    )
    assert generator.sample_next(["nothing"]) is None


def test_sample_next_reports_backoff(settings):
    four = load_settings(context_length=4, min_context_length=1)
    corpus = {"d *": _rows({"x": 5})}
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient(corpus), four), four, seed=0
    )
    word, step = generator.sample_next(["a", "b", "c", "d"])
    assert word == "x"
    assert step.level == 1
    assert step.backed_off is True


def test_no_backoff_flag_when_the_full_context_hits(settings):
    corpus = {"ctx *": _rows({"a": 1})}
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient(corpus), settings), settings, seed=0
    )
    _, step = generator.sample_next(["ctx"])
    assert step.backed_off is False


# --- detokenisation (P-3) ------------------------------------------------------


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["the", "cat", "sat", "."], "the cat sat."),
        (["hello", ",", "world", "!"], "hello, world!"),
        (["he", "said", "(", "loudly", ")", "."], "he said (loudly)."),
        (["twenty", "-", "five"], "twenty-five"),
        (["do", "n't", "stop"], "do n't stop"),
        (["don", "'", "t"], "don't"),
        (["_START_", "a", "b", "_END_"], "a b"),
        ([], ""),
        (["a", "", "b"], "a b"),
        (["50", "%", "of", "it"], "50% of it"),
        (["$", "5", "each"], "$5 each"),
        (["a", ";", "b", ":", "c"], "a; b: c"),
        (['"', "Yes", '"', "."], '"Yes".'),
        (["she", "said", ",", '"', "no", '"', "."], 'she said, "no".'),
        (['"', "a", '"', "and", '"', "b", "c", '"'], '"a" and "b c"'),
        (["he", "said", '"', "wait"], 'he said "wait"'),
        (["he", "said", '"', "wait", "."], 'he said "wait."'),
        (["he", "said", '"', "."], "he said."),
        (['"', "a", '"', "b", '"', "!"], '"a" b!'),
    ],
)
def test_detokenise_punctuation_spacing(tokens, expected):
    assert detokenise(tokens) == expected


def test_detokenise_collapses_double_spaces():
    assert "  " not in detokenise(["a", "", "", "b"])


# --- lone double quote (D-009) -------------------------------------------------


def test_sample_next_never_follows_a_double_quote_with_one(settings):
    corpus = {"ctx *": _rows({'"': 900, "a": 100})}
    generator = ParagraphGenerator(
        NextWordModel(CorpusClient(corpus), settings), settings, seed=0
    )
    draws = [generator.sample_next(["ctx", '"'])[0] for _ in range(200)]
    assert set(draws) == {"a"}


def test_double_quote_is_kept_in_the_text_and_out_of_the_query(settings):
    client = CorpusClient(
        {
            "_START_ *": _rows({'"': 10**9, "Yes": 1}),
            "Yes *": _rows({"_END_": 1}),
        }
    )
    generator = ParagraphGenerator(NextWordModel(client, settings), settings, seed=0)
    sentence = generator.sentence()
    assert sentence.tokens == ('"', "Yes", ".")
    assert sentence.text == '"Yes."'
    assert all('"' not in q.split() for q in client.queries)
    assert sentence.stop_reason == STOP_END_TOKEN


def test_end_token_after_a_non_terminal_piece_closes_with_a_period(settings):
    client = CorpusClient(
        {
            "_START_ *": _rows({"Chapter": 1}),
            "Chapter *": _rows({"_END_": 1}),
        }
    )
    generator = ParagraphGenerator(NextWordModel(client, settings), settings, seed=0)
    sentence = generator.sentence()
    assert sentence.stop_reason == STOP_END_TOKEN
    assert sentence.tokens == ("Chapter", ".")
    assert sentence.text == "Chapter."
