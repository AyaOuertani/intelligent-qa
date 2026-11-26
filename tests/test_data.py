import pytest

from qa.data import (
    answer_offsets_match_context,
    build_passage_pool,
    is_answerable,
    load_split_definition,
    save_split_definition,
    split_articles_into_train_and_validation,
)
from qa.exploration import question_context_overlap, question_type, summarise_split

AI_CONTEXT = "Artificial intelligence was founded as an academic discipline in 1956."
PARIS_CONTEXT = "Paris is the capital of France."

EXAMPLES = [
    {"id": "q1", "title": "AI", "context": AI_CONTEXT,
     "question": "When was AI established as an academic field?",
     "answers": {"text": ["1956"], "answer_start": [65]}},
    {"id": "q2", "title": "AI", "context": AI_CONTEXT,
     "question": "Who funded the first AI lab?",
     "answers": {"text": [], "answer_start": []}},
    {"id": "q3", "title": "Paris", "context": PARIS_CONTEXT,
     "question": "What is the capital of France?",
     "answers": {"text": ["Paris"], "answer_start": [0]}},
]

def test_is_answerable_uses_empty_answer_list():
    assert is_answerable(EXAMPLES[0]) is True
    assert is_answerable(EXAMPLES[1]) is False

def test_answer_offsets_are_character_positions():
    assert answer_offsets_match_context(EXAMPLES[0])
    shifted = {**EXAMPLES[0], "answers": {"text": ["1956"], "answer_start": [64]}}
    assert not answer_offsets_match_context(shifted)

def test_article_split_has_no_overlap_and_covers_everything():
    titles = [f"article_{i}" for i in range(100)]
    train, validation = split_articles_into_train_and_validation(titles, 0.1, seed=42)
    assert len(validation) == 10
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(titles)

def test_article_split_is_reproducible_and_order_independent():
    titles = [f"article_{i}" for i in range(50)]
    first = split_articles_into_train_and_validation(titles, 0.2, seed=7)
    second = split_articles_into_train_and_validation(list(reversed(titles)), 0.2, seed=7)
    assert first == second

def test_invalid_validation_fraction_is_rejected():
    with pytest.raises(ValueError):
        split_articles_into_train_and_validation(["a", "b"], validation_fraction=1.5)

def test_passage_pool_deduplicates_paragraphs_and_links_questions():
    passages, question_to_passage = build_passage_pool(EXAMPLES)
    assert len(passages) == 2
    assert question_to_passage["q1"] == question_to_passage["q2"] == "AI__p0"
    assert question_to_passage["q3"] == "Paris__p0"

def test_split_definition_round_trip(tmp_path):
    path = tmp_path / "splits.json"
    save_split_definition(path, ["A"], ["B"], seed=42)
    loaded = load_split_definition(path)
    assert loaded["train_titles"] == ["A"] and loaded["validation_titles"] == ["B"]
    assert loaded["seed"] == 42
    
# Small explicit list: tests stay offline and the expected values are easy to check by hand.
TEST_STOPWORDS = frozenset({"what", "when", "is", "the", "of", "was", "as", "an"})


def test_question_type():
    assert question_type("When was AI established?") == "when"
    assert question_type("Name the capital of France.") == "other"


def test_overlap_counts_only_content_words():
    # content words: capital, france -> both in the context
    assert question_context_overlap(EXAMPLES[2]["question"], PARIS_CONTEXT, TEST_STOPWORDS) == 1.0
    # content words: ai, established, academic, field -> only "academic" in the context
    assert question_context_overlap(
        EXAMPLES[0]["question"], AI_CONTEXT, TEST_STOPWORDS
    ) == pytest.approx(0.25)


def test_overlap_of_question_made_only_of_stopwords_is_zero():
    assert question_context_overlap("What is the?", PARIS_CONTEXT, TEST_STOPWORDS) == 0.0


def test_nltk_english_stopwords_are_loaded():
    from qa.stopwords import load_english_stopwords

    try:
        stopwords = load_english_stopwords()
    except LookupError:
        pytest.skip("NLTK stopwords corpus not available (no internet)")
    assert {"the", "of", "when", "what", "not"} <= stopwords
    assert "france" not in stopwords and "capital" not in stopwords


def test_summarise_split_counts():
    summary = summarise_split(EXAMPLES)
    assert summary == {"questions": 3, "articles": 2, "paragraphs": 2,
                       "unanswerable": 1, "unanswerable_share": pytest.approx(1 / 3)}
