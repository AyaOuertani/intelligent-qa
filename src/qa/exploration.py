from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from qa.stopwords import load_english_stopwords

WORD_PATTERN = re.compile(r"[a-z0-9]+")
QUESTION_WORDS = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")

#Lowercase and keep runs of letters/digits
def simple_word_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())

#The first wh-word found in the question, or 'other'
def question_type(question: str) -> str:
    for token in simple_word_tokens(question):
        if token in QUESTION_WORDS:
            return token
    return "other"

#Share of the question's content words that also appear in the context (0 to 1).
def question_context_overlap(question: str, context: str, stopwords: frozenset[str] | None = None) -> float:
    if stopwords is None:
        stopwords = load_english_stopwords()
    question_words = {w for w in simple_word_tokens(question) if w not in stopwords}
    if not question_words:
        return 0.0
    context_words = set(simple_word_tokens(context))
    return len(question_words & context_words) / len(question_words)


#Headline counts for one split: questions, articles, paragraphs, unanswerable share
def summarise_split(examples: Iterable[dict]) -> dict:
    n_questions = n_unanswerable = 0
    titles, paragraphs = set(), set()
    for example in examples:
        n_questions += 1
        n_unanswerable += len(example["answers"]["text"]) == 0
        titles.add(example["title"])
        paragraphs.add((example["title"], example["context"]))
    return {
        "questions": n_questions,
        "articles": len(titles),
        "paragraphs": len(paragraphs),
        "unanswerable": n_unanswerable,
        "unanswerable_share": n_unanswerable / n_questions if n_questions else 0.0,
    }

#Its job is to count how many questions belong to each question type
def count_question_types(examples: Iterable[dict]) -> Counter:
    return Counter(question_type(example["question"]) for example in examples)
