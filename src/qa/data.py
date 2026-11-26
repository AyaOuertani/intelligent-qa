from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

SQUAD_V2_HF_NAME = "rajpurkar/squad_v2"

#Download (or read from cache)
def load_squad_v2():
    from datasets import load_dataset
    return load_dataset(SQUAD_V2_HF_NAME)

#SQuAD 2.0 marks an unanswerable question with an empty answer list
def is_answerable(exemple: dict):
    return len(exemple["answers"]["text"]) > 0

#True if every gold answer is found in the context at its stated character offset.
def answer_offsets_match_context(example: dict) -> bool:
    context = example["context"]
    answers = example["answers"]
    return all(
        context[start : start + len(text)] == text
        for text, start in zip(answers["text"], answers["answer_start"])
    )
    

#Split ARTICLE titles (not questions) into train and validation sets
def split_articles_into_train_and_validation(titles: Iterable[str], validation_fraction: float = 0.1, seed: int = 42) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    unique_titles = sorted(set(titles))
    random.Random(seed).shuffle(unique_titles)
    n_validation = max(1, round(len(unique_titles) * validation_fraction))
    validation_titles = sorted(unique_titles[:n_validation])
    train_titles = sorted(unique_titles[n_validation:])
    return train_titles, validation_titles

#Collect the unique paragraphs (passages) and link each question to its gold passage.
def build_passage_pool(examples: Iterable[dict]) -> tuple[list[dict], dict[str, str]]:
    passages: list[dict] = []
    passage_id_by_key: dict[tuple[str, str], str] = {}
    paragraphs_seen_per_title: dict[str, int] = {}
    question_to_passage: dict[str, str] = {}

    for example in examples:
        key = (example["title"], example["context"])
        if key not in passage_id_by_key:
            index = paragraphs_seen_per_title.get(example["title"], 0)
            paragraphs_seen_per_title[example["title"]] = index + 1
            passage_id = f"{example['title']}__p{index}"
            passage_id_by_key[key] = passage_id
            passages.append(
                {"passage_id": passage_id, "title": example["title"], "context": example["context"]}
            )
        question_to_passage[example["id"]] = passage_id_by_key[key]

    return passages, question_to_passage

#Save which articles belong to which split, so every experiment uses the same split
def save_split_definition(path: str | Path, train_titles: list[str], validation_titles: list[str], seed: int) -> None:
    definition = {
        "dataset": SQUAD_V2_HF_NAME,
        "seed": seed,
        "train_titles": train_titles,
        "validation_titles": validation_titles,
        "test": "official SQuAD 2.0 validation (dev) set",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8")

def load_split_definition(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))