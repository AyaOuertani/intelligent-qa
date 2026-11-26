from qa.data import (
    answer_offsets_match_context,
    load_squad_v2,
    save_split_definition,
    split_articles_into_train_and_validation,
)
from qa.exploration import summarise_split

SEED = 42
VALIDATION_FRACTION = 0.1
SPLIT_PATH = "configs/squad_v2_article_splits.json"


def main() -> None:
    squad = load_squad_v2()
    original_train, official_dev = squad["train"], squad["validation"]
    
    bad_offsets = sum(not answer_offsets_match_context(ex) for ex in original_train)
    print(f"Training examples whose answer offsets do not match the context: {bad_offsets}")
    
    train_titles, validation_titles = split_articles_into_train_and_validation(original_train["title"], VALIDATION_FRACTION, SEED)
    leaked = set(train_titles + validation_titles) & set(official_dev["title"])
    print(f"Articles shared with the official dev (test) set: {len(leaked)}")

    validation_set = set(validation_titles)
    train = original_train.filter(lambda ex: ex["title"] not in validation_set)
    validation = original_train.filter(lambda ex:ex["title"] in validation_set)
    
    for name, split in [("train", train), ("validation", validation), ("test (official dev)", official_dev)]:
        print(f"{name:>20}: {summarise_split(split)}")
    
    save_split_definition(SPLIT_PATH, train_titles, validation_titles, SEED)
    print(f"Saved split definition to {SPLIT_PATH}")
    

if __name__ == "__main__":
    main()