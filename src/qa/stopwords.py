from __future__ import annotations

from functools import lru_cache


#Return NLTK's English stopwords, downloading the corpus on first use if missing.
@lru_cache(maxsize=1)
def load_english_stopwords() -> frozenset[str]:

    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("english")
    except LookupError:  # corpus not downloaded yet on this machine
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return frozenset(words)