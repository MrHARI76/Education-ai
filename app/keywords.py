"""Find the key concepts of a document with YAKE, then remove near-duplicates.

YAKE ("Yet Another Keyword Extractor") scores phrases using simple
statistics (how often and where a phrase appears); it needs no training
and no internet. A LOWER YAKE score means a MORE important phrase.

YAKE often returns near-identical phrases ("neural network" and "neural
networks"), so we embed every phrase and drop any that is too similar
(cosine > 0.85) to a better-scored phrase already kept.
"""

import re

import yake

from app.embeddings import embed_texts

MAX_TEXT_CHARS = 200_000  # YAKE gets slow on huge inputs; the start of a doc is enough
SIMILARITY_THRESHOLD = 0.85

# Words that make a phrase a "fragment" if it starts or ends with them.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "it", "its", "into", "than", "then", "also", "can", "will",
    "such", "which", "who", "what", "when", "where", "how", "not", "no", "we", "you",
}


def is_useful_phrase(phrase: str) -> bool:
    """Reject numbers, very short tokens and phrases that start/end with a stopword."""
    words = phrase.lower().split()
    if not words or len(phrase) < 3:
        return False
    if all(re.fullmatch(r"[\d\W_]+", w) for w in words):  # only digits/punctuation
        return False
    return words[0] not in STOPWORDS and words[-1] not in STOPWORDS


def dedupe_by_meaning(phrases: list[str]) -> list[str]:
    """Keep phrases in order, skipping any too similar to one already kept.

    `phrases` must be sorted best-first, so the higher-scored one survives.
    """
    if not phrases:
        return []
    vectors = embed_texts(phrases)
    kept: list[int] = []
    for i in range(len(phrases)):
        # Vectors are normalised, so the dot product is the cosine similarity.
        if all(float(vectors[i] @ vectors[j]) <= SIMILARITY_THRESHOLD for j in kept):
            kept.append(i)
    return [phrases[i] for i in kept]


def extract_keywords(text: str, top: int = 40) -> list[str]:
    """Return up to `top` key phrases (1–3 words), most important first."""
    extractor = yake.KeywordExtractor(lan="en", n=3, top=top, dedupLim=0.9)
    scored = extractor.extract_keywords(text[:MAX_TEXT_CHARS])
    scored.sort(key=lambda pair: pair[1])  # lower score = more important
    phrases = [phrase for phrase, _score in scored if is_useful_phrase(phrase)]
    return dedupe_by_meaning(phrases)
