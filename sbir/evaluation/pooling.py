"""Candidate pooling for labelling.

A golden set graded from one retriever's output measures nothing: the labels
inherit that retriever's blind spots, and it scores full marks against itself.
The pool is therefore built from three deliberately different strategies, and
the judge never sees which one produced a candidate.
"""

from __future__ import annotations

import pandas as pd

from .. import dataset, derive

DENSE_DEPTH = 30
LEXICAL_DEPTH = 25
SUBQUERY_DEPTH = 8


def corpus_text(frame: pd.DataFrame) -> pd.Series:
    """Lower-cased title and abstract, built once and reused across queries."""
    return (frame["title"].fillna("") + " " + frame["abstract"].fillna("")).str.lower()


def lexical(text: pd.Series, query: str, terms: dict,
            depth: int = LEXICAL_DEPTH) -> list[int]:
    """A keyword retriever, used only to widen the pool.

    Deliberately not part of the product. Its job here is to surface awards a
    dense retriever might rank low, so that if the embedding is missing
    something the labels can still record it.
    """
    words = [
        w for w in derive._WORD.findall(query.lower())
        if w not in derive.STOPWORDS
    ]
    if not words:
        return []

    score = pd.Series(0.0, index=text.index)
    for word in set(words):
        weight = derive.term_weight(word, terms)
        score += text.str.contains(word, regex=False).astype(float) * weight
    hits = score[score > 0]
    return [int(i) for i in hits.nlargest(depth).index]


def subqueries(query: str) -> list[str]:
    """Adjacent content-word pairs, retrieved separately.

    A long query averages its concepts into one vector, which can bury awards
    that match one concept strongly. Retrieving the parts recovers them.
    """
    words = [
        w for w in derive._WORD.findall(query.lower())
        if w not in derive.STOPWORDS
    ]
    return [f"{a} {b}" for a, b in zip(words, words[1:])] or words


def build(engine, text: pd.Series, terms: dict, query: str) -> dict:
    """Union of the three strategies, with per-source provenance kept.

    Pooling is always unfiltered. Filters are a product feature; applying them
    here would narrow what can ever be labelled.
    """
    from .. import store

    sources: dict[str, list[int]] = {}

    points = store.query(engine.client, engine.embedder.encode_one(query), DENSE_DEPTH)
    sources["dense"] = [int(p.id) for p in points]

    sources["lexical"] = lexical(text, query, terms)

    sub: list[int] = []
    for phrase in subqueries(query)[:6]:
        pts = store.query(engine.client, engine.embedder.encode_one(phrase), SUBQUERY_DEPTH)
        sub.extend(int(p.id) for p in pts)
    sources["subquery"] = sub

    pooled: list[int] = []
    for ids in sources.values():
        for doc in ids:
            if doc not in pooled:
                pooled.append(doc)

    return {"pooled": pooled, "sources": {k: sorted(set(v)) for k, v in sources.items()}}


def load_frame():
    frame = dataset.load()
    return frame, corpus_text(frame)
