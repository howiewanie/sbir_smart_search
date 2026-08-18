"""Retrieval metrics.

Graded relevance throughout: 2 means clearly on topic, 1 adjacent, 0 not
relevant. Anything outside the judged pool counts as 0, which is the standard
pooling assumption and the reason the pool is built from several different
retrieval configurations rather than from the system under test.
"""

from __future__ import annotations

import math

# Only grade 2 counts towards recall. Adjacent awards still carry weight in
# nDCG, where partial credit is the point, but calling them "found" would
# flatter any system that returns loosely related work.
RELEVANT_GRADE = 2


def recall_at_k(ranked: list[int], labels: dict[int, int], k: int) -> float | None:
    total = sum(1 for g in labels.values() if g >= RELEVANT_GRADE)
    if not total:
        return None
    found = sum(1 for doc in ranked[:k] if labels.get(doc, 0) >= RELEVANT_GRADE)
    return found / total


def precision_at_k(ranked: list[int], labels: dict[int, int], k: int) -> float:
    if not k:
        return 0.0
    return sum(1 for doc in ranked[:k] if labels.get(doc, 0) >= RELEVANT_GRADE) / k


def mrr(ranked: list[int], labels: dict[int, int]) -> float:
    for position, doc in enumerate(ranked, 1):
        if labels.get(doc, 0) >= RELEVANT_GRADE:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[int], labels: dict[int, int], k: int) -> float | None:
    def dcg(grades: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg([labels.get(doc, 0) for doc in ranked[:k]])
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if not ideal:
        return None
    return actual / ideal


def evaluate(ranked: list[int], labels: dict[int, int],
             ks: tuple[int, ...] = (10, 20, 50)) -> dict:
    out: dict[str, float | None] = {"mrr": mrr(ranked, labels)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, labels, k)
        out[f"precision@{k}"] = precision_at_k(ranked, labels, k)
    out["ndcg@10"] = ndcg_at_k(ranked, labels, 10)
    return out


def aggregate(per_query: dict[str, dict]) -> dict:
    """Mean over queries, skipping metrics that were undefined for a query."""
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    summary = {}
    for name in names:
        values = [m[name] for m in per_query.values() if m.get(name) is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
    return summary
