"""Running a retrieval configuration over the golden set."""

from __future__ import annotations

import time

from .. import store
from . import labels as label_store
from . import metrics, pooling
from .queries import GOLDEN_QUERIES


def build_pool(engine) -> dict:
    frame, text = pooling.load_frame()
    terms = store.load_corpus_terms()
    lookup = {int(i): (t, a) for i, t, a in
              zip(frame.index, frame["title"], frame["abstract"])}

    pool = {}
    for query in GOLDEN_QUERIES:
        built = pooling.build(engine, text, terms, query["text"])
        pool[query["id"]] = {
            "query": query["text"],
            "pooled": built["pooled"],
            "sources": built["sources"],
            "candidates": [
                {"id": doc,
                 "title": lookup.get(doc, ("", ""))[0],
                 "abstract": (lookup.get(doc, ("", ""))[1] or "")[:420]}
                for doc in built["pooled"]
            ],
        }
        print(f"  {query['id']}: {len(built['pooled'])} pooled "
              f"(dense {len(built['sources']['dense'])}, "
              f"lexical {len(built['sources']['lexical'])}, "
              f"subquery {len(set(built['sources']['subquery']))})")
    return pool


def rank(engine, query_text: str, depth: int = 50) -> list[int]:
    """A configuration's ranking for one query. This is the thing under test."""
    points = store.query(engine.client, engine.embedder.encode_one(query_text), depth)
    return [int(p.id) for p in points]


def run(engine, depth: int = 50, label_set: dict | None = None) -> dict:
    judged = label_set if label_set is not None else label_store.load_labels()
    if not judged:
        raise SystemExit("No labels yet. Build the pool and grade it first.")

    per_query, timings = {}, []
    for query in GOLDEN_QUERIES:
        marks = judged.get(query["id"])
        if not marks or not any(g >= metrics.RELEVANT_GRADE for g in marks.values()):
            continue
        started = time.perf_counter()
        ranked = rank(engine, query["text"], depth)
        timings.append((time.perf_counter() - started) * 1000)
        per_query[query["id"]] = metrics.evaluate(ranked, marks)

    return {
        "queries_scored": len(per_query),
        "summary": metrics.aggregate(per_query),
        "per_query": per_query,
        "latency_ms": {
            "mean": round(sum(timings) / len(timings), 1) if timings else None,
            "max": round(max(timings), 1) if timings else None,
        },
    }


def report(result: dict, title: str) -> str:
    s = result["summary"]
    lines = [
        f"{title}",
        f"  queries scored     {result['queries_scored']}",
        f"  recall@10          {s.get('recall@10')}",
        f"  recall@20          {s.get('recall@20')}",
        f"  recall@50          {s.get('recall@50')}",
        f"  nDCG@10            {s.get('ndcg@10')}",
        f"  MRR                {s.get('mrr')}",
        f"  precision@10       {s.get('precision@10')}",
        f"  latency mean/max   {result['latency_ms']['mean']} / {result['latency_ms']['max']} ms",
    ]
    return "\n".join(lines)
