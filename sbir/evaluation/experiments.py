"""Retrieval experiments run against the golden set.

Each variant builds its own index in its own directory so nothing here can
disturb the index the application serves, and so two variants can be compared
without rebuilding either.

Experiments run on a stratified sample rather than the full corpus. A full
build costs about 23 minutes per configuration at 128 tokens and roughly three
times that at 256, which is too slow to rerun after every change. The sample
keeps every graded award -- otherwise recall would be measured against awards
that are not present -- and fills the remainder at random so the retriever
still has to work through a realistic amount of unrelated material.
"""

from __future__ import annotations

import shutil
import time

import pandas as pd
from qdrant_client import QdrantClient, models

from .. import config, dataset
from ..embedder import Embedder
from . import metrics
from . import labels as label_store
from .queries import GOLDEN_QUERIES

EXPERIMENT_DIR = config.DATA_DIR / "experiments"
COLLECTION = "variant"
SAMPLE_SIZE = 50_000
SEED = 20260818


def sample(frame: pd.DataFrame, graded: set[int], size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Every graded award, plus random distractors up to ``size``."""
    keep = frame.index.isin(graded)
    graded_rows = frame[keep]
    remainder = frame[~keep]
    wanted = max(0, size - len(graded_rows))
    filler = remainder.sample(n=min(wanted, len(remainder)), random_state=SEED)
    return pd.concat([graded_rows, filler]).sort_index()


def build(frame: pd.DataFrame, name: str, max_seq_length: int,
          batch_size: int = 64) -> dict:
    """Embed the sample at a given window and store it under its own path."""
    path = EXPERIMENT_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(max_seq_length=max_seq_length)
    client = QdrantClient(path=str(path))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=embedder.dimension, distance=models.Distance.COSINE
        ),
    )

    texts = [dataset.embedding_text(row) for _, row in frame.iterrows()]
    ids = [int(i) for i in frame.index]
    started = time.time()

    for start in range(0, len(texts), batch_size):
        chunk_ids = ids[start:start + batch_size]
        vectors = embedder.encode(texts[start:start + batch_size], batch_size=batch_size)
        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(id=i, vector=v.tolist(), payload={})
                for i, v in zip(chunk_ids, vectors)
            ],
        )
        if start and start % 10_000 == 0:
            rate = start / (time.time() - started)
            print(f"    {start:,}/{len(texts):,}  {rate:.0f}/s", flush=True)

    elapsed = time.time() - started
    client.close()
    return {
        "name": name,
        "max_seq_length": max_seq_length,
        "awards": len(texts),
        "build_seconds": round(elapsed, 1),
        "rate": round(len(texts) / elapsed, 1),
    }


def score(name: str, max_seq_length: int, depth: int = 50) -> dict:
    """Run the golden queries against a built variant."""
    judged = label_store.load_labels()
    embedder = Embedder(max_seq_length=max_seq_length)
    client = QdrantClient(path=str(EXPERIMENT_DIR / name))

    per_query, timings = {}, []
    for query in GOLDEN_QUERIES:
        marks = judged.get(query["id"])
        if not marks or not any(g >= metrics.RELEVANT_GRADE for g in marks.values()):
            continue
        started = time.perf_counter()
        points = client.query_points(
            collection_name=COLLECTION,
            query=embedder.encode_one(query["text"]),
            limit=depth,
            with_payload=False,
        ).points
        timings.append((time.perf_counter() - started) * 1000)
        per_query[query["id"]] = metrics.evaluate([int(p.id) for p in points], marks)

    client.close()
    return {
        "queries_scored": len(per_query),
        "summary": metrics.aggregate(per_query),
        "per_query": per_query,
        "latency_ms": {"mean": round(sum(timings) / len(timings), 1)} if timings else {},
    }


def experiment_one(sample_size: int = SAMPLE_SIZE) -> dict:
    """Does the 128-token window lose retrievable content?

    79.4% of awards exceed 128 tokens and the median award has 49% of its text
    embedded, with the truncated tail skewed towards commercialisation and
    transition language. If that tail carries retrievable signal, widening the
    window should raise recall.
    """
    graded = {doc for marks in label_store.load_labels().values() for doc in marks}
    print(f"Loading corpus ({len(graded):,} graded awards must be present)...")
    frame = dataset.load()
    subset = sample(frame, graded, sample_size)
    present = sum(1 for doc in graded if doc in subset.index)
    print(f"Sample: {len(subset):,} awards, {present:,}/{len(graded):,} graded present\n")

    results = {}
    for name, window in (("w128", 128), ("w256", 256)):
        print(f"Building {name} (max_seq_length={window})...")
        built = build(subset, name, window)
        print(f"  built in {built['build_seconds'] / 60:.1f} min at {built['rate']}/s")
        scored = score(name, window)
        results[name] = {**built, **scored}
        s = scored["summary"]
        print(f"  recall@50 {s['recall@50']}  recall@10 {s['recall@10']}  "
              f"nDCG@10 {s['ndcg@10']}  P@10 {s['precision@10']}\n", flush=True)

    return results


def report(results: dict) -> str:
    base, variant = results["w128"], results["w256"]
    lines = [
        "Experiment 1 - embedding window, on a stratified sample",
        f"  sample size        {base['awards']:,} awards",
        "",
        f"  {'metric':<18} {'128 tok':>10} {'256 tok':>10} {'delta':>10}",
    ]
    for key in ("recall@10", "recall@20", "recall@50", "ndcg@10", "precision@10", "mrr"):
        a, b = base["summary"][key], variant["summary"][key]
        lines.append(f"  {key:<18} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")
    lines += [
        "",
        f"  {'build minutes':<18} {base['build_seconds'] / 60:>10.1f} "
        f"{variant['build_seconds'] / 60:>10.1f} "
        f"{(variant['build_seconds'] - base['build_seconds']) / 60:>+10.1f}",
        f"  {'embed rate /s':<18} {base['rate']:>10.1f} {variant['rate']:>10.1f}",
    ]
    return "\n".join(lines)
