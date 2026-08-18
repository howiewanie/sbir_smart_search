"""Turn the award CSV into a searchable vector collection."""

from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
from qdrant_client import models
from tqdm import tqdm

from . import config, dataset, store
from .embedder import Embedder

# Kept in the payload so results can be rendered without touching the CSV again.
PAYLOAD_FIELDS = (
    "title", "abstract", "company", "agency", "branch", "phase", "program",
    "year", "amount", "state", "city", "topic_code", "contract", "website",
    "employees", "hubzone", "women_owned", "disadvantaged", "pi_name",
    "contact_email", "award_date",
)


def _payload(row: pd.Series) -> dict:
    payload = {field: row[field] for field in PAYLOAD_FIELDS}
    payload["year"] = int(row["year"])
    payload["amount"] = float(row["amount"])
    payload["employees"] = int(row["employees"])
    for flag in ("hubzone", "women_owned", "disadvantaged"):
        payload[flag] = bool(row[flag])
    # Lower-cased company gives us exact company lookups without a scan.
    payload["company_key"] = row["company"].lower()
    return payload


def company_map(frame: pd.DataFrame) -> dict[str, list[int]]:
    """Company name -> the point ids it owns.

    Company lookups are exact-name work, not similarity work, so resolving them
    through a plain dictionary is both faster and more predictable than asking
    the vector store for a full-text scan.
    """
    mapping: dict[str, list[int]] = {}
    for point_id, name in zip(frame.index, frame["company"]):
        if name:
            mapping.setdefault(name, []).append(int(point_id))
    return mapping


def build(since: int | None = None, limit: int | None = None,
          batch_size: int = config.BATCH_SIZE) -> dict:
    frame = dataset.load(since=since, limit=limit)
    total = len(frame)
    if total == 0:
        raise SystemExit("No awards matched the requested filters.")

    embedder = Embedder()
    client = store.connect()

    print(f"Indexing {total:,} awards")
    print(f"  model    {embedder.model_name} ({embedder.dimension}d, {embedder.device})")
    print(f"  backend  {store.describe_backend()}")

    store.recreate_collection(client, embedder.dimension)

    texts = [dataset.embedding_text(row) for _, row in frame.iterrows()]
    started = time.time()
    done = 0

    with tqdm(total=total, unit="award", smoothing=0.05) as bar:
        for start in range(0, total, batch_size):
            chunk = frame.iloc[start:start + batch_size]
            vectors = embedder.encode(texts[start:start + batch_size], batch_size=batch_size)
            store.upsert(client, (
                models.PointStruct(id=int(idx), vector=vector.tolist(), payload=_payload(row))
                for (idx, row), vector in zip(chunk.iterrows(), vectors)
            ))
            done += len(chunk)
            bar.update(len(chunk))

    elapsed = time.time() - started
    store.save_companies(company_map(frame))
    facets = dataset.build_facets(frame)
    meta = {
        "model": embedder.model_name,
        "dimension": embedder.dimension,
        "max_seq_length": embedder.max_seq_length,
        "awards": done,
        "since": since,
        "limit": limit,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(elapsed, 1),
        "facets": facets,
    }
    store.save_meta(meta)

    rate = done / elapsed if elapsed else 0
    print(f"\nIndexed {done:,} awards in {timedelta(seconds=int(elapsed))} ({rate:.0f}/s)")
    cov = facets["coverage"]
    print(f"Coverage: {cov['first_year']}-{cov['complete_through']}, "
          f"{facets['totals']['companies']:,} companies, "
          f"${facets['totals']['funding'] / 1e9:.1f}B awarded")
    if cov["partial_years"]:
        thin = ", ".join(f"{y} ({cov['by_year'][y]:,})" for y in cov["partial_years"])
        print(f"Partial years, excluded from the coverage range: {thin}")
    return meta
