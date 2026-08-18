"""The graded relevance judgements.

Grades are stored in the repository rather than in ``data/`` because they are
the slowest thing here to reproduce and the easiest thing to get wrong. They
should be reviewed and corrected by hand over time; the format is deliberately
plain so that correcting one is a one-line edit.

RUBRIC
------
2  Clearly on topic. An analyst researching this query would expect to see this
   award, and would be surprised by its absence. The award's own purpose is the
   thing asked about.

1  Adjacent. Shares a technology, platform or application with the query but is
   not about the query. Useful context, not an answer. A component supplier to
   the field, an adjacent application of the same technique, or the right
   technology aimed at a different problem.

0  Not relevant. Retrieved because of shared vocabulary rather than shared
   subject.

Only grade 2 counts towards recall. Grade 1 earns partial credit in nDCG, which
is where "reasonable but not what I asked for" belongs.
"""

from __future__ import annotations

import json
from pathlib import Path

LABELS_PATH = Path(__file__).resolve().parent / "golden" / "labels.json"
POOL_PATH = Path(__file__).resolve().parent / "golden" / "pool.json"


def load_labels() -> dict[str, dict[int, int]]:
    if not LABELS_PATH.exists():
        return {}
    raw = json.loads(LABELS_PATH.read_text())
    return {
        qid: {int(doc): int(grade) for doc, grade in judged.items()}
        for qid, judged in raw.get("labels", {}).items()
    }


def save_labels(labels: dict[str, dict[int, int]], provenance: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance,
        "labels": {
            qid: {str(doc): grade for doc, grade in sorted(judged.items())}
            for qid, judged in sorted(labels.items())
        },
    }
    LABELS_PATH.write_text(json.dumps(payload, indent=1))


def load_pool() -> dict[str, dict]:
    if not POOL_PATH.exists():
        return {}
    return json.loads(POOL_PATH.read_text())


def save_pool(pool: dict[str, dict]) -> None:
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(pool, indent=1))


def coverage(labels: dict[str, dict[int, int]]) -> dict:
    """How much of the golden set is actually usable."""
    graded = {q: len(v) for q, v in labels.items()}
    relevant = {q: sum(1 for g in v.values() if g >= 2) for q, v in labels.items()}
    adjacent = {q: sum(1 for g in v.values() if g == 1) for q, v in labels.items()}
    return {
        "queries": len(labels),
        "judgements": sum(graded.values()),
        "relevant": sum(relevant.values()),
        "adjacent": sum(adjacent.values()),
        "queries_without_relevant": [q for q, n in relevant.items() if n == 0],
        "per_query_relevant": relevant,
    }
