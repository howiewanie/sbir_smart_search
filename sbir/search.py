"""Query planning and ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from qdrant_client import models

from . import config, dataset, store
from .embedder import Embedder

SORT_OPTIONS = ("relevance", "newest", "oldest", "amount")
MODES = ("auto", "semantic", "company")

# Vector search is cheap; pulling extra candidates lets the re-ranker do its job
# without a second round trip.
OVERSAMPLE = 4
MIN_POOL = 200
MAX_POOL = 2000

# Embedded Qdrant evaluates filters point by point in Python, which costs about
# five times more than an unfiltered scan. Pulling a wider unfiltered pool and
# narrowing it here is dramatically cheaper, and it returns the same answer:
# every award excluded from the pool scored below everything inside it, so if
# the pool still holds enough matches they are the true top matches. When the
# filter is selective enough that it does not, we pay for the exact query.
PREFILTER_POOL = 3000


@dataclass
class Filters:
    agency: list[str] = field(default_factory=list)
    branch: list[str] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    program: list[str] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    amount_min: float | None = None
    amount_max: float | None = None
    women_owned: bool = False
    hubzone: bool = False
    disadvantaged: bool = False

    def matches(self, payload: dict) -> bool:
        """Same predicate as :meth:`to_qdrant`, applied in Python.

        Used on the company path, where candidates arrive by id lookup rather
        than through a filtered vector query.
        """
        for name in ("agency", "branch", "phase", "program", "state"):
            values = [v for v in getattr(self, name) if v]
            if values and payload.get(name) not in values:
                return False

        year, amount = payload.get("year", 0), payload.get("amount", 0.0)
        if self.year_min is not None and year < self.year_min:
            return False
        if self.year_max is not None and year > self.year_max:
            return False
        if self.amount_min is not None and amount < self.amount_min:
            return False
        if self.amount_max is not None and amount > self.amount_max:
            return False

        for flag in ("women_owned", "hubzone", "disadvantaged"):
            if getattr(self, flag) and not payload.get(flag):
                return False
        return True

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []

        for name in ("agency", "branch", "phase", "program", "state"):
            values = [v for v in getattr(self, name) if v]
            if values:
                must.append(models.FieldCondition(
                    key=name, match=models.MatchAny(any=values)
                ))

        if self.year_min is not None or self.year_max is not None:
            must.append(models.FieldCondition(
                key="year", range=models.Range(gte=self.year_min, lte=self.year_max)
            ))
        if self.amount_min is not None or self.amount_max is not None:
            must.append(models.FieldCondition(
                key="amount", range=models.Range(gte=self.amount_min, lte=self.amount_max)
            ))

        for flag in ("women_owned", "hubzone", "disadvantaged"):
            if getattr(self, flag):
                must.append(models.FieldCondition(
                    key=flag, match=models.MatchValue(value=True)
                ))

        return models.Filter(must=must) if must else None


def _recency(year: int, current_year: int) -> float:
    age = max(0, current_year - year)
    return max(0.0, 1.0 - age / config.RECENCY_HALF_LIFE_YEARS)


class SearchEngine:
    """Loads the model and index once, then answers queries."""

    def __init__(self):
        meta = store.load_meta()
        if meta is None:
            raise RuntimeError(
                "No index found. Run `python -m sbir index` to build one."
            )
        self.meta = meta
        self.embedder = Embedder(
            model_name=meta.get("model", config.MODEL_NAME),
            max_seq_length=meta.get("max_seq_length", config.MAX_SEQ_LENGTH),
        )
        self.client = store.connect()
        self.total = store.count(self.client)
        if self.total == 0:
            raise RuntimeError(
                "The index is empty. Run `python -m sbir index` to build one."
            )
        self.company_ids = store.load_companies()
        self.company_names = [(name.lower(), name) for name in self.company_ids]

    @property
    def facets(self) -> dict:
        return self.meta.get("facets", {})

    def stats(self) -> dict:
        return {
            "awards": self.total,
            "model": self.embedder.model_name,
            "device": self.embedder.device,
            "backend": store.describe_backend(),
            "built_at": self.meta.get("built_at"),
            "years": self.facets.get("years"),
            "totals": self.facets.get("totals", {}),
        }

    def search(self, query: str, filters: Filters | None = None, limit: int = 20,
               offset: int = 0, sort: str = "relevance", mode: str = "auto") -> dict:
        started = time.perf_counter()
        filters = filters or Filters()
        query = (query or "").strip()
        sort = sort if sort in SORT_OPTIONS else "relevance"
        mode = mode if mode in MODES else "auto"

        if mode == "company" and query:
            hits = self._company_hits(query, filters)
            if sort == "relevance":
                sort = "newest"
            return self._page(hits, query, mode, sort, limit, offset, started, False)

        qfilter = filters.to_qdrant()
        wanted = offset + limit

        if query:
            pool_size = min(MAX_POOL, max(MIN_POOL, wanted * OVERSAMPLE))
            hits = self._semantic_hits(query, filters, qfilter, pool_size, wanted)
            self._rank(hits, query)
        else:
            # No query text: this is a browse, so fetch a slice and order by
            # whatever the caller asked for.
            pool_size = min(MAX_POOL, max(MIN_POOL, wanted * 2))
            records = store.scroll(self.client, pool_size, qfilter)
            hits = [self._hit(r.payload, None) for r in records]
            if sort == "relevance":
                sort = "newest"

        return self._page(hits, query, mode, sort, limit, offset, started,
                          len(hits) >= MAX_POOL)

    def _page(self, hits: list[dict], query: str, mode: str, sort: str,
              limit: int, offset: int, started: float, truncated: bool) -> dict:
        if sort == "newest":
            hits.sort(key=lambda h: (h["year"], h["amount"]), reverse=True)
        elif sort == "oldest":
            hits.sort(key=lambda h: (h["year"], -h["amount"]))
        elif sort == "amount":
            hits.sort(key=lambda h: h["amount"], reverse=True)

        return {
            "query": query,
            "mode": mode,
            "sort": sort,
            "total": len(hits),
            "truncated": truncated,
            "offset": offset,
            "limit": limit,
            "took_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": hits[offset:offset + limit],
        }

    def _semantic_hits(self, query: str, filters: Filters,
                       qfilter: models.Filter | None, pool_size: int,
                       wanted: int) -> list[dict]:
        vector = self.embedder.encode_one(query)

        if qfilter is None:
            points = store.query(self.client, vector, pool_size)
            return [self._hit(p.payload, p.score) for p in points]

        probe = store.query(self.client, vector, PREFILTER_POOL)
        hits = [
            self._hit(p.payload, p.score)
            for p in probe if filters.matches(p.payload)
        ]
        if len(hits) >= max(wanted, 1):
            return hits[:MAX_POOL]

        points = store.query(self.client, vector, pool_size, qfilter)
        return [self._hit(p.payload, p.score) for p in points]

    def resolve_companies(self, query: str, limit: int = 25) -> list[str]:
        """Company names matching every word of the query, best (shortest) first.

        Matching on words rather than a raw substring means "luna innovations"
        still finds "LUNA INNOVATIONS INCORPORATED".
        """
        words = dataset.tokenize(query)
        if not words:
            return []
        matched = [
            original for lowered, original in self.company_names
            if all(word in lowered for word in words)
        ]
        matched.sort(key=lambda name: (len(name), name))
        return matched[:limit]

    def _company_hits(self, query: str, filters: Filters) -> list[dict]:
        ids: list[int] = []
        for name in self.resolve_companies(query):
            ids.extend(self.company_ids.get(name, ()))
        records = store.retrieve(self.client, ids[:MAX_POOL])
        return [
            self._hit(r.payload, None)
            for r in records if filters.matches(r.payload)
        ]

    def _rank(self, hits: list[dict], query: str) -> None:
        """Blend similarity, recency and a light keyword bonus, then sort."""
        current_year = datetime.now().year
        terms = set(dataset.tokenize(query))
        weight = config.RECENCY_WEIGHT

        for hit in hits:
            similarity = hit["similarity"] or 0.0
            recency = _recency(hit["year"], current_year)
            bonus = 0.0
            if terms:
                title_terms = set(dataset.tokenize(hit["title"]))
                overlap = len(terms & title_terms) / len(terms)
                # A title that literally contains the query words is usually
                # what the user meant, but it should not outrank meaning.
                bonus = 0.05 * overlap
            hit["score"] = round((1 - weight) * similarity + weight * recency + bonus, 6)

        hits.sort(key=lambda h: h["score"], reverse=True)

    @staticmethod
    def _hit(payload: dict, similarity: float | None) -> dict:
        hit = dict(payload)
        hit["similarity"] = round(similarity, 6) if similarity is not None else None
        hit["score"] = hit["similarity"]
        return hit

    def companies(self, prefix: str, limit: int = 10) -> list[str]:
        """Company name suggestions for the search box, most prolific first."""
        if len(prefix.strip()) < 2:
            return []
        names = self.resolve_companies(prefix, limit=200)
        names.sort(key=lambda name: -len(self.company_ids.get(name, ())))
        return names[:limit]
