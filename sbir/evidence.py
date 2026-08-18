"""Selection of the award set that the whole intelligence page rests on.

Retrieval returns a ranked list. That list is not evidence: it repeats the same
project across phases, lets one prolific company occupy a quarter of the view,
and has no defensible end. This module turns it into a bounded, deduplicated,
company-diversified set that the page, the aggregates and the brief all read
from, so every number on the page describes the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class EvidenceSet:
    awards: list[dict]
    considered: int
    duplicates_removed: int
    capped_companies: list[str]

    def __len__(self) -> int:
        return len(self.awards)

    @property
    def sources(self) -> dict:
        """What the UI needs to describe the set honestly."""
        return {
            "size": len(self.awards),
            "considered": self.considered,
            "duplicates_removed": self.duplicates_removed,
            "companies_capped": len(self.capped_companies),
            "per_company_cap": config.EVIDENCE_PER_COMPANY,
        }


ABSTRACT_KEY_CHARS = 240


def _title_key(award: dict) -> tuple[str, str]:
    return award.get("company", ""), award.get("title", "").strip().lower()


def _abstract_key(award: dict) -> tuple[str, str] | None:
    """Second identity for a project, based on how it describes itself.

    Firms often re-file continuing work under a reworded title while reusing the
    abstract verbatim, so matching titles alone lets one project occupy several
    slots. Comparing the opening of the abstract catches those.
    """
    abstract = " ".join((award.get("abstract") or "").split()).lower()
    if len(abstract) < 80:
        return None
    return award.get("company", ""), abstract[:ABSTRACT_KEY_CHARS]


def select(candidates: list[dict], size: int = config.EVIDENCE_SIZE,
           per_company: int = config.EVIDENCE_PER_COMPANY) -> EvidenceSet:
    """Reduce ranked candidates to the evidence set.

    Two passes over the ranking, in order:

    1. Collapse repeats of the same project. A Phase I and Phase II award with
       the same title from the same company is one piece of evidence, not two;
       the better-scoring record is kept and the other phases are recorded on it.
    2. Cap how many awards any single company contributes, so the ecosystem view
       reflects a field rather than whichever firm files most often.

    The cap is relaxed rather than enforced blindly: if capping cannot fill the
    set, the best remaining awards are added back. A thin field should return
    what exists, not an artificially short page.
    """
    deduped: list[dict] = []
    by_title: dict[tuple[str, str], dict] = {}
    by_abstract: dict[tuple[str, str], dict] = {}
    duplicates = 0

    for award in candidates:
        title_key = _title_key(award)
        abstract_key = _abstract_key(award)
        existing = by_title.get(title_key)
        if existing is None and abstract_key is not None:
            existing = by_abstract.get(abstract_key)

        if existing is None:
            item = dict(award)
            item["related_awards"] = 1
            item["related_funding"] = float(award.get("amount") or 0.0)
            item["phases_seen"] = [award.get("phase")] if award.get("phase") else []
            deduped.append(item)
            by_title[title_key] = item
            if abstract_key is not None:
                by_abstract[abstract_key] = item
            continue

        duplicates += 1
        existing["related_awards"] += 1
        existing["related_funding"] += float(award.get("amount") or 0.0)
        phase = award.get("phase")
        if phase and phase not in existing["phases_seen"]:
            existing["phases_seen"].append(phase)
        # Register the alternate spelling so a third filing collapses too.
        by_title.setdefault(title_key, existing)
        if abstract_key is not None:
            by_abstract.setdefault(abstract_key, existing)

    selected: list[dict] = []
    overflow: list[dict] = []
    per_company_count: dict[str, int] = {}
    capped: set[str] = set()

    for award in deduped:
        company = award.get("company", "")
        seen = per_company_count.get(company, 0)
        if company and seen >= per_company:
            capped.add(company)
            overflow.append(award)
            continue
        per_company_count[company] = seen + 1
        selected.append(award)
        if len(selected) >= size:
            break

    if len(selected) < size:
        selected.extend(overflow[: size - len(selected)])

    return EvidenceSet(
        awards=selected[:size],
        considered=len(candidates),
        duplicates_removed=duplicates,
        capped_companies=sorted(capped),
    )
