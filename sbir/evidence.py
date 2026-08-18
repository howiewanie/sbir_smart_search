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


def _project_key(award: dict) -> tuple[str, str]:
    return award.get("company", ""), award.get("title", "").strip().lower()


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
    best: dict[tuple[str, str], dict] = {}
    duplicates = 0

    for award in candidates:
        key = _project_key(award)
        existing = best.get(key)
        if existing is None:
            item = dict(award)
            item["related_awards"] = 1
            item["related_funding"] = float(award.get("amount") or 0.0)
            item["phases_seen"] = [award.get("phase")] if award.get("phase") else []
            best[key] = item
            continue

        duplicates += 1
        existing["related_awards"] += 1
        existing["related_funding"] += float(award.get("amount") or 0.0)
        phase = award.get("phase")
        if phase and phase not in existing["phases_seen"]:
            existing["phases_seen"].append(phase)

    deduped = list(best.values())

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
