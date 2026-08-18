"""Deterministic analysis of an evidence set.

Everything here is arithmetic over award records. No model is involved, which
is the point: the figures on the intelligence page are auditable by
construction, and the product keeps working with no API key and no network.

Language is the only thing a model would ever be asked for later, and it would
read these structures rather than the corpus.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from . import derive
from .evidence import EvidenceSet

# A theme needs to appear in enough of the evidence to be a pattern rather than
# a single award's vocabulary.
MIN_THEME_AWARDS = 3
MAX_THEMES = 8

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _distribution(awards: list[dict], field: str) -> list[dict]:
    """Count and funding by a categorical field, largest first."""
    counts: Counter[str] = Counter()
    funding: dict[str, float] = defaultdict(float)
    for award in awards:
        value = award.get(field) or ""
        if not value:
            continue
        counts[value] += 1
        funding[value] += float(award.get("amount") or 0.0)
    return [
        {"name": name, "awards": count, "funding": round(funding[name], 2)}
        for name, count in counts.most_common()
    ]


def _timeline(awards: list[dict], coverage: dict) -> dict:
    """Activity by year, stopping where the data stops being trustworthy.

    Trailing years that are still filling up are excluded rather than plotted,
    because a partial year draws a collapse that the awards themselves do not
    show. They are reported separately so the omission is visible.
    """
    complete_through = coverage.get("complete_through")
    counts: Counter[int] = Counter()
    funding: dict[int, float] = defaultdict(float)
    excluded = 0

    for award in awards:
        year = award.get("year")
        if not isinstance(year, int):
            continue
        if complete_through is not None and year > complete_through:
            excluded += 1
            continue
        counts[year] += 1
        funding[year] += float(award.get("amount") or 0.0)

    if not counts:
        return {"points": [], "excluded_partial": excluded, "complete_through": complete_through}

    years = range(min(counts), max(counts) + 1)
    return {
        "points": [
            {"year": y, "awards": counts.get(y, 0), "funding": round(funding.get(y, 0.0), 2)}
            for y in years
        ],
        "excluded_partial": excluded,
        "complete_through": complete_through,
    }


def topic_progression(candidates: list[dict]) -> dict[str, int]:
    """Projects advanced from Phase I to Phase II *within this technology area*.

    Corpus-wide progression counts are dominated by a handful of firms that win
    SBIR awards across every field; ranking on them puts a generalist with one
    tangentially related award above the specialist with three. Counting only
    the projects retrieved for this query answers the question the user is
    actually asking, which is who has advanced work in *this* area.

    Computed over the retrieved candidates rather than the trimmed evidence set,
    because deduplication deliberately collapses the two phases into one row.
    """
    projects: dict[tuple[str, str], set[str]] = defaultdict(set)
    for award in candidates:
        company, title = award.get("company") or "", award.get("title") or ""
        if company and title:
            projects[(company, title.strip().lower())].add(award.get("phase") or "")

    advanced: Counter[str] = Counter()
    for (company, _), phases in projects.items():
        if "Phase I" in phases and "Phase II" in phases:
            advanced[company] += 1
    return advanced


def _companies(awards: list[dict], stats: dict, advanced: dict[str, int]) -> list[dict]:
    """Companies in the evidence, enriched with what the corpus knows about them."""
    in_evidence: dict[str, dict] = {}
    for award in awards:
        name = award.get("company") or ""
        if not name:
            continue
        entry = in_evidence.setdefault(
            name, {"company": name, "awards_here": 0, "funding_here": 0.0, "titles": []}
        )
        entry["awards_here"] += 1
        entry["funding_here"] += float(award.get("amount") or 0.0)
        if award.get("title"):
            entry["titles"].append(award["title"])

    out = []
    for name, entry in in_evidence.items():
        corpus = stats.get(name, {})
        out.append({
            **entry,
            "funding_here": round(entry["funding_here"], 2),
            "total_awards": corpus.get("awards"),
            "total_funding": corpus.get("funding"),
            "phase_i": corpus.get("phase_i"),
            "phase_ii": corpus.get("phase_ii"),
            "progressed": corpus.get("progressed"),
            "topic_progressed": int(advanced.get(name, 0)),
            "first_year": corpus.get("first_year"),
            "last_year": corpus.get("last_year"),
            "state": corpus.get("state", ""),
            "website": corpus.get("website", ""),
        })

    # Presence in this evidence decides the order; corpus history only breaks
    # ties. Sorting the other way round surfaces the biggest SBIR recipients
    # rather than the companies working on the thing that was asked about.
    out.sort(key=lambda c: (c["awards_here"], c["topic_progressed"],
                            c["total_awards"] or 0), reverse=True)
    return out


def _stem(label: str) -> frozenset[str]:
    return frozenset(w.rstrip("s") for w in label.split())


def _themes(awards: list[dict], terms: dict) -> list[dict]:
    """Phrases that are common in this evidence and uncommon in the corpus.

    Weighting by inverse document frequency is what stops the list filling with
    words every SBIR abstract contains. Each theme keeps the ids of the awards
    it came from, so the UI can open it onto its own evidence.

    Selection is greedy with two rejections, because the raw ranking describes
    the same idea several times over: a phrase whose words are already covered
    ("aerial" after "unmanned aerial") adds nothing, and neither does a phrase
    drawn from substantially the same awards as one already chosen.
    """
    if not terms.get("df"):
        return []

    seen: dict[str, set[int]] = defaultdict(set)
    for award in awards:
        text = f"{award.get('title', '')} {award.get('abstract', '')[:1200]}"
        for term in derive.phrases(text):
            seen[term].add(award["id"])

    scored = []
    for term, ids in seen.items():
        if len(ids) < MIN_THEME_AWARDS:
            continue
        # Multi-word phrases are more informative at equal frequency, and this
        # keeps a bigram ahead of the unigrams it contains.
        specificity = 1.0 + 0.5 * (len(term.split()) - 1)
        scored.append({
            "label": term,
            "awards": len(ids),
            "award_ids": ids,
            "score": len(ids) * derive.term_weight(term, terms) * specificity,
        })

    scored.sort(key=lambda t: t["score"], reverse=True)

    chosen: list[dict] = []
    for theme in scored:
        stem = _stem(theme["label"])
        if any(stem <= _stem(c["label"]) or _stem(c["label"]) <= stem for c in chosen):
            continue
        if any(
            len(theme["award_ids"] & c["award_ids"])
            / len(theme["award_ids"] | c["award_ids"]) > 0.6
            for c in chosen
        ):
            continue
        chosen.append(theme)
        if len(chosen) >= MAX_THEMES:
            break

    return [
        {"label": t["label"], "awards": t["awards"], "award_ids": sorted(t["award_ids"])}
        for t in chosen
    ]


def analyse(evidence: EvidenceSet, candidates: list[dict], coverage: dict,
            company_stats: dict, corpus_terms: dict) -> dict:
    awards = evidence.awards
    funding = sum(float(a.get("amount") or 0.0) for a in awards)
    years = [a["year"] for a in awards if isinstance(a.get("year"), int)]

    agencies = _distribution(awards, "agency")
    advanced = topic_progression(candidates)
    companies = _companies(awards, company_stats, advanced)
    progressed = [c for c in companies if c["topic_progressed"] > 0]
    repeat = [c for c in companies if (c.get("total_awards") or 0) >= 3]

    payload = {
        "totals": {
            "awards": len(awards),
            "funding": round(funding, 2),
            "agencies": len(agencies),
            "companies": len(companies),
            "years": [min(years), max(years)] if years else None,
        },
        "agencies": agencies,
        "branches": _distribution(awards, "branch"),
        "phases": _distribution(awards, "phase"),
        "programs": _distribution(awards, "program"),
        "states": _distribution(awards, "state")[:10],
        "timeline": _timeline(awards, coverage),
        "companies": companies,
        "ecosystem": {
            "recurring": repeat[:8],
            "progressed": sorted(
                progressed,
                key=lambda c: (c["topic_progressed"], c["awards_here"]),
                reverse=True,
            )[:8],
        },
        "themes": _themes(awards, corpus_terms),
        "evidence": evidence.sources,
    }
    payload["reading"] = reading(payload)
    return payload


def reading(analysis: dict) -> list[str]:
    """Observations that follow from the figures.

    These sentences are templates filled with counted values. They describe
    historical funding, which is not the same as current demand — the last
    sentence of each concentrated-agency note says so, and the page renders
    this block separately from the evidence for that reason.
    """
    out: list[str] = []
    totals = analysis.get("totals") or {}
    n = totals.get("awards") or 0
    agencies = analysis.get("agencies") or []
    top = agencies[0] if agencies else None
    second = agencies[1] if len(agencies) > 1 else None

    if top and n:
        share = round((top["awards"] / n) * 100)
        line = (
            f"{top['name']} accounts for {top['awards']} of the {n} awards "
            f"examined ({share}%)"
        )
        line += (
            f", ahead of {second['name']} at {second['awards']}."
            if second else "."
        )
        line += (
            f" Historical concentration of this kind suggests {top['name']} "
            "is worth investigating as a route to market, though it does not "
            "evidence current procurement demand."
        )
        out.append(line)

    phases = {row["name"]: row["awards"] for row in analysis.get("phases") or []}
    phase_i = phases.get("Phase I", 0)
    phase_ii = phases.get("Phase II", 0)
    if phase_i + phase_ii > 0:
        rate = round((phase_ii / (phase_i + phase_ii)) * 100)
        if rate >= 40:
            out.append(
                f"Phase II awards make up {rate}% of this evidence, which "
                "points to work that has repeatedly cleared feasibility review "
                "rather than stalling at first-stage funding."
            )
        else:
            out.append(
                f"Phase I dominates at {100 - rate}% of the evidence, so much "
                "of this activity is early-stage exploration and relatively "
                "few efforts are visibly reaching Phase II here."
            )

    progressed = (analysis.get("ecosystem") or {}).get("progressed") or []
    if progressed:
        lead = progressed[0]
        out.append(
            f"{len(progressed)} of the firms here have carried work in this "
            f"area from Phase I into Phase II, led by {_title_case(lead['company'])} "
            f"with {lead['topic_progressed']}. Progression within the same "
            "technology area is the clearest maturity signal the award record "
            "offers, so these are reasonable firms to examine first."
        )

    points = [p for p in (analysis.get("timeline") or {}).get("points") or [] if p.get("awards")]
    if len(points) >= 6:
        half = len(points) // 2
        early = sum(p["awards"] for p in points[:half])
        late = sum(p["awards"] for p in points[half:])
        year_from = points[half]["year"]
        year_to = points[-1]["year"]
        if late > early * 1.3:
            out.append(
                f"Activity is weighted towards the recent half of the record "
                f"({late} awards from {year_from}-{year_to} against {early} "
                "before), which is consistent with sustained and possibly "
                "growing interest."
            )
        elif early > late * 1.3:
            out.append(
                f"Most of this activity predates {year_from} ({early} awards "
                f"before, {late} from {year_from}-{year_to}). The available "
                "evidence points to a field that was funded more heavily in "
                "the past than recently."
            )
        else:
            out.append(
                "Award activity is spread fairly evenly across the record "
                "rather than clustering in one period, indicating steady "
                "rather than episodic funding."
            )

    if n >= 30:
        out.append(
            f"Strong historical evidence: {n} closely related awards across "
            f"{totals.get('agencies', 0)} agencies and {totals.get('companies', 0)} companies."
        )
    else:
        out.append(
            f"Limited historical evidence: only {n} closely related awards "
            "were identified, so treat the reading above as tentative."
        )
    return out


def _title_case(text: str) -> str:
    return re.sub(
        r"\w\S*",
        lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(),
        text or "",
    )
