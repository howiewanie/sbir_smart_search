"""Artifacts derived from the CSV rather than from the vectors.

These are pure functions of the award table, so they can be rebuilt in a couple
of minutes without re-embedding anything. Keeping them separate from the index
means a bug in an aggregate never costs a 23-minute rebuild.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import pandas as pd

from . import dataset, store

# Terms rarer than this are noise (typos, one-off part numbers); terms in more
# than this share of the corpus are boilerplate ("phase", "commercial").
MIN_DOC_FREQ = 25
MAX_DOC_RATIO = 0.12
MAX_VOCAB = 80_000

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")

# Words that survive an IDF filter but say nothing about a technology.
STOPWORDS = frozenset("""
the and for with that this from will are was were has have had can could would
its their there these those such other than then they them our out into within
been being over under more most less least new novel high low large small using
use used uses very much many both each also may might must shall should any all
per via non pre post based approach approaches method methods system systems
technology technologies development develop developed developing research
program programs project projects phase proposed propose proposal effort work
data results result provide provides provided demonstrate demonstrated
demonstration objective objectives goal goals capability capabilities design
designed performance improve improved improvement increase increased reduce
reduced reduction cost costs time current currently potential significant
critical key important required require requirements need needs application
applications innovation innovative commercial commercialization military
government agency benefits addition addressed address final feasibility
successful success test testing tests prototype team company inc llc
which who whom whose what when where why how while because although however
therefore thus hence about above below between through during before after
sbir sttr proposes proposing proposal phase-i phase-ii topic offeror
develops developing demonstrates enables enabling allows allowing provides
including include includes well being able ability make makes made
first second third one two three four five several various different
state-of-the-art art state prior existing conventional traditional
not nor none does did doing done process processes order orders way ways
given upon toward towards across along among against without between
level levels type types form forms part parts area areas field fields
range ranges rate rates size sizes number numbers amount amounts value values
""".split())


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


def company_stats(frame: pd.DataFrame) -> dict[str, dict]:
    """Corpus-wide facts about each company.

    Phase progression is the most credible proxy for technical maturity in this
    dataset and it is a join, not a judgement: a project is counted as having
    progressed when the same company holds both a Phase I and a Phase II award
    under the same title. Company names are used exactly as they appear --
    normalising them merges genuinely different firms (measured: 0.9% collapse,
    with false merges among them).
    """
    projects: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for company, title, phase in zip(frame["company"], frame["title"], frame["phase"]):
        if company and title:
            projects[company][title.lower()].add(phase)

    stats: dict[str, dict] = {}
    grouped = frame.groupby("company", sort=False)
    for company, rows in grouped:
        if not company:
            continue
        phases = rows["phase"].value_counts()
        titles = projects.get(company, {})
        progressed = sum(
            1 for phase_set in titles.values()
            if "Phase I" in phase_set and "Phase II" in phase_set
        )
        stats[company] = {
            "awards": int(len(rows)),
            "funding": float(rows["amount"].sum()),
            "phase_i": int(phases.get("Phase I", 0)),
            "phase_ii": int(phases.get("Phase II", 0)),
            "progressed": progressed,
            "first_year": int(rows["year"].min()),
            "last_year": int(rows["year"].max()),
            "agencies": sorted({a for a in rows["agency"] if a}),
            "state": next((s for s in rows["state"] if s), ""),
            "website": next((w for w in rows["website"] if w), ""),
        }
    return stats


def phrases(text: str) -> set[str]:
    """Unigrams and adjacent bigrams, stopwords removed.

    Bigrams matter because a one-word theme is rarely a theme: "aerial" is a
    fragment, "unmanned aerial" is a subject. Bigrams are formed only from
    adjacent surviving words, so removing a stopword joins its neighbours
    rather than inventing a phrase across a clause boundary.
    """
    words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
    out = set(words)
    out.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


def corpus_terms(frame: pd.DataFrame) -> dict:
    """Document frequency for the corpus vocabulary.

    Themes are "what is unusual about this evidence", which needs a background
    distribution to compare against. Storing document frequencies lets that
    comparison happen at query time without a model.
    """
    counts: Counter[str] = Counter()
    total = 0
    for title, abstract in zip(frame["title"], frame["abstract"]):
        counts.update(phrases(f"{title} {abstract[:1200]}"))
        total += 1

    kept = {
        term: freq for term, freq in counts.items()
        if MIN_DOC_FREQ <= freq <= total * MAX_DOC_RATIO
    }
    if len(kept) > MAX_VOCAB:
        kept = dict(Counter(kept).most_common(MAX_VOCAB))
    return {"documents": total, "df": kept}


def term_weight(term: str, terms: dict) -> float:
    """Inverse document frequency, with unknown terms treated as rare."""
    total = terms.get("documents", 1) or 1
    df = terms.get("df", {}).get(term, 1)
    return math.log(total / max(df, 1))


def build_all(frame: pd.DataFrame | None = None) -> dict[str, int]:
    frame = dataset.load() if frame is None else frame

    companies = company_map(frame)
    store.save_companies(companies)

    stats = company_stats(frame)
    store.save_company_stats(stats)

    terms = corpus_terms(frame)
    store.save_corpus_terms(terms)

    return {
        "companies": len(companies),
        "company_stats": len(stats),
        "vocabulary": len(terms["df"]),
    }
