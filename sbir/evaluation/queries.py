"""The golden query set.

Written in the shape a user would actually type -- a technology or problem
description, not a keyword string. Chosen to spread across the agencies and
technical areas the corpus actually contains, and to include a few queries that
are deliberately awkward: acronym-led, multi-constraint, or phrased in
commercial rather than technical language.
"""

from __future__ import annotations

GOLDEN_QUERIES: list[dict] = [
    {"id": "q01", "text": "autonomous inspection of power grid infrastructure using drones",
     "note": "multi-concept: autonomy + inspection + utility asset"},
    {"id": "q02", "text": "battery thermal management for electric aircraft",
     "note": "two engineering domains combined"},
    {"id": "q03", "text": "industrial control system cybersecurity",
     "note": "SCADA/ICS, jargon-led"},
    {"id": "q04", "text": "satellite communications ground terminals",
     "note": "broad, high-volume area"},
    {"id": "q05", "text": "semiconductor advanced packaging and heterogeneous integration",
     "note": "narrow technical vocabulary"},
    {"id": "q06", "text": "hypersonic vehicle thermal protection materials",
     "note": "materials science, DoD-heavy"},
    {"id": "q07", "text": "mRNA vaccine manufacturing and delivery",
     "note": "HHS-heavy, recent-skewed"},
    {"id": "q08", "text": "portable water purification for forward operating bases",
     "note": "application phrased operationally, not technically"},
    {"id": "q09", "text": "additive manufacturing of metal aerospace components",
     "note": "process + application"},
    {"id": "q10", "text": "underwater autonomous vehicle navigation without GPS",
     "note": "negative constraint in the query"},
    {"id": "q11", "text": "quantum sensing and magnetometry",
     "note": "emerging field, sparse evidence expected"},
    {"id": "q12", "text": "machine learning for medical imaging diagnostics",
     "note": "ML applied to a domain"},
    {"id": "q13", "text": "radiation hardened electronics for space environments",
     "note": "NASA/DoD overlap"},
    {"id": "q14", "text": "soldier exoskeleton for load carriage",
     "note": "Army-specific application"},
    {"id": "q15", "text": "counter-UAS detection and mitigation",
     "note": "acronym-led query"},
    {"id": "q16", "text": "solid state lithium batteries",
     "note": "short, high-precision technical query"},
    {"id": "q17", "text": "hydrogen fuel cells for unmanned systems",
     "note": "energy + platform"},
    {"id": "q18", "text": "RF spectrum sensing and interference mitigation",
     "note": "signals domain"},
    {"id": "q19", "text": "precision agriculture crop monitoring sensors",
     "note": "USDA, low-volume agency"},
    {"id": "q20", "text": "carbon capture sorbent materials",
     "note": "DOE/EPA, materials"},
    {"id": "q21", "text": "digital twin for predictive maintenance of machinery",
     "note": "commercial phrasing"},
    {"id": "q22", "text": "free space optical communications",
     "note": "specific modality"},
    {"id": "q23", "text": "neural interface for prosthetic limb control",
     "note": "medical device + control"},
    {"id": "q24", "text": "wildfire detection and monitoring from satellites",
     "note": "cross-agency, application-led"},
]

BY_ID = {q["id"]: q for q in GOLDEN_QUERIES}


def get(query_id: str) -> dict:
    return BY_ID[query_id]
