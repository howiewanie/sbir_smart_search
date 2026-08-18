# From search tool to research intelligence: architecture and roadmap

This is a planning document, not a specification of shipped behaviour. It records what the
system does today, what was measured on the live 201,204-award index, and the order in
which changes should be attempted.

Every number below was measured against the current index on a four-core CPU. Measurements
are dated because the corpus and the code will both move.

---

## 1. Where the system stands

The application is a relevance search engine. A query is embedded with
`all-MiniLM-L6-v2`, compared against the corpus by cosine similarity, re-scored with a
recency and title-keyword blend, filtered, and rendered as a list of award cards.

```
query ──► embed ──► cosine over 201k vectors ──► blend(similarity, recency, title overlap)
                                                      │
                                    structured filters │
                                                      ▼
                                              ranked award cards
```

Roughly 1,100 lines across eight modules. The layering is already close to what an
intelligence product needs: `dataset` (ingest and normalise), `embedder`, `indexer`,
`store` (Qdrant, embedded or server), `search` (planning and ranking), `api`, `cli`, and a
dependency-free front end. The seams are in sensible places.

What it cannot do is answer a question. It returns documents; the user does the analysis.

---

## 2. Measurements that should drive the design

### 2.1 There is no relevance elbow

Cosine score against rank, and the count of awards above two candidate thresholds:

| Query | rank 1 | rank 50 | rank 200 | n ≥ 0.50 | n ≥ 0.45 |
| --- | --- | --- | --- | --- | --- |
| autonomous grid inspection using drones | 0.686 | 0.512 | 0.469 | 74 | 324 |
| battery thermal management | 0.754 | 0.624 | 0.535 | 327 | 708 |
| satellite communications | 0.804 | 0.612 | 0.561 | 559 | ≥1000 |
| industrial control system cybersecurity | 0.750 | 0.543 | 0.479 | 127 | 335 |
| semiconductor advanced packaging | 0.734 | 0.574 | 0.488 | 167 | 378 |

The decay is smooth. For one query the curve passes 0.50 at rank 74 and for another at
rank 559, and that gap tracks how *generic the query wording is*, not how much the
government actually funded.

This is the central obstacle to the headline metrics. "42 relevant awards, $31M funding"
requires a definition of *relevant* that holds still across queries, and a raw bi-encoder
threshold is not one. Picking 0.5 and shipping it would produce a confident-looking number
that silently means something different on every query.

### 2.2 Half of each award is invisible to retrieval

`MAX_SEQ_LENGTH` is 128 tokens, chosen to bring a full index build from ~72 minutes down
to ~23. Against a 3,000-award sample:

- median award text: **260 tokens**, p90 **392**
- awards exceeding the window: **79.4%**
- share of award text actually embedded: **median 49%**

Of the truncated awards, the term "commercial" appears only past the cutoff in 20.2% of
cases and "phase ii" in 23.3%. SBIR abstracts are structured with the problem statement
first and commercialisation and transition potential last, so the truncation is not
removing random text — it is systematically removing the part that speaks to market
application. That is precisely the content a strategy product needs.

### 2.3 Lexical retrieval is not the missing piece

Rare technical terms, counted as how many of the dense top-K actually contain the term:

| Term | occurrences in corpus | in dense top-50 | in dense top-200 |
| --- | --- | --- | --- |
| GaN | 1,011 | 50 | 199 |
| SiC | 1,734 | 50 | 196 |
| LIDAR | 1,613 | 50 | 198 |
| hyperspectral | 991 | 48 | 198 |
| perovskite | 131 | 43 | 70 |
| photoacoustic | 116 | 45 | 68 |

The bi-encoder already resolves domain jargon. Adding BM25 to "fix keyword search" would
be solving a problem this corpus does not have. Hybrid retrieval stays on the shelf until
a failure appears that it addresses — most plausibly multi-constraint queries, which the
golden set will expose if they are real.

### 2.4 Duplication is real in the corpus but mild in the result list

41,730 company+title groups hold more than one record; **42.9% of the corpus** sits inside
such a group. That sounds alarming for evidence quality, but at the top of a result list it
is much weaker:

| Query (dense top-20) | distinct projects | distinct companies |
| --- | --- | --- |
| autonomous grid inspection using drones | 20/20 | 15/20 |
| battery thermal management | 18/20 | 13/20 |
| satellite communications | 18/20 | 16/20 |
| industrial control system cybersecurity | 17/20 | 15/20 |

Project-level duplication barely bites. Company-level repetition is the real crowding: one
firm can occupy a quarter of the visible evidence. Diversity work should therefore target
companies, not titles, and it is a refinement rather than a blocker.

### 2.5 Deterministic signals are richer than expected

- **40,541 projects** carry both a Phase I and a Phase II record, across **11,954
  companies**. Phase progression — the most credible available proxy for technical maturity
  — is a database join, not a model output.
- Award amounts are clean: **1.6% missing**, median Phase I $100k, Phase II $750k. Funding
  aggregates are computable exactly.
- Concentration: 38% of companies hold exactly one award; the top 1% hold 29% of all
  awards. "Recurring recipient" is a meaningful, checkable label.
- 20 of 41 export columns are currently unindexed. `RI Name` (research institution, 16.7%
  filled) is the STTR university partner and is a genuine ecosystem signal.

### 2.6 The current result count is not a count

`SearchEngine.search` returns a *pool*, capped at 200–2,000 candidates, and reports
`total = len(hits)`. The UI renders that as "Showing 20 of 200 matches". The 200 is the
cap, not a measurement of how many awards match. Nothing downstream should aggregate over
it until the evidence set is defined properly.

---

## 3. Proposed architecture

The existing pipeline becomes the retrieval half of a longer, inspectable chain. Each stage
is a pure function over the previous stage's output so it can be tested and replayed alone.

```
query
  │
  ▼
query understanding ......... intent + deterministic filters (years, agency, phase)
  │
  ▼
retrieval ................... dense over the corpus, structured filters
  │
  ▼
candidate ranking ........... reranking where it earns its latency
  │
  ▼
EVIDENCE SET ................ the contract: a bounded, defensible, cited set of awards
  │
  ├──────────────► deterministic aggregation ─► counts, sums, agency mix, phase mix,
  │                                              time series, company rollups
  │
  ├──────────────► external enrichment (optional, isolated, may fail)
  │
  ▼
grounded synthesis .......... themes, summary, interpretation — over the evidence set only
  │
  ▼
validation .................. every number and name re-checked against source records
  │
  ▼
intelligence page ──► (optional) 1-page brief ──► PDF / Markdown
```

The evidence set is the load-bearing idea. It is the single object the page, the
aggregates, the synthesis, the brief, and the evaluation harness all read from. Get its
definition right and the rest of the product is assembly; get it wrong and every number on
the page inherits the error.

### What the evidence set must guarantee

1. Bounded and explicit — a defined size and selection rule, never "everything above 0.5".
2. Every item traceable to a source record id.
3. Deduplicated at project level, diversified at company level.
4. Stable — the same query returns the same set, so the brief matches the page.
5. Honest in the UI about what it is. "Across the 40 most relevant awards" is defensible;
   "42 relevant awards exist" is not, unless a calibrated gate justifies it.

---

## 4. Deterministic versus generated

The split is not stylistic. It decides what can be wrong.

**Deterministic — computed from records, never generated**

Award counts, funding sums, agency and branch distribution, phase mix, program split,
awards over time, company rollups, Phase I→II progression, recurring recipients, first and
most recent award dates, research-institution partners, and every field on an evidence
card (title, company, agency, phase, amount, date, contract number).

**Generated — reading only the evidence set**

The 2–4 sentence orientation summary, theme labels over clustered evidence, the strategic
interpretation, and the brief's prose. Each claim carries the record ids it rests on.

**Never generated**

Numbers, company names, agency names, dates, amounts. If a model produces one, it is a bug,
and validation should reject the output rather than display it.

This is also the cheap path to trust: the numbers on the page are auditable by construction,
so the model is only ever responsible for language and judgement.

---

## 5. UX concept

One input. The complexity stays behind the page.

```
                What are you exploring?
      ┌──────────────────────────────────────────┐
      │ Describe a technology, company, or idea… │  [ Research → ]
      └──────────────────────────────────────────┘
```

Then a single scrolling page, ordered so it is useful before it is finished:

1. **Orientation** — title, 2–4 sentence summary, and four deterministic figures
   (awards examined, identified funding, agencies, companies).
2. **Government signals** — funding over time, awards by agency, phase mix. Charts only
   where the shape carries information the sentence cannot.
3. **Evidence** — award cards, collapsed, expandable to the abstract and full provenance.
4. **Companies and ecosystem** — recurring recipients, Phase I→II progression, university
   partners.
5. **Themes** — a handful of clusters, each opening onto its supporting awards.
6. **Interpretation** — visually separated from everything above, in calibrated language.
7. **Generate 1-Page Brief** — appears once the page is complete.

Three rules keep the machinery invisible:

- Nothing on the page names a technique. Loading copy describes work ("Searching historical
  awards…", "Analysing funding patterns…"), never components.
- Facts and interpretation never share a visual treatment. Interpretation sits on its own
  surface and reads "historical awards suggest", not "the Air Force wants this".
- Evidence strength is stated in evidence, not probability: "18 closely related awards
  across 4 programs", never "confidence 0.87".

Stream the page in the order above. Orientation and evidence can render as soon as
retrieval and aggregation finish; synthesis fills in after. Perceived latency matters more
than total latency, and the deterministic half is the fast half.

A separate `?debug=1` view can expose scores, timings, stage outputs and token costs. That
is where retrieval diagnostics live — never in the product surface.

---

## 6. Baseline

Freeze the current behaviour as **B0** before changing anything:

- dense-only retrieval, `all-MiniLM-L6-v2`, 128 tokens, cosine
- rank blend 0.90 similarity + 0.10 recency + 0.05 title overlap
- no dedup, no reranking, no synthesis
- ~350–500 ms per query warm; ~7 ms of that is embedding

B0 is the number every experiment reports against. Without it, "better" is an opinion.

---

## 7. Evaluation

Build now, because retrieval changes are about to be made and there is currently no way to
tell whether one helped.

**Golden set.** 25–30 research queries in the shape real users will type — "autonomous
infrastructure inspection", "battery thermal management for electric aircraft",
"industrial control system cybersecurity". For each, 10–20 awards graded 2 (clearly on
topic), 1 (adjacent), 0 (not). Pool candidates from several retrieval configurations before
grading so the labels are not biased toward the system that produced them.

**Retrieval metrics.** Recall@50 as the primary signal — the evidence set is drawn from the
top of the list, so what matters is whether the right awards are available to select.
nDCG@10 for ordering. MRR for the "is the first card any good" question.

**Deterministic fact validation.** Every number and name in generated output is re-derived
from source records and compared. This is a test, not a judge, and it should run in CI.

**Latency and cost.** Per stage, recorded on every request from the beginning. Retrofitting
observability after the fact is how systems become unexplainable.

**Deferred.** LLM-as-judge for synthesis quality, the human report rubric, and reranker
A/B comparisons — none are meaningful until there is synthesis to judge and a golden set to
judge it against.

---

## 8. Experiments, in order

**Experiment 1 — the embedding window.** The measured deficiency in §2.2. Compare 128
tokens against 256, and against chunking each award into overlapping passages with
max-pooling to the award. Metric: Recall@50 and nDCG@10 on the golden set. Cost: index build
time (23 min → ~72 min at 256 tokens) and, for chunking, roughly 2.5× the vectors and a
larger index. Keep only if recall moves materially; if 256 tokens captures most of the gain,
prefer it over chunking for the operational simplicity.

**Experiment 2 — a calibrated relevance gate.** The obstacle in §2.1. A cross-encoder over
the top ~200 candidates produces a query-independent relevance decision in a way raw cosine
does not, which is what makes a headline count defensible. Metric: nDCG@10, plus agreement
between the gate and human grading on where the evidence set should stop. Cost: several
hundred milliseconds on CPU — measure before assuming it is affordable, and consider running
it only for the evidence set rather than the full page.

**Experiment 3 — company-level diversity.** From §2.4: 13–16 distinct companies per 20
results. Apply MMR or a per-company cap during evidence selection. Metric: distinct
companies and distinct agencies at fixed size, with Recall@50 watched for regression.
Cheap, and it improves the ecosystem view directly.

**Experiment 4 — query understanding.** Real queries carry constraints in prose ("Navy
funding since 2020"). Extract deterministic filters before retrieval instead of letting the
embedding absorb them. Metric: Recall@50 on constraint-bearing golden queries.

**Deferred — hybrid BM25.** §2.3 shows the corpus does not need it today. Revisit only if
the golden set produces failures that lexical matching would catch.

---

## 9. Staged delivery

**Stage 1 — foundation.** Golden set, retrieval metrics harness, B0 frozen, per-stage
timing. No user-visible change.

**Stage 2 — evidence set.** Introduce the evidence-set contract with project dedup and
company diversity. Run Experiments 1 and 3.

**Stage 3 — deterministic intelligence.** Aggregation over the evidence set and the first
version of the intelligence page: orientation figures, charts, evidence cards, ecosystem,
all computed rather than generated. This is genuinely useful with no model involved, and it
de-risks the product from LLM availability.

**Stage 4 — grounded synthesis.** Summary, themes, interpretation over the evidence set,
behind deterministic validation. Requires an LLM API key.

**Stage 5 — the brief.** Structured report built from the assembled evidence, validated,
rendered to PDF. No new research pass.

**Stage 6 — external enrichment.** Recent public signals as a separate, clearly-attributed
evidence source that is allowed to fail without degrading the page.

Stage 3 is the point at which the product stops being a search engine. It is worth reaching
before any model is wired in.
