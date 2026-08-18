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
application.

> **Superseded by Experiment 1 (§8).** The inference drawn here — that the discarded tail
> therefore carries retrievable signal — was tested and is wrong. Widening the window to
> 256 tokens made retrieval *worse* on every metric. The tail is largely boilerplate that
> every SBIR abstract shares, so including it makes awards look more alike rather than
> more distinguishable. The truncation remains a real limit on what can be *displayed*;
> it is not a retrieval deficiency.

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
cap, not a measurement of how many awards match.

Re-measured 2026-08-18, the behaviour is worse than "a cap": the reported total moves with
the page size, because the pool is sized from `offset + limit`.

| requested `limit` | reported `total` |
| --- | --- |
| 5 | 200 |
| 20 | 200 |
| 50 | 200 |
| 100 | 400 |

The same query "matches" 200 or 400 awards depending on how many results the caller asked
to see. Nothing downstream should aggregate over it until the evidence set is defined.

### 2.7 The corpus stops roughly three years short of today

This is the finding with the largest product consequences, and it was missed on the first
pass.

| Award year | Awards |
| --- | --- |
| 2021 | 6,878 |
| 2022 | 6,646 |
| 2023 | 6,283 |
| 2024 | **23** |
| 2025 | **0** |

The latest parseable `Proposal Award Date` is 2024-10-01, and only 27 records fall in the
final year of the corpus. Against a mid-2026 wall clock the usable corpus ends in **2023**.

Three consequences:

1. **"What has changed recently" cannot be answered from SBIR data at all.** External
   enrichment is not the optional garnish it was filed as in §9 — it is the only possible
   source for that section. Until it exists, the section should be absent rather than
   filled with three-year-old awards.
2. **A funding-over-time chart is actively misleading.** Plotted naively, every query shows
   a collapse to near zero in 2024. That is the export's cutoff, not a change in government
   behaviour, and it is exactly the kind of confident-looking artefact a strategy user would
   act on. Either end the axis at the last complete year and label it, or omit the chart.
3. **The recency term in ranking is mistuned.** `0.10 × recency` currently treats a 2023
   award as new. The decay is defined against `datetime.now()`, so the whole corpus drifts
   further into the past every day the export is not refreshed.

### 2.8 Company names need no entity-resolution layer

Expected to be a problem; measured, and it is not. Of 32,402 distinct company strings,
aggressive suffix normalisation (stripping Inc/LLC/Corporation/Technologies) collapses only
**305**, or 0.9%.

Worse, spot-checking the collapses shows the normaliser causing damage rather than fixing
it: it merges `ENGINEERING TECHNOLOGIES LLC` with `Engineering Inc`, and
`RESEARCH APPLICATIONS, INC.` with `Research Applications Corporation` — plausibly
different firms. The naive fix is net negative.

Company rollups should therefore key on the **exact** company string. Fuzzy entity
resolution is a technique this corpus does not need, and skipping it removes a whole class
of silent aggregation errors.

### 2.9 Field coverage bounds what can be claimed

| Field | Filled |
| --- | --- |
| Agency, Phase, Program, State, City, Award Amount | 100% |
| Abstract | 86.4% |
| Contract | 74.9% |
| Branch | 65.2% |
| Company Website | 53.7% |
| Topic Code | 45.7% |
| RI Name (STTR research partner) | 16.7% |

Agency, phase, program, state and funding aggregates are safe to state flatly. Branch-level
claims cover roughly two thirds of awards and need saying so. `Topic Code` at 45.7% is too
sparse to be the primary key for phase-progression joins — company+title reaches 40,541
linked projects where company+topic reaches 29,856. `RI Name` is a real STTR ecosystem
signal but only for the sixth of the corpus that carries it.

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
2. **Government signals** — awards by agency, phase mix, and activity over time. Charts only
   where the shape carries information the sentence cannot, and the time axis must stop at
   the last complete year in the corpus (§2.7) or it will draw a funding collapse that does
   not exist.
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

### B0 measured, 2026-08-18

24 golden queries, 1,671 graded judgements, 743 awards graded relevant.

| Metric | B0 |
| --- | --- |
| Recall@10 | 0.312 |
| Recall@20 | 0.540 |
| **Recall@50** | **0.761** |
| nDCG@10 | 0.909 |
| MRR | 1.000 |
| Precision@10 | 0.854 |
| Latency mean / max | 415 / 531 ms |

**This reorders the experiment plan.** MRR is 1.000 — the first result is relevant on
every one of the 24 queries — and precision@10 is 0.854. The top of the ranking is
already close to saturated, so a reranker has almost nothing to reorder. Experiment 2 as
originally written aimed at exactly that, and the measurement says it is the wrong target.

The headroom is entirely in coverage: a quarter of the known-relevant awards never appear
in the top 50. Worst cases are carbon capture sorbents (0.39), quantum sensing and
magnetometry (0.55) and satellite ground terminals (0.58). Carbon capture is a genuine
failure rather than an artefact — 36 awards are graded relevant and only 14 are retrieved
in 50 slots.

One caveat on the metric: recall@50 is bounded above where a query has more than 50
relevant awards. Additive manufacturing (53 relevant) can reach at most 0.94. That affects
the three largest queries and none of the weak ones.

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

**Experiment 1 — the embedding window.** The measured deficiency in §2.2.

Run it on a **stratified 50k-award sample rather than the full corpus**. A full re-index at
256 tokens costs ~72 minutes per configuration, and the hypothesis — that truncation hides
retrievable content — is testable at a fraction of that. Sampling to 50k puts a build at
roughly 6 minutes at 128 tokens and 15 at 256, so three configurations fit in under an
hour and the experiment can be rerun after every change instead of once.

The sample must contain the graded awards for every golden query, plus a random remainder,
so recall is measured against a realistic amount of distractor material.

| Configuration | What it tests |
| --- | --- |
| 128 tokens (B0) | current behaviour |
| 256 tokens | does the truncated half carry retrievable signal |
| overlapping chunks, max-pooled to the award | does passage granularity beat a longer window |

Metric: Recall@50 primary, nDCG@10 secondary. Promote to a full-corpus build only if the
sample shows a material gain. If 256 tokens captures most of it, prefer that over chunking
— chunking roughly 2.5×'s the vector count and complicates every downstream id join.

#### Result, 2026-08-18: hypothesis rejected, keep 128

50,000-award stratified sample containing all 1,651 graded awards.

| Window | Recall@10 | Recall@50 | nDCG@10 | P@10 | MRR | Build | Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.2823 | 0.7439 | 0.8594 | 0.7875 | 0.9792 | 2.5 min | 339/s |
| 96 | 0.2974 | 0.7870 | 0.8745 | 0.8083 | 0.9792 | 3.3 min | 255/s |
| **128** | **0.3122** | **0.8116** | **0.9085** | **0.8542** | **1.0000** | 4.2 min | 196/s |
| 256 | 0.2900 | 0.7632 | 0.8729 | 0.8000 | 0.9514 | 10.4 min | 80/s |

128 tokens is the best setting on every metric, and the curve is a clean inverted U rather
than a plateau. Widening to 256 costs 2.5× the build time and loses 0.048 recall@50;
narrowing to 96 loses 0.025 and to 64 loses 0.068.

The explanation is that `all-MiniLM-L6-v2` mean-pools over the window. Past roughly 128
tokens an SBIR abstract turns into commercialisation and transition boilerplate that every
award shares, so the extra tokens pull each vector towards the corpus mean and make awards
harder to tell apart. More text is not more signal when the additional text is common to
everything.

The chunking variant was not run. It was worth trying only if a longer window had helped,
and it did not.

**Consequence for the roadmap.** Two of the four planned experiments are now closed by
measurement: reranking, because the ranking is already saturated (§6), and the embedding
window, because the current setting is optimal. The remaining recall gap is not explained
by either. The live candidates are company-level diversity (§8.3) and query understanding
(§8.4), and the most informative unexplored option is a different embedding model, since
the model — not the window — is now the binding constraint.

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

**Stage 4 — grounded LLM synthesis.** Cancelled. The page already states what the
figures show using templates filled with counted values. An LLM key is not required.

**Stage 5 — the takeaway file.** Done. The same deterministic payload is laid out as a
downloadable PDF and a Word document (`GET /api/research.pdf`, `GET /api/research.docx`),
covering every section of the on-screen report. Print still produces a compact brief of
the page. No second research pass, no model.

**Stage 6 — external enrichment.** Cancelled for the same reason as Stage 4: it needs a
search/news API key, and the product is the historical award record. Recent public
signals stay off the page rather than being approximated from old awards.

Stages 1–3 plus file export are the shipped product. They run with no API key.

### Credentials are not required

Checked 2026-08-18: no LLM key and no web-search key is present, and none is needed.
The award data itself comes from the official SBIR.gov **CSV export**, not from a live
SBIR API — that API has not been reliable. If it becomes usable, fetch can be pointed
at it later; until then `python -m sbir setup` downloads the bulk file.
