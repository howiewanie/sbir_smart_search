# AGENTS.md

## Cursor Cloud specific instructions

SBIR Smart Search is a semantic search engine over the public SBIR/STTR award database.
It is a single Python service: a FastAPI app (`sbir/api.py`) that serves both a JSON API
and the static front end in `web/`. There is no build step and no JavaScript toolchain.

Commands, configuration and architecture are documented in `README.md` — read it first
rather than rediscovering the CLI. The notes below only cover things that are specific to
this VM or that are easy to get wrong.

### Environment

- The virtualenv is at `/workspace/.venv`, built with **Python 3.11**, and the update
  script keeps it current. Run everything through `./.venv/bin/python`.
- 3.11 is deliberate. The current `requirements.txt` works on 3.10+, but the pre-rewrite
  version pinned `torch<2.2`, which has no 3.12 wheels. Staying on 3.11 keeps the update
  script working on any branch. Do not "upgrade" the venv to 3.12 without checking which
  `requirements.txt` the branch actually has.

### Data and index live in `data/` and are NOT in git

`data/` is gitignored and already populated in this VM:

- `data/award_data.csv` — the 350 MB SBIR.gov export
- `data/qdrant/` — the embedded vector index, ~1.2 GB
- `data/index_meta.json`, `data/companies.json` — index metadata and the company lookup

The full index is already built (201,204 awards, 1983-2024). **Rebuilding takes about
23 minutes**, so do not run `python -m sbir index` casually. Use `--limit` or `--since`
when you only need something to test against.

If you change how documents are embedded or what goes into the payload, the existing
index becomes stale and you do have to rebuild. `data/companies.json` maps company name
to point ids and must stay consistent with the index; it is written by `sbir.indexer` at
build time, and it can be regenerated on its own (point ids are deterministic) with:

```python
from sbir import dataset, indexer, store
store.save_companies(indexer.company_map(dataset.load()))
```

### Embedded Qdrant takes an exclusive lock

Only one process may open `data/qdrant` at a time. If `serve` is running, a second
command that touches the index (`search`, `index`, `status` on the collection) fails with
"already accessed by another instance of Qdrant client". Stop the server first. Long-lived
services here are run under tmux, so kill the `sbir-web` session before doing index work
and restart it afterwards.

### Performance characteristics worth preserving

Embedded Qdrant evaluates filters point by point in Python, which measured roughly five
times the cost of an unfiltered scan (2.5 s versus 0.45 s). `SearchEngine._semantic_hits`
therefore pulls a wide unfiltered pool and filters in Python, only falling back to a
filtered vector query when the pool runs dry. This is exact, not an approximation, and it
was verified against ground-truth filtered queries. Keep the fallback if you touch it.

Embedding uses `max_seq_length=128` rather than the model's 256 default; that is what
takes a full index build from ~72 minutes down to ~23.

### Testing

Validate changes by running the app and exercising the API, including
`GET /api/research`, `GET /api/research.pdf`, and `GET /api/research.docx`.
Those last two are the same deterministic payload laid out as files; they do
not call a language model. `./.venv/bin/ruff check sbir/` is available for a
quick lint (ruff is installed in the venv but is not a project dependency).
