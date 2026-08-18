# AGENTS.md

## Cursor Cloud specific instructions

SBIR Smart Search is a **command-line** semantic search tool (no web/GUI). Two entry
points read/write a Qdrant vector database:

- `indexer.py` — embeds SBIR/STTR award CSVs and upserts vectors into Qdrant.
- `search.py` — embeds a query and runs semantic (technology) or company search.

`config.py` holds the model name (`sentence-transformers/all-MiniLM-L6-v2`), Qdrant
host/port, collection name (`sbir_awards`), and CSV column mappings.

### Environment (already provisioned by the update script)

- Use **Python 3.11** (installed via the deadsnakes PPA). The pinned deps in
  `requirements.txt` (`torch<2.2.0`) have **no Python 3.12 wheels**, so 3.11 is required.
- Dependencies live in the repo-local virtualenv at `/workspace/.venv`. Run everything
  with `./.venv/bin/python ...`.
- `requirements.txt` is under-pinned, which pulls incompatible transitive versions.
  The update script pins two compat versions on top of it (do not "upgrade" these):
  - `transformers==4.40.2` — newer transformers use a pytree API missing in `torch 2.1.2`.
  - `qdrant-client==1.11.3` — client >=1.12 removed `QdrantClient.search()`, which
    `search.py` still calls. 1.11.3 keeps `.search()` and works with the 1.19 server.
- The embedding model is cached under `~/.cache/huggingface` (first load downloads it
  from huggingface.co; egress is available).

### Qdrant vector DB (must be started per boot; NOT in the update script)

There is no Docker in this VM. The native Qdrant **v1.19.0** binary is installed at
`/opt/qdrant/qdrant`. Start it (defaults to `localhost:6333`) in a tmux terminal, using a
runtime dir for its `./storage`:

```bash
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s qdrant -c /workspace/.qdrant-runtime -- bash -lc '/opt/qdrant/qdrant'
curl -fsS http://localhost:6333/healthz   # -> "healthz check passed"
```

`indexer.py`/`search.py` fail with a connection error if Qdrant is not running.

### Running / smoke-testing

- `search.py` requires the `sbir_awards` collection to already be populated (run indexing
  first), otherwise it raises "collection not found".
- A local end-to-end smoke test lives at `dev_smoke_test.py` (gitignored, kept in the VM):
  it indexes `sample_data/sample_awards.csv` and exercises the real `SBIRSearch.search()`
  path for both a technology and a company query:
  `./.venv/bin/python dev_smoke_test.py` → prints `SMOKE TEST: PASS`.

### Known pre-existing code bugs (out of scope for environment setup — not fixed here)

- `indexer.py` has a **syntax error** near line 67 (`self.model =` is left dangling,
  followed by a stray module-level `model = SentenceTransformer(...)`), so
  `python indexer.py` will not run until fixed.
- `search.py`'s interactive `main()` calls `display_distribution`, `display_results_page`,
  and `export_results`, which are **not defined** on `SBIRSearch`; the connection and the
  core `search()` method work, but the interactive result display crashes.

### No lint config or automated test suite

The repo ships no linter config and no unit/integration tests. Validate changes by running
the smoke test above (and, once the bugs above are fixed, the `indexer.py`/`search.py` CLIs).
