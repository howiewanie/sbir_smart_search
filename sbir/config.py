"""Runtime settings.

Every value can be overridden with an environment variable of the same name
prefixed with ``SBIR_`` -- e.g. ``SBIR_MODEL_NAME=BAAI/bge-small-en-v1.5``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Official bulk export. Roughly 350 MB. This is a downloaded CSV, not a live
# SBIR.gov API — that API has not been reliable. Re-point this if a working
# API appears; until then setup/fetch pull this file.
AWARD_DATA_URL = os.environ.get(
    "SBIR_AWARD_DATA_URL",
    "https://data.www.sbir.gov/awarddatapublic/award_data.csv",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SBIR_DATA_DIR", PROJECT_ROOT / "data"))

CSV_PATH = DATA_DIR / "award_data.csv"
QDRANT_PATH = DATA_DIR / "qdrant"
INDEX_META_PATH = DATA_DIR / "index_meta.json"
COMPANIES_PATH = DATA_DIR / "companies.json"
COMPANY_STATS_PATH = DATA_DIR / "company_stats.json"
CORPUS_TERMS_PATH = DATA_DIR / "corpus_terms.json"

# Evidence set shape. Small enough to read, wide enough to aggregate over.
EVIDENCE_SIZE = int(os.environ.get("SBIR_EVIDENCE_SIZE", 40))
EVIDENCE_CANDIDATES = int(os.environ.get("SBIR_EVIDENCE_CANDIDATES", 400))
EVIDENCE_PER_COMPANY = int(os.environ.get("SBIR_EVIDENCE_PER_COMPANY", 3))

# Set this to use a standalone Qdrant server instead of the embedded store,
# e.g. SBIR_QDRANT_URL=http://localhost:6333
QDRANT_URL = os.environ.get("SBIR_QDRANT_URL") or None
QDRANT_API_KEY = os.environ.get("SBIR_QDRANT_API_KEY") or None

COLLECTION_NAME = os.environ.get("SBIR_COLLECTION_NAME", "sbir_awards")

# all-MiniLM-L6-v2 is small, fast on CPU and good enough for this corpus.
# Swap in any sentence-transformers model; the index records which one was
# used so a mismatch is caught instead of silently returning nonsense.
MODEL_NAME = os.environ.get("SBIR_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Award text is front-loaded: the title plus the opening of the abstract says
# almost everything. Truncating at 128 tokens is ~4x faster than the model's
# 256-token default and barely moves result quality.
MAX_SEQ_LENGTH = int(os.environ.get("SBIR_MAX_SEQ_LENGTH", 128))
BATCH_SIZE = int(os.environ.get("SBIR_BATCH_SIZE", 64))

# How much a recent award is favoured over an older one, 0.0 - 1.0.
RECENCY_WEIGHT = float(os.environ.get("SBIR_RECENCY_WEIGHT", 0.10))

# Awards stop losing recency credit after this many years.
RECENCY_HALF_LIFE_YEARS = float(os.environ.get("SBIR_RECENCY_HALF_LIFE", 20))

HOST = os.environ.get("SBIR_HOST", "127.0.0.1")
PORT = int(os.environ.get("SBIR_PORT", 8000))
