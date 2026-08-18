"""HTTP API and static hosting for the web UI."""

from __future__ import annotations

import csv
import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .search import Filters, SearchEngine

WEB_DIR = config.PROJECT_ROOT / "web"

EXPORT_FIELDS = (
    "score", "year", "title", "company", "agency", "branch", "phase", "program",
    "amount", "city", "state", "topic_code", "contract", "pi_name", "website",
    "abstract",
)

app = FastAPI(title="SBIR Smart Search", version="2.0.0", docs_url="/api/docs")

_engine: SearchEngine | None = None
_engine_error: str | None = None


def engine() -> SearchEngine:
    """Load the model and index on first use so startup stays quick."""
    global _engine, _engine_error
    if _engine is None:
        try:
            _engine = SearchEngine()
            _engine_error = None
        except Exception as exc:
            _engine_error = str(exc)
            raise HTTPException(status_code=503, detail=_engine_error)
    return _engine


def _filters(
    agency: list[str], branch: list[str], phase: list[str], program: list[str],
    state: list[str], year_min: int | None, year_max: int | None,
    amount_min: float | None, amount_max: float | None,
    women_owned: bool, hubzone: bool, disadvantaged: bool,
) -> Filters:
    return Filters(
        agency=agency, branch=branch, phase=phase, program=program, state=state,
        year_min=year_min, year_max=year_max,
        amount_min=amount_min, amount_max=amount_max,
        women_owned=women_owned, hubzone=hubzone, disadvantaged=disadvantaged,
    )


@app.get("/api/stats")
def stats():
    try:
        return {"ready": True, **engine().stats()}
    except HTTPException:
        return {"ready": False, "detail": _engine_error}


@app.get("/api/facets")
def facets():
    return engine().facets


@app.get("/api/companies")
def companies(q: str = "", limit: int = Query(10, ge=1, le=50)):
    return {"companies": engine().companies(q, limit)}


@app.get("/api/search")
def search(
    q: str = "",
    mode: str = "auto",
    sort: str = "relevance",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    agency: list[str] = Query(default=[]),
    branch: list[str] = Query(default=[]),
    phase: list[str] = Query(default=[]),
    program: list[str] = Query(default=[]),
    state: list[str] = Query(default=[]),
    year_min: int | None = None,
    year_max: int | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    women_owned: bool = False,
    hubzone: bool = False,
    disadvantaged: bool = False,
):
    return engine().search(
        q,
        _filters(agency, branch, phase, program, state, year_min, year_max,
                 amount_min, amount_max, women_owned, hubzone, disadvantaged),
        limit=limit, offset=offset, sort=sort, mode=mode,
    )


@app.get("/api/export.csv")
def export(
    q: str = "",
    mode: str = "auto",
    sort: str = "relevance",
    limit: int = Query(500, ge=1, le=2000),
    agency: list[str] = Query(default=[]),
    branch: list[str] = Query(default=[]),
    phase: list[str] = Query(default=[]),
    program: list[str] = Query(default=[]),
    state: list[str] = Query(default=[]),
    year_min: int | None = None,
    year_max: int | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    women_owned: bool = False,
    hubzone: bool = False,
    disadvantaged: bool = False,
):
    payload = engine().search(
        q,
        _filters(agency, branch, phase, program, state, year_min, year_max,
                 amount_min, amount_max, women_owned, hubzone, disadvantaged),
        limit=limit, offset=0, sort=sort, mode=mode,
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in payload["results"]:
        writer.writerow(row)
    buffer.seek(0)

    slug = "".join(c if c.isalnum() else "_" for c in (q or "browse"))[:40] or "results"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sbir_{slug}.csv"'},
    )


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
