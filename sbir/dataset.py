"""Fetching and cleaning the SBIR.gov award export."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from . import config

# SBIR.gov ships human-readable headers. Map them onto the field names used
# everywhere else in the project. Anything not listed here is dropped.
COLUMNS = {
    "Company": "company",
    "Award Title": "title",
    "Agency": "agency",
    "Branch": "branch",
    "Phase": "phase",
    "Program": "program",
    "Contract": "contract",
    "Topic Code": "topic_code",
    "Award Year": "year",
    "Award Amount": "amount",
    "Proposal Award Date": "award_date",
    "Abstract": "abstract",
    "City": "city",
    "State": "state",
    "Company Website": "website",
    "Number Employees": "employees",
    "HUBZone Owned": "hubzone",
    "Women Owned": "women_owned",
    "Socially and Economically Disadvantaged": "disadvantaged",
    "PI Name": "pi_name",
    "Contact Email": "contact_email",
}

TEXT_FIELDS = (
    "company", "title", "agency", "branch", "phase", "program", "contract",
    "topic_code", "abstract", "city", "state", "website", "pi_name",
    "contact_email", "award_date",
)
FLAG_FIELDS = ("hubzone", "women_owned", "disadvantaged")

FACET_FIELDS = ("agency", "branch", "phase", "program", "state")


def download(url: str = config.AWARD_DATA_URL, dest: Path = config.CSV_PATH,
             force: bool = False) -> Path:
    """Download the award export, skipping the transfer if we already have it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"Using cached {dest} ({size_mb:.0f} MB). Pass --force to re-download.")
        return dest

    print(f"Downloading {url}")
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _to_amount(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _to_flag(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().eq("Y").fillna(False)


def load(path: Path = config.CSV_PATH, since: int | None = None,
         limit: int | None = None) -> pd.DataFrame:
    """Read the export into a tidy frame with predictable dtypes.

    Rows without a title *and* an abstract are dropped -- there is nothing to
    embed and they would only pollute results.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m sbir fetch` first."
        )

    present = {c: COLUMNS[c] for c in pd.read_csv(path, nrows=0).columns if c in COLUMNS}
    missing = set(COLUMNS.values()) - set(present.values())
    frame = pd.read_csv(path, usecols=list(present), low_memory=False).rename(columns=present)
    for field in missing:
        frame[field] = pd.NA

    for field in TEXT_FIELDS:
        frame[field] = (
            frame[field].astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .fillna("")
        )
    for field in FLAG_FIELDS:
        frame[field] = _to_flag(frame[field])

    frame["amount"] = _to_amount(frame["amount"])
    frame["employees"] = pd.to_numeric(frame["employees"], errors="coerce").fillna(0).astype(int)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")

    frame = frame[(frame["title"] != "") | (frame["abstract"] != "")]
    frame = frame[frame["year"].notna()]
    frame["year"] = frame["year"].astype(int)
    frame = frame[frame["year"].between(1980, 2100)]

    if since is not None:
        frame = frame[frame["year"] >= since]
    frame = frame.sort_values("year", ascending=False, kind="stable")
    if limit is not None:
        frame = frame.head(limit)

    return frame.reset_index(drop=True)


def embedding_text(row: pd.Series) -> str:
    """The string that actually gets embedded for a given award."""
    parts = [row["title"], row["abstract"]]
    return ". ".join(p for p in parts if p)[:2000]


# A trailing year holding less than this share of recent typical volume is
# treated as still filling up rather than as a real decline in funding.
COMPLETE_YEAR_RATIO = 0.4
RECENT_WINDOW_YEARS = 10


def coverage(frame: pd.DataFrame) -> dict:
    """Work out how far the data actually runs.

    The export trails real-world activity, and by an amount that changes every
    time it is refreshed. Nothing should hardcode a cutoff year: the last
    trustworthy year is derived here and everything downstream reads it.

    ``complete_through`` is the most recent year whose award count looks like a
    normal year. Years after it are reported separately as partial, because
    plotting them as-is draws a funding collapse that is an artefact of the
    export rather than anything the government did.
    """
    by_year = frame["year"].value_counts().sort_index()
    counts = {int(y): int(n) for y, n in by_year.items()}
    if not counts:
        return {"first_year": None, "last_year": None, "complete_through": None,
                "partial_years": [], "by_year": {}}

    years = sorted(counts)
    first_year, last_year = years[0], years[-1]

    window = [counts[y] for y in years if y > last_year - RECENT_WINDOW_YEARS]
    reference = float(pd.Series(window).median()) if window else 0.0

    complete_through = last_year
    if len(window) >= 3 and reference > 0:
        threshold = reference * COMPLETE_YEAR_RATIO
        for year in reversed(years):
            if counts[year] >= threshold:
                complete_through = year
                break

    return {
        "first_year": first_year,
        "last_year": last_year,
        "complete_through": complete_through,
        "partial_years": [y for y in years if y > complete_through],
        "by_year": counts,
    }


def build_facets(frame: pd.DataFrame) -> dict:
    """Distinct filter values plus a few headline numbers for the UI."""
    facets = {
        field: sorted(v for v in frame[field].unique() if v)
        for field in FACET_FIELDS
    }
    facets["coverage"] = coverage(frame)
    facets["years"] = [facets["coverage"]["first_year"], facets["coverage"]["last_year"]]
    facets["totals"] = {
        "awards": int(len(frame)),
        "companies": int(frame["company"].nunique()),
        "funding": float(frame["amount"].sum()),
    }
    return facets


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
