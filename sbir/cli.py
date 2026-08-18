"""Command line entry point: fetch data, build the index, search, serve the UI."""

from __future__ import annotations

import argparse
import textwrap

from . import config, dataset, indexer, store


def _money(value: float) -> str:
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def cmd_fetch(args) -> None:
    dataset.download(force=args.force)


def cmd_index(args) -> None:
    if not config.CSV_PATH.exists():
        dataset.download()
    indexer.build(since=args.since, limit=args.limit, batch_size=args.batch_size)


def cmd_setup(args) -> None:
    dataset.download(force=args.force)
    indexer.build(since=args.since, limit=args.limit)
    print("\nReady. Start the web UI with:  python -m sbir serve")


def cmd_status(args) -> None:
    meta = store.load_meta()
    print(f"Data file   {config.CSV_PATH}"
          f"{'' if config.CSV_PATH.exists() else '  (missing -- run: python -m sbir fetch)'}")
    print(f"Vector store {store.describe_backend()}")
    if not meta:
        print("Index        none -- run: python -m sbir index")
        return
    facets = meta.get("facets", {})
    totals = facets.get("totals", {})
    cov = facets.get("coverage", {})
    print(f"Index        {meta['awards']:,} awards, built {meta['built_at']}")
    print(f"Model        {meta['model']} ({meta['dimension']}d)")
    print(f"Coverage     {cov.get('first_year')}-{cov.get('complete_through')}, "
          f"{totals.get('companies', 0):,} companies, "
          f"{_money(totals.get('funding', 0))} awarded")
    if cov.get("partial_years"):
        thin = ", ".join(f"{y} ({cov['by_year'][str(y)]:,})" if str(y) in cov.get("by_year", {})
                         else str(y) for y in cov["partial_years"])
        print(f"Still filling {thin}")


def cmd_search(args) -> None:
    from .search import Filters, SearchEngine

    engine = SearchEngine()
    filters = Filters(
        agency=args.agency, phase=args.phase, program=args.program, state=args.state,
        year_min=args.year_min, year_max=args.year_max, amount_min=args.amount_min,
    )
    payload = engine.search(
        " ".join(args.query), filters, limit=args.limit, sort=args.sort, mode=args.mode
    )

    print(f"\n{payload['total']} matches in {payload['took_ms']}ms\n")
    for rank, hit in enumerate(payload["results"], 1):
        print(f"{rank:>3}. {hit['title'][:92]}")
        print(f"     {hit['company']}  |  {hit['agency']}  |  {hit['phase']}  "
              f"|  {hit['year']}  |  {_money(hit['amount'])}")
        if args.abstracts and hit["abstract"]:
            body = textwrap.fill(hit["abstract"][:400], width=92,
                                 initial_indent="     ", subsequent_indent="     ")
            print(body + ("..." if len(hit["abstract"]) > 400 else ""))
        print()


def cmd_derive(args) -> None:
    from . import derive

    print("Rebuilding derived artifacts from the CSV...")
    for name, count in derive.build_all().items():
        print(f"  {name:15} {count:,}")


def cmd_eval(args) -> None:
    from .evaluation import harness
    from .evaluation import labels as label_store

    if args.action == "coverage":
        cov = label_store.coverage(label_store.load_labels())
        print(f"queries      {cov['queries']}")
        print(f"judgements   {cov['judgements']:,}")
        print(f"relevant     {cov['relevant']:,}")
        print(f"adjacent     {cov['adjacent']:,}")
        if cov["queries_without_relevant"]:
            print(f"unusable     {cov['queries_without_relevant']}")
        return

    if args.action == "show":
        # For reviewing and correcting grades, which the label file invites.
        pool = label_store.load_pool()
        graded = label_store.load_labels()
        entry = pool.get(args.query)
        if entry is None:
            raise SystemExit(f"Unknown query {args.query!r}. Known: {sorted(pool)}")
        marks = graded.get(args.query, {})
        print(f"{args.query}: {entry['query']}")
        for candidate in entry["candidates"]:
            grade = marks.get(candidate["id"], 0)
            print(f"  [{grade}] {candidate['id']:>6} {candidate['title'][:78]}")
        return

    from .search import SearchEngine

    engine = SearchEngine()
    if args.action == "pool":
        label_store.save_pool(harness.build_pool(engine))
        print("Pool written. Grade it before running.")
        return

    print(harness.report(harness.run(engine, depth=args.depth), "B0 dense baseline"))


def cmd_serve(args) -> None:
    import uvicorn

    if store.load_meta() is None:
        print("No index yet. Building one is a one-time step:\n"
              "  python -m sbir setup\n")
    print(f"SBIR Smart Search  ->  http://{args.host}:{args.port}\n")
    uvicorn.run("sbir.api:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sbir",
        description="Semantic search over the public SBIR/STTR award database.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="download the data and build the index")
    p.add_argument("--since", type=int, help="only index awards from this year onward")
    p.add_argument("--limit", type=int, help="only index the N most recent awards")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("fetch", help="download the award CSV from SBIR.gov")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("index", help="build the vector index from the CSV")
    p.add_argument("--since", type=int, help="only index awards from this year onward")
    p.add_argument("--limit", type=int, help="only index the N most recent awards")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("serve", help="run the web UI")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.PORT)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("search", help="search from the terminal")
    p.add_argument("query", nargs="+")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--mode", choices=("auto", "semantic", "company"), default="auto")
    p.add_argument("--sort", choices=("relevance", "newest", "oldest", "amount"),
                   default="relevance")
    p.add_argument("--agency", action="append", default=[])
    p.add_argument("--phase", action="append", default=[])
    p.add_argument("--program", action="append", default=[])
    p.add_argument("--state", action="append", default=[])
    p.add_argument("--year-min", type=int)
    p.add_argument("--year-max", type=int)
    p.add_argument("--amount-min", type=float)
    p.add_argument("--abstracts", action="store_true", help="print abstract snippets")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="show what is downloaded and indexed")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("derive", help="rebuild company stats and term frequencies")
    p.set_defaults(func=cmd_derive)

    p = sub.add_parser("eval", help="internal retrieval evaluation")
    p.add_argument("action", choices=("pool", "run", "coverage", "show"))
    p.add_argument("--query", help="query id, for `show`")
    p.add_argument("--depth", type=int, default=50)
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
