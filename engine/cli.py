"""Command line: `python -m engine.cli spec ./my-notes` drafts a spec from a folder of files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import Engine
from .ingest import SUPPORTED_EXTS, IngestError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kae", description="Knowledge AI Engine command line")
    sub = parser.add_subparsers(dest="cmd", required=True)

    spec = sub.add_parser("spec", help="Ingest a folder of notes and print a draft spec as Markdown")
    spec.add_argument("folder", type=Path)
    spec.add_argument("--name", default=None, help="Project name (defaults to the folder name)")
    spec.add_argument("--provider", default=None, help="offline | openai | anthropic | ollama")
    spec.add_argument("--json", action="store_true", help="Print the spec as JSON instead of Markdown")

    search = sub.add_parser("search", help="Search a folder of notes")
    search.add_argument("folder", type=Path)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)

    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")

    engine = Engine()
    project = engine.create_project(getattr(args, "name", None) or args.folder.name, "cli")
    for path in sorted(args.folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
            try:
                engine.ingest_bytes(project["id"], path.name, path.read_bytes(), "cli")
            except IngestError as exc:
                print(f"skipped {path.name}: {exc}", file=sys.stderr)

    if args.cmd == "search":
        for hit in engine.search(project["id"], args.query, args.k):
            print(f"[{hit['score']:.2f}] {hit['filename']} ({hit['chunk_id']})\n  {hit['text'][:200]!r}\n")
        return 0

    run = engine.start_run(project["id"], "cli", args.provider)
    for w in run["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps(run["spec"], indent=2) if args.json else engine.export_markdown(run["id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
