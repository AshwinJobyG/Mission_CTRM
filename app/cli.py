"""Command-line interface for the knowledge assistant.

Usage:
    python -m app.cli config                 # show active configuration
    python -m app.cli ingest [FOLDER] [--reset] [--no-recursive]
    python -m app.cli ask "your question"    # one-shot question
    python -m app.cli chat                   # interactive chat loop
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .embeddings import OllamaError
from .ingest import ingest_folder
from .rag import answer_question


def _print_answer(ans) -> None:
    print("\n" + ans.answer + "\n")
    print("Sources:")
    print(ans.format_sources())
    print()


def cmd_config(_: argparse.Namespace) -> int:
    print(config.summary())
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        report = ingest_folder(
            folder=args.folder,
            recursive=not args.no_recursive,
            reset=args.reset,
        )
    except OllamaError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        ans = answer_question(args.question, top_k=args.top_k)
    except OllamaError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    _print_answer(ans)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    print("Knowledge assistant — type your question (Ctrl-C or 'exit' to quit).\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        try:
            ans = answer_question(question, top_k=args.top_k)
        except OllamaError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue
        _print_answer(ans)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.cli", description="Local RAG knowledge assistant (PS-019)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="Show active configuration").set_defaults(func=cmd_config)

    p_ing = sub.add_parser("ingest", help="Read a folder and build the vector DB")
    p_ing.add_argument("folder", nargs="?", default=None, help="Folder to ingest (default: DATA_DIR)")
    p_ing.add_argument("--reset", action="store_true", help="Clear the collection first")
    p_ing.add_argument("--no-recursive", action="store_true", help="Do not descend into subfolders")
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=config.TOP_K)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive chat loop")
    p_chat.add_argument("--top-k", type=int, default=config.TOP_K)
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
