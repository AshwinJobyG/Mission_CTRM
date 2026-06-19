"""CLI for the JIRA connector — contract calls, vector indexing, and RAG chat.

Direct connector calls:
    python -m jira_connector.cli health
    python -m jira_connector.cli fetch NGPOWER-46
    python -m jira_connector.cli search "build failure" --project NGPOWER --status Open

Vector RAG (the architecture flow): build the index once, then ask/chat.
    python -m jira_connector.cli index --project NGPOWER
    python -m jira_connector.cli ask "what is the status of NGPOWER-46?"
    python -m jira_connector.cli chat

`index` embeds tickets into a Chroma vector DB via the ION embeddings endpoint;
`ask`/`chat` retrieve semantically and answer via the ION LLM with a confidence
score. Set ION_LLM_API_URL / ION_LLM_API_KEY / ION_LLM_MODEL / ION_LLM_EMBED_MODEL.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import rag, vectorstore
from .embeddings import EmbeddingError, check as embed_check
from .errors import JiraError
from .fetch import fetch
from .health import health
from .index import build_index
from .llm import LLMError
from .search import search


def _scope(args: argparse.Namespace) -> dict:
    scope: dict = {}
    if getattr(args, "project", None):
        scope["projects"] = args.project
    if getattr(args, "status", None):
        scope["statuses"] = args.status
    if getattr(args, "max_results", None):
        scope["max_results"] = args.max_results
    return scope


def cmd_health(_: argparse.Namespace) -> int:
    print(json.dumps(health(), indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        print(json.dumps(fetch(args.key), indent=2))
    except JiraError as exc:
        print(f"[{exc.kind}] {exc.message}", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        chunks = search(args.query, _scope(args))
    except JiraError as exc:
        print(f"[{exc.kind}] {exc.message}", file=sys.stderr)
        return 1
    for c in chunks[: args.top_k]:
        prov = c["provenance"]
        print(f"[{c['score']:.3f}] {c['chunk_id']} ({prov['ticket']}/{prov['field']})")
        print(f"      {c['text'][:120]}...")
    return 0


def cmd_embed_check(_: argparse.Namespace) -> int:
    try:
        print(embed_check())
    except EmbeddingError as exc:
        print(f"[embeddings] {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    try:
        report = build_index(
            projects=args.project,
            max_tickets=args.max_tickets,
            reset=not args.no_reset,
        )
    except (JiraError, EmbeddingError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


def _print_rag(ans: rag.RagAnswer) -> None:
    print("\n" + ans.answer + "\n")
    print(f"Confidence: {ans.confidence_pct}% ({ans.confidence_band})")
    if ans.per_source:
        print("Sources:")
        for ticket, sim in ans.per_source:
            url = next((s.url for s in ans.sources if s.ticket == ticket), "")
            print(f"  [{ticket}] {round(sim * 100)}%  {url}")
    print()


def _ask(question: str, top_k: int, model: str | None) -> int:
    try:
        ans = rag.answer(question, top_k=top_k, model=model)
    except EmbeddingError as exc:
        print(f"[embeddings] {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"[llm] {exc}", file=sys.stderr)
        return 2
    _print_rag(ans)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    return _ask(args.question, args.top_k, args.model)


def cmd_chat(args: argparse.Namespace) -> int:
    print(f"JIRA RAG assistant — {vectorstore.count()} chunks indexed. 'exit' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            return 0
        _ask(q, args.top_k, args.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jira_connector.cli", description="JIRA connector CLI + vector RAG.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Connector liveness").set_defaults(func=cmd_health)

    pf = sub.add_parser("fetch", help="Fetch a ticket by key")
    pf.add_argument("key")
    pf.set_defaults(func=cmd_fetch)

    ps = sub.add_parser("search", help="JQL search -> ranked chunks (debug)")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=8)
    ps.add_argument("--project", action="append")
    ps.add_argument("--status", action="append")
    ps.add_argument("--max-results", type=int, dest="max_results", default=None)
    ps.set_defaults(func=cmd_search)

    sub.add_parser("embed-check", help="Test the configured embeddings backend").set_defaults(func=cmd_embed_check)

    pi = sub.add_parser("index", help="Build the vector DB from JIRA tickets")
    pi.add_argument("--project", action="append", help="Project key to index (repeatable)")
    pi.add_argument("--max-tickets", type=int, default=200)
    pi.add_argument("--no-reset", action="store_true", help="Add to the existing index")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="Ask a question (vector RAG + confidence)")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, default=6)
    pa.add_argument("--model", default=None, help="Override ION chat model")
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("chat", help="Interactive RAG chat")
    pc.add_argument("--top-k", type=int, default=6)
    pc.add_argument("--model", default=None, help="Override ION chat model")
    pc.set_defaults(func=cmd_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
