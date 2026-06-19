"""CLI for the JIRA connector — direct contract calls + an Ollama-backed chatbot.

Direct connector calls:
    python -m jira_connector.cli health
    python -m jira_connector.cli fetch CXC-1234
    python -m jira_connector.cli search "build failure" --project CXC --status Open

ION-LLM-backed RAG over JIRA (answers grounded in ticket context, with citations):
    python -m jira_connector.cli ask "latest on the CXC 4.8.2 build failure" --project CXC
    python -m jira_connector.cli chat --project CXC          # interactive loop

`ask`/`chat` use the company-hosted ION LLM — set ION_LLM_API_URL / ION_LLM_API_KEY /
ION_LLM_MODEL in the environment (see .env.jira.example).
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import JiraError
from .fetch import fetch
from .health import health
from .llm import LLMError, answer
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


def _print_sources(chunks) -> None:
    seen: dict[str, str] = {}
    for c in chunks:
        prov = c["provenance"]
        seen.setdefault(prov.get("ticket", "?"), prov.get("url", ""))
    if seen:
        print("\nSources:")
        for ticket, url in seen.items():
            print(f"  [{ticket}] {url}")


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


def _answer(question: str, scope: dict, top_k: int, model: str | None) -> int:
    try:
        chunks = search(question, scope)[:top_k]
    except JiraError as exc:
        print(f"[{exc.kind}] {exc.message}", file=sys.stderr)
        return 1
    try:
        reply = answer(question, chunks, model=model)
    except LLMError as exc:
        print(f"[llm] {exc}", file=sys.stderr)
        return 2
    print("\n" + reply)
    _print_sources(chunks)
    print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    return _answer(args.question, _scope(args), args.top_k, args.model)


def cmd_chat(args: argparse.Namespace) -> int:
    scope = _scope(args)
    print("JIRA escalation assistant (Ollama). Type a question; 'exit' to quit.")
    if scope:
        print(f"Scope: {scope}")
    print()
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
        _answer(q, scope, args.top_k, args.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jira_connector.cli", description="JIRA connector CLI + Ollama chatbot.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Connector liveness").set_defaults(func=cmd_health)

    pf = sub.add_parser("fetch", help="Fetch a ticket by key")
    pf.add_argument("key")
    pf.set_defaults(func=cmd_fetch)

    def add_scope(sp):
        sp.add_argument("--project", action="append", help="Restrict to project (repeatable)")
        sp.add_argument("--status", action="append", help="Restrict to status (repeatable)")
        sp.add_argument("--max-results", type=int, dest="max_results", default=None)

    ps = sub.add_parser("search", help="Search -> ranked chunks")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=8)
    add_scope(ps)
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="Ask a question, answered by Ollama over JIRA context")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, default=6)
    pa.add_argument("--model", default=None, help="Override Ollama chat model")
    add_scope(pa)
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("chat", help="Interactive Ollama-backed chat over JIRA")
    pc.add_argument("--top-k", type=int, default=6)
    pc.add_argument("--model", default=None, help="Override Ollama chat model")
    add_scope(pc)
    pc.set_defaults(func=cmd_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
