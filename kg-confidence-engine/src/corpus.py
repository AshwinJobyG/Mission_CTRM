"""Load, validate, and summarize the synthetic knowledge corpus.

Phase 1 foundation. Everything downstream (retrieval, graph, decision,
confidence, access, eval) is measured against this corpus, so the loader
validates the schema strictly but treats *intentionally* dangling references
as a reported gap signal rather than a crash — dangling refs are a seeded
condition the confidence model later detects.

Run ``python -m src.corpus`` to print corpus statistics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"

# ---- controlled vocabularies -------------------------------------------------

NODE_TYPES = {"incident", "ticket", "comment", "resolution", "runbook", "doc"}
STATUSES = {"open", "in_progress", "resolved", "wontfix", "verified", "deprecated"}
# trust ordering, highest first: resolution > runbook/doc > ticket/incident > comment
SOURCE_TIERS = {"resolution", "runbook", "doc", "incident", "ticket", "comment"}
SECURITY_LABELS = {"public", "internal", "restricted", "hr_only"}
REL_TYPES = {
    "duplicate_of",
    "relates_to",
    "caused_by",
    "resolved_by",
    "blocks",
    "supersedes",
    "contradicts",
}

REQUIRED_FIELDS = {
    "id",
    "type",
    "title",
    "body",
    "status",
    "source_tier",
    "date",
    "author",
    "security_label",
    "links",
}


@dataclass(frozen=True)
class Edge:
    """A typed, directed edge declared by a node's ``links``."""

    src: str
    target: str
    rel: str


class CorpusValidationError(ValueError):
    """Raised when the corpus violates the schema (hard error, not a gap)."""


class Corpus:
    """In-memory view of the corpus with validation and structural helpers."""

    def __init__(self, nodes: list[dict], meta: dict | None = None):
        self.meta = meta or {}
        self._raw: list[dict] = list(nodes)
        # Keyed by id; last-wins on duplicates (duplicates are flagged by validate()).
        self.nodes: dict[str, dict] = {n["id"]: n for n in nodes if "id" in n}
        self.order: list[str] = [n["id"] for n in nodes if "id" in n]

    # ---- loading ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = CORPUS_PATH, *, validate: bool = True) -> "Corpus":
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            nodes, meta = raw, {}
        else:
            nodes = raw["nodes"]
            meta = {k: v for k, v in raw.items() if k != "nodes"}
        corpus = cls(nodes, meta)
        if validate:
            corpus.validate()
        return corpus

    # ---- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[dict]:
        for nid in self.order:
            yield self.nodes[nid]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def get(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    @property
    def ids(self) -> set[str]:
        return set(self.nodes)

    # ---- edges ------------------------------------------------------------

    def edges(self) -> list[Edge]:
        """All declared edges (including dangling ones)."""
        out: list[Edge] = []
        for node in self:
            for link in node.get("links", []):
                out.append(Edge(node["id"], link["target"], link["rel"]))
        return out

    def is_dangling(self, edge: Edge) -> bool:
        return edge.target not in self.nodes

    def dangling_refs(self) -> list[Edge]:
        """Edges whose target id is absent from the corpus (a gap signal)."""
        return [e for e in self.edges() if self.is_dangling(e)]

    def contradiction_pairs(self) -> list[tuple[str, str]]:
        """Unordered node pairs joined by a ``contradicts`` edge."""
        seen: set[frozenset[str]] = set()
        pairs: list[tuple[str, str]] = []
        for e in self.edges():
            if e.rel != "contradicts" or self.is_dangling(e):
                continue
            key = frozenset((e.src, e.target))
            if key not in seen:
                seen.add(key)
                pairs.append((e.src, e.target))
        return pairs

    def supersedes_pairs(self) -> list[tuple[str, str]]:
        """(newer, older) pairs joined by a ``supersedes`` edge."""
        return [
            (e.src, e.target)
            for e in self.edges()
            if e.rel == "supersedes" and not self.is_dangling(e)
        ]

    def by_security_label(self, *labels: str) -> list[dict]:
        wanted = set(labels)
        return [n for n in self if n["security_label"] in wanted]

    # ---- validation -------------------------------------------------------

    def validate(self) -> list[Edge]:
        """Validate the schema. Returns dangling edges (reported, not fatal).

        Raises :class:`CorpusValidationError` on any hard schema violation:
        missing/duplicate ids, missing required fields, invalid enum values,
        malformed links, or unparsable dates. Dangling references are the one
        *intentional* exception and are returned for reporting.
        """
        errors: list[str] = []
        seen_ids: set[str] = set()

        for i, node in enumerate(self._raw):
            loc = node.get("id", f"<index {i}>")

            missing = REQUIRED_FIELDS - set(node)
            if missing:
                errors.append(f"{loc}: missing required fields {sorted(missing)}")
                continue

            nid = node["id"]
            if nid in seen_ids:
                errors.append(f"{nid}: duplicate id")
            seen_ids.add(nid)

            if node["type"] not in NODE_TYPES:
                errors.append(f"{nid}: invalid type {node['type']!r}")
            if node["status"] not in STATUSES:
                errors.append(f"{nid}: invalid status {node['status']!r}")
            if node["source_tier"] not in SOURCE_TIERS:
                errors.append(f"{nid}: invalid source_tier {node['source_tier']!r}")
            if node["security_label"] not in SECURITY_LABELS:
                errors.append(f"{nid}: invalid security_label {node['security_label']!r}")
            if not str(node.get("title", "")).strip():
                errors.append(f"{nid}: empty title")
            if not str(node.get("body", "")).strip():
                errors.append(f"{nid}: empty body")

            try:
                date.fromisoformat(node["date"])
            except (ValueError, TypeError):
                errors.append(f"{nid}: invalid date {node.get('date')!r} (want YYYY-MM-DD)")

            links = node.get("links", [])
            if not isinstance(links, list):
                errors.append(f"{nid}: links must be a list")
                continue
            for link in links:
                if not isinstance(link, dict) or {"target", "rel"} - set(link):
                    errors.append(f"{nid}: malformed link {link!r}")
                    continue
                if link["rel"] not in REL_TYPES:
                    errors.append(f"{nid}: invalid rel {link['rel']!r} -> {link['target']}")

        if errors:
            raise CorpusValidationError(
                "Corpus failed validation:\n  - " + "\n  - ".join(errors)
            )

        # Referential integrity: every non-dangling target must exist.
        # Dangling refs are intentional and returned (not raised).
        return self.dangling_refs()

    # ---- stats ------------------------------------------------------------

    def stats(self) -> dict:
        type_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for node in self:
            type_counts[node["type"]] = type_counts.get(node["type"], 0) + 1
            label_counts[node["security_label"]] = (
                label_counts.get(node["security_label"], 0) + 1
            )
            status_counts[node["status"]] = status_counts.get(node["status"], 0) + 1
        for e in self.edges():
            rel_counts[e.rel] = rel_counts.get(e.rel, 0) + 1
        return {
            "node_count": len(self),
            "edge_count": len(self.edges()),
            "type_counts": type_counts,
            "rel_counts": rel_counts,
            "label_counts": label_counts,
            "status_counts": status_counts,
            "dangling": self.dangling_refs(),
            "contradictions": self.contradiction_pairs(),
            "supersedes": self.supersedes_pairs(),
        }


# ---- CLI report --------------------------------------------------------------

def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _print_report(corpus: Corpus) -> None:
    s = corpus.stats()
    print("=" * 70)
    print("CORPUS STATS — kg-confidence-engine")
    if corpus.meta.get("scenario"):
        print(f"scenario: {corpus.meta['scenario']}")
    print("=" * 70)
    print(f"nodes: {s['node_count']}    edges: {s['edge_count']}")
    print()
    print(f"nodes by type:     {_fmt_counts(s['type_counts'])}")
    print(f"edges by relation: {_fmt_counts(s['rel_counts'])}")
    print(f"status:            {_fmt_counts(s['status_counts'])}")
    print(f"security labels:   {_fmt_counts(s['label_counts'])}")

    print()
    print(f"dangling references ({len(s['dangling'])}):")
    for e in s["dangling"]:
        print(f"  - {e.src} --{e.rel}--> {e.target}   (target absent)")

    print()
    print(f"contradiction pairs ({len(s['contradictions'])}):")
    for a, b in s["contradictions"]:
        print(f"  - {a}  <contradicts>  {b}")

    print()
    print(f"supersedes pairs ({len(s['supersedes'])}):")
    for newer, older in s["supersedes"]:
        print(f"  - {newer}  supersedes  {older}")

    restricted = corpus.by_security_label("restricted", "hr_only")
    print()
    print(f"restricted / hr_only nodes ({len(restricted)}):")
    for n in restricted:
        print(f"  - {n['id']} [{n['security_label']}] {n['title']}")

    # Seeded-condition checklist (Phase 1 definition of done)
    print()
    print("seeded-condition check (Phase 1 DoD):")
    checks = [
        ("30-50 nodes", 30 <= s["node_count"] <= 50),
        (">=1 contradiction pair", len(s["contradictions"]) >= 1),
        (">=1 dangling reference", len(s["dangling"]) >= 1),
        (">=1 superseded/stale node", len(s["supersedes"]) >= 1),
        (">=2 restricted/hr_only nodes", len(restricted) >= 2),
    ]
    all_ok = True
    for label, ok in checks:
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print()
    print("RESULT:", "ALL SEEDED CONDITIONS PRESENT" if all_ok else "MISSING CONDITIONS")
    print("=" * 70)
    return all_ok


def main(argv: Iterable[str] | None = None) -> int:
    try:
        corpus = Corpus.load()
    except CorpusValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    ok = _print_report(corpus)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
