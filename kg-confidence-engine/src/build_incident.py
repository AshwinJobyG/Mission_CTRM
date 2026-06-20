"""Phase B — author and emit the synthetic P0-incident knowledge graph.

Single source of truth for the heterogeneous incident corpus. Running this
module regenerates three artifacts under ``data/`` and then verifies them:

* ``data/incident_corpus.json`` — the typed, evidence-bearing knowledge graph,
  loadable via ``Corpus.load_named("incident")`` / ``--corpus incident``.
* ``data/incident_manifest.json`` — the ground-truth node/edge listing plus the
  gold causal spine (impact → root cause). ``eval_causal.json`` derives its gold
  paths from this, so the manifest is the contract.
* ``data/transcripts/*.md`` — the synthetic meeting/postmortem passages, in the
  voice of the real NGPOWER transcripts (Maneesh / Anupam / Rinit / Franziska).

The incident spine encoded here (rediscovered by Stage 3.5, scored by
path-completeness):

    Risk(coupling) ← raised by Anupam in Grooming 14-May
      → Decision(drop NGPOWER-49) prioritised by Maneesh, decided in Grooming
        → Ticket NGPOWER-49 (safeguard, Dropped) owned by Rinit
          → Incident NGPOWER-145 (P0) caused by absent 49 + no-recovery 50,
            impacting EU Power, resulting in 4 action items.

Run: ``python -m src.build_incident``  (writes + verifies)
     ``python -m src.build_incident --verify-only``  (verify existing files)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .corpus import Corpus, INCIDENT_CORPUS_PATH
from .schema import CAUSAL_TRAVERSAL_RELS, node_type

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
MANIFEST_PATH = DATA_DIR / "incident_manifest.json"


# ===========================================================================
# Synthetic transcripts (real-voice reference; the evidence passages quote them)
# ===========================================================================

TRANSCRIPTS: dict[str, str] = {
    "grooming_14may.md": """# Backlog grooming — 14 May 2026
**Attendees:** Maneesh Sama (PM), Anupam Sengupta (tech lead), Rinit Jain (dev)

**Maneesh:** We're tight on the EPEX connectivity go-live. I want to know what
we can drop from this iteration without moving that date.

**Rinit:** NGPOWER-45 (volume generation) is done. NGPOWER-49 — decoupling
trade persistence from volume generation — I have the change ready behind a
flag, but it needs another iteration to test the failover paths.

**Anupam:** I want to flag a risk before we drop 49. Right now trade
persistence is transactionally coupled to volume generation. If we drop 49,
that coupling stays. A database failover *mid-batch* can lose trades that were
generated but never persisted. 49 is the safeguard for exactly that.

**Maneesh:** Understood, but the EPEX connectivity date is the priority this
iteration. Let's drop 49 and hold the date. I'll own that call.

**Anupam:** Then we should at least pull NGPOWER-50 — the replay/recovery path —
so we can reconstruct lost trades if it happens.

**Maneesh:** Note it, but 50 stays in backlog for now. We revisit next grooming.
""",
    "design_call.md": """# Persistence / volume-generation design call — 8 May 2026
**Attendees:** Anupam Sengupta (tech lead), Rinit Jain (dev)

**Anupam:** The core design issue is that persistence and volume generation
share one transaction. NGPOWER-49 makes persistence its own transaction — that
is the safeguard. Without it, a failover between generation and commit drops the
trade silently.

**Rinit:** So 49 is really the mitigation for the coupling risk, not just a
refactor.

**Anupam:** Exactly. Treat 49 as the risk mitigation, not a nice-to-have.
""",
    "incident_bridge.md": """# NGPOWER-145 incident bridge — 28 May 2026
**Attendees:** Maneesh Sama (PM), Anupam Sengupta (tech lead), Rinit Jain (dev),
Franziska Vogel (EU Power product owner)

**Franziska:** EU Power is missing roughly 40 trades from last night's EPEX run.
This is a P0 — settlement is blocked.

**Anupam:** There was an EPEX database failover at 02:14, mid-batch. Persistence
is still coupled to volume generation, so the in-flight trades were generated
but never committed.

**Rinit:** And we have no replay path — NGPOWER-50 is still in backlog — so we
can't reconstruct them automatically.

**Maneesh:** This traces straight back to dropping NGPOWER-49 at the 14-May
grooming. We need the postmortem to capture that.
""",
    "postmortem.md": """# NGPOWER-145 postmortem — 29 May 2026
**Author:** Anupam Sengupta

**Summary:** P0 — ~40 EU Power trades lost during an EPEX database failover
mid-batch on 28 May.

**Root cause:** Trade persistence was transactionally coupled to volume
generation. The safeguard that would have decoupled them, NGPOWER-49, was
dropped at the 14-May grooming to protect the EPEX connectivity date. A failover
mid-batch therefore lost trades that were generated but never persisted.

**Contributing cause:** No replay/recovery path existed (NGPOWER-50, still in
Backlog), so the lost trades could not be reconstructed automatically.

**Action items:**
1. Pull NGPOWER-50 (replay/recovery) into the current iteration.
2. Prioritise NGPOWER-19 (idempotent persistence retries).
3. Add a failover-during-batch regression test.
4. Grooming rule: a safeguard ticket cannot be dropped without explicit risk
   sign-off from the tech lead.
""",
    "iteration_review.md": """# Iteration review (reference) — 2 May 2026
**Attendees:** Maneesh Sama (PM), Anupam Sengupta, Rinit Jain

Reference transcript retained for voice/texture. Covers NGPOWER-11, NGPOWER-22
and NGPOWER-43 status and the EPEX connectivity milestone. No incident content.
""",
    "pbr_rdx_reference.md": """# PBR-RDX reference data walkthrough (reference) — 6 May 2026
**Attendees:** Anupam Sengupta, Rinit Jain

Reference transcript retained for voice/texture. Covers EPEX reference-data
loading (PBR-RDX) and how volume generation consumes it. No incident content.
""",
}


# ===========================================================================
# Nodes
# ===========================================================================

def _ev(source_id: str, passage: str) -> dict:
    return {"source_id": source_id, "passage": passage}


# Evidence passages reused on multiple edges, quoted from the transcripts above.
EV_GROOM_RISK = _ev(
    "MEET-grooming-14may",
    "Anupam: If we drop 49, that coupling stays. A database failover mid-batch "
    "can lose trades that were generated but never persisted. 49 is the safeguard.")
EV_GROOM_DROP = _ev(
    "MEET-grooming-14may",
    "Maneesh: the EPEX connectivity date is the priority this iteration. Let's "
    "drop 49 and hold the date. I'll own that call.")
EV_TICKET49_DROP = _ev(
    "NGPOWER-49",
    "Comment (Maneesh, 14-May): Dropped this iteration per grooming to protect "
    "the EPEX connectivity date.")
EV_TICKET49_OWNER = _ev(
    "NGPOWER-49",
    "Assignee: Rinit Jain. Comment (Rinit): decoupling change was ready behind a "
    "flag, pending failover testing.")
EV_DESIGN_MIT = _ev(
    "MEET-design-call",
    "Anupam: NGPOWER-49 makes persistence its own transaction — treat 49 as the "
    "risk mitigation, not a nice-to-have.")
EV_PM_CAUSE49 = _ev(
    "DOC-postmortem",
    "Root cause: the safeguard NGPOWER-49 that would have decoupled persistence "
    "was dropped at the 14-May grooming; a failover mid-batch lost unpersisted "
    "trades.")
EV_PM_CAUSE50 = _ev(
    "DOC-postmortem",
    "Contributing cause: no replay/recovery path existed (NGPOWER-50, still "
    "Backlog), so lost trades could not be reconstructed.")
EV_BRIDGE_IMPACT = _ev(
    "MEET-incident-bridge",
    "Franziska: EU Power is missing roughly 40 trades from last night's EPEX "
    "run. Settlement is blocked — this is a P0.")
EV_PM_ACTIONS = _ev(
    "DOC-postmortem",
    "Action items: pull NGPOWER-50, prioritise NGPOWER-19, add failover "
    "regression test, add grooming sign-off rule.")


def _person(pid, name, role, st):
    return {
        "id": pid, "node_type": "person", "title": name, "name": name,
        "role": role, "team": "NGPOWER", "jira_account_id": pid.lower(),
        "date": "2026-05-01", "security_label": "internal",
        "searchable_text": st, "links": [],
    }


def _ticket(key, summary, status, assignee, priority, st, *, source_tier="ticket"):
    return {
        "id": key, "node_type": "ticket", "title": f"{key}: {summary}",
        "key": key, "summary": summary, "status": status, "assignee": assignee,
        "priority": priority, "source_tier": source_tier,
        "date": "2026-05-10", "security_label": "internal",
        "searchable_text": st, "links": [],
    }


def build_nodes() -> list[dict]:
    nodes: list[dict] = []

    # --- People ---
    nodes += [
        _person("PERSON-maneesh", "Maneesh Sama", "PM",
                "Maneesh Sama product manager PM NGPOWER prioritised dropping "
                "NGPOWER-49 to protect EPEX connectivity date grooming 14 May"),
        _person("PERSON-anupam", "Anupam Sengupta", "tech_lead",
                "Anupam Sengupta tech lead raised coupling risk trade persistence "
                "volume generation NGPOWER-49 safeguard grooming postmortem author"),
        _person("PERSON-rinit", "Rinit Jain", "developer",
                "Rinit Jain developer owner NGPOWER-49 NGPOWER-45 decoupling "
                "persistence change ready behind flag"),
        _person("PERSON-franziska", "Franziska Vogel", "customer",
                "Franziska Vogel EU Power product owner reported 40 trades lost "
                "EPEX failover settlement blocked P0"),
    ]

    # --- Team / Project / Product ---
    nodes += [
        {"id": "TEAM-ngpower", "node_type": "team", "title": "NGPOWER delivery team",
         "name": "NGPOWER", "members": ["PERSON-maneesh", "PERSON-anupam", "PERSON-rinit"],
         "date": "2026-05-01", "security_label": "internal",
         "searchable_text": "NGPOWER delivery team Maneesh Anupam Rinit EPEX power trading",
         "links": []},
        {"id": "PRODUCT-epex", "node_type": "product", "title": "EPEX connectivity",
         "name": "EPEX connectivity", "date": "2026-05-01", "security_label": "internal",
         "searchable_text": "EPEX connectivity power exchange go-live milestone trade submission",
         "links": []},
        {"id": "PRODUCT-persistence", "node_type": "product", "title": "Trade persistence service",
         "name": "Trade persistence service", "date": "2026-05-01", "security_label": "internal",
         "searchable_text": "trade persistence service commits trades volume generation coupling",
         "links": []},
    ]

    # --- Tickets (real backlog keys + synthetic incident) ---
    nodes += [
        _ticket("NGPOWER-45", "Volume generation for EPEX trades", "resolved",
                "PERSON-rinit", "Medium",
                "NGPOWER-45 volume generation EPEX trades done resolved Rinit"),
        _ticket("NGPOWER-49", "Decouple trade persistence from volume generation (transactional safeguard)",
                "wontfix", "PERSON-rinit", "High",
                "NGPOWER-49 decouple trade persistence volume generation transactional "
                "safeguard dropped wontfix grooming 14 May Rinit High priority"),
        _ticket("NGPOWER-50", "Trade replay / recovery path after failover", "open",
                "unassigned", "High",
                "NGPOWER-50 trade replay recovery path failover backlog open no recovery"),
        _ticket("NGPOWER-19", "Idempotent persistence retries", "open",
                "unassigned", "Medium",
                "NGPOWER-19 idempotent persistence retries backlog prioritise"),
        _ticket("NGPOWER-11", "EPEX reference-data loader hardening", "in_progress",
                "PERSON-rinit", "Medium",
                "NGPOWER-11 EPEX reference data loader hardening in progress"),
        _ticket("NGPOWER-22", "Volume-generation throughput tuning", "open",
                "unassigned", "Low",
                "NGPOWER-22 volume generation throughput tuning backlog"),
        _ticket("NGPOWER-43", "Settlement export formatting", "in_progress",
                "PERSON-rinit", "Low",
                "NGPOWER-43 settlement export formatting in progress"),
    ]

    # --- Incident ---
    nodes.append({
        "id": "NGPOWER-145", "node_type": "incident",
        "title": "NGPOWER-145: P0 — EU Power trades lost during EPEX failover",
        "key": "NGPOWER-145", "summary": "P0: ~40 EU Power trades lost during an EPEX "
        "database failover mid-batch", "status": "resolved", "assignee": "PERSON-anupam",
        "priority": "P0", "source_tier": "incident", "date": "2026-05-28",
        "security_label": "internal",
        "searchable_text": "NGPOWER-145 P0 incident EU Power trades lost EPEX database "
        "failover mid-batch persistence coupling NGPOWER-49 dropped root cause why",
        "links": []})

    # --- Meetings ---
    nodes += [
        {"id": "MEET-grooming-14may", "node_type": "meeting",
         "title": "Backlog grooming — 14 May 2026", "date": "2026-05-14",
         "participants": ["PERSON-maneesh", "PERSON-anupam", "PERSON-rinit"],
         "transcript": "transcripts/grooming_14may.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "backlog grooming 14 May drop NGPOWER-49 EPEX date Anupam "
         "raised coupling risk Maneesh prioritised pull NGPOWER-50 declined",
         "links": []},
        {"id": "MEET-design-call", "node_type": "meeting",
         "title": "Persistence / volume-generation design call — 8 May 2026",
         "date": "2026-05-08", "participants": ["PERSON-anupam", "PERSON-rinit"],
         "transcript": "transcripts/design_call.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "design call persistence volume generation coupling NGPOWER-49 "
         "safeguard own transaction mitigation",
         "links": []},
        {"id": "MEET-incident-bridge", "node_type": "meeting",
         "title": "NGPOWER-145 incident bridge — 28 May 2026", "date": "2026-05-28",
         "participants": ["PERSON-maneesh", "PERSON-anupam", "PERSON-rinit", "PERSON-franziska"],
         "transcript": "transcripts/incident_bridge.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "incident bridge NGPOWER-145 EU Power 40 trades lost EPEX "
         "failover 02:14 mid batch no replay path traces to dropping NGPOWER-49",
         "links": []},
    ]

    # --- Documents ---
    nodes += [
        {"id": "DOC-postmortem", "node_type": "document",
         "title": "NGPOWER-145 postmortem", "date": "2026-05-29", "author": "PERSON-anupam",
         "transcript": "transcripts/postmortem.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "postmortem NGPOWER-145 root cause persistence coupled volume "
         "generation safeguard NGPOWER-49 dropped no recovery NGPOWER-50 action items",
         "links": []},
        {"id": "DOC-iteration-review", "node_type": "document",
         "title": "Iteration review (reference)", "date": "2026-05-02",
         "transcript": "transcripts/iteration_review.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "iteration review reference NGPOWER-11 NGPOWER-22 NGPOWER-43 "
         "EPEX connectivity milestone status",
         "links": []},
        {"id": "DOC-pbr-rdx", "node_type": "document",
         "title": "PBR-RDX reference-data walkthrough (reference)", "date": "2026-05-06",
         "transcript": "transcripts/pbr_rdx_reference.md", "source_tier": "doc",
         "security_label": "internal",
         "searchable_text": "PBR-RDX reference data EPEX loading volume generation walkthrough",
         "links": []},
    ]

    # --- Decision / Risk / Action items / Customer ---
    nodes += [
        {"id": "RISK-coupling", "node_type": "risk",
         "title": "Trade persistence transactionally coupled to volume generation",
         "date": "2026-05-08", "severity": "high", "security_label": "internal",
         "searchable_text": "risk trade persistence transactionally coupled volume "
         "generation failover mid batch loses trades raised by Anupam mitigation NGPOWER-49",
         "links": []},
        {"id": "DEC-drop49", "node_type": "decision",
         "title": "Drop NGPOWER-49 to protect EPEX connectivity go-live date",
         "date": "2026-05-14", "decided_by": "PERSON-maneesh", "security_label": "internal",
         "searchable_text": "decision drop NGPOWER-49 protect EPEX connectivity go-live "
         "date grooming 14 May prioritised by Maneesh despite coupling risk",
         "links": []},
        {"id": "AI-pull50", "node_type": "action_item",
         "title": "Pull NGPOWER-50 (replay/recovery) into current iteration",
         "date": "2026-05-29", "owner": "PERSON-rinit", "security_label": "internal",
         "searchable_text": "action item pull NGPOWER-50 replay recovery path current iteration",
         "links": []},
        {"id": "AI-prioritise19", "node_type": "action_item",
         "title": "Prioritise NGPOWER-19 (idempotent persistence retries)",
         "date": "2026-05-29", "owner": "PERSON-rinit", "security_label": "internal",
         "searchable_text": "action item prioritise NGPOWER-19 idempotent persistence retries",
         "links": []},
        {"id": "AI-regression", "node_type": "action_item",
         "title": "Add a failover-during-batch regression test",
         "date": "2026-05-29", "owner": "PERSON-rinit", "security_label": "internal",
         "searchable_text": "action item failover during batch regression test",
         "links": []},
        {"id": "AI-grooming-rule", "node_type": "action_item",
         "title": "Grooming rule: safeguards cannot be dropped without risk sign-off",
         "date": "2026-05-29", "owner": "PERSON-maneesh", "security_label": "internal",
         "searchable_text": "action item grooming rule safeguard cannot be dropped without "
         "tech lead risk sign off",
         "links": []},
        {"id": "CUST-eupower", "node_type": "customer",
         "title": "EU Power (EU Power Trading GmbH)", "date": "2026-05-28",
         "security_label": "restricted",
         "searchable_text": "EU Power Trading GmbH tenant customer impacted 40 trades lost "
         "settlement blocked EPEX",
         "links": []},
    ]
    return nodes


# ===========================================================================
# Edges  (src, rel, target, evidence|None)
# ===========================================================================

# The causal spine — every edge here is in CAUSAL_TRAVERSAL_RELS and carries
# evidence. Stage 3.5 walks these; path-completeness scores them.
SPINE_EDGES = [
    ("NGPOWER-145", "CAUSED_BY", "NGPOWER-49", EV_PM_CAUSE49),
    ("NGPOWER-145", "CAUSED_BY", "NGPOWER-50", EV_PM_CAUSE50),
    ("NGPOWER-49", "DROPPED_IN", "DEC-drop49", EV_TICKET49_DROP),
    ("NGPOWER-49", "OWNED_BY", "PERSON-rinit", EV_TICKET49_OWNER),
    ("DEC-drop49", "DECIDED_IN", "MEET-grooming-14may", EV_GROOM_DROP),
    ("DEC-drop49", "PRIORITIZED_BY", "PERSON-maneesh", EV_GROOM_DROP),
    ("MEET-grooming-14may", "RAISED_RISK", "RISK-coupling", EV_GROOM_RISK),
    ("RISK-coupling", "RAISED_RISK", "PERSON-anupam", EV_GROOM_RISK),
]

# Supporting typed edges (context, not part of the scored causal spine).
SUPPORT_EDGES = [
    ("RISK-coupling", "MITIGATED_BY", "NGPOWER-49", EV_DESIGN_MIT),
    ("NGPOWER-145", "IMPACTED", "CUST-eupower", EV_BRIDGE_IMPACT),
    ("NGPOWER-145", "RESULTED_IN", "AI-pull50", EV_PM_ACTIONS),
    ("NGPOWER-145", "RESULTED_IN", "AI-prioritise19", EV_PM_ACTIONS),
    ("NGPOWER-145", "RESULTED_IN", "AI-regression", EV_PM_ACTIONS),
    ("NGPOWER-145", "RESULTED_IN", "AI-grooming-rule", EV_PM_ACTIONS),
    ("MEET-design-call", "DESCRIBES", "RISK-coupling", EV_DESIGN_MIT),
    ("DOC-postmortem", "DESCRIBES", "NGPOWER-145", None),
    ("MEET-incident-bridge", "DESCRIBES", "NGPOWER-145", EV_BRIDGE_IMPACT),
    ("AI-pull50", "DESCRIBES", "NGPOWER-50", None),
    ("AI-prioritise19", "DESCRIBES", "NGPOWER-19", None),
    ("NGPOWER-45", "ASSIGNED_TO", "PERSON-rinit", None),
    ("NGPOWER-49", "ASSIGNED_TO", "PERSON-rinit", None),
    ("PERSON-maneesh", "MEMBER_OF", "TEAM-ngpower", None),
    ("PERSON-anupam", "MEMBER_OF", "TEAM-ngpower", None),
    ("PERSON-rinit", "MEMBER_OF", "TEAM-ngpower", None),
    ("TEAM-ngpower", "SUPPORTS", "PRODUCT-epex", None),
    ("TEAM-ngpower", "SUPPORTS", "PRODUCT-persistence", None),
    ("DOC-iteration-review", "DESCRIBES", "PRODUCT-persistence", None),
    ("DOC-pbr-rdx", "DESCRIBES", "PRODUCT-epex", None),
    # EVIDENCED_BY edges make the evidence visible as graph structure too.
    ("DEC-drop49", "EVIDENCED_BY", "MEET-grooming-14may", None),
    ("RISK-coupling", "EVIDENCED_BY", "MEET-grooming-14may", None),
    ("NGPOWER-145", "EVIDENCED_BY", "DOC-postmortem", None),
]

ALL_EDGES = SPINE_EDGES + SUPPORT_EDGES


def attach_edges(nodes: list[dict]) -> list[dict]:
    """Attach each edge to its source node's ``links`` list."""
    by_id = {n["id"]: n for n in nodes}
    for src, rel, tgt, ev in ALL_EDGES:
        if src not in by_id:
            raise ValueError(f"edge source not a node: {src}")
        link = {"target": tgt, "rel": rel}
        if ev:
            link["evidence"] = ev
        by_id[src]["links"].append(link)
    return nodes


# ===========================================================================
# Emit + verify
# ===========================================================================

def _corpus_dict(nodes: list[dict]) -> dict:
    return {
        "scenario": "NGPOWER P0 incident — trade-persistence loss (synthetic)",
        "notes": "Heterogeneous knowledge graph for causal 'why' reasoning. "
                 "Generated by src/build_incident.py. Causal edges carry evidence.",
        "nodes": nodes,
    }


def _manifest(nodes: list[dict]) -> dict:
    return {
        "scenario": "NGPOWER P0 incident",
        "impact_node": "NGPOWER-145",
        "root_cause_node": "RISK-coupling",
        "personas": {
            "raised_risk": "PERSON-anupam",
            "made_the_call": "PERSON-maneesh",
            "owned_the_code": "PERSON-rinit",
            "customer_impacted": "PERSON-franziska",
        },
        "gold_causal_spine": [
            {"src": s, "rel": r, "target": t, "has_evidence": bool(e)}
            for (s, r, t, e) in SPINE_EDGES
        ],
        "nodes": [
            {"id": n["id"], "node_type": node_type(n), "title": n["title"]}
            for n in nodes
        ],
        "edges": [
            {"src": s, "rel": r, "target": t, "has_evidence": bool(e)}
            for (s, r, t, e) in ALL_EDGES
        ],
    }


def write_artifacts() -> dict:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, text in TRANSCRIPTS.items():
        (TRANSCRIPT_DIR / fname).write_text(text, encoding="utf-8")

    nodes = attach_edges(build_nodes())
    INCIDENT_CORPUS_PATH.write_text(json.dumps(_corpus_dict(nodes), indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(_manifest(nodes), indent=2), encoding="utf-8")
    return {"nodes": len(nodes), "edges": len(ALL_EDGES),
            "transcripts": len(TRANSCRIPTS)}


def verify() -> bool:
    """Load the emitted corpus, rebuild the graph, and assert the spine + evidence."""
    corpus = Corpus.load(INCIDENT_CORPUS_PATH)
    edges = corpus.edges()
    edge_index = {(e.src, e.rel, e.target): e for e in edges}
    ok = True

    print("=" * 72)
    print("INCIDENT GRAPH — verification")
    print("=" * 72)
    print(f"nodes: {len(corpus)}   edges: {len(edges)}   "
          f"dangling: {len(corpus.dangling_refs())}")

    # node-type census
    counts: dict[str, int] = {}
    for n in corpus:
        counts[node_type(n)] = counts.get(node_type(n), 0) + 1
    print("node_type census: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\ncausal spine (impact → root cause), each step must carry evidence:")
    for src, rel, tgt, _ev in SPINE_EDGES:
        e = edge_index.get((src, rel, tgt))
        present = e is not None
        has_ev = bool(e and e.evidence)
        flag = "ok " if (present and has_ev) else "XX "
        ok = ok and present and has_ev
        ev_src = e.evidence[0] if (e and e.evidence) else "—"
        print(f"  [{flag}] {src} --{rel}--> {tgt}   evidence@{ev_src}")

    # every causal-traversal-rel edge in the corpus must have evidence
    print("\nall causal-rel edges carry evidence:")
    missing_ev = [(e.src, e.rel, e.target) for e in edges
                  if e.rel in CAUSAL_TRAVERSAL_RELS and not e.evidence]
    if missing_ev:
        ok = False
        for s, r, t in missing_ev:
            print(f"  [XX ] {s} --{r}--> {t}  (NO EVIDENCE)")
    else:
        print("  [ok ] every causal-traversal edge has evidence")

    # no dangling refs in the incident corpus (it is meant to be complete)
    if corpus.dangling_refs():
        ok = False
        print("\n[XX ] unexpected dangling references:")
        for e in corpus.dangling_refs():
            print(f"      {e.src} --{e.rel}--> {e.target}")

    print("\nRESULT:", "SPINE PRESENT & FULLY EVIDENCED ✅" if ok else "VERIFICATION FAILED ❌")
    print("=" * 72)
    return ok


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--verify-only" not in argv:
        stats = write_artifacts()
        print(f"wrote incident_corpus.json ({stats['nodes']} nodes, "
              f"{stats['edges']} edges), incident_manifest.json, "
              f"{stats['transcripts']} transcripts\n")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
