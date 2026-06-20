"""Schema constants for the heterogeneous knowledge graph (Phase A).

Single source of truth for the entity-node-type and edge-relation vocabularies.
This module is the *additive* migration that lets the corpus and graph hold
typed entities (Person, Team, Ticket, Meeting, Decision, Risk, Action Item, …)
and typed causal edges (CAUSED_BY, DECIDED_IN, RAISED_RISK, …) **without
changing how the original corpus loads or scores**.

Backward-compatibility contract
-------------------------------
* Legacy nodes carry no ``node_type`` and default to :data:`DEFAULT_NODE_TYPE`
  (``"record"``). Their validation path is unchanged (see ``corpus.validate``).
* Legacy edges use the original lower-case relation set
  (:data:`LEGACY_REL_TYPES`); they remain valid.
* New typed relations (:data:`CAUSAL_REL_TYPES`) are upper-case by convention so
  they never collide with the legacy lower-case set (e.g. legacy ``blocks`` vs
  new ``BLOCKS`` are distinct strings and both accepted).

Two distinct "type" axes exist and must not be confused:
* ``type``       — the legacy *document* classification field
                   (incident/ticket/comment/resolution/runbook/doc). Unchanged.
* ``node_type``  — the new *entity* type (record/person/team/…). Optional;
                   absent ⇒ ``"record"``.
"""

from __future__ import annotations

# ---- entity (node_type) vocabulary ------------------------------------------

DEFAULT_NODE_TYPE = "record"

# Heterogeneous entity types instantiated by the P0-incident graph (Phase B).
ENTITY_NODE_TYPES = {
    DEFAULT_NODE_TYPE,  # legacy / default — homogeneous record nodes
    "person",
    "team",
    "project",
    "product",
    "ticket",
    "meeting",
    "document",
    "decision",
    "risk",
    "action_item",
    "incident",   # incident as a first-class entity in the causal graph
    "customer",   # impacted tenant / customer
}

# Fields a typed entity node must carry. Far smaller than the legacy record
# contract: an entity is identified structurally and made retrievable through
# its synthesized ``searchable_text`` (resolves conflict C3). Type-specific
# fields (role, assignee, participants, …) are validated by Phase B's loader,
# not by the structural schema check here.
ENTITY_REQUIRED_FIELDS = {"id", "node_type", "title", "searchable_text"}


# ---- edge (relation) vocabulary ---------------------------------------------

# Original relations — must remain valid for the legacy corpus unchanged.
LEGACY_REL_TYPES = {
    "duplicate_of",
    "relates_to",
    "caused_by",
    "resolved_by",
    "blocks",
    "supersedes",
    "contradicts",
}

# New typed relations for the heterogeneous / causal graph (Phase B/C).
# Upper-case by convention. Includes the directional variants used by the
# incident spine (OWNED_BY, IMPACTED) alongside the Phase-A enumerated set.
CAUSAL_REL_TYPES = {
    "OWNS",
    "OWNED_BY",
    "MEMBER_OF",
    "SUPPORTS",
    "DESCRIBES",
    "ASSIGNED_TO",
    "IMPACTED_BY",
    "IMPACTED",
    "CAUSED_BY",
    "DECIDED_IN",
    "RAISED_RISK",
    "MITIGATED_BY",
    "PRIORITIZED_BY",
    "DROPPED_IN",
    "BLOCKS",
    "RESULTED_IN",
    "EVIDENCED_BY",
}

# Full accepted vocabulary = legacy ∪ causal.
REL_TYPES = LEGACY_REL_TYPES | CAUSAL_REL_TYPES

# The subset of relations along which Stage 3.5 (Phase C) walks backward from an
# impact to its root cause(s). Defined here so traversal and eval share one list.
CAUSAL_TRAVERSAL_RELS = {
    "CAUSED_BY",
    "DROPPED_IN",
    "DECIDED_IN",
    "RAISED_RISK",
    "PRIORITIZED_BY",
    "OWNED_BY",
}


# ---- helpers ----------------------------------------------------------------

def node_type(node: dict) -> str:
    """Entity type of a node, defaulting legacy nodes to ``"record"``."""
    return node.get("node_type", DEFAULT_NODE_TYPE)


def is_record(node: dict) -> bool:
    """True for legacy/homogeneous record nodes (the unchanged path)."""
    return node_type(node) == DEFAULT_NODE_TYPE
