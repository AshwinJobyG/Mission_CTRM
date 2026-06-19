"""Offline demo / smoke check using a captured fixture (no live network).

Runs the real normalize -> chunk -> rank pipeline against a saved JIRA response,
so the demo works even with no connectivity. For a full live demo, use the MCP
tools (jira_fetch / jira_search) against a real instance instead.

    python -m tests.offline_demo
"""

from __future__ import annotations

import json
from pathlib import Path

from jira_connector.chunk import chunks_from_record
from jira_connector.fetch import normalize_issue
from jira_connector.rank import score_chunks

FIXTURE = Path(__file__).parent / "fixtures" / "issue_CXC-1234.json"


def main() -> int:
    issue = json.loads(FIXTURE.read_text())
    url = f"https://acme.atlassian.net/browse/{issue['key']}"

    record = normalize_issue(issue, url)
    print("=== fetch(CXC-1234) normalized record ===")
    print(json.dumps(record, indent=2)[:1200], "...\n")

    chunks = chunks_from_record(record)
    ranked = score_chunks(chunks, "4.8.2 build failure pricing", {record["id"]: record["updated"]})
    print("=== search() top ranked chunks ===")
    for c in ranked[:3]:
        prov = c["provenance"]
        print(f"[{c['score']:.3f}] {c['chunk_id']} ({prov['ticket']}/{prov['field']})")
        print(f"      {c['text'][:90]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
