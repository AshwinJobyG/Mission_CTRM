# JIRA MCP Connector

> **Scope:** ONE source connector for the RAG-based Escalation Context Management
> Assistant (ION Hackathon **PS-003**). This document covers **only** the JIRA
> connector — a standalone MCP source over a live JIRA Cloud/Server instance.
> The MCP Controller, vector index, context assembler, LLM, and privacy layer
> are **out of scope** and treated as external consumers.
>
> This file is both the **phased build guide** (execute it phase by phase in 24h)
> and the component's **living README** afterward.

---

## 0. What this connector is

A standalone MCP source implementing the project's **uniform connector contract**
over a live JIRA REST API. Exactly three methods:

| Method | Signature | Returns |
|--------|-----------|---------|
| `search` | `search(query, scope)` | ranked **chunks** (ACL-scoped, provenance per chunk) |
| `fetch` | `fetch(id)` | full normalized **record** + provenance |
| `health` | `health()` | `up` \| `degraded` \| `down` |

### Design rules (true for every phase)

1. **Callable in isolation.** The controller is just an external consumer; the
   connector must run and be tested standalone.
2. **Provenance is first-class.** Every record and every chunk carries "which
   ticket / which field / what URL / when retrieved." An escalation answer
   without "which ticket said this" is useless downstream.
3. **Access control at the source.** `scope` (ACL/project constraints) is applied
   **inside the JQL**, so out-of-scope data is never fetched — not filtered after.
4. **Downstream-ready shape.** `search` returns chunked text + metadata so a
   vector index / context assembler could consume it without reshaping.
5. **Secrets from env/secrets only.** No tokens in code or git, ever.

### The contract is language-agnostic

The three method names, their inputs, and the normalized record/chunk schemas
(below) are the **public contract**. The reference implementation stack is a
recommendation, not part of the contract — any language that can speak HTTP and
expose an MCP server satisfies it.

---

## Recommended stack (and why)

| Concern | Choice | Why |
|---------|--------|-----|
| Language | **Python 3.11** | Fastest path in 24h; rich JIRA/HTTP ecosystem; matches the rest of the assistant. |
| MCP | **`mcp` (official Python SDK, FastMCP)** | Exposes the three methods as MCP tools with minimal boilerplate. |
| HTTP | **`httpx`** | Sync + async, timeouts, connection reuse, clean error model. |
| JIRA API | **Raw REST via `httpx`** (not the `jira` lib) | Full control over fields, JQL, pagination, and error codes; fewer surprises than a heavy wrapper. |
| Config | **`pydantic-settings` + `.env`** | Typed config, env-only secrets, clear validation errors. |
| Tests/fixtures | **`pytest` + `respx`** | Replay captured real responses so the demo doesn't depend on live network. |

> The contract stays language-agnostic; only this reference implementation is
> Python.

---

## Phase 0 — Setup & Auth

**Goal:** A skeleton that authenticates to a real JIRA and proves reachability
before any feature code.

### Concrete steps

1. Create the project skeleton (package + entrypoint + config + tests dir).
2. Pin dependencies (`httpx`, `mcp`, `pydantic-settings`, `pytest`, `respx`).
3. Define the **config/env strategy** — all secrets via environment, validated
   at startup, never hardcoded:

   | Env var | Meaning | Example |
   |---------|---------|---------|
   | `JIRA_BASE_URL` | Instance base URL | `https://acme.atlassian.net` |
   | `JIRA_AUTH_MODE` | `cloud` or `server` | `cloud` |
   | `JIRA_EMAIL` | Account email (Cloud only) | `bot@acme.com` |
   | `JIRA_API_TOKEN` | API token (Cloud) | *(secret)* |
   | `JIRA_PAT` | Personal Access Token (Server/DC) | *(secret)* |
   | `JIRA_TIMEOUT_MS` | Per-call timeout | `4000` |
   | `JIRA_DEGRADED_MS` | up→degraded latency threshold | `1500` |

4. Implement auth header construction for **both** modes (details below).
5. Write the **connectivity smoke test**: call `GET /rest/api/3/myself` (Cloud)
   or `/rest/api/2/myself` (Server) and print the authenticated account.

### JIRA REST auth: two modes

| | **Cloud** | **Server / Data Center** |
|---|-----------|--------------------------|
| Credential | Email + **API token** | **Personal Access Token (PAT)** |
| HTTP auth | Basic: `base64(email:api_token)` | Bearer: `Authorization: Bearer <PAT>` |
| API base | `/rest/api/3` | `/rest/api/2` |
| Create token | id.atlassian.com → API tokens | JIRA → Profile → Personal Access Tokens |

**Recommendation:** target **Cloud + API token** as the primary path (most
common for hackathon instances, simplest to provision), but keep the auth header
behind a single switch on `JIRA_AUTH_MODE` so Server/DC works by changing config
only.

> **Production note:** in the full system these credentials are owned by the
> **controller's auth broker** (per-user OAuth / token exchange), and the
> connector receives a short-lived token + the caller's identity in `scope`. For
> the hackathon we read a single service token from env — but isolate auth in one
> module so swapping to a broker later is a localized change.

### Files created/touched

```
jira_connector/
  __init__.py
  config.py          # typed settings, env-only secrets
  auth.py            # build auth header for cloud|server (the broker seam)
  client.py          # thin httpx wrapper: base_url, auth, timeout
  smoke_test.py      # GET /myself connectivity check
.env.example         # documents the env vars above (no secrets)
requirements.txt
README -> this file
```

### Definition of done

- [ ] `.env.example` lists every var; real `.env` is git-ignored.
- [ ] Config fails loudly if a required secret is missing.
- [ ] Auth header builds correctly for the selected mode.
- [ ] Smoke test prints the authenticated user from `/myself`.
- [ ] No secret appears in code, logs, or git history.

### Manual verification

```
# expect: 200 + your account displayName/accountId
python -m jira_connector.smoke_test
```
If this fails, **stop** — nothing downstream can work until auth + reachability
are green.

---

## Phase 1 — `health()`

**Goal:** Cheapest proof that auth + reachability work; the connector's liveness
signal for the controller.

### Concrete steps

1. Issue a lightweight authenticated request (reuse `/myself`).
2. Measure latency and map the outcome to a state.
3. Return a small status object (state + latency + checked_at + detail).

### State mapping

| Condition | State |
|-----------|-------|
| 200 within `JIRA_DEGRADED_MS` (default 1500 ms) | `up` |
| 200 but slower than `JIRA_DEGRADED_MS`, OR HTTP **429** (rate-limited) | `degraded` |
| Auth failure (401/403), connection error, or timeout (`JIRA_TIMEOUT_MS`) | `down` |

> The **up→degraded** flip is the latency threshold `JIRA_DEGRADED_MS`. The
> **→down** flip is the hard `JIRA_TIMEOUT_MS` per-call timeout (or any
> auth/connection error). Keep `DEGRADED_MS < TIMEOUT_MS`.

Return shape:

```json
{ "state": "up", "latency_ms": 240, "checked_at": "2026-06-19T10:00:00Z", "detail": null }
```

### Files created/touched

```
jira_connector/health.py
tests/test_health.py        # respx: simulate 200-fast, 200-slow, 429, 401, timeout
```

### Definition of done

- [ ] Returns exactly one of `up | degraded | down`.
- [ ] 429 and slow-but-OK both yield `degraded`.
- [ ] 401/403/timeout/connection-error yield `down`.
- [ ] Never raises — always returns a state.

### Manual verification

```
python -c "from jira_connector.health import health; print(health())"   # -> up
```
Then temporarily set a tiny `JIRA_TIMEOUT_MS=1` and confirm it returns `down`.

---

## Phase 2 — `fetch(id)`

**Goal:** Resolve a single ticket key (e.g. `CXC-1234`) into the **normalized
record** — the connector's public data contract.

### Concrete steps

1. `GET /rest/api/3/issue/{key}?fields=...&expand=renderedFields` requesting only
   the fields the schema needs (don't pull everything).
2. Fetch comments (inline via `fields=comment` or `/issue/{key}/comment`).
3. Map raw JIRA JSON → the normalized record below.
4. Attach the **provenance block** (this is mandatory, not optional).

### Normalized record schema (PUBLIC CONTRACT — keep stable)

```json
{
  "id": "CXC-1234",
  "summary": "4.8.2 build fails on pricing module",
  "description": "Full text of the description field...",
  "status": "In Progress",
  "assignee": { "name": "Jane Doe", "account_id": "5b10..." },
  "reporter": { "name": "John Roe", "account_id": "5b11..." },
  "priority": "High",
  "created":  "2026-04-20T09:12:00Z",
  "updated":  "2026-04-22T14:03:00Z",
  "labels":   ["build-failure", "pricing"],
  "links":    [ { "type": "blocks", "id": "CXC-1300" } ],
  "comments": [
    { "author": "Jane Doe", "created": "2026-04-21T08:00:00Z",
      "body": "Reproduced on CI node 3..." }
  ],
  "provenance": {
    "source": "jira",
    "url": "https://acme.atlassian.net/browse/CXC-1234",
    "retrieved_at": "2026-06-19T10:05:00Z"
  }
}
```

**Notes**
- Every field present even if empty (use `null`/`[]`) so consumers don't branch
  on missing keys.
- Timestamps normalized to ISO-8601 UTC.
- `assignee`/`reporter` may be `null` (unassigned).
- `provenance.url` is the human-browsable ticket URL — the citation downstream
  will show this.

### Files created/touched

```
jira_connector/schema.py    # the normalized record/chunk definitions (the contract)
jira_connector/fetch.py     # raw issue JSON -> normalized record
tests/fixtures/issue_CXC-1234.json   # captured real response (see fixtures note)
tests/test_fetch.py
```

### Definition of done

- [ ] `fetch("CXC-1234")` returns a record matching the schema exactly.
- [ ] Comments, labels, links, assignee/reporter all mapped.
- [ ] Provenance block present with valid `url` + `retrieved_at`.
- [ ] Unknown/not-found key returns a typed `not-found` error (see taxonomy), not
      a crash.

### Manual verification

```
python -c "import json,jira_connector.fetch as f; print(json.dumps(f.fetch('CXC-1234'), indent=2))"
```
Eyeball that `summary`, `status`, latest comment, and the `provenance.url` are
correct against the ticket in a browser.

---

## Phase 3 — `search(query, scope)`

**Goal:** Natural-language query → ranked **chunks** (not raw tickets),
ACL-scoped at the source, each chunk carrying provenance.

### NL → JQL strategy (start simple)

For 24h, keyword + filters is enough; semantic re-ranking is a later layer.

1. Extract keywords from `query` (stopword-strip; keep quoted phrases).
2. Build JQL combining keyword `text ~` search with `scope` filters:

   ```sql
   text ~ "build failure pricing"
     AND project IN (CXC)            -- from scope
     AND status IN ("Open","In Progress")   -- optional from scope/query
   ORDER BY updated DESC
   ```

3. Page results (`maxResults`, cap for the demo, e.g. 20 tickets).
4. **Note for later:** the JQL keyword hit-set becomes the candidate pool that a
   semantic re-ranker (embeddings over chunks) would re-order. Keep relevance
   scoring isolated so it can be swapped from lexical → semantic.

### `scope` applies ACL at the source

`scope` carries the caller's allowed projects / constraints. **Inject them into
the JQL** (`project IN (...)`, label/visibility filters) so out-of-scope tickets
are **never fetched**. Example `scope`:

```json
{ "projects": ["CXC"], "statuses": ["Open","In Progress"], "max_results": 20 }
```

> Access control is enforced in the query, not by filtering results afterward.
> In production the controller/privacy layer supplies `scope` from the caller's
> identity; here we pass it explicitly.

### Return RANKED CHUNKS (not raw tickets)

Long fields (`description`, each comment) are chunked; short tickets may be a
single chunk. Each chunk is independently citable.

```json
[
  {
    "chunk_id": "CXC-1234#desc#0",
    "text": "The 4.8.2 build fails during the pricing module link step...",
    "score": 0.87,
    "provenance": {
      "source": "jira",
      "ticket": "CXC-1234",
      "field": "description",
      "url": "https://acme.atlassian.net/browse/CXC-1234",
      "retrieved_at": "2026-06-19T10:08:00Z"
    }
  },
  {
    "chunk_id": "CXC-1234#comment#2",
    "text": "Reproduced on CI node 3; rollback of PR #812 fixes it.",
    "score": 0.81,
    "provenance": { "source": "jira", "ticket": "CXC-1234",
      "field": "comment", "url": "...#comment-...", "retrieved_at": "..." }
  }
]
```

**Ranking (24h baseline):** lexical score (JIRA text-match order) blended with
recency (`updated`). Document the scoring so it's swappable. Return chunks sorted
by `score` descending.

### Files created/touched

```
jira_connector/jql.py       # query + scope -> JQL string
jira_connector/chunk.py     # field text -> chunks (size/overlap), attach provenance
jira_connector/rank.py      # baseline lexical+recency scoring (swap seam)
jira_connector/search.py    # orchestrates: jql -> issues -> chunks -> ranked
tests/fixtures/search_buildfail.json
tests/test_search.py        # incl. a scope-enforcement test (out-of-project excluded)
```

### Definition of done

- [ ] NL query produces valid JQL with `scope` filters applied.
- [ ] Out-of-scope projects are **absent from results** (proven by test).
- [ ] Output is a ranked list of chunks, each with provenance + `score`.
- [ ] Long descriptions/comments are chunked, not returned whole.

### Manual verification

```
python -c "import jira_connector.search as s; \
  print(s.search('latest on CXC 4.8.2 build failure', {'projects':['CXC']})[0])"
```
Confirm the top chunk is from a relevant ticket and its `provenance.url` resolves.
Then run with a `scope` for a project you can't see and confirm zero results.

---

## Phase 4 — Hardening & demo

**Goal:** Make the three methods robust and demoable. (This is **P1** — see time
budget; do it only after P0 works end to end.)

### Concrete steps

1. **Per-call timeout** — enforce `JIRA_TIMEOUT_MS` on every request.
2. **Graceful degradation** — on rate-limit/slowness, `health()` reports
   `degraded`; `search`/`fetch` return partial results or a typed error rather
   than crashing.
3. **Basic caching** — in-memory TTL cache keyed by normalized query (and by key
   for `fetch`). Enough to make repeated demo queries instant; not a real cache
   layer.

### Error taxonomy (what each returns)

| Class | Trigger | `fetch`/`search` result | `health` |
|-------|---------|-------------------------|----------|
| `auth` | 401 / 403 | typed `auth` error, no data | `down` |
| `rate-limit` | 429 | cached result if available, else typed `rate-limit` error | `degraded` |
| `not-found` | 404 / unknown key | typed `not-found`, empty result | n/a |
| `timeout` | exceeds `JIRA_TIMEOUT_MS` | typed `timeout` error | `down` |
| `upstream` | 5xx | typed `upstream` error | `degraded`/`down` |

Errors are **returned as typed objects**, never leaked as raw stack traces — the
controller needs a stable failure shape.

### Demo queries (against real tickets)

| # | Query | `scope` | Expected output shape |
|---|-------|---------|-----------------------|
| 1 | "latest on the CXC 4.8.2 build failure" | `{projects:["CXC"]}` | ranked chunks; top chunk from the build-failure ticket w/ provenance URL |
| 2 | `fetch("CXC-1234")` | — | full normalized record incl. comments + provenance |
| 3 | "open high-priority escalations in CXC" | `{projects:["CXC"],statuses:["Open"]}` | chunks only from Open CXC tickets; out-of-scope absent |

### Fixtures note (so the demo doesn't depend on live network)

During Phases 2–3, **capture 3–5 real JIRA responses** (one `/myself`, one issue,
one search) into `tests/fixtures/*.json`. Replay them with `respx` in tests and
keep a `--offline` demo path. This means a flaky conference network can't break
the live demo. Capturing fixtures is cheap insurance, not gold-plating.

### Definition of done

- [ ] Every method honors the per-call timeout.
- [ ] Each error class returns its typed shape (table above).
- [ ] Repeated identical query is served from cache.
- [ ] All three demo queries produce the expected shapes — live **and** from
      fixtures offline.

### Manual verification

Run each demo query live, then disconnect network and run the offline/fixture
path — outputs should match in shape.

---

## Gold-plating — explicitly OUT for 24h

Skip these unless P0+P1 are fully done:

- Semantic/embedding re-ranking inside the connector (lexical is fine; re-ranking
  is an upstream layer).
- Persistent/Redis cache (in-memory TTL is enough).
- Webhook/push or incremental sync (pull-on-demand only).
- OAuth 3LO / per-user token exchange (env service token + a clean auth seam).
- Attachment download/parsing, sprint/board APIs, custom-field mapping beyond the
  schema.
- Full pagination of huge result sets (cap results for the demo).

---

## Phase order & time budget (24h)

| Phase | Focus | Priority | Budget |
|-------|-------|----------|--------|
| 0 | Setup & Auth + smoke test | **P0** | 3h |
| 1 | `health()` | **P0** | 2h |
| 2 | `fetch(id)` + normalized schema | **P0** | 5h |
| 3 | `search(query, scope)` → ranked chunks, ACL in JQL | **P0** | 7h |
| — | Capture fixtures (during P2/P3) | P0 | (folded in) |
| 4 | Hardening: timeouts, errors, cache | **P1** | 4h |
| — | Demo prep + dry run (live + offline) | P1 | 2h |
| — | Buffer / overflow | — | 1h |

**P0 = health + fetch + search working end-to-end against real tickets.** Phase 4
hardening is P1. If time is short, ship P0 with the demo running off fixtures.
