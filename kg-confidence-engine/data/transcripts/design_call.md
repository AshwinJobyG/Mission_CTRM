# Persistence / volume-generation design call — 8 May 2026
**Attendees:** Anupam Sengupta (tech lead), Rinit Jain (dev)

**Anupam:** The core design issue is that persistence and volume generation
share one transaction. NGPOWER-49 makes persistence its own transaction — that
is the safeguard. Without it, a failover between generation and commit drops the
trade silently.

**Rinit:** So 49 is really the mitigation for the coupling risk, not just a
refactor.

**Anupam:** Exactly. Treat 49 as the risk mitigation, not a nice-to-have.
