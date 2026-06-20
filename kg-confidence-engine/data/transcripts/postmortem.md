# NGPOWER-145 postmortem — 29 May 2026
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
