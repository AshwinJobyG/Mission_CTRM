# Backlog grooming — 14 May 2026
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
