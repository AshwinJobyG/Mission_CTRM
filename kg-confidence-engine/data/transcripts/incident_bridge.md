# NGPOWER-145 incident bridge — 28 May 2026
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
