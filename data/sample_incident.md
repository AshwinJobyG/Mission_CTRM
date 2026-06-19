# Incident Report: INC-2041 — Pricing Service Outage

**Owner:** Trading Platform Team · **Date:** 2026-04-22 · **Severity:** High

## Summary

The commodities pricing service returned stale quotes for 38 minutes after a
cache node failed and the fallback did not trigger.

## Root cause

The Redis health-check timeout was set higher than the request timeout, so the
client kept routing to the dead node instead of failing over.

## Resolution

- Lowered the health-check timeout below the request timeout.
- Added an alert on cache-node heartbeat gaps.

## Lessons learned

- Always set health-check timeouts shorter than request timeouts.
- Synthetic checks should validate *freshness* of quotes, not just liveness.
