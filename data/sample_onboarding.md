# Onboarding: New Engineer Setup

**Owner:** Platform Team · **Last updated:** 2026-05-10

New engineers should complete these steps in their first week:

1. Request access to the internal GitLab via the IT portal (approval ~1 day).
2. Install the standard toolchain: Python 3.11, Docker, and the `ctrm-cli`.
3. Clone the `mission-ctrm` repository and run `make setup`.
4. Read the architecture overview in Confluence space "CTRM-ARCH".
5. Pair with your onboarding buddy on a starter ticket within 3 days.

## Common pitfalls

- VPN must be connected before requesting database credentials.
- The staging environment resets every night at 02:00 UTC; do not rely on
  persisted test data there.
