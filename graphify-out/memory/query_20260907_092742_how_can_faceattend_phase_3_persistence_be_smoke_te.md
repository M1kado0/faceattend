---
type: "query"
date: "2026-09-07T09:27:42.270683+00:00"
question: "How can FaceAttend Phase 3 persistence be smoke-tested before Phase 4?"
contributor: "graphify"
---

# Q: How can FaceAttend Phase 3 persistence be smoke-tested before Phase 4?

## Answer

Run uv run python scripts/test_local_persistence.py. It creates a fresh temporary SQLite database and uses three synthetic normalized vectors to verify template persistence across restart, same-person exact matching, unknown rejection, liveness-gated attendance, duplicate idempotency, person/template erasure, and preserved append-only audit history. It stores no camera image or real biometric embedding. Full verification reached 253 passing tests.