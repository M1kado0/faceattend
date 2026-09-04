# Local-First Desktop Migration Progress

## Phase 0 and Initial Phase 1 Slice — 2026-09-04

### Outcome

The project owner accepted ADR-001 and authorized an incremental migration to a
PySide6 desktop product over a headless Python CV/application core. The current
web implementation remains intact. This slice establishes repository safety,
records the pre-change baseline, and introduces only framework-independent
types, interfaces, and workflow states.

### Preserved Dirty State

The pre-change working tree contained user work in backend routes, frontend
attendance/liveness files, ML serving and liveness modules, tests, and README.
`ml/liveness/deepfake_detect.py` was deleted and `ml/liveness/head_pose.py` was
untracked. None of those changes were overwritten or promoted as validated.

### Baseline Evidence

Commands were run before migration edits with
`UV_CACHE_DIR=/tmp/faceguard-uv-cache`:

- `uv run pytest`: 59 passed, 3 third-party deprecation warnings.
- `uv run ruff check .`: passed.
- `uv run mypy .`: failed before checking the full tree. Reported missing type
  information for `jose`, `passlib`, and `faiss`, plus duplicate discovery of
  `frontend/public-site/services/api_client.py` as `api_client` and
  `services.api_client`.
- `uv run pytest --cov=backend --cov=ml --cov=frontend --cov=tests
  --cov-report=term-missing`: 59 passed; 60% aggregate measured coverage. This
  includes test modules inside covered packages and is not yet the desired
  core-logic-only coverage metric.

These are baseline limitations, not migration regressions.

### Repository Tracking Policy

- Ignore only the root `/models/` artifact directory, preventing model weights
  from being committed without hiding `backend/db/models/` source code.
- Track `AGENTS.md`, `.github/`, and `docs/` as project instructions and durable
  architecture/research records.
- Track `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.json`, and Markdown
  query memory. Keep generated Graphify visualizations and caches local.
- Continue ignoring `.agents/`, `.codex/`, `.mcp.json`, and `images/` as local
  tooling/configuration or sensitive media.

### Headless Boundary Added

The first `src/faceattend/` slice defines:

- explicit registration and attendance workflow states;
- transition guards that prevent matching/template capture before active and
  passive liveness;
- typed frame, face, quality, pose, liveness, embedding, match, registration,
  and attendance results;
- protocols for detection, alignment, embedding, head pose, active/passive
  liveness, analysis, and matching.

No protocol imports Qt, FastAPI, or persistence code. No model behavior has
been extracted yet.

### Characterization Coverage

Existing tests already characterize liveness-before-embedding, model-version
filtering, identity-not-matched behavior, duplicate attendance, registration
deletion, and relevant audit calls. New characterization tests lock the current
no-face, multiple-face, and exactly-one-face behavior without changing the
existing ML implementation.

### Next Gate

Post-change verification produced:

- `uv run pytest`: 65 passed, adding six focused tests without regressing the
  original 59;
- `uv run ruff check .`: passed;
- `uv run mypy src`: passed for all six new headless source files;
- `uv run mypy .`: the same five baseline errors in four files remained;
- `uv run pytest --cov=faceattend --cov-report=term-missing tests/unit`: 93%
  for the initial headless package. Protocol-only method bodies account for the
  uncovered lines.

The repository is ready for Phase 2 extraction at the interface level. Phase 2
must first correct and validate head pose; the untracked implementation must not
be copied unchanged. Repository readiness does not imply that current CV model
quality, thresholds, or liveness security have been validated.
