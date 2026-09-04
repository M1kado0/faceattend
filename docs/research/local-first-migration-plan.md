# FaceAttend Local-First Migration Plan

**Last updated:** 2026-09-04  
**Current phase:** Phase 2 — CV-core extraction and head-pose validation  
**Overall status:** IN_PROGRESS

This is the tracking checklist for the approved local-first hybrid direction:
PySide6 desktop UI + headless Python CV/application core + SQLite + evaluation
tools. The existing web implementation remains preserved until parity is
verified and removal is explicitly approved.

## Status rules

- `COMPLETED` — work is implemented and verified by the stated checks.
- `IN_PROGRESS` — the current active workstream.
- `PENDING` — not started, but required later.
- `BLOCKED` — cannot proceed without a named decision or external dependency.
- `DEFERRED` — intentionally postponed; not part of the current critical path.

Only one phase should be `IN_PROGRESS` at a time. Update this file after each
implementation turn with the date, files changed, checks run, and the next gate.

## Phase checklist

### Phase 0 — Protect and characterize the existing project

**Status: COMPLETED**

- [x] Preserve the dirty working tree and record unfinished liveness changes.
- [x] Run baseline tests, lint, type checking, and coverage.
- [x] Correct the recursive `models/` ignore collision.
- [x] Define tracking rules for instructions, ADRs, research notes, and durable
      Graphify artifacts.
- [x] Confirm existing tests cover liveness ordering, model-version filtering,
      identity rejection, duplicate attendance, deletion, and audit calls.

**Gate:** Existing behavior is documented and no user work was overwritten.

### Phase 1 — Define the headless package boundary

**Status: COMPLETED (initial foundation)**

- [x] Add `src/faceattend/` package discovery.
- [x] Add explicit registration and attendance workflow states.
- [x] Enforce active-liveness then passive-liveness ordering in state contracts.
- [x] Add typed frame, face, quality, pose, liveness, embedding, match,
      registration, and attendance values.
- [x] Add replaceable protocols for detection, alignment, embedding, head pose,
      active/passive liveness, analysis, and matching.
- [x] Add headless contract tests and current face-count characterization tests.

**Gate:** 65 tests pass; Ruff passes; `mypy src` passes; no runtime was removed.

### Phase 2 — Extract and validate the CV core

**Status: IN_PROGRESS**

- [ ] Validate head pose using MediaPipe transformation matrices.
- [ ] Implement and compare calibrated canonical-3D `solvePnP` pose estimation.
- [ ] Test pose signs, units, neutral/turn directions, smoothing, and
      hysteresis.
- [ ] Add temporal face continuity and fail-closed substitution checks.
- [ ] Extract model lifecycle behind headless detector, aligner, embedder, and
      passive-PAD adapters.
- [ ] Remove misleading embedding model metadata and preserve model checksums.
- [ ] Keep current MiniFASNetV2 as a replaceable baseline pending evaluation.

**Next gate:** Head-pose tests pass on synthetic/controlled landmark fixtures;
then extract existing CV behavior without importing FastAPI or Qt.

### Phase 3 — Add local SQLite persistence

**Status: PENDING**

- [ ] Define migrations and enable SQLite foreign keys/WAL where appropriate.
- [ ] Add people, consent, enrollment, templates, sessions, attendance,
      liveness, model/configuration, and audit records.
- [ ] Store normalized float32 embeddings as versioned BLOBs with quality and
      pose metadata; do not store raw images by default.
- [ ] Implement deletion, uniqueness constraints, transactions, and restart
      reload tests.
- [ ] Implement exact NumPy matching with threshold and ambiguity-margin policy.
- [ ] Keep a matcher seam for measured future FAISS adoption.

### Phase 4 — Build the PySide6 application shell

**Status: PENDING**

- [ ] Add Qt Widgets UI without placing model logic in widgets.
- [ ] Give one camera worker exclusive `cv2.VideoCapture` ownership.
- [ ] Give one inference worker ownership of loaded models.
- [ ] Use a capacity-one latest-frame buffer and Qt signals/slots.
- [ ] Keep preview and inference rates independent; drop stale frames.
- [ ] Add explicit cancellation, shutdown, camera-loss, and model-failure states.
- [ ] Keep headless tests independent of Qt.

### Phase 5 — Implement enrollment

**Status: PENDING**

- [ ] Require one tracked face with quality gates.
- [ ] Add randomized ordered active challenge with neutral transitions,
      dwell, timeout, and continuity checks.
- [ ] Make circular sectors represent measured pose bins, never timer progress.
- [ ] Run temporal passive PAD on selected frames or a short window.
- [ ] Capture 3–5 diverse normalized templates and persist consent/audit data.
- [ ] Confirm ordinary successful/cancelled enrollment leaves no raw frames.

### Phase 6 — Implement attendance check-in

**Status: PENDING**

- [ ] Preserve mandatory active + passive liveness for the first desktop version.
- [ ] Add quality, no-face, multiple-face, continuity, unknown, ambiguous,
      duplicate, camera-failure, and model-failure states.
- [ ] Match only registered consenting identities using calibrated thresholds.
- [ ] Record attendance only after both liveness checks and an explicit action.
- [ ] Add integration tests for every terminal outcome and audit event.

### Phase 7 — Build reproducible evaluation

**Status: PENDING**

- [ ] Recognition: genuine/impostor scores, ROC/DET, FMR/FNMR, EER, rank-1,
      unknown rejection, ambiguity, and condition breakdowns.
- [ ] PAD: APCER by attack type, BPCER, ACER where comparable, failure-to-
      process, cross-camera/domain results, and latency.
- [ ] Active liveness: completion, timeout, replay/print acceptance, pose
      stability, substitution resistance, and completion time.
- [ ] Runtime: FPS, dropped frames, stage p50/p95/p99, cold/warm startup,
      memory, CPU, and end-to-end workflow time.
- [ ] Record hardware, camera, model hashes, configuration, thresholds, split,
      seed, and commit for every result.

### Phase 8 — Verify parity and portfolio quality

**Status: PENDING**

- [ ] Demonstrate enrollment, liveness-gated check-in, matching, unknown and
      ambiguity rejection, duplicate handling, deletion, audit, and history.
- [ ] Compare desktop results with the preserved web pipeline.
- [ ] Meet or document progress toward >80% core-logic coverage.
- [ ] Produce an evidence-backed demo sequence and limitations statement.

### Phase 9 — Retire obsolete architecture

**Status: DEFERRED — EXPLICIT APPROVAL REQUIRED**

- [ ] Preserve historical web/crawler context in documentation and Graphify.
- [ ] Obtain approval before deleting or mass-moving frontend, FastAPI, service,
      crawler, PostgreSQL, Redis, MinIO, Qdrant, or deployment scaffolding.
- [ ] Remove only after desktop parity, reproducibility, and rollback are
      demonstrated.

## Current decision record

- **Architecture:** local-first hybrid; desktop UI is primary, headless core is
  reusable, current web runtime is preserved temporarily.
- **Liveness policy:** active and passive are mandatory during registration and
  check-in until a separate threat-model ADR changes it.
- **Storage:** SQLite metadata/audit plus versioned float32 embeddings; no raw
  enrollment images by default.
- **Search:** exact normalized NumPy search initially; FAISS only if measured
  scale justifies it.
- **Models:** current InsightFace and MiniFASNet remain baselines, not proven
  final choices; licensing and target-camera evaluation remain open.

## Latest execution record

**2026-09-04 — Phase 0 and initial Phase 1 foundation completed**

- Tests: 65 passed (59 original + 6 focused tests).
- Ruff: passed.
- `mypy src`: passed.
- Full `mypy .`: five pre-existing errors remain in four files.
- Headless unit coverage: 93% under the focused command.
- Graphify code graph synchronized; completion evidence saved to Graphify
  memory.
- No GUI, SQLite migration, model extraction, or destructive cleanup performed.

## Update template

Copy this block into the latest execution record after each turn:

```text
DATE — Phase N — STATUS
- Outcome:
- Files changed:
- Tests/checks:
- Evidence or measurements:
- Known limitations:
- Graphify/documentation update:
- Next gate:
- Approval required:
```
