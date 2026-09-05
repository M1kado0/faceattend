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

- [ ] Validate MediaPipe transformation-matrix extraction against actual
      Face Landmarker output. The current legacy options disable matrix output.
- [x] Implement canonical-3D `solvePnP` pose estimation with documented
      camera-intrinsic fallback.
- [x] Test pose signs, degree units, neutral/turn directions, smoothing, and
      hysteresis with deterministic synthetic fixtures.
- [x] Add temporal face continuity and fail-closed substitution checks.
- [ ] Extract model lifecycle behind headless detector, aligner, embedder, and
      passive-PAD adapters.
- [ ] Remove misleading embedding model metadata and preserve model checksums
      during adapter extraction.
- [ ] Keep current MiniFASNetV2 as a replaceable baseline pending evaluation.

**Next gate:** Compare the new estimator with actual MediaPipe transformation
matrices, then extract existing CV behavior without importing FastAPI or Qt.

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

**2026-09-04 — Phase 2 pose and continuity foundation (IN_PROGRESS)**

- Added headless canonical `solvePnP` and transformation-matrix pose helpers;
  both return documented degree-valued yaw, pitch, and roll.
- Added deterministic projections recovering neutral, yaw, pitch, and roll
  within 0.2 degrees, matrix extraction tests, invalid-input tests,
  exponential smoothing, and enter/exit hysteresis tests.
- Added `FaceContinuityTracker` with bounded no-face gaps, monotonic timestamps,
  track-ID checks, multiple-face rejection, center-jump limits, and area-change
  limits.
- Added `LazyModel` and model metadata contracts; concrete InsightFace and
  MiniFASNet adapters remain pending.
- Verification: 80 tests passed; Ruff passed; `mypy src` passed; full `mypy .`
  retained the same five baseline errors; focused headless coverage is 92%.
- Evidence boundary: synthetic pose tests validate implementation mathematics,
  not camera accuracy. Actual MediaPipe matrix output and target-camera error
  are still unmeasured. Phase 2 remains `IN_PROGRESS`.

**2026-09-04 — Controlled pose-validation helper (IN_PROGRESS)**

- Added a headless validation summary that compares estimated yaw/pitch/roll
  against controlled reference poses and reports per-axis mean/max error and
  pass rate at a chosen tolerance.
- Added tests for passing samples, empty datasets, invalid tolerances, and
  non-finite estimates.
- Evidence boundary: this helper evaluates recorded/reference pose estimates;
  it does not collect webcam data or establish real-camera accuracy.
- Phase 2 remains `IN_PROGRESS`; a real-camera fixture and MediaPipe matrix
  comparison are still required.

**2026-09-04 — Model lifecycle and detector adapter example (IN_PROGRESS)**

- Added a thread-safe lazy model loader and an injectable `InsightFaceDetector`
  adapter under the headless vision package.
- The adapter converts raw InsightFace faces into typed `FaceObservation`
  values and exposes model metadata without importing Qt, FastAPI, or storage.
- Added fake-model tests proving one-time loading and field conversion without
  downloading model weights.
- The adapter's provisional quality values are detection placeholders; a
  dedicated quality policy remains to be extracted and validated.
- Verification: focused detector/pose tests passed, Ruff passed, and `mypy src`
  passed. Phase 2 remains `IN_PROGRESS`.

**2026-09-04 — Direct InsightFace recognition adapter (IN_PROGRESS)**

- Replaced the partial embedding function with `InsightFaceEmbedder`, which
  accepts an aligned HWC `uint8` BGR image and calls InsightFace's recognition
  model directly through `get_feat`.
- Added one-time lazy loading, L2 normalization, finite/zero-output checks,
  injectable model loading, and explicit model metadata.
- Verification: embedding, detector, and pose tests passed (23 total); Ruff
  and `mypy src` passed. Real `buffalo_l` inference and checksum discovery are
  still pending; Phase 2 remains `IN_PROGRESS`.

**2026-09-04 — Five-point alignment adapter (IN_PROGRESS)**

- Replaced the incomplete alignment draft with `InsightFaceAligner` using
  InsightFace's ArcFace `norm_crop` convention.
- The adapter validates an HWC `uint8` BGR frame and exactly five finite 2D
  landmarks, then returns a contiguous 112x112 crop for the embedder.
- Added alignment shape, dtype, invalid-landmark, and invalid-frame tests.
- Verification: alignment, embedding, and detector tests passed (13 total);
  Ruff and `mypy src` passed. Phase 2 remains `IN_PROGRESS`.

**2026-09-04 — Headless recognition smoke-test script (IN_PROGRESS)**

- Added `scripts/test_headless_pipeline.py` to exercise a local image through
  detector → five-point alignment → direct recognition embedding.
- The script prints face count, detector score, landmark shape, aligned image
  shape, embedding dimension/dtype/norm, and model metadata; it writes no image
  or embedding data.
- Verification: Ruff passed. Real execution requires a local `buffalo_l`
  detector pack, one test image, and the recognition ONNX path.

**2026-09-05 — Direct detector model ownership (IN_PROGRESS)**

- Changed `InsightFaceDetector` to load only `models/det_10g.onnx` through the
  InsightFace model-zoo adapter instead of constructing `FaceAnalysis`, which
  loaded unrelated landmark, gender/age, and recognition models.
- Updated the smoke test to accept `--detector-model`; the embedder remains the
  sole owner of `models/w600k_r50.onnx`.
- Verification: detector, alignment, and embedding tests passed (13 total);
  Ruff and `mypy src` passed; the real local image smoke test completed with
  one face, five keypoints, a 112x112 crop, and a normalized 512-D embedding.
- Limitation: detector quality fields remain provisional and model checksums
  are still supplied as configuration rather than discovered automatically.

**2026-09-05 — Visual pipeline diagnostics (IN_PROGRESS)**

- Extended `scripts/test_headless_pipeline.py` with optional OpenCV previews and
  diagnostic image output for the original frame, detected landmarks, and
  aligned crop.
- The embedding stage remains numeric; the script prints its dimension, dtype,
  norm, and model metadata instead of fabricating an image visualization.
- Default execution remains non-visual and writes no files.

**2026-09-05 — Temporary diagnostic image output (IN_PROGRESS)**

- The smoke-test script now saves original, detected, and aligned images to a
  generated temporary directory by default and prints only that directory
  path, not image data.
- `--save-dir` remains available when a persistent diagnostic location is
  explicitly desired; `--show` remains optional for interactive OpenCV windows.

**2026-09-05 — MiniFASNet passive-liveness adapter (IN_PROGRESS)**

- Added a headless `MiniFASNetPassiveLivenessDetector` that wraps the existing
  MiniFASNetV2 implementation without changing its preprocessing or score
  semantics.
- The adapter accepts aligned workflow frames plus one face per frame, lazily
  loads the model, exposes model metadata, and returns typed passive-liveness
  evidence.
- Temporal windows use the minimum frame score, so one failed frame cannot be
  hidden by a higher-scoring frame.
- Added fake-model tests for pass/fail windows, one-time loading, invalid input,
  and invalid scores. Verification: 18 focused tests passed; Ruff and
  `mypy src` passed. Real MiniFASNet weights and attack-protocol evaluation
  remain pending.

**2026-09-05 — Real MiniFASNet smoke test (IN_PROGRESS)**

- Loaded `models/MiniFASNetV2.onnx` successfully through the headless adapter
  and evaluated one detected face from an existing local static image.
- Result: decision `failed`, score `0.0811316`, threshold `0.85`.
- Evidence boundary: this static image is not a bona-fide live-camera sample,
  so the result only confirms model loading and output plumbing. It is not a
  model-quality or threshold-calibration result.
- Next evidence required: a consented live-camera bona-fide sample and
  attack-specific print/display/replay samples under a documented protocol.

**2026-09-05 — Webcam headless-pipeline smoke-test mode (IN_PROGRESS)**

- Extended `scripts/test_headless_pipeline.py` with a webcam mode using
  `--camera`, `--camera-index`, and `--frames`. It captures a bounded sequence,
  requires exactly one detected face per frame, evaluates the passive-liveness
  window, then aligns and embeds the final frame.
- The script now adds both the repository root and `src/` to `sys.path`, so it
  can use the new `faceattend` adapters together with the existing root-level
  `ml` MiniFASNet implementation when run directly with `uv`.
- Image mode was re-run successfully: one face, five landmarks, a 112x112
  aligned crop, and a normalized 512-D embedding. MiniFASNet returned
  `failed` with score `0.081132` against threshold `0.85`; this remains a
  plumbing/static-image smoke result, not liveness quality evidence.
- Verification: Ruff passed for the script. No webcam run was performed in
  this turn; a consented live-camera sample is still required.
- Next gate: run the webcam mode locally, inspect frame-count/face-continuity
  behavior, and collect bona-fide plus attack samples under a documented
  protocol before changing thresholds or liveness policy.

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
