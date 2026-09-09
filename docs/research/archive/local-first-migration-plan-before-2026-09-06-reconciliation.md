# Historical migration-plan snapshot

Archived on 2026-09-06 before checklist reconciliation. This preserves the
previous plan and execution records verbatim below; completion claims and
“next gate” statements are historical, not current verification.

Use [the current plan](../local-first-migration-plan.md) for authoritative
statuses. In particular, the former frame-evidence and active-evaluation
completion claims were narrowed after inspecting their runtime integration.

---

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

- [x] Validate MediaPipe transformation-matrix extraction against actual
      Face Landmarker output; camera phase captures remain calibration evidence.
- [x] Implement canonical-3D `solvePnP` pose estimation with documented
      camera-intrinsic fallback.
- [x] Test pose signs, degree units, neutral/turn directions, smoothing, and
      hysteresis with deterministic synthetic fixtures.
- [x] Add temporal face continuity and fail-closed substitution checks.
- [x] Extract model lifecycle behind headless detector, aligner, embedder, and
      passive-PAD adapters.
- [x] Remove misleading embedding model metadata and compute model checksums
      when configured model files are present.
- [x] Keep current MiniFASNetV2 as a replaceable baseline pending evaluation.
- [x] Add calibration-aware active-liveness phases with configurable baseline,
      hysteresis, dwell, timeout, and fail-closed outcomes.
- [x] Add typed per-frame `FrameEvidence` shared by active evaluation and
      temporal passive-PAD processing.
- [x] Connect MediaPipe action evidence to the session-backed evaluator and
      bounded temporal PAD session.
- [x] Add a headless camera worker that owns OpenCV capture and forwards frames
      into an evidence processor without importing Qt.

**Next gate:** Connect the evidence processor to real camera/MediaPipe runtime,
then collect labeled bona-fide and attack PAD trials. Qt GUI integration remains
Phase 4 work.

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

## Randomized active-liveness plan

**Status: IN_PROGRESS**

The first implementation will use randomized active challenges. The green-oval
coverage experience is deferred and is not a security or enrollment gate.

### Challenge session design

**Status: IN_PROGRESS**

- [x] Define `BLINK`, `TURN_LEFT`, and `TURN_RIGHT` as the initial challenges.
- [x] Add `LOOK_UP` and `SMILE` as supported, opt-in challenge actions.
- [ ] Keep `LOOK_UP`, `LOOK_DOWN`, and `SMILE` disabled until separately
      validated.
- [x] Generate a fresh challenge sequence for every session.
- [x] Use secure per-session randomness in production and deterministic seeds only
      in tests.
- [x] Store the generated sequence in session and audit metadata.
- [x] Prevent immediate duplicate challenges.
- [x] Define per-challenge and total-session timeouts.

### Frame-evidence boundary

**Status: COMPLETE**

- [x] Represent each frame with timestamp, face count, tracking identity,
      quality, lighting, PAD score, motion, and optional embedding evidence.
- [x] Reject repeated or backwards timestamps.
- [x] Reject no-face and multiple-face frames.
- [x] Reject face substitution and implausible tracking jumps.
- [x] Add duplicate-frame detection as a replay/camera-failure signal.
- [x] Keep the evidence/session logic independent of Qt and FastAPI.

Evidence boundary note: completion means the headless contract and tests are
implemented. Real-camera attack validation and Qt wiring remain later gates.

**2026-09-05 — Frame-evidence boundary completion (MEASURED)**

- Outcome: Completed the typed evidence contract, validation rules, temporal
  continuity integration, and duplicate-frame signal.
- Files changed: `src/faceattend/vision/types.py`,
  `src/faceattend/vision/evidence.py`, MediaPipe/PAD adapters, and tests.
- Tests/checks: 54 focused evidence/session/PAD tests passed; Ruff and
  `mypy src` passed.
- Evidence or measurements: deterministic tests cover all checklist failure
  modes. No biometric images were added.
- Known limitations: duplicate fingerprints are a warning signal, not complete
  replay protection; real camera attack trials remain pending.
- Next gate: connect a real MediaPipe processor to the camera worker and
  collect labeled PAD trials.

**2026-09-05 — Active challenge evaluation completion (MEASURED)**

- Outcome: Completed the headless active evaluator contract with operational
  MediaPipe-matrix pose input, explicit boundary yaw inversion, neutral gating
  before and between directional actions, hysteresis, dwell, valid-frame count,
  blink transition evidence, and fail-closed terminal states.
- Files changed: `src/faceattend/vision/active_liveness.py` and focused tests.
- Tests/checks: 48 active/session/MediaPipe tests passed; Ruff and `mypy src`
  passed.
- Evidence or measurements: deterministic tests cover neutral gating, ordered
  turns, look-up, blink/smile actions, wrong direction, insufficient dwell,
  timeout, no-face, multiple-face, and face substitution.
- Known limitations: camera trials are not a substitute for attack evaluation;
  SolvePnP remains diagnostic-only and real threshold performance is pending.
- Next gate: integrate the evaluator with the camera worker and run repeated
  bona-fide plus labeled replay/print trials.

### Active challenge evaluation

**Status: COMPLETE**

- [x] Use MediaPipe transformation-matrix pose operationally; keep solvePnP
      diagnostic-only.
- [x] Apply the calibrated physical yaw mapping only at the challenge boundary.
- [x] Require neutral before accepting a directional challenge.
- [x] Require return to neutral before the next directional challenge.
- [x] Use enter/exit hysteresis, minimum dwell, and a valid-frame count.
- [x] Detect blinks as open → closed → open transitions.
- [x] Fail closed on timeout, wrong direction, insufficient dwell, face
      substitution, no face, and multiple faces.

Active-evaluation note: completion covers the headless evaluator and its
deterministic tests. Real-camera false-failure and attack-resistance rates are
still evaluation work, not implementation completion.

### Passive PAD and matching order

**Status: IN_PROGRESS**

- [x] Run MiniFASNetV2 over multiple selected frames or a short temporal window.
- [x] Record median score, minimum score, suspicious-frame count, and
      failure-to-process count.
- [x] Keep active and passive liveness mandatory.
- [ ] Complete active liveness before passive PAD, then extract embeddings and
      perform identity matching.
- [x] Reject unknown and ambiguous identities before attendance recording.

Matching note: workflow-state contracts prevent matching before both liveness
stages and make unknown/ambiguous outcomes terminal. The actual local embedding
matcher and attendance persistence path remain pending.

**2026-09-05 — Temporal PAD aggregation and ordering gate (IN_PROGRESS)**

- Outcome: MiniFASNet temporal evaluation now reports median/minimum scores,
  suspicious-frame count, and inference-failure count. Active-plus-passive
  ordering remains enforced by the workflow state machine.
- Files changed: `src/faceattend/vision/passive_liveness.py`, typed evidence
  values, and unit tests.
- Tests/checks: 62 focused tests passed; Ruff and `mypy src` passed.
- Evidence or measurements: deterministic windows cover passing scores,
  suspicious low scores, and per-frame inference failures. Unknown and
  ambiguous attendance states cannot transition to persistence.
- Known limitations: no local embedding matcher or attendance persistence
  service has been implemented yet; real PAD calibration and attack trials are
  still pending.
- Next gate: implement the local matcher/persistence workflow only after
  labeled PAD evaluation is available.

### Headless tests and real-camera validation

**Status: IN_PROGRESS**

- [x] Test deterministic challenge generation and every supported challenge.
- [x] Test neutral gating, wrong direction, dwell, timeout, blink noise,
      continuity, substitution, and duplicate frames.
- [ ] Capture repeated bona-fide trials under varied lighting, distance,
      glasses, and natural movement.
- [ ] Test print, phone/display replay, and prerecorded challenge attacks.
- [ ] Measure completion rate, false-failure rate, latency, and attack
      acceptance rate.
- [x] Keep biometric recordings local and untracked by default.

Validation note: deterministic headless coverage is complete. Real-camera
trials, attack acceptance, and rate/latency measurements remain pending until
consented labeled captures are available. Local trial directories are ignored
by Git (`local-biometric-recordings/` and `pad-trials/`).

**2026-09-05 — Headless validation checkpoint (IN_PROGRESS)**

- Outcome: Added deterministic coverage for all supported challenge actions and
  verified the existing neutral, direction, dwell, timeout, blink-noise,
  continuity, substitution, and duplicate-frame behavior.
- Files changed: challenge-session tests and `.gitignore` recording-directory
  policy.
- Tests/checks: 63 focused tests passed; Ruff and `mypy src` passed; Graphify
  updated.
- Evidence or measurements: only deterministic/synthetic evidence was used.
- Known limitations: varied bona-fide captures, print/display/replay attacks,
  completion and false-failure rates, latency, and attack acceptance remain
  unmeasured.
- Next gate: collect consented labeled local trials and run the evaluation
  protocol without committing biometric recordings.

### Explicitly rejected for the first version

- [ ] Do not use green-oval completion as a security gate.
- [ ] Do not copy proxy-yaw calculations or hard-coded thresholds from examples.
- [ ] Do not use global random seeding in production.
- [ ] Do not accept a session because only some challenges passed.
- [ ] Do not claim Apple Face ID-level security.

**2026-09-05 — Challenge session foundation (IN_PROGRESS)**

- Added a headless `ChallengeSession` that selects a fresh 2–3 action sequence
  from the initial blink/left/right pool using secure randomness by default.
- Added explicit per-challenge and total-session timeout states, strict
  timestamp ordering, exact-sequence completion, reset behavior, and an
  audit-friendly immutable snapshot containing the selected sequence and
  completed actions.
- Kept CV evidence interpretation outside this class; pose, blink, PAD, and
  continuity remain responsibilities of the active-liveness pipeline.
- Verification: 12 focused session/liveness tests passed; Ruff and mypy passed
  for the new module. Full project checks were not rerun in this slice.
- Next gate: connect session lifecycle to the existing active-liveness evaluator
  without weakening active-plus-passive policy, then add frame-evidence and
  neutral/continuity integration tests.

**2026-09-05 — Session-backed challenge evaluator (IN_PROGRESS)**

- Added `LOOK_UP` and `SMILE` challenge actions while keeping them opt-in until
  their camera reliability is validated.
- Connected `ChallengeSession` to `ActiveLivenessChallengeEvaluator` through an
  optional session-backed mode. The session owns randomized sequence lifecycle
  and timeouts; the evaluator owns pose/action evidence, dwell, and fail-closed
  face checks.
- Added integration tests for exact ordered completion with neutral transitions,
  look-up plus smile actions, and face-track substitution rejection.
- Verification: 15 focused session/liveness tests passed; Ruff and `mypy src`
  passed. Full project checks were not rerun in this slice.
- Next gate: add blink evidence integration and frame-evidence/temporal PAD
  boundaries before wiring the session into application services.

## Current decision record

- **Architecture:** local-first hybrid; desktop UI is primary, headless core is
  reusable, current web runtime is preserved temporarily.
- **Liveness policy:** active and passive are mandatory during registration and
  check-in until a separate threat-model ADR changes it.
- **Active liveness:** randomized challenge-response is the first implementation
  target; green-oval pose coverage is deferred as optional future UX.
- **Initial challenge set:** blink and left/right turns; up/down and smile require
  separate validation before activation.
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

**2026-09-05 — MediaPipe pose-matrix comparison (IN_PROGRESS)**

- Enabled `output_facial_transformation_matrixes` in the legacy MediaPipe
  Face Landmarker options so real matrix output is available during active
  liveness processing.
- Added a headless `PoseComparison` result that compares MediaPipe matrix
  Euler angles with the calibrated canonical-face `solvePnP` estimator and
  reports absolute per-axis error.
- Active challenge decisions now use the calibrated `solvePnP` pose. Matrix
  values are diagnostic only until real-camera agreement is measured; direct
  `solvePnP` remains the fallback for runtimes that expose no matrix.
- Added deterministic neutral, left/right yaw, up/down pitch, roll, and
  invalid-input comparison tests. Verification: 36 focused tests passed,
  Ruff passed, and `mypy src` passed.
- Evidence boundary: synthetic rotation matrices validate extraction and
  conventions, not MediaPipe camera accuracy. A consented webcam comparison
  and attack-protocol measurements are still required.
- Next gate: run the active-liveness path with a local Face Landmarker model,
  collect matrix-versus-solvePnP errors for controlled poses, and decide
  whether matrix output can become the primary estimator.

**2026-09-05 — Standalone MediaPipe import fix (IN_PROGRESS)**

- Added repository-root and `src/` path bootstrapping for direct
  `python -m ml.liveness.mediapipe_active` execution.
- Verification: the module starts without the previous `faceattend` import
  error; 36 MediaPipe/head-pose tests passed and Ruff passed.
- Limitation: this module has no command-line video runner, so launching it
  only validates imports. Matrix-versus-`solvePnP` output requires invoking
  `MediaPipeActiveLivenessChecker.check()` with a decoded video or adding a
  dedicated diagnostic script.

**2026-09-05 — Webcam pose-comparison diagnostic (IN_PROGRESS)**

- Added `scripts/compare_head_pose_webcam.py` as a focused diagnostic rather
  than expanding the full detector/liveness smoke test.
- The script captures a bounded webcam sequence, runs MediaPipe Face
  Landmarker with transformation matrices enabled, compares matrix and
  calibrated `solvePnP` yaw/pitch/roll values per frame, and prints mean
  absolute errors. It writes no frames or biometric images.
- It reports no-face, multiple-face, missing-matrix, camera-read, and pose
  errors separately and returns failure when no comparable frame is produced.
- Ruff passed and `--help` ran successfully. A one-frame local smoke attempt
  was blocked by macOS camera permission (`camera access has been denied`);
  no pose measurements were collected.
- Next gate: grant camera permission and run the diagnostic through neutral,
  left/right yaw, up/down pitch, and roll movements; record errors without
  changing the estimator or liveness thresholds yet.

**2026-09-05 — Real stationary pose comparison (IN_PROGRESS)**

- A 60-frame camera-1 run produced stable MediaPipe matrix values around
  yaw `2°`, pitch `-9°`, roll `-3°`, but the solvePnP output reported pitch
  around `173°`, yielding roughly `181°` pitch error while the head was still.
- This is measured evidence that the previous canonical 3D model orientation
  was wrong for image coordinates: eye and mouth vertical positions were
  inverted, creating the alternate upside-down solvePnP solution.
- Corrected the canonical model so image y increases downward (eyes above
  mouth/chin). Deterministic pose tests still pass (21 tests).
- A follow-up camera run in this environment was blocked by macOS camera
  permission, so corrected real-camera error is not yet measured.
- Next gate: rerun `compare_head_pose_webcam.py` after camera permission is
  available and verify stationary pitch is near zero and matrix/solvePnP
  errors are acceptably small before using pose for liveness decisions.

**2026-09-05 — solvePnP mirrored-solution mitigation (IN_PROGRESS)**

- A second 60-frame run showed the previous vertical correction changed the
  failure from a 173° pitch branch to a roughly 177° roll branch. MediaPipe
  remained stable near neutral, so solvePnP was still selecting a mirrored
  solution rather than estimating the camera pose reliably.
- Restored the conventional +y-up canonical face model (which projects eyes
  above mouth/chin in image coordinates) and added a neutral positive-depth
  extrinsic initial guess to iterative `solvePnP`.
- Deterministic pose tests remain green (21 tests) and Ruff passes. A new
  camera measurement is required before claiming the mirrored branch is
  resolved.

**2026-09-05 — solvePnP validation remains failed (IN_PROGRESS)**

- A corrected 60-frame camera-1 run still produced unstable solvePnP poses:
  pitch frequently near `170°` and roll near `177°`, with occasional large
  yaw/roll jumps, while MediaPipe remained stable near neutral.
- Mean absolute errors were approximately yaw `22°`, pitch `168°`, and roll
  `30°`. This is decisive evidence that the current six-point monocular
  solvePnP configuration is not suitable for liveness decisions.
- Do not calibrate thresholds or use these solvePnP angles operationally.
  The next investigation must inspect landmark correspondence, reprojection
  error, camera intrinsics, and rotation-convention conversion; MediaPipe
  matrix output remains the only currently stable runtime signal.

**2026-09-05 — MediaPipe matrix pose becomes operational (IN_PROGRESS)**

- Added `MediaPipeMatrixHeadPoseEstimator` to the headless vision package.
- Updated the MediaPipe turn challenge to use matrix-derived yaw/pitch/roll
  directly. `solvePnP` now runs only for comparison diagnostics and never
  supplies an operational fallback.
- Missing or invalid transformation matrices fail closed with explicit
  `head_pose_unavailable` reasons.
- Added left/right yaw, up/down pitch, and invalid/missing matrix tests in
  both headless pose and MediaPipe active-liveness test suites.
- Verification: 46 focused tests passed; Ruff and `mypy src` passed.
- Required manual evidence remains: run the webcam diagnostic through
  neutral → left → neutral → right → neutral → up → down → roll and record
  MediaPipe values. Threshold tuning is deferred until that evidence exists.

**2026-09-05 — Labeled movement sequence matrix evidence (IN_PROGRESS)**

- A 180-frame camera-1 run covered the requested neutral/turn/pitch/roll
  sequence. MediaPipe matrix output was continuous and responsive: observed
  ranges were approximately yaw `-42°..36°`, pitch `-48°..3°`, and roll
  `-17°..24°`.
- The matrix signal visibly followed the large left/right and pitch changes;
  stationary portions were comparatively smooth. This supports MediaPipe as
  the provisional operational pose signal for active liveness.
- SolvePnP remained invalid (mean errors approximately yaw `21°`, pitch
  `172°`, roll `13°`) and is still diagnostic-only.
- Limitation: the diagnostic does not timestamp manual phase labels, so these
  ranges do not establish exact per-direction thresholds or challenge
  completion rates. Threshold tuning remains pending a labeled protocol with
  neutral dwell and each requested direction recorded separately.

**2026-09-05 — Phase-labeled pose calibration tooling (IN_PROGRESS)**

- Extended `scripts/compare_head_pose_webcam.py` with `--phase` labels
  (`neutral`, `left`, `right`, `up`, `down`, `roll`) and optional `--csv`
  angle-only output. Each run now prints MediaPipe median, min, max, and
  range for the selected phase.
- Added `scripts/summarize_pose_phases.py` to aggregate labeled captures and
  print candidate directional enter/exit thresholds plus a three-frame dwell
  candidate. These are suggestions only and are not applied automatically.
- Added active-challenge tests for directional dwell and near-threshold noise
  rejection. Verification: 48 focused tests passed; Ruff and `mypy src`
  passed.
- Manual work required: capture each phase separately into one CSV, run the
  summary script, then validate candidate thresholds on fresh captures and
  measure bona-fide completion versus false-failure rates.

**2026-09-05 — Initial labeled phase measurements (IN_PROGRESS)**

- Captured neutral, left, right, up, and down phases into `pose.csv`.
- MediaPipe medians were approximately: neutral `(yaw 0.15°, pitch
  -6.86°, roll -1.38°)`; left `(58.03°, -15.45°, -21.66°)`; right
  `(-53.90°, -8.90°, 10.79°)`; up `(-2.35°, -19.66°, 0.47°)`; down
  `(-4.23°, 12.90°, -0.56°)`.
- Directional separation is clear, but the physical labels imply yaw sign is
  reversed relative to the current challenge implementation: user-left was
  positive yaw and user-right negative yaw. This must be resolved through an
  explicit camera/mirror convention before challenge thresholds are changed.
- Roll capture is still missing. Candidate thresholds from the summarizer
  are provisional only; they do not establish false-failure or attack
  acceptance rates.
- The solvePnP comparison errors remain invalid and do not affect this
  MediaPipe measurement.

**2026-09-05 — Roll phase measurements (IN_PROGRESS)**

- Captured separate roll-left and roll-right samples. MediaPipe roll medians
  were approximately `-28.98°` (left tilt; range `-40.58..-12.25°`) and
  `22.11°` (right tilt; range `0.78..45.69°`).
- Roll direction is clearly separable, but yaw/pitch drift was present during
  both captures (left median yaw `12.13°`, right median yaw `-4.41°`). Treat
  these as directional evidence, not final thresholds.
- Roll calibration should gate on the roll axis while rejecting samples with
  excessive simultaneous yaw/pitch or poor face quality. SolvePnP errors remain
  diagnostic-only and irrelevant to this MediaPipe result.

**2026-09-05 — Explicit physical-yaw mapping (IN_PROGRESS)**

- Added `invert_yaw_for_challenge` to `ActiveLivenessConfig`, defaulting to
  `True` from the labeled camera evidence.
- Raw MediaPipe yaw remains unchanged for diagnostics. Only the yaw passed to
  `BlinkTurnLeftRightChallenge.observe()` is mapped at the challenge boundary.
- Added a test covering inverted and non-inverted mappings. Verification: 49
  focused tests passed; Ruff and `mypy src` passed.
- Threshold values and camera-specific mapping still require validation on
  fresh labeled captures and should not be treated as final security policy.

**2026-09-05 — Calibration-aware challenge evaluator (IN_PROGRESS)**

- Added a headless `ActiveLivenessChallengeEvaluator` with configurable
  neutral baseline, directional enter/exit thresholds, dwell duration, and
  timeout. It supports neutral, left/right yaw, up/down pitch, and optional
  left/right roll phases.
- Added explicit outcomes for completion, timeout, insufficient dwell, wrong
  direction, no face, multiple faces, and face substitution.
- Added automatic SHA-256 discovery for present detector, embedding, and
  passive-liveness model files while preserving explicit checksum overrides.
- Verification: 44 focused vision tests passed; Ruff passed. Repeated
  bona-fide trials across lighting, distance, glasses, and natural movement
  remain manual evidence, not completed results.
- Local `pose.csv`, `roll_left.csv`, and `roll_right.csv` remain untracked and
  must not be committed by default.

**2026-09-05 — Calibration evaluator completion checkpoint (IN_PROGRESS)**

- Added ordered challenge evaluation for neutral, yaw, pitch, and roll phases
  with explicit completion, timeout, wrong-direction, insufficient-dwell,
  no-face, multiple-face, and face-substitution outcomes.
- Added configurable physical baseline and hysteresis/dwell settings without
  promoting the captured candidate thresholds to policy.
- Added SHA-256 model-file discovery for detector, embedding, and MiniFASNet
  adapters; explicit checksum overrides remain supported.
- Verification: full headless/unit/characterization run passed (84 tests),
  followed by the timeout regression test (6 active-liveness tests); Ruff and
  `mypy src` passed.
- Repeated bona-fide trials and false-failure measurements remain pending and
  require manual captures under varied conditions.

**2026-09-05 — Apple Face ID enrollment and active-liveness research (IN_PROGRESS)**

- Apple documentation describes Face ID enrollment/authentication as using the
  TrueDepth camera: infrared imagery, a depth map, randomized capture patterns,
  attention detection, and Secure Enclave processing. The system stores a
  mathematical representation captured across varied poses; this is not
  equivalent to a monocular RGB webcam plus head-pose landmarks.
- The Face-ID-style head movement and green coverage indicator are therefore
  appropriate UX inspiration for FaceAttend enrollment, but the visual motion
  alone is not Apple-level liveness or spoof resistance.
- Research on interactive face PAD supports randomized challenge-response
  (head turns, blinks, gaze/visual stimuli, or speech) as a useful defense
  against simple photo and some replay attacks, while surveys and dynamic-face
  authentication studies report that prerecorded video, high-quality displays,
  masks, and deepfake/re-enactment attacks can still challenge RGB-only systems.
- FaceAttend should keep two separate headless components: a measured pose
  coverage tracker that earns green enrollment sectors and a randomized active
  challenge evaluator that validates ordered responses, dwell, timing, and
  continuity. Coverage must never be implemented as timer-driven progress.
- Provisional decision: retain the current active challenge design, but make
  its challenge sequence randomized and combine it with temporal passive PAD.
  Do not claim Face ID-equivalent security. Validate attack acceptance and
  bona-fide failure rates on FaceAttend's target camera before tuning policy.
- References: [Apple biometric security](https://support.apple.com/guide/security/biometric-security-sec067eb0c9e/web),
  [Apple Face ID Security Guide](https://www.apple.com/business-docs/FaceID_Security_Guide.pdf),
  [RGB face PAD survey](https://arxiv.org/abs/2010.04145),
  [dynamic face authentication study](https://www.sciencedirect.com/science/article/pii/S0167404822000281),
  and [ISO/IEC 30107-3](https://www.iso.org/standard/79520.html).

## Update template

**2026-09-05 — External challenge-repository comparison (IN_PROGRESS)**

- Reviewed `AbdulazeezAde/face-liveness-check`, `amoghgg/face-biometrics-api`,
  and `Anantu-Rajesh/rule-based-liveness-detection` as design references, not
  as validated security implementations.
- Best ideas to borrow: immutable per-frame evidence, a stateful short-lived
  session, cryptographically shuffled challenge sequences, explicit landmark
  activity detectors, neutral gating between turns, consecutive-frame/dwell
  requirements, duplicate-frame indicators, quality/PAD aggregation, and
  liveness-before-embedding matching.
- Ideas not to copy unchanged: proxy yaw thresholds, global random seeding,
  debug-print-driven state, mutable default dictionaries, optional/passive-only
  policy modes, FAISS for a tiny local database, and embedding computation in
  every frame before liveness has passed.
- FaceAttend decision: abandon the green-oval completion requirement for the
  first implementation. Use randomized active challenges with MediaPipe matrix
  pose, blink state, continuity, dwell, neutral transitions, temporal MiniFASNet
  evidence, and calibrated thresholds. Keep the green-oval concept only as a
  possible future UX experiment, not as a security requirement.
- The external repositories do not establish attack resistance by themselves;
  FaceAttend still needs target-camera bona-fide and attack trials, explicit
  thresholds, and ISO/IEC 30107-oriented reporting.

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

**2026-09-05 — MediaPipe action evidence bridge (IN_PROGRESS)**

- Outcome: Connected real Face Landmarker blink transitions and smile
  blendshape scores to the headless `ChallengeSession` evaluator through a
  dedicated adapter. MediaPipe now requests blendshape output explicitly.
- Files changed: `ml/liveness/mediapipe_active.py` and
  `ml/tests/test_mediapipe_active.py`.
- Tests/checks: 42 focused tests passed; Ruff and `mypy src` passed.
- Evidence or measurements: deterministic fixtures verify blink open-close-open
  transitions, smile threshold plus consecutive frames, missing/weak
  blendshape rejection, and session-backed smile completion.
- Known limitations: PAD is not integrated; the legacy video checker still
  uses its fixed challenge path. Real-camera blendshape calibration remains
  pending.
- Graphify/documentation update: pending `graphify update .` after this record.
- Next gate: add a runtime session path for all selected actions, then proceed
  to temporal passive-PAD integration only after action evidence is validated.
- Approval required: none for this headless, non-destructive change.

**2026-09-05 — Runtime active + temporal passive session (IN_PROGRESS)**

- Outcome: Added a runtime composition that feeds MediaPipe action evidence
  into the active `ChallengeSession` evaluator while collecting a bounded
  passive-PAD frame window. PAD finalization is fail-closed until active
  liveness completes.
- Files changed: `ml/liveness/mediapipe_active.py`,
  `src/faceattend/vision/passive_liveness.py`, exports, and focused tests.
- Tests/checks: 51 focused tests passed; Ruff and `mypy src` passed.
- Evidence or measurements: tests verify session completion from a MediaPipe
  smile event, PAD cannot finalize early, temporal windows are bounded, and
  stale frames are rejected.
- Known limitations: the legacy video checker remains unchanged and PAD model
  inference still occurs only at finalization; GUI/camera wiring is pending.
- Graphify/documentation update: pending `graphify update .` after this record.
- Next gate: integrate the runtime composition with the local camera worker,
  then benchmark temporal MiniFASNet behavior on bona-fide and attack clips.
- Approval required: none for this headless, non-destructive change.

**2026-09-05 — Shared frame-evidence boundary (IN_PROGRESS)**

- Outcome: Added immutable typed `FrameEvidence` data shared by active
  challenge evaluation and temporal passive PAD. MediaPipe builds one record
  per frame; the active evaluator and PAD session consume that same record.
- Files changed: `src/faceattend/vision/types.py`, active/passive headless
  adapters, exports, and focused tests.
- Tests/checks: 55 focused tests passed; Ruff and `mypy src` passed.
- Evidence or measurements: tests cover completed action evidence, missing
  face, multiple faces, unavailable pose, blink/smile evidence, stale frames,
  bounded PAD windows, and fail-closed PAD ordering.
- Known limitations: camera-worker integration and real-camera PAD trials are
  still pending; `FrameEvidence` carries data only and performs no inference.
- Graphify/documentation update: pending `graphify update .` after this record.
- Next gate: connect the shared evidence path to the camera worker, then run
  real temporal MiniFASNet calibration and attack trials.
- Approval required: none for this headless, non-destructive change.

**2026-09-05 — Camera evidence worker and PAD trial gate (IN_PROGRESS)**

- Outcome: Added a headless camera worker that exclusively owns OpenCV capture,
  timestamps frames, forwards them to an evidence processor, and releases the
  device on exit. Qt remains out of this boundary.
- Files changed: `src/faceattend/camera/worker.py`, its package export, and
  `tests/unit/test_camera_worker.py`.
- Tests/checks: 57 focused tests passed; Ruff and `mypy src` passed.
- Evidence or measurements: the worker tests verify evidence forwarding,
  sequence IDs, unavailable-camera failure, and guaranteed capture release.
  A real webcam smoke attempt was blocked by macOS camera permission
  (`OpenCV: camera access has been denied` for camera index 1).
- Known limitations: no valid bona-fide calibration or print/screen attack
  trial was recorded in this environment. Existing still-image samples are
  not a substitute for labeled temporal trials.
- Graphify/documentation update: pending `graphify update .` after this record.
- Next gate: grant camera permission and collect repeated labeled genuine,
  print, and screen-replay windows before selecting PAD thresholds.
- Approval required: user must provide/authorize camera access and consented
  attack-trial data; no code approval is needed for this headless boundary.

**2026-09-05 — Bona-fide MiniFASNet webcam smoke test (MEASURED RESULT)**

- Outcome: A 10-frame webcam run completed successfully through detection,
  temporal passive liveness, alignment, and embedding.
- Evidence or measurements: one face was detected in every frame (`1/1` for
  all 10 frames); final detector score was `0.8253`; MiniFASNet score was
  `0.924292` against the provisional `0.85` threshold; embedding dimension
  was `512` with L2 norm `1.000000`.
- Known limitations: this is one bona-fide smoke session, not calibration or
  security evidence. No print, screen-replay, or other attack samples were
  measured, and the short window cannot estimate false-failure rates.
- Graphify/documentation update: pending `graphify update .` after this record.
- Next gate: collect repeated genuine and labeled attack windows under varied
  conditions before changing the PAD threshold.

