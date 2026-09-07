# FaceAttend Local-First Migration Plan

**Last reconciled:** 2026-09-07

**Current phase:** Phase 3 — ready for SQLite, audit persistence, and local matching
**Overall status:** PHASE 2 COMPLETED; PHASE 3 PENDING

Target: PySide6 desktop UI + headless Python CV/application core + SQLite +
reproducible evaluation. Preserve the web implementation until verified parity
and explicit removal approval.

This is the authoritative checklist. The [original plan and execution
history](archive/local-first-migration-plan-before-2026-09-06-reconciliation.md)
are preserved unchanged as a historical snapshot, including rejected pose
approaches, camera measurements, earlier test counts, and previous completion
claims. Where those claims conflict with this checklist, use this checklist.

## How to track progress

- `[x]`: the specifically worded deliverable exists with supporting evidence.
  A tested helper is not automatically an integrated or camera-validated feature.
- `[ ]`: remaining work, including partial implementations needing integration.
- `COMPLETED`: the stated phase scope passed its gate.
- `IN_PROGRESS`: active implementation; Phase 2 is the current workstream.
- `PENDING`: later phase not yet entered; early reusable pieces may already exist.
- `BLOCKED`: identify the exact missing permission, data, or decision.
- `DEFERRED`: intentionally outside the current implementation.

Keep one authoritative checkbox per deliverable. Later phases reuse Phase 2
components rather than rebuilding challenge engines. Record test commands,
results, limitations, and next gate after each turn. Never turn a proposed
experiment or a passing synthetic test into measured security effectiveness.

## Phase overview

| Phase | Scope | Status |
|---|---|---|
| 0 | Protect existing work and establish baselines | COMPLETED — historical baseline |
| 1 | Headless types, protocols, and workflow-state foundation | COMPLETED — foundation only |
| 2 | CV adapters, frame evidence, randomized sessions, headless runtime | COMPLETED |
| 3 | SQLite, audit persistence, exact local matcher | PENDING |
| 4 | Qt shell, camera/inference concurrency, session presentation | PENDING |
| 5 | Enrollment using the shared liveness runtime | PENDING |
| 6 | Attendance using the shared liveness runtime and matcher | PENDING |
| 7 | Reproducible recognition, liveness, and runtime evaluation | PENDING — pose diagnostics already exist |
| 8 | Parity, packaging, and portfolio evidence | PENDING |
| 9 | Approved retirement of obsolete infrastructure | DEFERRED — approval required |

## Where the randomized active-liveness plan belongs

It is now part of the numbered migration, not a second competing plan.

| Former subsection | Authoritative home |
|---|---|
| Challenge session design | Phase 2B; durable sequence audit in Phase 3 |
| Frame-evidence boundary | Phase 2C |
| Active challenge evaluation | Phase 2D |
| Passive PAD and matching order | Phase 2E for liveness; Phase 3 matcher; Phases 5–6 application gating |
| Headless tests and real-camera validation | Phase 2F regression/smoke gate; Phase 7 repeated trials and metrics |
| Challenge instructions and progress display | Phase 4 presentation; reused by Phases 5–6 |
| Green-oval completion | Deferred optional UX, not an enrollment/security gate |

## Current decisions and boundaries

- Follow [accepted ADR-001](../adr/001-local-desktop-cv-architecture.md) and
  [the feasibility note](python-desktop-cv-feasibility.md).
- Active and passive liveness remain mandatory for registration and attendance.
  Required decision order: active pass → passive pass → embedding → matching
  where applicable → persistence. Buffering frames during active evaluation
  does not mean PAD inference or its decision has already passed.
- Random challenges replace mandatory green-oval enrollment for the first version.
  Reuse session/evidence logic; do not create a parallel challenge implementation.
- Default random pool: BLINK, TURN_LEFT, TURN_RIGHT. LOOK_UP and SMILE are
  implemented as opt-in actions but not camera-validated defaults. LOOK_DOWN
  and roll exist as labeled evaluator phases, not randomized session actions.
- MediaPipe matrix pose is provisional primary; solvePnP remains diagnostic-only.
  Agreement between two estimators is not error against ground truth.
- Numeric pose, blink, smile, PAD, and matching thresholds remain provisional
  until validated. Threshold-policy changes still require the appropriate approval.
- SQLite and normalized NumPy matching are the planned local store/search.
  Keep multiple templates, model compatibility, unknown and ambiguity rejection.
- Raw images are not retained by default in the target product. Current
  diagnostic behavior is not yet proof that this privacy requirement is met.
- No partial-challenge success policy, global production random seeding, copied
  proxy-yaw thresholds, or Apple Face ID-equivalent security claims.
- Model licensing, consent, deletion, audit, and production legal review remain
  constraints. Do not edit legal documents or remove legacy code in this work.

## Phase 0 — Protect and characterize the existing project

**Status: COMPLETED — historical baseline, not a claim that today's tree is clean**

- [x] Preserve and record unfinished web/liveness work.
- [x] Record baseline tests, lint, type checking, and coverage.
- [x] Correct the recursive `models/` ignore collision.
- [x] Define tracking policy for instructions, documentation, and Graphify records.
- [x] Establish legacy characterization for liveness ordering, model-version
      filtering, rejection, duplicates, deletion, and audit calls.

Evidence: the 2026-09-04 execution record and Graphify memory record 65 passing
tests after the initial slice, with five pre-existing full-tree mypy errors.
These numbers are historical; rerun relevant checks for each new change.
The current dirty/untracked tree is preserved, not automatically committed.

## Phase 1 — Define the headless package boundary

**Status: COMPLETED — initial foundation**

- [x] Configure `src/faceattend/` package discovery.
- [x] Define typed CV/domain values and replaceable model/matcher protocols.
- [x] Define registration and attendance state transitions with liveness ordering.
- [x] Add headless contract and current face-count characterization tests.

Evidence: `vision/types.py`, `vision/interfaces.py`,
`application/states.py`, `tests/unit/test_headless_contracts.py`, and
`tests/characterization/test_current_face_selection.py`.
State transitions are contracts, not completed registration/attendance services.

## Phase 2 — Extract and integrate the headless CV core

**Status: COMPLETED**

### 2A — Model adapters and pose foundations

- [x] Add lazy detector, aligner, embedder, and MiniFASNet adapters with injectable
      model loading, normalized embeddings, metadata, and checksum discovery.
- [x] Add dedicated MediaPipe matrix estimator and deterministic pose tests.
- [x] Enable MediaPipe transformation/blendshape output; use matrix pose in the
      legacy turn checker and fail closed there on missing/invalid matrices.
- [x] Keep canonical solvePnP available for comparison, with its failed webcam
      validation recorded; do not promote it to operational fallback.
- [x] Add synthetic pose validation, smoothing/hysteresis helpers, and labeled
      webcam/CSV diagnostics.
- [x] Compose the real headless face-analysis/evidence processor using these
      adapters: `HeadlessFaceAnalyzer` and `create_face_analyzer` now provide
      detection, MediaPipe matrix/action evidence, and separate gated alignment/
      embedding. See [usage and boundaries](headless-face-processor.md).
- [x] Verify model compatibility/checksums at the composed runtime boundary:
      explicit pinned artifacts, rehashed files, adapter identity, baseline ONNX
      tensor contracts, and alignment/embedding output checks. This verifies
      compatibility with selected files, not provenance or model effectiveness.

### 2B — Randomized challenge session

- [x] Implement secure per-session sampling, unique actions, exact sequence
      completion, immutable snapshots, and per-challenge/session timeout fields.
- [x] Support BLINK, TURN_LEFT, TURN_RIGHT; keep LOOK_UP and SMILE opt-in.
- [x] Expose an injectable random source for deterministic tests.
- [x] Store selected/completed actions in the session snapshot.
- [x] Reconcile evaluator/session clocks, reset, and terminal-state lifecycle:
      session timeouts are checked before every evidence path, including waiting
      for neutral and dwell; terminal results remain sticky.
- [x] Test supported actions through evidence-driven evaluation, not only
      direct `complete_current()` calls on the sequence container.

Durable audit writing belongs to Phase 3 and its application use to Phases 5–6;
an “audit-friendly” snapshot is not an audit event.

### 2C — Frame-evidence boundary and continuity

- [x] Define data-only `FrameEvidence` with timestamped frame, face count,
      optional tracking/pose/quality/lighting/motion/PAD/action/embedding evidence.
- [x] Build evidence in the MediaPipe adapter; expose active and PAD consumers.
- [x] Validate face-count consistency and selected signal ranges.
- [x] Add standalone monotonic-time/track/jump continuity checks and tests.
- [x] Add recent exact-pixel duplicate hashing and a duplicate evidence flag.
- [x] Keep headless evidence/session modules independent of Qt and FastAPI.
- [x] Connect `FaceContinuityTracker` to the real evidence/session path and
      propagate loss, substitution, and implausible-jump outcomes.
- [x] Consume `FrameEvidence.failure_reason` in both active and passive consumers;
      exact recent duplicates fail the attempt, with synthetic policy tests and
      no claim of full replay protection.
- [x] Compute real face-crop quality/lighting signals and enforce configurable
      exposure, blur, size, centering, and landmark-visibility gates.
- [x] Add advisory eye darkness/glare and lower-face texture warnings; do not
      classify glasses or masks from these heuristics.
- [x] Review the three reference repositories and document borrowed/rejected
      ideas in [quality/session research](frame-quality-and-session-policy.md).
- [ ] Calibrate quality thresholds and evaluate occlusion-warning false positives
      on repeated held-out camera trials (lighting, pose, glasses, masks, distance).
- [x] Unify timestamp/count/pose sources and validate missing/nonfinite evidence;
      reject stale/repeated/backward input through the composed path.
- [x] Convert no-face/multiple-face/PAD input errors to explicit terminal session
      results. Active observation precedes PAD buffering; abort/finalize clear it.

Track IDs and geometric continuity alone do not prove biometric identity
continuity. Test substitution limits explicitly; never label this complete
identity-substitution protection.

### 2D — Active challenge evidence and evaluation

- [x] Add configurable neutral baseline, directional enter/exit thresholds,
      dwell duration, minimum valid frames, and explicit failure statuses.
- [x] Add initial/between-direction neutral gating logic.
- [x] Add MediaPipe EAR blink counting and smile blendshape action bridge.
- [x] Preserve raw pose and implement configurable physical-yaw inversion in the
      legacy challenge boundary.
- [x] Apply and test yaw mapping exactly once at the new session boundary;
      preserve raw pose in the shared evidence.
- [x] Ensure the real processor supplies matrix pose only; reject missing/invalid
      matrices without solvePnP fallback in the new session path.
- [x] Complete regression coverage for neutral return, enter/exit-band behavior,
      dwell/frame-count interaction, wrong direction, timeouts, and reset.
- [x] Require observed open → closed → open blink evidence after session/challenge
      start; the counter now begins unarmed and must observe fresh open evidence.
- [x] Test action freshness/reset after face loss and challenge transitions,
      blink noise, missing blendshapes, and sustained-smile behavior.
- [x] Validate LOOK_UP/SMILE on fresh proposal camera trials before enabling them
      by default. Validation rejected default enablement for now.

`scripts/run_liveness_trial.py --challenge look_up|smile` now provides an
explicit experimental path that cannot alter the default BLINK/TURN_LEFT/
TURN_RIGHT pool. The checkbox remains open until the repeated fresh-camera
protocol in `docs/research/liveness-trial-protocol.md` has measured both actions.

First `LOOK_UP` proposal attempt (`lookup-normal-01`) was invalid: only two frames
were processed before `face_missing_too_long`. Investigation showed continuity
was incorrectly inferring disappearance from slow valid-to-valid inference gaps.
That clock-boundary defect is fixed and regression-tested; the attempt is retained
as diagnostic evidence, not counted as a LOOK_UP success/failure trial.

Post-fix proposal evidence: LOOK_UP completed active liveness in 10/16 attempts
(62.5%); three attempts timed out and three terminated on multiple-face evidence.
Nine of 16 passed the complete active+PAD pipeline. Median completed duration was
3.36 s. This is functional but too failure-prone for the default pool.

SMILE completed active liveness in all 10 intended positive attempts, but only
4/10 passed the complete pipeline because MiniFASNet rejected one or more smiling
frames in six attempts. The no-smile negative control timed out correctly. The
already-smiling/no-initial-neutral control nevertheless completed active liveness,
showing that the current neutral arming rule is not reliable on live blendshape
scores. It was rejected only by the old active-frame PAD policy. The runtime now
separates active challenges from a post-active neutral hold, so smiling challenge
frames are never used as PAD input. SMILE must nevertheless remain disabled by
default pending a corrected neutral-expression arming rule and fresh validation
of this new staged policy.

These are single-participant, single-camera proposal results with repeated trial
IDs and retry attempts. They are engineering evidence, not held-out validation,
population performance, or a security claim.

Do not tune defaults from one person's pose capture or hide a convention bug by
changing thresholds.

### 2E — Temporal passive PAD and liveness ordering

- [x] Run MiniFASNet over provided windows; expose median/minimum scores,
      suspicious-frame count, and failure-to-process count.
- [x] Add bounded temporal frame buffering with strict timestamp ordering.
- [x] Compose active evaluation and PAD buffering; block PAD finalization until
      active completion in `MediaPipeLivenessSession`. Active frames are excluded
      from the dedicated PAD window.
- [x] Define/test minimum usable window size or duration, sampling, and evidence
      retention. After active completion, the runtime presents `Face the camera`,
      then `Hold still`; it requires neutral pose and expression, samples at 200 ms,
      requires at least five samples spanning one second, and caps the in-memory
      window at ten frames. Losing neutral restarts the provisional PAD window.
- [x] Aggregate post-active MiniFASNet median/minimum scores, suspicious frames,
      and processing failures; keep every sampled pixel only in memory and clear
      the window on success, failure, timeout, or cancellation.
- [x] Retain the sharpest accepted neutral evidence as a one-use, post-liveness
      embedding candidate. The runtime can invoke an injected extractor only after
      both gates pass and then clears the pixels; failure/cancellation also clear
      them. Extraction of multiple enrollment templates remains a registration task.
- [x] Integrate continuity, quality, duplicate policy, inference-error handling,
      sticky overall failure, and abort/finalize buffer cleanup.
- [x] Add application-owned cancellation and complete reset/terminal lifecycle
      across the processor, action bridge, evaluator, and session.
- [x] Add an end-to-end headless orchestration test proving embedding/matching
      callbacks cannot run on active or PAD failure.

Actual identity scoring is Phase 3. Attendance recording is Phase 6.
A state-machine test is not evidence that those runtime paths already exist.

### 2F — Headless camera connection and completion gate

- [x] Add synchronous `CameraEvidenceWorker` with capture ownership, timestamps,
      injected evidence-processor seam, and fake-camera tests.
- [x] Add focused tests for adapters, pose, sessions, action evidence, temporal PAD,
      and evidence helpers.
- [x] Supply a real detection → MediaPipe matrix/action evidence processor with
      the callable signature expected by the camera worker.
- [x] Connect that processor's evidence to continuity/quality enforcement,
      active session decisions, and gated PAD; exercise camera → real adapter
      wrappers → session → PAD using synthetic SDK outputs.
- [x] Bound retained results and test read failure, processor failure, stop, and
      cleanup. Streaming callbacks may consume every result while the returned
      diagnostic list has explicit bounded capacity, including zero retention.
- [x] Run integrated headless regression tests for missing/multiple faces,
      invalid pose, timestamp errors, duplicate policy, substitution/jumps,
      each supported action, and both liveness failure gates.
- [x] Run a consented real-camera end-to-end smoke session and record configuration,
      failure reasons, and limitations without retaining raw images by default.
- [x] Protect root pose CSVs and local trial/diagnostic directories with explicit
      ignore rules. Runtime trials retain aggregate JSONL metrics only; temporary
      frame buffers are cleared on success, failure, and cancellation.

**Phase 2 exit gate:** complete the integration and regression items above,
record an actual camera-session smoke result, preserve legacy behavior, and
document remaining calibration/security uncertainty. No GUI or SQLite is needed
to close this headless gate. Repeated effectiveness trials are tracked in Phase 7
and must precede threshold promotion or security claims.

The first staged-policy smoke (`staged-smoke-01`) was a measured low-light
fail-closed result, not a startup defect or successful liveness pass. Camera 1
correctly produced `no_face` in a dark room; active liveness remained in progress,
PAD was not run, no embedding was extracted, and the runtime terminated.

The normally lit rerun (`staged-smoke-normal-01`) completed the full Phase 2 path:
randomized TURN_LEFT → BLINK, post-active neutral hold, five-to-ten-frame temporal
PAD, and 512-dimensional embedding extraction. MiniFASNet passed with median
0.999887 and minimum 0.999366, with zero suspicious frames and zero processing
failures. The run processed 62 frames in 14.254 seconds. This closes the Phase 2
engineering gate; it remains one participant/camera proposal smoke, not threshold
calibration, population evidence, or a security benchmark.

## Phase 3 — SQLite, audit persistence, and exact local matching

**Status: PENDING**

- [ ] Add migrations, foreign keys, transactions, and appropriate concurrency/WAL.
- [ ] Add people, consent, enrollment/templates, attendance sessions/records,
      liveness attempts, model/configuration versions, and audit records.
- [ ] Persist challenge sequence, outcomes, failure reasons, and configuration
      version in audit records; do not store raw biometric payloads in logs.
- [ ] Store normalized float32 embedding BLOBs with dimensions/model/checksum/
      quality/pose metadata; implement reload and compatibility filtering.
- [ ] Implement exact NumPy person-level matching, threshold and ambiguity policy,
      with a replaceable search interface; FAISS only for measured need.
- [ ] Test deletion, erasure of templates, uniqueness/idempotency, restart,
      rollback, unknown identities, and ambiguous matches.
- [ ] Define safe migration of existing PostgreSQL/FAISS metadata without deleting
      or silently reinterpreting incompatible templates.

## Phase 4 — PySide6 shell and runtime concurrency

**Status: PENDING**

- [ ] Reuse Phase 2 capture/processor interfaces in camera and inference workers;
      keep one camera owner and one owner of loaded inference models.
- [ ] Add Qt Widgets views with no inference/model logic in widgets.
- [ ] Use capacity-one latest-frame backpressure and signals/slots, not networking.
- [ ] Separate preview and inference rates; timestamp results and drop stale frames.
- [ ] Present randomized instructions, neutral guidance, progress, and failures
      from session evidence; the GUI never awards liveness progress.
- [ ] Add explicit mode transitions, cancellation, shutdown, camera/model errors,
      and no-camera/model states; verify repeated start/stop and responsiveness.
- [ ] Keep the CV/application core runnable and testable without Qt.

The existing synchronous worker is a reusable foundation, not completion of
Qt threading, bounded preview delivery, or responsive GUI operation.

## Phase 5 — Enrollment application workflow

**Status: PENDING**

- [ ] Coordinate explicit consent, quality/single-face checks, and the shared
      randomized-liveness runtime; do not implement another challenge engine.
- [ ] Require active pass then temporal PAD pass before template extraction.
- [ ] Select 3–5 diverse high-quality normalized templates with pose metadata.
- [ ] Persist identity/templates/consent/audit atomically with failure rollback.
- [ ] Test cancellation, retry, no/multiple faces, liveness rejection, model and
      database failures; clear ordinary raw-frame buffers on all exit paths.

Green-oval sectors are deferred optional guidance, not required implementation.
Pose diversity remains an enrollment concern without requiring that animation.

## Phase 6 — Attendance check-in application workflow

**Status: PENDING**

- [ ] Reuse the same mandatory active → passive runtime for explicit session check-in.
- [ ] Extract embeddings only after both passes; match registered consenting,
      model-compatible identities using the approved calibrated policy.
- [ ] Reject unknown and ambiguous identities before any attendance write.
- [ ] Enforce duplicate handling in the database and return explicit outcomes.
- [ ] Write check-in and required audit events with transaction/failure handling.
- [ ] Integration-test all terminal outcomes, including no/multiple faces,
      continuity loss, liveness failure, unknown/ambiguous/duplicate, and camera,
      model, or database failure.

Existing `application/states.py` rejection transitions remain useful tests, but
the local matcher and persistence services still need implementation.

## Phase 7 — Reproducible evaluation and threshold validation

**Status: IN_PROGRESS — metrics-only trial protocol/harness exists; physical trials remain**

Develop the following tooling as needed during Phase 2, but track each result
here. Formal final evaluation follows integration of the application workflows.

- [x] Add synthetic pose-summary helper, labeled webcam angle capture, and
      candidate-threshold summarizer. Candidates are not promoted automatically.
- [x] Record preliminary user-provided directional pose measurements and one
      10-frame bona-fide passive-PAD/embedding smoke result in historical records.
- [x] Ignore `local-biometric-recordings/` and `pad-trials/` directories.
- [x] Define trial manifests, consent/retention rules, condition/attack labels,
      participant/session-disjoint splits where applicable, and a diagnostic mode
      that cannot record attendance.
- [ ] Capture repeated bona-fide attempts across lighting, distance, glasses,
      camera placement, and natural movement.
- [ ] Collect print, phone/display replay, prerecorded/fixed-challenge replay,
      substitution, and frozen-frame trials; document unevaluated attack types.
- [ ] Measure active completion, false-failure and timeout rates, wrong-direction/
      dwell failures, completion latency, and attack acceptance with denominators.
- [ ] Evaluate temporal PAD APCER per attack type, BPCER, compatible ACER,
      failure-to-process, cross-camera/domain behavior, and window-policy effects.
- [ ] Compare MiniFASNet with at least one reproducible lightweight alternative
      as required by ADR-001; check licensing, preprocessing, and CPU cost.
- [ ] Calibrate on validation data; freeze thresholds/configuration and evaluate
      fresh held-out trials. Report uncertainty and sample-size limitations.
- [ ] Evaluate recognition score distributions, ROC/DET, FMR/FNMR, EER, rank-1,
      unknown rejection, ambiguity, template aggregation, and condition breakdowns.
- [ ] Measure FPS, stale/dropped frames, p50/p95/p99 stage latency, cold/warm
      startup, memory, CPU, and registration/check-in time.
- [ ] Record hardware/camera, model hashes, configuration, thresholds, dataset/
      split, seeds, and commit for every result; keep raw recordings local-only.
- [x] Verify debug scripts and local output paths follow retention/ignore policy:
      root pose CSVs are explicitly ignored, pipeline images default to the OS
      temporary directory, and liveness trials default to an ignored local folder.

The executable local protocol is documented in
`docs/research/liveness-trial-protocol.md`; `scripts/run_liveness_trial.py`
records aggregate metrics and model checksums without an attendance-writing path.
Physical trials and threshold promotion remain unchecked until measured.

The prior 0.924292 PAD score at threshold 0.85 was one genuine smoke window,
not a calibrated threshold or proof of attack resistance. Pose estimator
disagreement must not be reported as ground-truth head-pose error.

## Phase 8 — Parity, packaging, and portfolio evidence

**Status: PENDING**

- [ ] Demonstrate enrollment, liveness-gated recognition/check-in, unknown/
      ambiguity rejection, duplicates, deletion, audit, history, and failure recovery.
- [ ] Compare behavior against preserved legacy characterization.
- [ ] Run full tests/Ruff/type checks; separate baseline failures from regressions.
- [ ] Measure core coverage against >80% target; do not reuse an old slice's number.
- [ ] Verify clean-start model loading/packaging, model provenance/licensing,
      local configuration, and recoverable installation on the initial target.
- [ ] Produce reproducible demo/results and an explicit RGB-webcam limitations
      statement; no production-security claims without supporting evaluation.

## Phase 9 — Retire obsolete architecture

**Status: DEFERRED — EXPLICIT APPROVAL REQUIRED**

- [ ] Preserve old web/crawler history and a recoverable migration boundary.
- [ ] Obtain approval before removing/mass-moving frontend, FastAPI/ML service,
      crawler, PostgreSQL, queue/storage, or deployment infrastructure.
- [ ] Remove only after parity and rollback are demonstrated.
- [ ] Clean obsolete dependencies/instructions and refresh README/Graphify.

## Reconciliation record — 2026-09-06

**Outcome:** Integrated the randomized-liveness work into Phases 2–7 and separated
helper completion from runtime integration and real-camera validation.

**Evidence inspected:** Graphify report/query and September 4 memory; accepted
ADR; current `src/faceattend/` modules, MediaPipe bridge, focused tests,
`.gitignore`, and the previous plan. Existing Graphify memory predates concrete
adapters and later session work, so it is historical rather than current status.

**Corrections:**

- Frame-evidence and active-evaluation “COMPLETE” labels reopened for integration.
- Sequence snapshot stays checked; durable audit persistence is unchecked.
- Standalone continuity and duplicate detection stay checked; runtime enforcement
  is unchecked.
- Legacy yaw mapping stays checked; new session-boundary mapping is unchecked.
- Neutral/dwell/blink mechanisms stay checked at foundation level; full boundary
  and transition regression coverage is unchecked.
- Temporal aggregation stays checked; full liveness → matching → attendance is
  unchecked. Unknown/ambiguous state names alone do not complete that workflow.
- LOOK_UP/SMILE exclusion from the default pool is acknowledged; LOOK_DOWN is
  not misrepresented as an implemented randomized action.
- Removed mandatory green-oval enrollment from the current checklist.
- Local recording-directory ignore rules stay checked; root pose CSV protection
  and debug-image cleanup remain open.
- Preserved all prior execution history in the linked snapshot rather than
  silently rewriting past reports.

**Scope:** Documentation reconciliation only; no implementation or threshold changes.
**Graphify:** Saved a dated reconciliation memory with the reopened integration
gates and phase mapping. Ran `graphify update .` successfully: 1,623 nodes and
2,637 edges. This refresh is AST/code-only; it does not semantically re-extract
the revised Markdown. The new memory and this plan carry the documentation
corrections; old memory records remain historical.
**Checks:** Ran `UV_CACHE_DIR=/tmp/faceguard-uv-cache uv run pytest tests/unit
tests/characterization ml/tests/test_mediapipe_active.py -q`: **116 passed**.
Warnings: Albumentations version lookup could not resolve its host; InsightFace
alignment uses a deprecated scikit-image API. This was a focused regression run,
not full-suite coverage, a fresh lint/type audit, or a camera/security trial.
**Next gate:** Phase 2C/2D failure propagation, continuity and physical-yaw mapping,
then Phase 2E/2F integrated session/camera validation.
**Approval:** No deletion or model/policy change authorized by this edit.

### 2026-09-07 — Phase 2E — Dedicated neutral PAD window

- Outcome: changed the composed runtime from PAD-on-active-frames to a staged
  cascade: randomized active challenges, face-camera guidance, stable neutral
  hold, a five-to-ten-frame temporal MiniFASNet window, then a one-use neutral
  embedding candidate. Continuity, timestamp, one-face, and quality checks remain
  active across the stage boundary.
- Files: `ml/liveness/mediapipe_active.py`, `src/faceattend/vision/types.py`,
  `src/faceattend/vision/passive_liveness.py`,
  `src/faceattend/vision/face_analyzer.py`, `scripts/run_liveness_trial.py`, and
  focused unit/integration tests.
- Tests/checks: full pytest **244 passed** with seven upstream dependency
  warnings; Ruff passed; `mypy src` passed for 23 source files.
- Evidence: deterministic tests prove that active frames do not enter PAD, a
  non-neutral pose or expression restarts collection, PAD runs only after the
  neutral dwell/window criteria, and transient frames clear on every terminal
  path. This is implementation evidence, not PAD-security validation.
- Limitations: default neutral-expression threshold and one-second window remain
  proposal settings. The new staged flow needs fresh bona-fide and presentation-
  attack trials. Existing SMILE/LOOK_UP results used the superseded PAD policy.
- Graphify/documentation: refresh after final verification.
- Next gate: run a consented real-camera session using the staged flow, then
  calibrate neutral/PAD settings on separate proposal and validation captures.
- Approval: no model, legal, retention, or destructive repository decision made.

### 2026-09-07 — Phase 2F — First staged real-camera smoke

- Outcome: executed consented proposal trial `staged-smoke-01` on camera index 1
  with normal lighting declared in the CLI metadata but an actually dark room,
  normal distance, no glasses, natural movement, and randomized TURN_LEFT →
  BLINK → TURN_RIGHT. It failed closed on frame 1 with `no_face`; no challenge
  completed. The declared lighting label is inaccurate and must not be used as a
  normal-light sample.
- Configuration: CPUExecutionProvider; default active configuration; default
  post-active neutral hold; MiniFASNet threshold 0.85; detector, embedding,
  landmarker, and PAD checksums were captured in the ignored JSONL trial record.
- Evidence: runtime phase `failed`; active `in_progress`; passive failure reason
  `no_face`; one observed frame; 4244 ms including model/camera startup; no PAD
  score and no embedding.
- Privacy: no raw image was written. Only aggregate metrics were appended under
  ignored `local-biometric-recordings/`.
- Limitation: this confirms fail-closed behavior under extreme low light. It does
  not establish a startup-acquisition defect because the camera could not expose
  a detectable face. A bounded pre-session acquisition state remains a possible
  Phase 4 UX improvement, not a Phase 2 blocker.
- Next gate: rerun under genuinely normal light. Do not tune thresholds from this
  failed low-light attempt.

### 2026-09-07 — Phase 2F — Successful staged real-camera smoke

- Outcome: consented proposal trial `staged-smoke-normal-01` completed on camera
  index 1 under normal light/distance, without glasses, using natural movement.
  The default secure generator selected TURN_LEFT → BLINK and both completed.
- Measured result: runtime and active status `completed`; temporal MiniFASNet
  passed with median 0.999887, minimum 0.999366, zero suspicious frames, and zero
  processing failures; a 512-dimensional embedding was extracted only afterward.
  The run observed 62 frames and took 14254 ms end to end.
- Configuration: CPUExecutionProvider, default randomized challenge and staged
  neutral-PAD configuration, MiniFASNet threshold 0.85, with all four model
  checksums recorded in the local ignored metrics file.
- Privacy: no camera images were retained; aggregate metrics remain under ignored
  `local-biometric-recordings/`.
- Evidence boundary: this is a measured project smoke result showing functional
  composition and ordering. It is not calibration, a false-reject estimate,
  attack resistance evidence, or population/security performance.
- Phase decision: Phase 2 is complete. Phase 3 may begin after the user chooses
  to proceed; repeated bona-fide and attack trials remain Phase 7 work.

## Execution-record template

Append new records here; keep the phase checklist above authoritative.

### 2026-09-07 — Phase 2D–2F — Edge cases, PAD lifecycle, and trial harness

- Outcome: completed deterministic edge coverage for neutral return, hysteresis,
  evaluator reset, fresh blink transitions, and sustained smile. Added a bounded,
  sampled PAD window with minimum count/duration, sticky cancellation, and early
  suspicious-evidence retention. Added a single liveness gate proving embedding
  and matching callbacks cannot execute after active or passive failure. Camera
  evidence can now stream through a callback while retained diagnostics are
  bounded (or disabled), with cleanup tests for stop/read/processor failures.
- Files: active/action/session/PAD/camera modules and unit tests; `.gitignore`;
  `scripts/run_liveness_trial.py`; `docs/research/liveness-trial-protocol.md`; and
  this plan.
- Tests/checks: full pytest **238 passed** with six upstream SWIG/InsightFace
  deprecation warnings; Ruff passed; `mypy src` passed for 23 source files; the
  trial command-line help smoke check passed.
- Evidence: deterministic synthetic regression evidence only. The trial runner
  provides consent, condition, split, attack-label, model-checksum, liveness,
  quality, and latency fields while retaining no camera images.
- Limitations: no consented real-camera session was run by the automated agent;
  no repeated bona-fide or attack trials exist yet. Current quality/PAD thresholds
  remain baselines. Bounded PAD retention preserves the earliest samples and must
  later be compared with alternative temporal sampling policies.
- Graphify/documentation: documentation and code graph refresh follow final checks.
- Next gate: user-operated real-camera smoke, then proposal/validation/test trials
  for quality and liveness calibration plus print/phone/prerecorded attacks.
- Approval: physical trials require participant consent and access to their camera
  and presentation instruments; no model/policy/deletion change was made.

### 2026-09-06 — Phase 2B — Evaluator/session lifecycle completed

- Outcome: aligned evaluator and randomized-session clocks. The evaluator now
  observes the shared session before no-face, multiple-face, neutral, action,
  dwell, or mismatch branches. Completion can reuse the already-observed frame
  timestamp without a false duplicate-time failure; terminal states remain
  sticky. Added evidence-driven action tests for blink, smile, left/right turns,
  and look-up, plus session-timeout tests while waiting for neutral and during
  directional dwell.
- Files: `src/faceattend/vision/active_liveness.py`,
  `src/faceattend/vision/challenge_session.py`,
  `tests/unit/test_challenge_session.py`, and this plan.
- Tests: focused session/active tests **27 passed**; full `uv run pytest -q`
  **183 passed** with six dependency/deprecation warnings. Ruff check passed;
  `mypy src` passed (21 files).
- Evidence level: deterministic synthetic/evidence-driven tests. No camera or
  attack effectiveness claim.
- Limitations: evaluator reset still resets evaluator-local state only; a new
  randomized session should be created/restarted by its owning application
  service. Face continuity, quality, duplicate policy, and yaw mapping remain
  separate integration gates.
- Graphify/documentation: saved this result to Graphify and refreshed the code
  graph; documentation semantic extraction is not performed by code-only update.
- Next gate: connect failure propagation and continuity/physical-yaw mapping in
  the composed runtime, then integrate active and temporal PAD decisions.
- Approval: none; no model, threshold, database, GUI, legal, or legacy removal
  changes.

### 2026-09-06 — Phase 2A — Headless processor and model boundary completed

- Outcome: implemented real detection/MediaPipe evidence composition, pinned
  file and ONNX compatibility checks, native MediaPipe lifecycle, and separate
  liveness-result-gated alignment/embedding with output validation.
- Files: `src/faceattend/vision/face_analyzer.py`,
  `tests/unit/test_face_analyzer.py`, this plan, and the linked usage note.
- Tests: observed failing tests before each new capability, then passing tests.
  Final `NO_ALBUMENTATIONS_UPDATE=1 UV_CACHE_DIR=/tmp/faceguard-uv-cache uv run
  pytest -q`: **176 passed**, six existing dependency deprecation warnings.
  Scoped Ruff check/format, `mypy src` (21 files), and `git diff --check` passed.
- Native smoke: selected local ONNX tensor layouts passed validation; SCRFD on a
  synthetic blank image returned `no_face`; native MediaPipe returned zero faces
  and closed. Sandbox graphics initialization aborted; approved out-of-sandbox
  rerun succeeded. No webcam or biometric images were used/saved.
- Limits: smoke hashes were computed from local files for this diagnostic, not
  independently authenticated. Runtime use requires explicitly selected pins.
  No real-face composition, camera calibration, PAD attack trial, or production
  security claim. Keep model files immutable while lazy models are in use.
- Integration: raw yaw stays unchanged; continuity/quality/duplicate enforcement,
  session-bound gate binding, durable audit, and camera/Qt lifecycle remain open.
  Liveness result objects must come from trusted same-session orchestration;
  they are not authentication tokens.
- Graphify: saved a new dated Phase 2A completion note and refreshed the code
  graph with `graphify update .`; earlier audit notes remain historical.
  The CLI refresh is AST-only, not semantic re-extraction of documentation.
- Next gate: Phase 2C/2D evidence failure propagation, continuity, and physical-yaw
  mapping, then wire the composed processor into active/temporal-PAD decisions.
- Approval: no threshold, model selection, database, legal, or legacy-runtime
  changes. Camera/security effectiveness still requires consented trials.

### 2026-09-06 — Phase 2C — Guarded evidence and measured quality implemented

- Outcome: connected continuity, freshness/sequence validation, upstream failures,
  exact duplicate rejection, measured quality, and active-first PAD buffering.
  Added sticky overall session results and abort/finalize pixel cleanup. Raw yaw
  remains unchanged in evidence and is inverted once at the challenge boundary.
- Files: `vision/quality.py`, `evidence.py`, `continuity.py`, `types.py`,
  `face_detector.py`, `face_analyzer.py`, `active_liveness.py`,
  `challenge_session.py`, `passive_liveness.py`, MediaPipe composition, focused
  unit/integration tests, this plan, processor usage and quality research notes.
- Tests: `NO_ALBUMENTATIONS_UPDATE=1 UV_CACHE_DIR=/tmp/faceguard-uv-cache uv run
  pytest -q --tb=short`: **220 passed**, six existing dependency warnings.
  `uv run ruff check .`, `uv run mypy src` (22 files), and `git diff --check`
  passed. Changed Python files formatted with Ruff.
- Evidence: synthetic images and model-SDK outputs; the complete camera-worker
  → actual adapter wrappers → evidence/session → PAD path is exercised headlessly.
  No personal images, real-camera trials, or attack measurements were collected.
- Research: reviewed pinned code in all three requested repositories. Borrowed
  crop-level quality and evidence boundaries; rejected categorical glasses/mask
  labels from simple texture/colour rules, fake frame clocks, and failed-pose
  neutral fallbacks. See [comparison/policy](frame-quality-and-session-policy.md).
- Limitations: quality defaults need held-out calibration; occlusion warnings
  are heuristic, not detectors. Exact duplicates do not detect general video
  replay; geometry alone cannot prove identity continuity. PAD minimum window,
  retention/sampling, cancellation and complete lifecycle remain Phase 2E tasks.
- Graphify: saved a dated completion/policy memory and ran the AST-only update;
  the research Markdown remains the authoritative source of the reference review.
- Next gate: minimum usable temporal PAD evidence and cancellation/lifecycle,
  then consented calibration and attack trials. Phase 2 remains IN_PROGRESS.
- Approval: no model swap, recognition-threshold change, legal edits, GUI,
  database implementation, deletion, commit or push was performed.

```text
DATE — Phase N/subsection — STATUS
- Outcome:
- Files changed:
- Tests/checks (exact command and result):
- Evidence level (code inspection / synthetic test / measured camera trial):
- Known limitations:
- Graphify/documentation update:
- Next gate:
- Approval required:
```
