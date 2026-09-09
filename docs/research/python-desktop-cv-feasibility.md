# Python Desktop CV Migration Feasibility

**Date:** 2026-09-04  
**Status:** Architecture research; no migration authorized

## Conclusion

Recommend a **local-first hybrid**: PySide6 for the desktop experience, backed
by a headless Python application/CV core and separate evaluation commands. The
current HTTP frontend/backend/ML split is not justified by the present
single-device, CV-portfolio goal. A single giant Qt/OpenCV loop is also rejected
because it would make experiments and testing harder.

## Runtime Boundaries

```text
Qt main thread: views, overlays, workflow presentation
Camera worker: exclusive camera ownership and frame timestamps
Inference worker: model ownership and latest-frame processing
Application controller: registration/check-in state transitions
Persistence worker/repositories: SQLite transactions
Evaluation package: headless datasets, metrics, and benchmarks
```

Use a bounded latest-frame mailbox rather than a growing FIFO. Dropping stale
preview frames is preferable to increasing real-time latency. Start with Qt
worker objects moved to `QThread`; introduce a process boundary only if profiling
shows that native inference still starves the GUI or isolation is required.

OpenCV `VideoCapture` in the camera worker is the simplest first implementation
because the CV pipeline already consumes NumPy arrays. Qt Multimedia's
`QCamera`/`QVideoSink` is a valid alternative when camera portability or native
preview behavior outweighs frame-conversion complexity.

## Registration Recommendation

1. Require exactly one sufficiently large, sharp, well-exposed face.
2. Track one face consistently across the sequence.
3. Issue randomized, ordered pose/blink challenges.
4. Estimate pose using MediaPipe's canonical-face transformation matrix or a
   calibrated canonical 3D model; do not reuse image coordinates as 3D points.
5. Smooth pose over time and require directional thresholds, dwell, return to
   neutral, ordering, and timeout constraints.
6. Update the circular UI from accepted pose bins, not elapsed time.
7. Run passive PAD over multiple selected frames or a short window.
8. Keep 3-5 quality-filtered embeddings spanning useful pose/lighting variation.
9. Persist the enrollment, template metadata, consent event, and audit event in
   one application transaction where practical.

Coverage improves enrollment diversity. It is not by itself secure liveness.
Randomized challenge-response plus passive PAD raises the cost of simple replay,
but monocular RGB cannot claim Face ID-equivalent assurance.

## Check-In Recommendation

Keep the current active-plus-passive policy during the first desktop version.
Use a shorter randomized challenge than enrollment, followed by temporal passive
PAD, alignment, embedding, exact identity search, ambiguity rejection, session
validation, duplicate handling, and an audit event.

Passive-only check-in is a possible later usability experiment, not the initial
architecture decision. It requires attack-specific evidence and a separate ADR.

## Recognition and Search

Keep InsightFace `buffalo_l` only for non-commercial research while its model
license is acceptable. Treat the hard-coded `arcface-r100-v1` identity as
incorrect model metadata. Benchmark a distributable alternative, such as the
OpenCV YuNet/SFace pair, if public portfolio packaging or commercial use matters.

Store 3-5 L2-normalized float32 templates per person. Score identity by the best
template initially; compare best-template, centroid, and quality-weighted fusion
experimentally. Calibrate a match threshold and a best-versus-second-best margin
on validation data.

FAISS is unnecessary at the expected scale. A synthetic local benchmark on the
current development machine measured exact 512-D NumPy search at approximately
3 microseconds for 100 templates, 16 microseconds for 1,000, and 522 microseconds
for 10,000. Retain a matcher interface so FAISS can be reintroduced if scale is
later demonstrated.

## Passive PAD

MiniFASNetV2 remains the incumbent baseline. Its upstream system uses model/scale
fusion and warns about camera-domain dependence, while FaceAttend currently uses
one model. Compare it under the same protocol with OpenVINO `anti-spoof-mn3`, a
small MobileNetV3 model with documented preprocessing and permissive original
model licensing. CDCN is a useful research baseline when the project can support
its heavier reproduction/training path.

No dataset score establishes deployment quality. Use target-camera bona-fide,
print, and display-replay samples plus an official cross-domain benchmark.

## Persistence and Privacy

Use SQLite with foreign keys and WAL mode for people, embedding metadata,
attendance sessions, attendance records, liveness attempts, model/configuration
versions, consent records, and audit events. Store each float32 embedding as a
BLOB with dimension, dtype, model version, quality, pose bin, and creation time;
load active templates into a contiguous NumPy matrix.

Use a dedicated persistence worker or connections owned by their respective
threads; do not share a default SQLite connection across Qt threads.

Do not retain raw registration frames by default. Re-enroll when changing the
embedding model. If research images are needed, collect them through a separate
opt-in protocol with purpose, encryption, retention, and deletion controls.

## Evaluation Gates

- Recognition: genuine/impostor distributions, ROC/DET, FMR/FNMR, EER,
  identification rank-1, unknown rejection, ambiguity margin, and condition
  breakdowns.
- Passive PAD: APCER by attack type, BPCER, ACER where appropriate, ROC/DET,
  failure-to-process, cross-camera/domain testing, and latency.
- Active liveness: bona-fide completion, attack acceptance, challenge timeout,
  pose stability/error, replay resistance, and completion time.
- System: capture FPS, per-stage median/p95/p99 latency, memory, CPU utilization,
  dropped/stale frames, cold/warm startup, and failure recovery.

## Security Boundary

The proposed RGB system can be evaluated against ordinary print and display
replay attacks. It may only partially mitigate prerecorded challenge videos and
high-quality displays. It does not inherently stop 3D masks, real-time deepfake
reenactment, virtual-camera injection, local database tampering, or compromise of
the host. Those require additional evidence, trusted capture/platform controls,
or stronger sensors.

## Migration Sequence

1. Fix repository tracking and preserve the current behavior with
   characterization tests.
2. Define headless CV and application interfaces.
3. replace and validate head-pose estimation.
4. Extract model lifecycle, liveness, alignment, embedding, and matching.
5. Add SQLite repositories and exact NumPy matching.
6. Build the PySide6 shell, camera worker, inference worker, and state machine.
7. Implement enrollment and check-in parity.
8. Build the evaluation suite and select thresholds/models from evidence.
9. Package and test on target hardware.
10. Only then archive or remove obsolete web and crawler components with
    explicit approval.

