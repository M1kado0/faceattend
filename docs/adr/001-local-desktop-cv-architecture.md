# ADR-001: Local-First Desktop CV Architecture

**Status:** Accepted
**Date:** 2026-09-04
**Deciders:** Project owner

## Context

FaceAttend is currently split across a Jinja2/HTMX web application, a FastAPI
backend, an HTTP ML service, PostgreSQL, and FAISS. The intended portfolio value
has shifted toward computer-vision engineering: head pose, liveness, face
recognition, calibration, real-time performance, and reproducible evaluation.

The present application is a single-user, self-owned attendance prototype. It
does not yet have the organizer, roster, remote administration, or multi-device
requirements that would justify its service boundaries. The project owner has
approved an incremental migration while requiring the web implementation to be
preserved until verified parity.

## Options Considered

1. Keep the web/service architecture — preserves a route to multi-device and
   remote operation, but consumes substantial effort outside the stated CV goal.
2. Replace everything with a monolithic Qt/OpenCV loop — removes infrastructure,
   but couples UI, capture, inference, persistence, and experimentation.
3. Use a local PySide6 desktop UI over a headless Python CV/application core —
   removes unnecessary network boundaries while keeping CV logic independently
   testable and usable by evaluation commands.

## Decision

Choose option 3: a **local-first hybrid** consisting of a PySide6 desktop UI and
a headless Python core. "Hybrid" here does not mean retaining the current web
stack in the primary product. It means separating interactive desktop delivery
from reusable headless CV and evaluation modules.

The initial runtime should use:

- one camera-owning worker;
- one long-lived inference worker with latest-frame backpressure;
- GUI updates only on Qt's main thread;
- SQLite for local domain metadata and audit events;
- 3-5 normalized embedding templates per person;
- exact NumPy cosine search for the expected small enrollment set;
- active and passive liveness for both registration and check-in until a later
  threat-model evaluation and approved ADR justify changing that policy.

MiniFASNetV2 remains a baseline, not an accepted production PAD model. It must
be compared with at least one reproducible lightweight alternative and tested on
target-camera bona-fide and attack samples. The current head-pose estimator must
be replaced or corrected and validated before it contributes to liveness.

## Consequences

- CV inference avoids video encoding and inter-service HTTP overhead.
- The application becomes better aligned with a CV portfolio and reproducible
  local experiments.
- Remote access, centralized multi-device attendance, and browser portability
  are intentionally deferred.
- Desktop packaging, camera drivers, model distribution, and local database
  integrity become explicit engineering concerns.
- Local operation reduces network exposure but does not remove biometric
  privacy, consent, erasure, audit, or model-license obligations.
- The existing web and legacy components must be preserved until feature
  characterization and migration parity are complete.

## Assumptions Requiring Reconsideration

Revisit this decision if project evidence or requirements invalidate any of:

1. single-device/local attendance is the intended operating model;
2. non-commercial research use is acceptable or distributable model licenses
   are selected;
3. the target camera and hardware class;
4. the required presentation-attack threat model;
5. whether organizer/roster behavior is in scope.

## References

- [Feasibility research note](../research/python-desktop-cv-feasibility.md)
- [Qt for Python documentation](https://doc.qt.io/qtforpython-6/)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Silent Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [OpenVINO anti-spoof-mn3](https://docs.openvino.ai/2023.3/omz_models_model_anti_spoof_mn3.html)
- [ISO/IEC 30107-3](https://www.iso.org/standard/79520.html)
