---
type: "query"
date: "2026-09-06T19:16:40.856661+00:00"
question: "What did the 2026-09-07 Phase 2 liveness hardening complete?"
contributor: "graphify"
source_nodes: ["MediaPipeLivenessSession", "TemporalPassiveLivenessSession", "CameraEvidenceWorker"]
---

# Q: What did the 2026-09-07 Phase 2 liveness hardening complete?

## Answer

Deterministic tests now cover neutral return, hysteresis, reset, fresh open-closed-open blink evidence, and sustained smile. Temporal MiniFASNet sampling defaults to at least 3 samples spanning 200 ms at 100 ms intervals with a 30-sample cap, preserves early suspicious evidence, and clears frames on cancellation/failure/finalization. A shared gate prevents embedding and matching after either liveness failure. Camera evidence streams through callbacks with explicit bounded or zero result retention and cleanup on stop/read/processor failure. Root pose CSVs and local biometric trial outputs are ignored. A consented metrics-only trial harness and proposed protocol exist, but real-camera repeated bona-fide calibration and print/phone/prerecorded attack results remain unmeasured.

## Source Nodes

- MediaPipeLivenessSession
- TemporalPassiveLivenessSession
- CameraEvidenceWorker