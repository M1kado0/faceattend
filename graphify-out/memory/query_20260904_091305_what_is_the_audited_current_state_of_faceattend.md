---
type: "query"
date: "2026-09-04T09:13:05.333292+00:00"
question: "What is the audited current state of FaceAttend?"
contributor: "graphify"
source_nodes: ["FaceAttend", "MediaPipeActiveLivenessChecker", "ActiveLivenessChallenge", "FAISSStore", "Threshold Calibration"]
---

# Q: What is the audited current state of FaceAttend?

## Answer

FaceAttend is an early explicit self-check-in biometric attendance MVP. Active MediaPipe liveness and passive MiniFASNet must precede InsightFace buffalo_l embedding and user-scoped FAISS matching. No training or validated experiments exist. Current blockers are an untracked database model directory caused by the models ignore rule, invalid head-pose angle scaling, no roster or organizer domain, incomplete consent erasure portability rate-limiting and audit enforcement, uncalibrated thresholds, model licensing uncertainty, and extensive legacy crawler material. Preserve legacy history but label it out of scope.

## Source Nodes

- FaceAttend
- MediaPipeActiveLivenessChecker
- ActiveLivenessChallenge
- FAISSStore
- Threshold Calibration