---
type: "query"
date: "2026-09-06T19:50:54.114845+00:00"
question: "Should LOOK_UP and SMILE be enabled in the default FaceAttend challenge pool after proposal trials?"
contributor: "graphify"
source_nodes: ["ChallengeAction", "MediaPipeActionEvidence", "MiniFASNetPassiveLivenessDetector"]
---

# Q: Should LOOK_UP and SMILE be enabled in the default FaceAttend challenge pool after proposal trials?

## Answer

No. After excluding four invalid pre-fix continuity runs, LOOK_UP active liveness completed 10 of 16 attempts; three timed out and three ended on multiple-face evidence. Nine of 16 passed active plus PAD. Median completed duration was 3.36 seconds. SMILE active liveness completed all 10 intended positive attempts, but only 4 of 10 passed active plus PAD because MiniFASNet rejected smiling windows. The no-smile control timed out correctly, while the already-smiling/no-initial-neutral control incorrectly completed active liveness and was rejected only by PAD. Both actions were validated experimentally but remain disabled by default. Results are single-participant, single-camera proposal evidence, not held-out security evidence.

## Source Nodes

- ChallengeAction
- MediaPipeActionEvidence
- MiniFASNetPassiveLivenessDetector