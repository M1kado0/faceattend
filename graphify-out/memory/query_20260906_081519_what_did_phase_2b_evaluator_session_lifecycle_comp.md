---
type: "query"
date: "2026-09-06T08:15:19.841741+00:00"
question: "What did Phase 2B evaluator session lifecycle complete?"
contributor: "graphify"
source_nodes: ["ActiveLivenessChallengeEvaluator", "ChallengeSession", "FrameEvidence"]
---

# Q: What did Phase 2B evaluator session lifecycle complete?

## Answer

ActiveLivenessChallengeEvaluator now checks its shared ChallengeSession clock before every evidence branch, including neutral waiting, face failures, action matching, and dwell. complete_current accepts a timestamp already checked by the evaluator without double-observing it. Deterministic evidence-driven tests cover BLINK, SMILE, TURN_LEFT, TURN_RIGHT, LOOK_UP, session timeout while waiting for neutral, and challenge timeout during dwell. 183 full tests pass, Ruff and mypy src pass. Reset remains evaluator-local; application services must create/restart randomized sessions.

## Source Nodes

- ActiveLivenessChallengeEvaluator
- ChallengeSession
- FrameEvidence