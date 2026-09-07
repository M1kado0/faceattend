---
type: "query"
date: "2026-09-06T07:50:52.416108+00:00"
question: "How does the reconciled migration plan map randomized liveness to numbered phases as of 2026-09-06?"
contributor: "graphify"
source_nodes: ["ChallengeSession", "FrameEvidence", "MediaPipeLivenessSession", "CameraEvidenceWorker", "ActiveLivenessChallengeEvaluator"]
---

# Q: How does the reconciled migration plan map randomized liveness to numbered phases as of 2026-09-06?

## Answer

The authoritative docs/research/local-first-migration-plan.md integrates randomized sessions, frame evidence, active evaluation, temporal PAD and headless camera composition into Phase 2. Phase 3 owns SQLite, durable sequence audit and NumPy matching; Phase 4 owns Qt; Phases 5-6 reuse the liveness core; Phase 7 owns repeated trials and metrics. Phase 2 remains IN_PROGRESS. Code inspection found FrameEvidence failure_reason is ignored by active and PAD consumers, FaceContinuityTracker is not wired to the runtime, physical yaw inversion exists only on the legacy path, detector quality values are placeholders, and CameraEvidenceWorker has only an injected processor seam. Sequence snapshots are not durable audit records, and unknown/ambiguous state contracts are not local matching or attendance services. Existing adapters and helper tests remain completed at their stated scope. Focused regression command passed 116 tests on 2026-09-06, not camera or security validation. Earlier completion claims are preserved as historical in docs/research/archive/local-first-migration-plan-before-2026-09-06-reconciliation.md and superseded by the current checklist. Green-oval completion is deferred optional UX. Root pose CSVs are untracked but not ignored. No code, thresholds or legal documents changed.

## Source Nodes

- ChallengeSession
- FrameEvidence
- MediaPipeLivenessSession
- CameraEvidenceWorker
- ActiveLivenessChallengeEvaluator