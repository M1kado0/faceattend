---
type: "query"
date: "2026-09-06T08:07:41.369824+00:00"
question: "What did Phase 2A headless face composition complete on 2026-09-06?"
contributor: "graphify"
source_nodes: ["HeadlessFaceAnalyzer", "MediaPipeFrameLandmarker", "RuntimeModelManifest", "FrameEvidence", "CameraEvidenceWorker"]
---

# Q: What did Phase 2A headless face composition complete on 2026-09-06?

## Answer

HeadlessFaceAnalyzer now composes existing detector, aligner, embedder, MiniFASNet and a synchronous MediaPipeFrameLandmarker. It is callable by CameraEvidenceWorker, emitting raw matrix pose/action FrameEvidence or explicit face/landmark/pose failure. It does not run PAD or embedding per frame. Separate extract_embedding requires both passed kind/version-compatible liveness results and latest valid evidence; trusted application orchestration must still bind decisions to the same session. ModelArtifact and RuntimeModelManifest enforce explicit expected SHA256 pins, rehashed files, adapter metadata and baseline ONNX role/tensor contracts; alignment/embedding outputs are checked. Constructor and real factory validate these contracts. Provenance, label semantics, immutable weights and licensing remain separate requirements. 176 full-suite tests passed, scoped Ruff and mypy src passed. Native local SCRFD/MediaPipe blank-frame smoke succeeded outside sandbox after sandbox graphics abort. No camera, biometric recording, calibration or attack evaluation. Plan Phase 2A tasks checked; continuity, yaw mapping, quality, duplicate failure propagation and full active/PAD integration remain open. See docs/research/headless-face-processor.md.

## Source Nodes

- HeadlessFaceAnalyzer
- MediaPipeFrameLandmarker
- RuntimeModelManifest
- FrameEvidence
- CameraEvidenceWorker