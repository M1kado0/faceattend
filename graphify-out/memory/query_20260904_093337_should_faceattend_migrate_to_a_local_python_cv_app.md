---
type: "query"
date: "2026-09-04T09:33:37.429027+00:00"
question: "Should FaceAttend migrate to a local Python CV application?"
contributor: "graphify"
source_nodes: ["ADR-001: Local-First Desktop CV Architecture", "Python Desktop CV Migration Feasibility", "MediaPipe Active Liveness", "MiniFASNet v2 Passive Liveness", "Threshold Calibration"]
---

# Q: Should FaceAttend migrate to a local Python CV application?

## Answer

Recommend HYBRID: make the primary product a local PySide6 desktop application over a headless Python CV and evaluation core, while preserving the existing web stack until parity. Use one camera-owning worker, one long-lived inference worker with latest-frame backpressure, SQLite for metadata and audit events, 3-5 normalized templates per person, and exact NumPy cosine search at expected scale. Keep active plus passive liveness for registration and check-in until an evidence-backed ADR changes policy. Treat MiniFASNetV2 as a baseline and compare it with OpenVINO anti-spoof-mn3 on target-camera attacks. Replace and validate the current head-pose estimator. Do not retain raw images by default. Main losses are remote and multi-device operation; main gains are CV focus, lower latency, testability, and simpler reproducible experiments.

## Source Nodes

- ADR-001: Local-First Desktop CV Architecture
- Python Desktop CV Migration Feasibility
- MediaPipe Active Liveness
- MiniFASNet v2 Passive Liveness
- Threshold Calibration