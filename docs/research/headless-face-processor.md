# Using the headless face processor

`src/faceattend/vision/face_analyzer.py` now supplies the composition boundary.
It does not start a GUI, open a camera, write attendance, or save images.

## Public API

- `ModelArtifact(path, metadata)`: a selected local file with an expected SHA-256.
- `RuntimeModelManifest`: detector, embedding, passive PAD, and MediaPipe artifacts.
- `create_face_analyzer(manifest, quality_config=...)`: constructs the existing real adapters after
  verifying files, metadata, and the current baseline ONNX tensor interfaces.
- `processor.analyze(frame)`: detection only, matching the existing protocol.
- `processor(frame)`: detection plus MediaPipe matrix pose/action evidence.
- `processor.extract_embedding(evidence, active=..., passive=...)`: separate
  alignment/embedding step requiring both passed liveness results and matching
  liveness model versions. Rejects incompatible alignment/embedding outputs.
- `processor.close()`: closes native MediaPipe resources and prevents reuse.

The detector, embedder, and PAD adapters load lazily once per runtime instance;
MediaPipe loads once per capture stream. Use one owning inference thread and
create a fresh processor for a new stream. Do not share it concurrently.

## Selecting model files

The current contract supports SCRFD-10G's nine score/bbox/keypoint outputs,
112×112 BGR alignment and a normalized float32 512D embedding, plus the current
80×80 three-logit MiniFASNet export. Class index 1 means real in the existing PAD
wrapper; tensor shapes alone cannot establish that semantic interpretation.

Create each artifact explicitly, for example:

```python
from pathlib import Path
from faceattend.vision.face_analyzer import ModelArtifact, RuntimeModelManifest
from faceattend.vision.types import ModelMetadata

detector = ModelArtifact(
    Path("models/det_10g.onnx"),
    ModelMetadata("insightface-detector", "buffalo_l-det_10g", approved_detector_sha256),
)
# Construct embedding, passive, and landmarker artifacts the same way.
manifest = RuntimeModelManifest(detector, embedding, passive, landmarker)
```

The `approved_*` values are required inputs, not placeholders accepted by the
runtime. Obtain pins from the selected artifact record; verify provenance and
licensing separately. Computing a hash from a local file identifies that file
but does not prove it is trustworthy. Missing files, unknown hashes, mismatched
adapter identities, external ONNX initializer weights, and incompatible tensor
interfaces are rejected. MediaPipe checks its task archive at native loading.
Keep files immutable while the runtime is alive; this is not tamper-proof loading.

## Camera-worker seam

```python
from faceattend.camera.worker import CameraEvidenceWorker
from faceattend.vision.face_analyzer import create_face_analyzer

processor = create_face_analyzer(manifest)
try:
    evidence = CameraEvidenceWorker(camera_index=1).run(processor, max_frames=30)
    # Inspect structured outcomes, not raw images/embeddings in logs.
    print([item.failure_reason for item in evidence])
finally:
    processor.close()
```

This is a bounded diagnostic example, not a completed attendance workflow. It
keeps frames in memory until references are released. The existing camera worker
still accumulates returned evidence; do not run it unbounded for a GUI.
For repository scripts, use `PYTHONPATH=src uv run python ...` if the package is
not installed editable. Tests already configure the source path.

## Evidence and safety boundaries

Per-frame processing never runs PAD or embedding inference. It returns measured
face quality, raw matrix pose and blink/smile events, or explicit failures.
Invalid timestamps return failure evidence; model loading/input exceptions are
converted to terminal failures by the composed session callable. Native process
crashes are not recoverable Python exceptions. Raw yaw is mapped only at the
challenge adapter, not during evidence collection.

For the guarded path, pass a `MediaPipeLivenessSession` configured with
`processor=processor` to the camera worker. Its `EvidenceStreamGuard` enforces
freshness, continuity, exact duplicates, finite evidence, quality and failure
propagation before active observation and PAD buffering. Read the overall
`session.result`, not just the active result; finalize PAD only after active
completion. See [quality/session policy](frame-quality-and-session-policy.md)
for provisional defaults, warning semantics and synthetic test scope.

The embedding method checks stage kinds, passed decisions, model versions, and
that the supplied frame evidence is this processor's latest valid observation.
The application must bind both decisions to the same continuous session; plain
Python result objects are not security credentials. No matching, persistence,
consent, or audit workflow is implemented by this module.

## Verification on 2026-09-06

- 176 full-suite tests passed; scoped Ruff and `mypy src` passed.
- Synthetic SDK outputs exercise the real detector/aligner/embedder adapters.
- Real local artifact layout/checksum smoke succeeded, followed by SCRFD and
  MediaPipe processing a synthetic blank frame with no faces.
- MediaPipe native graphics initialization failed inside the sandbox but worked
  outside it. This environment limitation is not a handled Python exception.
- No webcam capture, real-face session, threshold calibration, or attack trial
  was performed by this change.
