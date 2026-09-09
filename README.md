# FaceAttend

> A local-first desktop face-attendance prototype built to explore real-time
> computer vision: face recognition, active liveness, passive presentation-attack
> detection, and auditable local attendance records.

FaceAttend is a **PySide6 desktop application**. It runs on one local computer
with a webcam, local model files, and a local SQLite database. It does **not**
need a browser, backend server, REST API, WebSocket connection, Docker, or
networked database.

## What it does

### Enrollment

1. The operator enters a person's name and records explicit consent.
2. The person completes randomized active-liveness instructions, such as a
   blink or a head movement.
3. The person faces the camera and holds still while the application collects a
   short neutral-frame window for passive liveness.
4. When both liveness checks pass, FaceAttend selects 3–5 quality/pose-diverse
   face templates and stores them locally.
5. Likely duplicate enrollment is blocked with a generic message; it does not
   disclose who may already be registered.

### Attendance check-in

1. The operator selects or creates an attendance session.
2. The person completes the same active-liveness and neutral passive-liveness
   sequence.
3. Only then does FaceAttend create an embedding and search compatible,
   consenting enrolled templates.
4. It records attendance only for a confident, unambiguous match. Unknown,
   ambiguous, duplicate, failed-liveness, no-face, and multiple-face outcomes
   are explicit rejections.

## Features

- Local PySide6 interface with live camera preview and structured guidance.
- MediaPipe landmarks, transformation-matrix head pose, blendshapes, blink
  evidence, face-quality checks, and randomized active challenges.
- MiniFASNetV2 temporal passive liveness over a short neutral-frame window.
- InsightFace `buffalo_l` face detection, alignment, and normalized 512D face
  embeddings.
- SQLite records for people, consent, templates, sessions, check-ins, model and
  configuration versions, liveness attempts, and audit events.
- Exact NumPy cosine matching with model-compatibility filtering, unknown
  rejection, ambiguity rejection, and duplicate check-in handling.
- Headless CV and application layers separated from Qt GUI code.
- A bounded latest-frame camera/inference pipeline to avoid stale-frame buildup.

## Architecture

```text
PySide6 desktop UI
        |
        v
camera worker -> newest-frame buffer -> inference worker
                                        |
                                        v
                              headless CV/application core
                              - detection and quality checks
                              - MediaPipe active liveness
                              - MiniFASNetV2 passive liveness
                              - alignment and embeddings
                              - matching and workflow decisions
                                        |
                                        v
                                      SQLite
```

The GUI presents instructions and results. It never decides whether liveness
has passed; those decisions come from the headless CV/application core.

## Requirements

- Python 3.11+
- A working webcam
- [uv](https://docs.astral.sh/uv/)
- The following local model artifacts in `models/`:

```text
models/
├── det_10g.onnx
├── w600k_r50.onnx
├── MiniFASNetV2.onnx
└── face_landmarker_v2_with_blendshapes.task
```

Model artifacts are intentionally not downloaded automatically. Review their
licenses and provenance before redistributing this project.

## Installation

```bash
git clone https://github.com/M1kado0/faceattend.git
cd faceattend
uv sync
cp .env.example .env
```

Edit `.env` for your webcam. The current development machine uses camera index
`1`, but yours may be `0` or another number:

```env
FACEATTEND_CAMERA_INDEX=1

# Optional local overrides
# FACEATTEND_DATABASE_PATH=./data/faceattend.sqlite3
# FACEATTEND_MODEL_DIR=./models
```

`.env` is ignored by Git. The app loads it automatically at startup.

## Run

```bash
uv run python -m faceattend
```

Or, after `uv sync`, use the installed command:

```bash
uv run faceattend
```

The first launch creates the local database at:

```text
data/faceattend.sqlite3
```

## Demo flow

1. Open the application.
2. Create an attendance session.
3. Select **Register**, enter a name, and grant explicit consent.
4. Follow the active-liveness instructions.
5. Face the camera and hold still for passive liveness.
6. Select **Check in**, choose the session, and repeat the liveness sequence.
7. Try another check-in to observe the duplicate-attendance response.

## Repository layout

```text
src/faceattend/
├── application/  # registration and attendance workflow coordination
├── camera/       # camera ownership and newest-frame buffering
├── gui/          # PySide6 windows, views, and Qt workers
├── persistence/  # SQLite schema, records, and repositories
├── vision/       # detection, quality, liveness, embeddings, matching
└── evaluation/   # reusable headless evaluation helpers

models/           # local model artifacts; not downloaded automatically
data/             # local SQLite data; do not commit
```

## Privacy and security boundaries

FaceAttend handles biometric data. Use it only with explicit consent and only
for legitimate, visible, user-initiated check-ins.

- Raw camera frames are held in memory for normal enrollment/check-in and are
  not stored by default.
- Embeddings, consent records, liveness outcomes, attendance records, model
  metadata, and audit events are stored locally in SQLite.
- The local database is sensitive biometric data. Protect the computer and
  database file, and provide deletion when it is no longer needed.
- This is an RGB-webcam prototype, **not Apple Face ID** and not a production
  biometric-security product.
- Active challenges and MiniFASNetV2 may help with basic presentation attacks,
  but they do not guarantee resistance to high-quality replays, deepfakes,
  masks, camera injection, application tampering, or database theft.
- Recognition and liveness thresholds are provisional project settings, not
  calibrated production-security guarantees.

## Status

FaceAttend is a functional portfolio/research prototype. The local desktop
workflow has been exercised with real webcam enrollment, restart/template
reload, check-in, and explicit rejection states.

It still needs rigorous, consented evaluation before any performance or
security claims: repeated bona-fide trials, presentation-attack trials,
threshold calibration, recognition metrics, latency measurements, and an
independent legal/privacy review before real deployment.

## License

This repository is currently marked as proprietary. Third-party model and
library licenses apply independently.
