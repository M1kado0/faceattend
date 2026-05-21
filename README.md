# FaceAttend — Liveness-Gated Face Attendance System

> FaceAttend is a Python-first attendance system that uses face registration, face recognition, and active/passive liveness checks to record attendance while reducing spoofing and proxy check-ins.

---

## Overview

FaceAttend is built around a controlled attendance flow:

1. **Register a person** with explicit consent and a face photo.
2. **Verify liveness at registration** so a printed photo or screen replay cannot be enrolled.
3. **Store face embeddings**, not raw face crops, for recognition.
4. **Create attendance sessions** for classes, labs, events, meetings, or workplace shifts.
5. **Verify liveness at check-in** using active and passive checks.
6. **Match the live face** against registered embeddings.
7. **Record attendance** with timestamp, session, confidence score, and audit trail.

The web surface is a **server-rendered FastAPI site** for registration, check-in, session management, and attendance review.

The web app renders server-side HTML using **Jinja2** templates with **HTMX** for interactivity. No React, no TypeScript, no build step. Pure Python end-to-end.

---

## Why Active And Passive Liveness?

Face attendance systems are easy to abuse if they only compare a face image against a database. Someone could hold up a printed photo, replay a video, or use another person’s image.

- **Passive liveness** analyzes a still image or short clip for spoofing signals such as printed photos, screens, masks, or deepfakes.
- **Active liveness** asks the person to complete a challenge such as blinking twice, turning their head, or smiling.

For the MVP, active liveness starts with **blink twice** using MediaPipe face landmarks. Passive liveness uses the MiniFASNet-based antispoofing path already in the ML layer.

Attendance should only be recorded after:

```text
active challenge completed
+ passive spoof check passed
+ face matched to a registered identity
= valid attendance check-in
```

---

## Core Flows

### Registration

1. Admin or user opens the registration page.
2. User provides identity details such as name, email, class/group, or employee/student ID.
3. User completes liveness capture.
4. System detects and embeds the face.
5. System stores the embedding and registration metadata.
6. Audit log records consent and biometric enrollment.

### Attendance Check-In

1. User opens a session check-in page.
2. System issues an active liveness challenge.
3. User records a short webcam clip and submits it.
4. ML service verifies active liveness and passive liveness.
5. Backend embeds the live face and searches the registered-face index.
6. If confidence is high enough, attendance is recorded.
7. UI shows success, duplicate check-in, failed liveness, or no registered match.

### Admin / Instructor Review

Admins or instructors can:

- create attendance sessions
- view check-ins by session
- export attendance records
- review failed or suspicious check-in attempts
- delete a person’s biometric data when required

---

## Architecture

```text
┌────────────────────┐
│      Web App       │
│ FastAPI + Jinja2   │
│       + HTMX       │
└─────────┬──────────┘
          │ HTTP
          ▼
┌────────────────────┐
│    Backend API     │
│ FastAPI + SQLModel │
│ Auth, Sessions,    │
│ Attendance, Audit  │
└─────────┬──────────┘
          │
          ├──────────────┐
          ▼              ▼
┌─────────────────┐  ┌──────────────────┐
│   ML Service    │  │   Vector Store    │
│ Face detect,    │  │ FAISS initially,  │
│ embed, liveness │  │ Qdrant/Milvus later│
└─────────────────┘  └──────────────────┘
          │
          ▼
┌────────────────────┐
│ PostgreSQL + Audit │
│ people, sessions,  │
│ check-ins, logs    │
└────────────────────┘
```

---

## Stack

| Layer | Tooling |
|---|---|
| Web UI | **FastAPI + Jinja2 + HTMX** — pure Python, no build step |
| Styling | **Tailwind CSS** via CDN, **DaisyUI** for pre-built components |
| Webcam capture | ~50 lines of vanilla **JavaScript** (only place JS is needed) |
| Forms | **WTForms** + Pydantic validation |
| Auth | **fastapi-users** (drop-in auth) |
| Backend API | **FastAPI** (Python 3.11+) |
| ORM | **SQLModel** (Pydantic + SQLAlchemy, same author as FastAPI) |
| RDBMS | **PostgreSQL** |
| Cache & rate limit | **Redis** |
| Face Detection | RetinaFace / SCRFD (ONNX) — via **insightface** library |
| Face Alignment | 5-point landmark alignment (included with insightface) |
| Embeddings | ArcFace / AdaFace — also via **insightface** |
| Passive liveness / Antispoofing | MiniFASNet / Silent-Face-Anti-Spoofing |
| Active liveness | MediaPipe face landmarks, starting with blink detection |
| Vector DB | **FAISS** to start → **Qdrant / Milvus** for scale |
| GPU Serving | Triton Inference Server or Ray Serve (later) |
| Queue | Celery + Redis for simplicity → Kafka when scaling |
| Storage | MinIO locally → S3 in production |
| Notifications | Email (Postmark) + webhooks |
| Monitoring | Prometheus + Grafana, OpenTelemetry |
| Orchestration | Docker Compose → Kubernetes |

---

## Core Features

### For users
- 🪪 **Face registration** with explicit consent
- ✅ **Liveness-gated check-in** for attendance sessions
- 📊 **Attendance result feedback**: success, duplicate, failed liveness, or no match
- 🗑 **Privacy controls** for export/deletion where applicable

### For admins / instructors
- 🧑‍🏫 **Attendance session management**
- 📋 **Attendance dashboard** by class, event, or shift
- 📤 **CSV/export support** for attendance records
- 🔎 **Review queue** for failed or suspicious check-ins

### Cross-cutting
- 🔐 **Authentication** for registration, check-in, and admin review
- 🛡 **Rate limiting** per IP, per account, per API key
- 🗝 **Encryption at rest** for biometric embeddings
- 📦 **Model versioning + re-embedding pipeline**

---

## Project Structure

```
faceattend/
├── backend/              # Backend API, attendance, registration, audit
│   ├── api/              # FastAPI app and routes
│   ├── db/               # SQLModel models and migrations
│   ├── indexer/          # Vector index management
│   ├── audit/            # Audit logging
│   └── CLAUDE.md
├── ml/                   # Inference pipeline
│   ├── liveness/         # Passive and active liveness
│   ├── pipeline/         # Face embedding and inference helpers
│   ├── serving/          # ML HTTP service
│   └── CLAUDE.md
├── frontend/             # FastAPI public site with Jinja2 + HTMX
│   ├── public-site/      # End-user-facing app
│   └── CLAUDE.md
├── infra/                # Docker, K8s, Terraform
├── scripts/              # Dev utilities and demo seed scripts
├── tests/
├── docs/
│   ├── adr/              # Architecture Decision Records
│   ├── legal/            # GDPR, EU AI Act notes
│   └── api/              # OpenAPI specs
├── CLAUDE.md             # Root instructions
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

Some older modules may still reflect the previous web-monitoring direction while the project is being migrated. New work should follow the attendance-system model described here.

---

## Getting Started

### Prerequisites (anticipated)
- Python 3.11+
- Docker & Docker Compose
- CUDA-capable GPU eventually (CPU is fine for Phases 0-5)
- `uv` for Python deps

### Quick start (target)
```bash
git clone https://github.com/your-org/faceattend.git
cd faceattend
cp .env.example .env
docker compose up
# Public site:   http://localhost:8000
# Backend API:   http://localhost:8002
```

---

## Legal, Privacy, And Safety

Biometric data is **Article 9 special category data** under GDPR — stricter protections required.

- ✅ **Lawful basis**: explicit consent at face registration
- ✅ **Data minimization**: store embeddings, not raw face images, beyond processing window
- ✅ **Right to erasure**: registration + embedding deletion when required
- ✅ **Right to portability**: data export available
- ✅ **Purpose limitation**: use registered faces only for explicit attendance check-ins
- ✅ **Audit trail**: every registration, liveness attempt, match decision, check-in, export, and deletion logged with actor, target, timestamp, and justification
- ✅ **EU AI Act**: face recognition may fall under high-risk; conformity assessment required pre-launch
- ✅ **No hidden surveillance**: check-ins must be explicit user actions

**A legal review is required before production deployment.**

---

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — Instructions for Claude Code (root)
- [`backend/CLAUDE.md`](./backend/CLAUDE.md) — API, attendance, registration, audit
- [`ml/CLAUDE.md`](./ml/CLAUDE.md) — Inference, liveness, clustering
- [`frontend/CLAUDE.md`](./frontend/CLAUDE.md) — Public site (Jinja2 + HTMX)
- [`docs/adr/`](./docs/adr) — Architecture Decision Records

---

## Current MVP Priorities

1. Active liveness with MediaPipe blink detection.
2. Passive liveness integration with registration/check-in.
3. Face registration with embedding storage.
4. Attendance session and check-in models.
5. Match live check-in face against registered embeddings.
6. Attendance dashboard and export.

---

## Demo Strategy

Use consented local data:

- register yourself or a consenting test user
- record active liveness through the webcam
- seed a small local set of registration/check-in examples
- show successful, duplicate, spoof-failed, and no-match check-ins

Do not demo with people who did not consent.

---

## License
