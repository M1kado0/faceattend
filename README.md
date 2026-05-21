# FaceGuard — Face Monitoring & Match Dashboard

> Protect users' photos online by finding where their images appear across the public internet — with built-in antispoofing, liveness verification, and continuous match monitoring.

---

## Overview

FaceGuard is a production-grade pipeline that:

1. **Crawls** public images from the open web at scale
2. **Verifies liveness** during enrollment (anti-spoofing)
3. **Detects and embeds** faces using a high-throughput ML inference stack
4. **Indexes** hundreds of millions of face embeddings in a vector database
5. **Scans enrolled faces** against the index and persists match results
6. **Monitors** the web continuously, updating each user's match dashboard as new images are indexed

The web surface is a **public site** where users enroll their face once and review dynamically updated match results.

The web app renders server-side HTML using **Jinja2** templates with **HTMX** for interactivity. No React, no TypeScript, no build step. Pure Python end-to-end.

---

## Why Liveness & Antispoofing?

This system processes biometric data with significant abuse potential. Without liveness checks, anyone could enroll a stranger's face and use the platform as a stalking tool.

- **At enrollment**: liveness proves the person adding a face physically owns it
- **During monitoring**: matches are generated only for liveness-verified enrollments, avoiding a separate manual search flow
- **Antispoofing models** reject printed photos, screen replays, masks, and deepfakes

Approach: **passive liveness** (single image/short clip analysis) as default, **active liveness** (blink/turn challenges) as fallback when passive confidence is low.

---

## Architecture

```
┌──────────────────┐
│   Public Site    │
│ FastAPI+Jinja2   │
│     + HTMX       │
└────────┬─────────┘
         │ (HTTP / JSON)
            ┌───────▼────────┐
            │  Backend API   │
            │   (FastAPI)    │
            │     Auth       │
            └───────┬────────┘
                    │
        ┌───────────┼───────────────────────┐
        │           │                       │
        ▼           ▼                       ▼
  ┌─────────┐  ┌─────────────────┐   ┌──────────────┐
  │ Liveness│  │  Face Inference │   │   Vector     │
  │  Check  │  │  Detect→Align   │   │   Search     │
  │ (ONNX)  │  │  →Embed (TRT)   │   │ Milvus/Qdrant│
  └─────────┘  └─────────────────┘   └──────┬───────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │ Match Results  │
                                   │  + Clustering  │
                                   └────────┬───────┘
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │   Takedown     │
                                   │   Workflow     │
                                   └────────────────┘

┌──────────────┐      ┌──────────────────┐    ┌────────────┐
│ Web Crawler  │─────▶│  Ingest Queue    │───▶│ Inference  │
│  Distributed │      │ (Kafka/Celery)   │    │  Workers   │
│   + Proxies  │      └──────────────────┘    └─────┬──────┘
└──────┬───────┘                                    │
       │            ┌─────────────┐                 │
       └───────────▶│ Image Store │                 │
                    │  (S3/MinIO) │◄────────────────┘
                    └─────────────┘

                                     ┌────────────────────┐
                                     │ Notification System│
                                     │  Email · Webhooks  │
                                     └────────────────────┘
```

---

## Stack

| Layer | Tooling |
|---|---|
| Web UI (public site) | **FastAPI + Jinja2 + HTMX** — pure Python, no build step |
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
| Liveness / Antispoofing | MiniFASNet / Silent-Face-Anti-Spoofing |
| Perceptual Hashing | pHash / dHash via **imagehash** |
| Vector DB | **FAISS** to start (in-memory, simple) → **Milvus / Qdrant** for scale |
| GPU Serving | Triton Inference Server or Ray Serve (later) |
| Crawling | **Scrapy** + Playwright, rotating proxies |
| Queue | Celery + Redis for simplicity → Kafka when scaling |
| Storage | MinIO locally → S3 in production |
| Notifications | Email (Postmark) + webhooks |
| Monitoring | Prometheus + Grafana, OpenTelemetry |
| Orchestration | Docker Compose → Kubernetes |

---

## Core Features

### For end users (public site)
- 🪪 **Account registration** with email verification
- 📸 **Face enrollment** with passive/active liveness check
- 📊 **Dynamic match dashboard** — see where your enrolled face appears, sorted by confidence
- 🔔 **Continuous monitoring** — dashboard updates and optional alerts for new matches
- 🗑 **Full GDPR controls** — export your data, delete your account, withdraw consent
- 💳 **Subscription tiers** (free + paid)

### Cross-cutting
- 🔐 **Authentication** for liveness-gated enrollment and match review
- 🛡 **Rate limiting** per IP, per account, per API key
- 🗝 **Encryption at rest** for biometric embeddings
- 📦 **Model versioning + re-embedding pipeline**

---

## Project Structure

```
faceguard/
├── backend/              # JSON API service (FastAPI)
│   ├── api/              # REST endpoints
│   ├── indexer/          # Vector index management
│   ├── monitor/          # Re-crawl & notification scheduler
│   ├── takedown/         # Future takedown utilities
│   ├── audit/            # Audit logging
│   └── CLAUDE.md
├── crawler/              # Distributed web crawler
│   ├── spiders/
│   └── CLAUDE.md
├── ml/                   # Inference pipeline
│   ├── detection/
│   ├── alignment/
│   ├── embedding/
│   ├── liveness/
│   ├── clustering/
│   ├── phash/
│   └── CLAUDE.md
├── frontend/             # FastAPI public site with Jinja2 + HTMX
│   ├── public-site/      # End-user-facing app
│   └── CLAUDE.md
├── infra/                # Docker, K8s, Terraform
├── scripts/
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

---

## Getting Started

### Prerequisites (anticipated)
- Python 3.11+
- Docker & Docker Compose
- CUDA-capable GPU eventually (CPU is fine for Phases 0-5)
- `uv` for Python deps

### Quick start (target)
```bash
git clone https://github.com/your-org/faceguard.git
cd faceguard
cp .env.example .env
docker compose up
# Public site:   http://localhost:8000
# Backend API:   http://localhost:8002
```

---

## Legal & Compliance

Biometric data is **Article 9 special category data** under GDPR — stricter protections required.

- ✅ **Lawful basis**: explicit consent at enrollment
- ✅ **Data minimization**: store embeddings, not raw face images, beyond processing window
- ✅ **Right to erasure**: full account + embedding deletion in <30 days
- ✅ **Right to portability**: data export available
- ✅ **Purpose limitation**: only monitor faces enrolled through liveness verification
- ✅ **Audit trail**: every enrollment, match scan, and biometric data operation logged with actor, target, timestamp, and justification
- ✅ **EU AI Act**: face recognition may fall under high-risk; conformity assessment required pre-launch
- ✅ **Robots.txt respect** at crawl time
- ✅ **Source attribution**: only publicly accessible URLs crawled

**A legal review is required before production deployment.**

---

## Documentation

- [`CLAUDE.md`](./CLAUDE.md) — Instructions for Claude Code (root)
- [`backend/CLAUDE.md`](./backend/CLAUDE.md) — API, indexer, monitor, audit
- [`crawler/CLAUDE.md`](./crawler/CLAUDE.md) — Crawler architecture & ethics
- [`ml/CLAUDE.md`](./ml/CLAUDE.md) — Inference, liveness, clustering
- [`frontend/CLAUDE.md`](./frontend/CLAUDE.md) — Public site (Jinja2 + HTMX)
- [`docs/adr/`](./docs/adr) — Architecture Decision Records

---

## License
