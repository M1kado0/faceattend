# AGENTS.md — Root Instructions for Claude Code

This is the **root** instruction file. It covers project-wide rules. Claude Code reads nested `CLAUDE.md` files automatically when working inside subdirectories — refer to those for component-specific rules.

| When working on... | Read this |
|---|---|
| Backend API, attendance, registration, audit | [`backend/AGENTS.md`](./backend/AGENTS.md) |
| Face detection, liveness, embeddings | [`ml/AGENTS.md`](./ml/AGENTS.md) |
| Public site (Jinja2 + HTMX) | [`frontend/AGENTS.md`](./frontend/AGENTS.md) |

---

## Project Summary

**FaceAttend** is now an attendance system with face registration, face recognition, and active/passive liveness checks.

The core product is:

```text
register a consenting face
verify active/passive liveness
match live check-in to registered embedding
record attendance with an audit trail
```

This is a biometric data product. Liveness is not optional. Attendance must never be recorded from a face image alone.

Project is in **early design phase** — no production code exists yet. Prefer clean structure over quick hacks.

**The developer is a beginner in frontend, knows FastAPI on the backend.** Pick simple, Python-first solutions wherever possible.

---

## Top-Level Architecture Principles

1. **Explicit check-ins only**: no hidden background surveillance or passive tracking.
2. **Liveness before attendance**: active and passive liveness must pass before recording a check-in.
3. **Registration before recognition**: attendance matching should search only registered, consenting identities.
4. **Embeddings in vector DB**: never store raw face crops in the vector store.
5. **Audit every biometric operation**: registration, liveness attempt, match decision, check-in, export, and deletion.
6. **Fail loudly**: ambiguous recognition, failed liveness, and duplicate check-ins should return explicit states.
7. **Python-first**: keep the stack Python wherever possible. The only frontend JavaScript should be webcam capture and minimal HTMX support.

---

## Repository Layout

```
faceattend/
├── backend/        # FastAPI API, attendance, registration, audit
├── ml/             # Face detection, embeddings, active/passive liveness
├── frontend/       # FastAPI public site (Jinja2 + HTMX)
│   └── public-site/
├── infra/          # Docker, K8s, Terraform
├── scripts/        # Dev utilities, benchmarks, migrations
├── tests/          # Cross-component integration tests
├── docs/
│   ├── adr/        # Architecture Decision Records
│   ├── legal/      # GDPR / EU AI Act compliance notes
│   └── api/        # OpenAPI specs
├── pyproject.toml  # Python deps (uv-managed)
├── docker-compose.yml
└── .env.example
```

Some existing files may still reflect the previous public-web monitoring/crawler direction. Treat them as legacy until migrated. New work should target the attendance-system model.

**Important**: `frontend/` is a Python web app that renders server-side HTML. It is not a JS/TS project.

---

## Service Ports (dev)

| Service | Port |
|---|---|
| Public site (web app) | 8000 |
| Backend API | 8002 |
| ML service | 8003 |
| PostgreSQL | 5432 |
| Redis | 6379 |
| MinIO | 9000 |
| Qdrant / Milvus | 6333 / 19530 |
| Prometheus | 9090 |
| Grafana | 3000 |

---

## Cross-Cutting Standards

### Python (all components)
- Python **3.11+** only
- **Type hints everywhere** — no untyped functions
- **`uv`** for dependency management
- Format with **`ruff format`**, lint with **`ruff check`**
- **`pytest`** for tests; target >80% coverage on core logic
- Async for IO-bound (FastAPI handlers, HTTP, DB); sync for CPU-bound (inference)
- **SQLModel** for ORM (Pydantic-native, FastAPI-friendly)

### HTML / Templates (frontend)
- Jinja2 with autoescape enabled
- HTMX for interactivity (no SPAs, no React)
- Tailwind CSS via CDN; DaisyUI for components
- Only place JavaScript is used: webcam capture (~50 lines, lives in `frontend/public-site/static/js/webcam.js`)

### Attendance Domain
- Use clear domain names: `Person`, `FaceRegistration`, `AttendanceSession`, `CheckIn`, `LivenessAttempt`.
- A check-in must belong to a known attendance session.
- A check-in must be an explicit user action.
- Active liveness and passive liveness must run before face matching.
- Low-confidence recognition must not record attendance automatically.
- Duplicate check-ins should be idempotent and return a clear duplicate state.
- Admin functionality must be protected by role-based access control.

### Git
- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- Branch naming: `feat/<short-name>`, `fix/<short-name>`
- Squash merge to `main`; require PR approval

### Secrets
- **Never** hardcode secrets — use `.env` locally, secrets manager in production
- **Never** commit `.env`, only `.env.example`

---

## Environment Variables (shared)

Defined in root `.env.example`. Each component reads what it needs.

```
# Shared
ENV=development                     # development | staging | production
LOG_LEVEL=info

# Database
POSTGRES_URI=postgresql+asyncpg://user:pass@localhost:5432/faceattend
REDIS_URI=redis://localhost:6379

# Vector DB
VECTOR_DB_BACKEND=faiss             # faiss | qdrant | milvus
VECTOR_DB_URI=...                   # not needed for faiss (uses local file)
VECTOR_DB_INDEX_PATH=./data/faiss.idx   # for faiss only

# Queue
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Storage
S3_ENDPOINT=http://localhost:9000   # MinIO in dev
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET_IMAGES=faceattend-images
S3_BUCKET_EXPORTS=faceattend-exports

# Auth
JWT_SECRET=changeme                 # 32+ random bytes in prod
JWT_ISSUER=faceattend
API_KEY_SECRET=...

# Service URLs (for inter-service calls)
BACKEND_API_URL=http://localhost:8002
ML_SERVICE_URL=http://localhost:8003

# ML
EMBEDDING_MODEL_PATH=./models/arcface_r100.onnx
EMBEDDING_MODEL_VERSION=arcface-r100-v1
LIVENESS_MODEL_PATH=./models/minifasnet_v2.onnx
LIVENESS_THRESHOLD_PASSIVE=0.85
LIVENESS_THRESHOLD_ACTIVE=0.95

# Compliance
DATA_RETENTION_DAYS=30
GDPR_DELETE_GRACE_HOURS=24
```

---

## Legal & Compliance (Project-Wide)

Biometric data is **GDPR Article 9 special category data**. Treat it accordingly:

- **Lawful basis**: explicit consent, captured at face registration with timestamped record
- **Liveness at registration**: prevents non-consensual enrollment
- **Liveness at check-in**: prevents spoofed attendance
- **No hidden surveillance**: attendance check-ins must be explicit user actions
- **Audit log**: every biometric operation logged with actor, target, action, timestamp
- **Right to erasure**: delete biometric registrations and index entries when required
- **Right to portability**: export attendance/person records where applicable
- **Annotate compliance-relevant code** with comment markers:
  - `# GDPR:` for data subject rights logic
  - `# AI-ACT:` for AI Act conformity logic
  - `# AUDIT:` for code that must write to audit log
  - `# PRIVACY:` for biometric minimization or consent-sensitive code

A legal review is required before production launch.

---

## What NOT to Do (Project-Wide)

- ❌ Do **not** store raw face crops in the vector DB — embeddings only
- ❌ Do **not** bypass the queue and call the indexer directly from another service
- ❌ Do **not** record attendance without liveness
- ❌ Do **not** record attendance from a static uploaded face image alone
- ❌ Do **not** skip liveness checks on registration or check-in endpoints
- ❌ Do **not** add a new endpoint without rate limiting
- ❌ Do **not** log raw biometric data anywhere
- ❌ Do **not** build hidden surveillance or background tracking
- ❌ Do **not** add crawler/public-web search/takedown work unless explicitly requested
- ❌ Do **not** commit `.env` or any file with real credentials
- ❌ Do **not** modify the audit log writer to make it optional — every biometric op must log
- ❌ Do **not** introduce React, Vue, or any SPA framework without an ADR — the project is Python-first by design

---

## ADR Process

Any significant technical decision (vector DB choice, model architecture, liveness policy, retention policy, etc.) gets an ADR in `docs/adr/`:

```
docs/adr/
├── 001-vector-db-choice.md
├── 002-liveness-passive-vs-active.md
├── 003-embedding-model-selection.md
├── 004-htmx-vs-spa.md
├── 005-attendance-retention-policy.md
└── ...
```

Template:
```markdown
# ADR-NNN: <Title>

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-XXX
**Date:** YYYY-MM-DD
**Deciders:** <names>

## Context
Why does this decision need to be made?

## Options Considered
1. Option A — pros/cons
2. Option B — pros/cons
3. Option C — pros/cons

## Decision
What did we decide and why?

## Consequences
What are the trade-offs? What becomes easier? Harder?

## References
Links to docs, benchmarks, discussions.
```

---

## Useful Commands

```bash
# Full stack up
docker compose up

# Run all tests
uv run pytest

# Format & lint everything
uv run ruff format . && uv run ruff check .

# Run individual services in dev
uv run uvicorn backend.api.main:app --reload --port 8002
uv run uvicorn ml.serving.api:app --reload --port 8003
uv run uvicorn frontend.public-site.main:app --reload --port 8000
```

---

## Where to ask "should I?" before doing

Some decisions need human input before Claude Code should proceed unilaterally:

- Adding a new third-party biometric/liveness vendor or commercial API
- Changing the embedding model (requires full re-embedding)
- Adding a new biometric category (e.g. iris, voice)
- Anything affecting consent, attendance retention, recognition thresholds, or audit logging
- Anything in `docs/legal/`
- Introducing JavaScript outside `frontend/public-site/static/js/webcam.js`
- Introducing a SPA framework (React/Vue/etc.) — this goes against the project's Python-first principle

For these, propose an ADR and wait for sign-off.

---

## Beginner Notes

The developer building this is learning. When generating code:

- **Prefer clarity over cleverness.** Verbose-but-obvious beats compact-but-magical.
- **Add comments** explaining *why*, not *what* (the code already shows the what).
- **Suggest the simplest working solution first**, then mention more advanced options as future improvements.
- **Use the standard library** when it's good enough. Don't pull in a dependency for things that take 5 lines of plain Python.
- **Link to docs** when introducing a new concept. Library docs > tutorials > random blog posts.
- **Don't optimize prematurely.** Working > fast > scalable, in that order.

A `docs/learnings/` folder exists for the developer to record what they've learned. Encourage that habit when helpful.

---

## Current MVP Priorities

1. Active liveness with MediaPipe blink detection.
2. Passive liveness integration with registration/check-in.
3. Face registration with embedding storage.
4. Attendance session and check-in models.
5. Match live check-in face against registered embeddings.
6. Attendance dashboard and export.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
