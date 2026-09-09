# FaceAttend

FaceAttend is a Python-first attendance system with webcam face registration,
face recognition, and active/passive liveness checks.

The MVP flow is intentionally small:

```text
register a consenting face
create an attendance session
check in with blink and head-turn liveness
match the live face to the registered template
record attendance
export attendance as CSV
```

No admin panel, no organizer mode, no background surveillance, no crawler, no
public-web search, no takedown flow.

---

## Current MVP

FaceAttend is built around explicit self check-ins.

1. A user creates an account or logs in.
2. The user registers their face with a short webcam video.
3. The backend verifies active liveness with a blink and head-turn challenge.
4. The backend samples frames from that same video and checks passive liveness.
5. The best live frame is embedded and stored in the vector index.
6. The user creates an attendance session.
7. The user checks in to that session with another webcam challenge video.
8. The backend verifies liveness again, embeds the best live frames, and matches
   against the registered face template.
9. Attendance is recorded only when liveness and identity match both pass.

Attendance is never recorded from a static uploaded image alone.

---

## Screenshots / Demo Flow

Add real screenshots after running the local demo:

| Step | Page | What to capture |
|---|---|---|
| 1 | `/face-registration` | Webcam oval, countdown, blink and head-turn challenge |
| 2 | `/sessions` | Created attendance session |
| 3 | `/check-in` | Session selector and liveness camera |
| 4 | `/attendance` | Recorded check-in and confidence |
| 5 | `/attendance.csv` | CSV export downloaded/opened |

Recommended files:

```text
docs/screenshots/01-face-registration.png
docs/screenshots/02-sessions.png
docs/screenshots/03-check-in.png
docs/screenshots/04-attendance.png
docs/screenshots/05-csv-export.png
```

Short demo video outline:

```text
0:00 Login
0:05 Register face with blink, left-turn countdown, right-turn countdown
0:20 Create attendance session
0:30 Check in with blink, left-turn countdown, right-turn countdown
0:45 Show attendance result
0:55 Show duplicate check-in behavior
1:05 Export CSV
```

Do not record or publish another person’s face without explicit consent.

---

## Liveness Policy

The current production-ish rule is:

```text
active_liveness_passed
AND passive_liveness_pass_ratio >= 0.8
AND face_visible_ratio >= 0.8
AND identity_similarity >= 0.75
```

Active liveness:

- implemented with MediaPipe Face Landmarker
- current challenge: `blink_turn_left_right`
- frontend gives live guidance only
- backend is the source of truth

Passive liveness:

- samples frames from the webcam video
- runs MiniFASNet-based passive spoof detection
- uses the best live frames for embedding and matching

Identity matching:

- registration stores embeddings, not raw face crops
- check-in searches the user’s registered face template
- low-confidence or wrong-person matches do not record attendance
- duplicate check-ins for the same session are idempotent

---

## Architecture

```text
┌────────────────────┐
│   Frontend Site    │
│ FastAPI + Jinja2   │
│ HTMX + webcam JS   │
└─────────┬──────────┘
          │ HTTP
          ▼
┌────────────────────┐
│    Backend API     │
│ FastAPI + SQLModel │
│ Auth, sessions,    │
│ check-ins, audit   │
└─────────┬──────────┘
          │
          ├──────────────┐
          ▼              ▼
┌─────────────────┐  ┌──────────────────┐
│   ML Service    │  │   Vector Store    │
│ face embedding, │  │ FAISS locally     │
│ active/passive  │  │                  │
│ liveness        │  │                  │
└─────────────────┘  └──────────────────┘
          │
          ▼
┌────────────────────┐
│ PostgreSQL          │
│ users, sessions,    │
│ registrations,      │
│ attendance records  │
└────────────────────┘
```

---

## Stack

| Layer | Tooling |
|---|---|
| Frontend | FastAPI, Jinja2, HTMX |
| Styling | Tailwind CSS CDN, DaisyUI |
| Webcam | Vanilla JavaScript in `frontend/public-site/static/js/webcam.js` |
| Backend API | FastAPI |
| ORM | SQLModel |
| Database | PostgreSQL |
| ML API | FastAPI |
| Active liveness | MediaPipe Face Landmarker |
| Passive liveness | MiniFASNet-style antispoofing |
| Embeddings | ArcFace/AdaFace pipeline |
| Vector search | FAISS locally |

---

## Project Structure

```text
faceattend/
├── backend/        # FastAPI API, auth, sessions, registrations, check-ins
├── ml/             # Face embeddings, active liveness, passive liveness
├── frontend/       # Server-rendered public site
│   └── public-site/
├── docs/           # ADRs, legal notes, screenshots, learnings
├── scripts/        # Dev utilities
├── tests/          # Cross-component tests
├── alembic.ini
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Local Demo

### 1. Configure environment

```bash
cp .env.example .env
```

Set real local values for:

```bash
POSTGRES_URI=postgresql+asyncpg://user:pass@localhost:5432/faceattend
ALEMBIC_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/faceattend
JWT_SECRET=changeme-use-32-plus-random-bytes-in-production
ML_SERVICE_URL=http://localhost:8003
BACKEND_API_URL=http://localhost:8002
```

### 2. Start dependencies

Use Docker Compose for Postgres/Redis/etc. if needed:

```bash
docker compose up
```

If you already run Postgres locally, make sure the `faceattend` database exists.

### 3. Run migrations

```bash
uv run alembic -c alembic.ini upgrade head
```

### 4. Start services

Use three terminals:

```bash
uv run uvicorn ml.serving.api:app --reload --port 8003
```

```bash
uv run uvicorn backend.api.main:app --reload --port 8002
```

```bash
uv run uvicorn frontend.public-site.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

### 5. Browser flow

```text
/register or /login
/face-registration -> register face with blink and turn countdowns
/sessions -> create an attendance session
/check-in -> select session and complete the liveness challenge
/attendance -> confirm record appears
/attendance.csv -> confirm export works
```

### 6. Failure cases to demo

- wrong face: check-in is rejected with identity mismatch
- same face, same session twice: duplicate/idempotent behavior
- no face: frontend asks user to position face in the oval
- face too far: frontend asks user to move closer
- failed blink: backend rejects liveness

---

## Useful Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run alembic -c alembic.ini upgrade head
uv run uvicorn ml.serving.api:app --reload --port 8003
uv run uvicorn backend.api.main:app --reload --port 8002
uv run uvicorn frontend.public-site.main:app --reload --port 8000
```

---

## Privacy And Safety

FaceAttend handles biometric data. Treat it as sensitive.

- Use only consenting test users.
- Do not demo with crawled or public celebrity images.
- Do not store raw face crops in the vector database.
- Do not record attendance without active and passive liveness.
- Do not add background tracking or hidden surveillance.
- Keep audit logging for biometric operations.
- Get legal review before any real production deployment.

---

## Current Status

Implemented:

- account login/register
- webcam face registration
- frontend liveness guidance and countdown
- backend active blink liveness
- passive liveness on sampled video frames
- embedding storage in FAISS
- attendance sessions
- liveness-gated check-in
- duplicate check-in handling
- attendance dashboard
- CSV export

Not in scope for this MVP:

- admin panel
- organizer/instructor mode
- reports page
- settings page
- crawler/search/takedown flows

---

## License
