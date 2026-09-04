# backend/AGENTS.md — Backend Services

> Read the root [`AGENTS.md`](../AGENTS.md) first for project-wide rules.

This directory contains all server-side Python services: API gateway, vector indexer, monitor/scheduler, takedown workflow, and audit logger.

---

## Layout

```
backend/
├── api/                    # FastAPI app
│   ├── main.py             # App entry point
│   ├── routes/
│   │   ├── auth.py         # POST /v1/auth/*
│   │   ├── users.py        # User profile, GDPR rights
│   │   ├── face_registrations.py # /v1/face-registrations
│   │   ├── check_ins.py          # /v1/check-ins
│   │   ├── attendance_records.py # /v1/attendance-records
│   │   ├── # takedown routes removed
│   │   ├── notifications.py
│   │   └── billing.py      # Stripe webhooks
│   ├── schemas.py          # Pydantic models
│   ├── auth/               # OAuth2 + JWT + API key logic
│   ├── middleware/         # Rate limiting, request logging
│   └── dependencies.py     # Shared FastAPI dependencies
├── indexer/                # Vector index management
│   ├── store.py            # Abstract VectorStore interface
│   ├── faiss_store.py      # FAISS impl
│   ├── milvus.py           # Milvus impl
│   ├── qdrant.py           # Qdrant impl
│   ├── ingest.py           # Queue consumer → index writer
│   └── reembed.py          # Re-embedding pipeline for model upgrades
├── monitor/                # Background services
│   ├── recrawl_scheduler.py
│   ├── match_diff.py       # Find new matches for enrolled users
│   └── notifier.py         # Send email / webhook alerts
├── takedown/               # DMCA workflow
│   ├── notice.py           # Generate DMCA notices from templates
│   ├── tracker.py          # Track status: requested → sent → resolved
│   └── platforms/          # Per-platform reporting integrations
├── audit/                  # GDPR-compliant audit logging
│   ├── logger.py
│   └── exporter.py         # Data export for right-to-portability
├── db/                     # SQLAlchemy models + Alembic migrations
│   ├── models/
│   └── migrations/
└── tests/
```

---

## API Conventions

- All endpoints use **FastAPI**
- Request/response schemas defined with **Pydantic v2** in `api/schemas.py`
- API versioning via URL prefix: `/v1/...`
- All endpoints return JSON
- Errors follow RFC 7807 (Problem Details for HTTP APIs)

### Authentication
- **End users**: OAuth2 password flow → JWT access + refresh tokens
- **Programmatic access**: `X-API-Key` header
- JWT secret from `JWT_SECRET` env var; tokens expire in 15min (access) / 7 days (refresh)

### Rate Limiting
Apply per-route, enforced by middleware. Defaults:
- Anonymous: 10 req/min per IP
- Authenticated user: 60 req/min per account
- Search endpoints: 20 req/hour per account (free tier), 200/hour (paid)
- API key: as configured per key

Use Redis as the rate limit backend (`slowapi` or custom).

## Endpoint Reference (target)

### Public (authenticated user)
```
POST  /v1/auth/register
POST  /v1/auth/login
POST  /v1/auth/refresh
POST  /v1/auth/logout

GET   /v1/users/me
PATCH /v1/users/me
DELETE /v1/users/me                    # GDPR right to erasure
GET   /v1/users/me/export              # GDPR right to portability

POST  /v1/face-registrations       # Body: {liveness_video}
GET   /v1/face-registrations
DELETE /v1/face-registrations/{id}

POST  /v1/check-ins                # Body: {liveness_video, session_id}
GET   /v1/attendance-records?since=...
GET   /v1/attendance-records/{id}


GET   /v1/notifications/settings
PATCH /v1/notifications/settings
POST  /v1/notifications/webhooks
```

### Webhooks
```
POST  /v1/webhooks/stripe              # Stripe events
```

---

## Liveness Verification on Face Registration & Check-In

Every face registration and check-in **must** validate liveness before forwarding to the inference pipeline.

```python
# Pseudocode for a face registration handler
async def create_face_registration(request: EnrollRequest, user: User = Depends(get_user)):
    # AUDIT: log attempt
    await audit.log(actor=user.id, action="face_registration.attempt")

    # 1. Verify liveness FIRST
    liveness_result = await ml_client.verify_liveness(request.liveness_blob)
    if not liveness_result.passed:
        await audit.log(actor=user.id, action="face_registration.liveness_failed",
                        meta={"score": liveness_result.score})
        raise HTTPException(403, "liveness_failed")

    # 2. Detect + embed face
    embedding = await ml_client.embed(request.photo)
    if embedding is None:
        raise HTTPException(400, "no_face_detected")

    # 3. Index
    await indexer.add(user_id=user.id, embedding=embedding,
                      model_version=settings.embedding_model_version)
    await audit.log(actor=user.id, action="face_registration.success")
```

**Search has the same pattern** — liveness first, then embed, then ANN query.

---

## Vector Index Abstraction

Never call Milvus or Qdrant directly outside `indexer/`. Always go through the `VectorStore` interface:

```python
class VectorStore(Protocol):
    async def add(self, *, image_id: str, embedding: np.ndarray,
                  metadata: dict) -> None: ...
    async def search(self, *, embedding: np.ndarray, top_k: int,
                     filter: dict | None = None) -> list[Match]: ...
    async def delete(self, *, image_id: str) -> None: ...
    async def delete_by_user(self, *, user_id: str) -> int: ...  # GDPR
```

Implementation chosen at startup via `VECTOR_DB_BACKEND` env var.

### Embedding format
- **512-dim float32** (ArcFace/AdaFace standard)
- L2-normalized at ingest time
- Stored with metadata: `image_id`, `source_url`, `crawled_at`, `face_bbox`, `embedding_model_version`, `phash`

---

## Audit Logging

Every biometric operation must call `audit.log()`. The audit log is append-only and lives in a dedicated PostgreSQL table with row-level retention guarantees.

Schema:
```python
{
    "id": uuid,
    "timestamp": datetime,
    "actor_id": str,          # User or system actor who did the thing
    "actor_type": str,        # "user" | "system"
    "action": str,            # e.g. "check_in.success", "face_registration.attempt"
    "target_id": str | None,  # User whose biometric data was touched
    "metadata": dict,         # Action-specific context
    "ip_address": str,
    "user_agent": str,
}
```

**Mark code with `# AUDIT:` whenever it should write to the audit log.**

---

## Attendance Review Workflow

```
[Check-In Submitted] -> [Liveness Verified] -> [Attendance Recorded]
                                          │
                                          ▼
                              [Generate DMCA Notice]
                                          │
                                          ▼
                              [Send to Platform / Host]
                                          │
                                          ▼
                              [Track: pending → resolved]
                                          │
                                          ▼
                              [Notify User of Outcome]
```

DMCA templates live in `takedown/notice.py`. Per-platform integrations in `takedown/platforms/` — start with manual email fallback, add API integrations incrementally (Instagram, X, Reddit, etc.).

---

## Database (PostgreSQL)

Use **SQLAlchemy 2.x** (async) + **Alembic** for migrations.

Core tables:
- `users` — accounts, hashed passwords, plan tier
- `face_registrations` — user -> face embedding ID(s) link
- `attendance_records` — attendance check-in results
- `audit_logs` — biometric and attendance audit trail
- `audit_log` — append-only audit trail
- `api_keys` — programmatic access tokens
- `webhooks` — user-configured webhook endpoints
- `clusters` — face cluster metadata

**Never** store raw face crops in PostgreSQL — only embeddings (in vector DB) and metadata.

---

## Testing

```bash
# Unit + integration
uv run pytest backend/tests/

# With coverage
uv run pytest backend/tests/ --cov=backend --cov-report=html

# Only fast tests
uv run pytest backend/tests/ -m "not slow"
```

Mock the inference service in API tests using `pytest-httpx` or a fixture.

---

## Common Tasks

### Add a new API endpoint
1. Define request/response schemas in `api/schemas.py`
2. Create route in appropriate `api/routes/*.py` file
3. Add rate limit decorator
4. Add audit logging if it touches biometric data
5. Add tests in `backend/tests/api/`
6. Update OpenAPI spec is auto-generated by FastAPI

### Add a new vector DB backend
1. Implement the `VectorStore` Protocol in `indexer/<backend>.py`
2. Wire into the factory in `indexer/store.py`
3. Add integration test in `backend/tests/indexer/`

### Add a new platform takedown integration
1. Create `takedown/platforms/<platform>.py`
2. Implement the `TakedownPlatform` interface
3. Register in `takedown/platforms/__init__.py`

---

## What NOT to Do (Backend-Specific)

- ❌ Do **not** add an endpoint that takes a photo without a liveness check
- ❌ Do **not** call the vector DB directly outside `indexer/`
- ❌ Do **not** return raw embeddings to clients (they're biometric data)
- ❌ Do **not** skip the audit log on any biometric operation
- ❌ Do **not** log PII or biometric blobs to stdout/files
- ❌ Do **not** use synchronous DB drivers — async only
