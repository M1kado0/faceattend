# Copilot instructions for this repository

## Build, test, and lint commands

This repo is Python-first (`uv`-managed) and does not have a JS/TS build pipeline.

```bash
# Start local stack (services, DB, Redis, MinIO, etc.)
docker compose up

# Run all tests
uv run pytest

# Run tests by component
uv run pytest backend/tests/
uv run pytest crawler/tests/
uv run pytest ml/tests/
uv run pytest frontend/public-site/tests/
uv run pytest frontend/admin-panel/tests/

# Run a single test
uv run pytest path/to/test_file.py::test_name

# Lint + format
uv run ruff format . && uv run ruff check .

# Type-check
uv run mypy .
```

## High-level architecture

- FaceAttend is split into independent services: `backend/` (JSON API), `crawler/`, `ml/`, and two server-rendered web apps in `frontend/` (public site + admin panel).
- The public site (`:8000`) and admin panel (`:8001`) are separate FastAPI apps using Jinja2 + HTMX, both calling backend API (`:8002`) over HTTP through shared API client code.
- Crawl/index path is queue-driven: crawler discovers images → pipelines dedupe/download/hash → emit queue events → ML/indexer process embeddings and write to vector storage.
- User-sensitive path enforces liveness before biometric registration/check-in: user capture (webcam) → liveness validation → embedding/recognition or face registration write.
- Takedown and monitoring are backend workflows layered on top of match discovery (notice rendering, platform submission/fallback, status tracking, notifications).

## Key conventions (repo-specific)

- Read `CLAUDE.md` at repo root plus nested component `CLAUDE.md` files (`backend/`, `crawler/`, `ml/`, `frontend/`) before substantial edits; these define project-specific rules.
- Graph-first exploration is expected: consult `graphify-out/GRAPH_REPORT.md` before broad codebase investigation.
- Liveness checks are non-negotiable for **face registration/check-in** endpoints; crawler ingest explicitly skips liveness.
- Every biometric read/write/recognition/delete path must write audit logs; keep `# AUDIT:` markers on audit-critical code.
- Frontend route handlers should return HTML/templates (including HTMX partials), not JSON API responses.
- In frontend apps, call backend via `frontend/shared/api_client/client.py` (`BackendClient`) rather than ad-hoc `httpx` calls in route handlers.
- Keep JavaScript constrained to `frontend/shared/static/js/webcam.js`; do not introduce SPA frameworks or broader frontend build tooling.
- Access vector backends via `backend/indexer` abstractions (`VectorStore`), not direct Milvus/Qdrant calls from API routes.
- Use compliance markers consistently where relevant: `# GDPR:`, `# AI-ACT:`, `# AUDIT:`, `# LEGAL:`.
