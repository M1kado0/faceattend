# crawler/AGENTS.md — Web Crawler

> Read the root [`AGENTS.md`](../AGENTS.md) first for project-wide rules.

This directory contains the distributed web crawler responsible for discovering public images from the open internet.

---

## Layout

```
crawler/
├── spiders/
│   ├── base.py             # BaseSpider with shared behavior
│   ├── <source>.py         # One spider per source type
│   └── __init__.py
├── middleware/
│   ├── proxy_rotation.py
│   ├── rate_limit.py
│   ├── robots_check.py
│   └── dedup.py
├── pipelines/
│   ├── image_download.py
│   ├── phash.py            # Perceptual hash for dedup
│   └── queue_emit.py       # Push to Kafka/Celery for inference
├── scheduler/
│   ├── recrawl.py          # Re-crawl logic
│   └── priority.py         # Crawl priority queue
├── settings.py
└── tests/
```

---

## Architecture

```
[Scheduler]
    │
    ▼
[Spider Pool]──▶[Proxy Pool]
    │              │
    ▼              ▼
[robots.txt]──[Rate Limiter]
    │
    ▼
[Image Download]
    │
    ▼
[Perceptual Hash]──▶[Dedup Check]──▶[Skip if seen]
    │
    ▼
[S3 / MinIO Storage]
    │
    ▼
[Emit to Inference Queue]
```

---

## Ethical & Legal Rules — Non-Negotiable

Crawling for a biometric product is legally sensitive. These rules are mandatory:

1. **Only crawl publicly accessible URLs.** Never authenticate. Never bypass paywalls. Never use stolen credentials.
2. **Respect `robots.txt`.** Always. Overrides require an entry in `docs/legal/crawl-overrides.md` with documented justification and legal sign-off.
3. **Respect `noindex`, `nofollow`, `X-Robots-Tag` headers.**
4. **Identify the crawler** with a clear `User-Agent`: `FaceAttendBot/1.0 (+https://faceattend.io/bot)`
5. **Provide an opt-out**: the bot URL must serve a page explaining what we do and how to block us.
6. **Crawl politely**: max 1 req/sec per domain unless their robots.txt explicitly allows more.
7. **Do not crawl** sites where minors are the primary subject (school sites, children's content platforms).
8. **Do not crawl** known revenge-porn or doxxing sites — these need a blocklist maintained in `crawler/blocklist.txt`.
9. **Honor takedown requests** at the source level — if a site asks to be removed, add to blocklist and purge their content from the index.

Mark legally relevant code with `# LEGAL:` for traceability.

---

## Spider Pattern

Subclass `BaseSpider`. Emit `ImageItem` objects.

```python
from crawler.spiders.base import BaseSpider
from crawler.items import ImageItem

class ExampleSpider(BaseSpider):
    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com/gallery"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        for img_url in response.css("img::attr(src)").getall():
            yield ImageItem(
                url=response.urljoin(img_url),
                source_page=response.url,
                spider=self.name,
            )
```

---

## ImageItem Schema

Emitted by spiders, processed by pipelines, eventually emitted to the inference queue.

```python
@dataclass
class ImageItem:
    url: str                  # Image URL
    source_page: str          # Page where the image was found
    spider: str               # Spider name
    crawled_at: datetime      # UTC timestamp
    phash: str | None = None  # Filled by phash pipeline
    s3_key: str | None = None # Filled by storage pipeline
    content_type: str | None = None
    bytes_size: int | None = None
```

---

## Proxy Rotation

Use a rotating residential proxy pool in production. Configure via env:

```
PROXY_PROVIDER=brightdata        # brightdata | smartproxy | custom
PROXY_USERNAME=...
PROXY_PASSWORD=...
PROXY_POOL_SIZE=100
```

Middleware in `middleware/proxy_rotation.py` assigns a fresh proxy per request, with sticky sessions for sites that require them (rare).

**Never** use proxies to evade rate limits a site has explicitly set. Proxies are for distribution and reliability, not abuse.

---

## Deduplication

Two layers of dedup:

1. **URL dedup** (cheap): if we've crawled this URL within `RECRAWL_INTERVAL_DAYS`, skip
2. **Content dedup** (expensive): compute pHash; if pHash already in index with same URL set, skip storing the bytes again (still update `last_seen` timestamp)

URL dedup state lives in Redis with a TTL. pHash dedup lives in PostgreSQL.

---

## Re-crawl Scheduling

`scheduler/recrawl.py` prioritizes which URLs to revisit:

- **High priority**: pages where matches were found for monitored users (re-crawl every 1-7 days)
- **Medium priority**: pages with high image churn (re-crawl every 7-30 days)
- **Low priority**: stable archive pages (re-crawl every 30-90 days)

Priority adjustments happen in the monitor service (`backend/monitor/`), which emits re-crawl events back to the crawler scheduler.

---

## Scaling

Designed for distributed operation:

- **Scrapy Cluster** or **Frontera** for distributed URL frontier
- **Redis** for shared dedup state and proxy pool coordination
- **S3/MinIO** for image storage (write-once, read-many)
- **Kafka** for emitting to inference queue

Each spider instance is stateless and horizontally scalable.

---

## Testing

```bash
# Unit tests for spiders, middlewares, pipelines
uv run pytest crawler/tests/

# Integration test against a fixture HTTP server
uv run pytest crawler/tests/integration/
```

Mock external HTTP with `pytest-httpx` or VCR. Never run real spiders against real sites in tests.

---

## Monitoring

Prometheus metrics emitted from each spider:
- `crawler_requests_total{spider,status}` — request counts
- `crawler_items_emitted_total{spider}` — image items emitted
- `crawler_dedup_skipped_total{spider,reason}` — dedup hits
- `crawler_proxy_errors_total{proxy_id}` — proxy failures
- `crawler_robots_blocked_total{spider}` — robots.txt blocks

Dashboards in Grafana under `infra/grafana/crawler.json`.

---

## Common Tasks

### Add a new spider
1. Create `crawler/spiders/<source>.py`
2. Subclass `BaseSpider`
3. Set `name`, `allowed_domains`, `start_urls`, `custom_settings`
4. Implement `parse()` to emit `ImageItem`
5. Register implicit (Scrapy auto-discovers)
6. Add tests
7. **Verify robots.txt compliance** in a manual smoke test before deploying

### Add a domain to the blocklist
1. Add to `crawler/blocklist.txt`
2. Run `scripts/purge_domain.py <domain>` to remove existing content
3. Log to `docs/legal/blocklist-decisions.md` with reason

### Tune crawl politeness
- Adjust `DOWNLOAD_DELAY` and `CONCURRENT_REQUESTS_PER_DOMAIN` per spider
- Lower limits are always safe; raising requires checking the site's `robots.txt`

---

## What NOT to Do (Crawler-Specific)

- ❌ Do **not** disable `ROBOTSTXT_OBEY` without a documented override
- ❌ Do **not** use credentials, cookies, or session tokens to access content
- ❌ Do **not** crawl sites in the blocklist
- ❌ Do **not** scrape children's content platforms or school sites
- ❌ Do **not** use proxies to evade explicit per-IP rate limits
- ❌ Do **not** commit proxy credentials — they go in env vars / secrets manager
- ❌ Do **not** store images that fail content moderation checks (CSAM, gore) — discard immediately and log the source for review