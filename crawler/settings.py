"""Scrapy settings — applies to every spider unless overridden via custom_settings."""

BOT_NAME = "faceattend"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

# LEGAL: identify ourselves clearly; never spoof.
USER_AGENT = "FaceAttendBot/1.0 (+https://faceattend.io/bot)"

# LEGAL: robots.txt is non-negotiable; overrides require docs/legal/crawl-overrides.md.
ROBOTSTXT_OBEY = True

# Politeness
DOWNLOAD_DELAY = 1.0
CONCURRENT_REQUESTS_PER_DOMAIN = 2
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Middleware
DOWNLOADER_MIDDLEWARES = {
    "crawler.middleware.robots_check.RobotsCheckMiddleware": 100,
    "crawler.middleware.proxy_rotation.ProxyRotationMiddleware": 350,
    "crawler.middleware.rate_limit.RateLimitMiddleware": 500,
    "crawler.middleware.dedup.DedupMiddleware": 900,
}

# Pipelines
ITEM_PIPELINES = {
    "crawler.pipelines.phash.PhashPipeline": 100,
    "crawler.pipelines.image_download.ImageDownloadPipeline": 200,
    "crawler.pipelines.queue_emit.QueueEmitPipeline": 900,
}

# Telemetry
EXTENSIONS = {
    "scrapy.extensions.telnet.TelnetConsole": None,
}

# Distributed crawl: scrapy-redis frontier
SCHEDULER = "scrapy_redis.scheduler.Scheduler"
DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
SCHEDULER_PERSIST = True
