"""BaseSpider — shared behavior for all FaceAttend spiders."""

from __future__ import annotations

from datetime import UTC, datetime

import scrapy

from crawler.items import ImageItem


class BaseSpider(scrapy.Spider):
    """Subclass and override `name`, `allowed_domains`, `start_urls`, `parse`."""

    custom_settings: dict = {
        "DOWNLOAD_DELAY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "ROBOTSTXT_OBEY": True,  # LEGAL: never disable without a documented override.
    }

    def make_image_item(self, *, url: str, source_page: str) -> ImageItem:
        return ImageItem(
            url=url,
            source_page=source_page,
            spider=self.name,
            crawled_at=datetime.now(UTC),
        )
