# src/data/benzinga_news.py

import os
import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BenzingaConfig:
    api_key: str = field(default_factory=lambda: os.environ["BENZINGA_API_KEY"])
    base_url: str = "https://api.benzinga.com/api/v2/news"
    page_size: int = 100           # Max allowed
    request_delay: float = 0.5    # Seconds between requests (rate limiting)
    max_retries: int = 3
    retry_backoff: float = 2.0    # Exponential backoff multiplier

    # Crude oil topic keywords (topics searches title + tags + body)
    topics: list = field(default_factory=lambda: [
        "crude oil", "WTI", "Brent", "OPEC",
        "refinery", "inventory", "EIA",
        "geopolitical tension", "oil supply",
        "petroleum", "barrel"
    ])

    # Equity tickers related to crude (futures symbol CL=F likely won't match)
    tickers: list = field(default_factory=lambda: ["USO", "XLE", "MRO", "CVX", "XOM"])

    # Importance levels to collect
    importance_levels: list = field(default_factory=lambda: ["low", "medium", "high"])


# ── Client ────────────────────────────────────────────────────────────────────

class BenzingaNewsClient:
    def __init__(self, config: BenzingaConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json"})

    def _build_params(
        self,
        date_from: str,
        date_to: str,
        page: int = 0,
        importance: Optional[str] = None,
        topics: Optional[str] = None,
        tickers: Optional[str] = None,
        topic_group_by: str = "or",
    ) -> dict:
        params = {
            "token": self.config.api_key,
            "dateFrom": date_from,
            "dateTo": date_to,
            "pageSize": self.config.page_size,
            "page": page,
            "displayOutput": "full",           # Full body text for NLP
            "sort": "created:asc",
            "topic_group_by": topic_group_by,
        }

        if tickers:
            params["tickers"] = tickers
        else:
            params["topics"] = topics or ",".join(self.config.topics)

        if importance:
            params["importance"] = importance

        return params

    def _get_with_retry(self, params: dict) -> list:
        for attempt in range(self.config.max_retries):
            try:
                resp = self.session.get(self.config.base_url, params=params, timeout=30)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return []   # No data for this window, not an error
                elif resp.status_code == 401:
                    raise ValueError("Benzinga API auth failed — check your API key.")
                else:
                    logger.warning(f"HTTP {resp.status_code} on attempt {attempt + 1}")

            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")

            if attempt < self.config.max_retries - 1:
                sleep_time = self.config.retry_backoff ** attempt
                time.sleep(sleep_time)

        logger.error("Max retries exceeded")
        return []

    def fetch_window(
        self,
        date_from: str,
        date_to: str,
        importance: Optional[str] = None,
        topics: Optional[str] = None,
        tickers: Optional[str] = None,
        topic_group_by: str = "or",
    ) -> list:
        """Fetch all articles in a date window, handling pagination automatically."""
        all_articles = []
        page = 0

        while True:
            params = self._build_params(date_from, date_to, page, importance, topics, tickers, topic_group_by)
            articles = self._get_with_retry(params)

            if not articles:
                break

            all_articles.extend(articles)
            logger.info(f"  Page {page}: fetched {len(articles)} articles")

            if len(articles) < self.config.page_size:
                break  # Last page

            page += 1
            time.sleep(self.config.request_delay)

        return all_articles

    def backfill(
        self,
        start_date: str,
        end_date: str,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Backfill historical articles in monthly windows.
        Runs two passes: topic-based + ticker-based, then deduplicates.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_articles = []
        current = start

        while current < end:
            window_end = min(current + timedelta(days=window_days), end)
            date_from = current.strftime("%Y-%m-%d")
            date_to = window_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching window: {date_from} → {date_to}")

            # Pass 1: keyword/topic search
            topic_articles = self.fetch_window(date_from, date_to, use_tickers=False)

            # Pass 2: equity ticker search (catches energy equity news)
            ticker_articles = self.fetch_window(date_from, date_to, use_tickers=True)

            all_articles.extend(topic_articles)
            all_articles.extend(ticker_articles)

            current = window_end + timedelta(days=1)
            time.sleep(self.config.request_delay)

        df = self._parse_articles(all_articles)
        df = df.drop_duplicates(subset="id")
        logger.info(f"Backfill complete: {len(df)} unique articles")
        return df

    def fetch_high_importance(self, date_from: str, date_to: str) -> pd.DataFrame:
        """
        Separate fetch for high-importance articles only.
        Used to construct event intensity / volatility spike features.
        """
        articles = self.fetch_window(date_from, date_to, importance="high", use_tickers=False)
        articles += self.fetch_window(date_from, date_to, importance="medium", use_tickers=False)
        df = self._parse_articles(articles).drop_duplicates(subset="id")
        df["importance_flag"] = True
        return df

    @staticmethod
    def _parse_articles(articles: list[dict]) -> pd.DataFrame:
        """Normalize raw API response into a flat DataFrame."""
        if not articles:
            return pd.DataFrame()

        records = []
        for a in articles:
            records.append({
                "id": a.get("id"),
                "title": a.get("title", ""),
                "teaser": a.get("teaser", ""),
                "body": a.get("body", ""),
                "author": a.get("author", ""),
                "url": a.get("url", ""),
                "created": a.get("created"),
                "updated": a.get("updated"),
                "channels": [c["name"] for c in a.get("channels", [])],
                "tags": [t["name"] for t in a.get("tags", [])],
                "tickers": [s["name"] for s in a.get("stocks", [])],
            })

        df = pd.DataFrame(records)
        df["created"] = pd.to_datetime(df["created"], format="%a, %d %b %Y %H:%M:%S %z", utc=True)
        df["date"] = df["created"].dt.date  # Daily key for merging with price series
        return df.sort_values("created").reset_index(drop=True)