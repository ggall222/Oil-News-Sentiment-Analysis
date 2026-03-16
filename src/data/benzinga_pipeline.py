# src/data/benzinga_pipeline.py
# Multi-strategy fetch pipeline with tier-aware storage

import pandas as pd
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from .benzinga_news import BenzingaNewsClient, BenzingaConfig
from .benzinga_queries import QUERY_STRATEGIES, QueryStrategy, QueryTier

logger = logging.getLogger(__name__)


class BenzingaMultiStrategyPipeline:
    """
    Runs all query strategies and stores results by tier.
    
    Output structure:
        data/raw/benzinga/
            broad/       YYYY-MM.parquet
            opec/        YYYY-MM.parquet
            disruption/  YYYY-MM.parquet
            macro/       YYYY-MM.parquet
    """

    def __init__(self, api_key: str, output_dir: str = "data/raw/benzinga"):
        self.client = BenzingaNewsClient(BenzingaConfig(api_key=api_key))
        self.output_dir = Path(output_dir)

    def run_backfill(
        self,
        start_date: str,
        end_date: str,
        strategies: Optional[list] = None,
        window_days: int = 30,
    ) -> dict:
        """
        Run backfill for specified strategies (default: all).
        Returns dict of {strategy_name: DataFrame}.
        """
        target_strategies = strategies or list(QUERY_STRATEGIES.keys())
        results = {}

        for name in target_strategies:
            strategy = QUERY_STRATEGIES[name]
            logger.info(f"\n{'='*60}")
            logger.info(f"Running strategy: {name.upper()} ({strategy.tier.value})")
            logger.info(f"Topics: {strategy.topics_param()}")

            df = self._backfill_strategy(strategy, start_date, end_date, window_days)
            df = self._tag_tier(df, strategy)

            self._save_by_month(df, strategy.name)
            results[name] = df

            logger.info(f"Strategy '{name}' complete: {len(df)} unique articles")

        return results

    def _backfill_strategy(
        self,
        strategy: QueryStrategy,
        start_date: str,
        end_date: str,
        window_days: int,
    ) -> pd.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        all_articles = []
        current = start

        while current < end:
            window_end = min(current + timedelta(days=window_days), end)
            date_from = current.strftime("%Y-%m-%d")
            date_to = window_end.strftime("%Y-%m-%d")

            # Topic-based fetch
            articles = self.client.fetch_window(
                date_from=date_from,
                date_to=date_to,
                topics=strategy.topics_param(),
                topic_group_by=strategy.topic_group_by,
            )
            all_articles.extend(articles)

            # Ticker-based fetch (broad tier only)
            if strategy.tickers and strategy.tier == QueryTier.BROAD:
                ticker_articles = self.client.fetch_window(
                    date_from=date_from,
                    date_to=date_to,
                    tickers=strategy.tickers_param(),
                )
                all_articles.extend(ticker_articles)

            current = window_end + timedelta(days=1)

        df = self.client._parse_articles(all_articles)
        return df.drop_duplicates(subset="id")

    @staticmethod
    def _tag_tier(df: pd.DataFrame, strategy: QueryStrategy) -> pd.DataFrame:
        """Tag each article with its source strategy for downstream filtering."""
        df = df.copy()
        df["query_strategy"] = strategy.name
        df["query_tier"] = strategy.tier.value
        return df

    def _save_by_month(self, df: pd.DataFrame, strategy_name: str) -> None:
        """Partition output by month for efficient incremental loading."""
        if df.empty:
            return

        out_dir = self.output_dir / strategy_name
        out_dir.mkdir(parents=True, exist_ok=True)

        df["year_month"] = df["created"].dt.to_period("M").astype(str)
        for period, group in df.groupby("year_month"):
            path = out_dir / f"{period}.parquet"
            group.drop(columns="year_month").to_parquet(path, index=False)
            logger.info(f"  Saved {len(group)} articles → {path}")