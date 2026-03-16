# src/features/news_features.py
# Downstream feature engineering from the parsed DataFrame

import numpy as np
import pandas as pd

# ── FinBERT — lazy-loaded on first use to avoid startup cost ──────────────────
_finbert = None


def _get_finbert():
    """Load FinBERT pipeline once and cache it for the process lifetime."""
    global _finbert
    if _finbert is None:
        from transformers import pipeline as hf_pipeline
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,        # return probabilities for all three classes
            truncation=True,
            max_length=512,
            device=-1,         # CPU; set device=0 for GPU if available
        )
    return _finbert


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each article with FinBERT (financial-domain BERT).

    Text strategy: title + teaser + first 300 chars of body.
    Prioritises the headline (highest information density) while still
    incorporating body context, within FinBERT's 512-token limit.

    Adds columns:
        finbert_positive  — P(positive)  [0, 1]
        finbert_negative  — P(negative)  [0, 1]
        finbert_neutral   — P(neutral)   [0, 1]
        sentiment         — positive - negative  [-1, 1]  (VADER-compatible)

    Idempotent: returns early if already scored (avoids duplicate columns).
    """
    if "finbert_positive" in df.columns:
        return df  # already scored — avoid re-running inference

    df = df.copy()

    # Build scoring text: title-heavy, body truncated
    df["text_for_scoring"] = (
        df["title"].fillna("") + ". " +
        df["teaser"].fillna("") + ". " +
        df["body"].fillna("").str[:300]
    ).str.strip()

    finbert = _get_finbert()
    texts = df["text_for_scoring"].tolist()

    # Batch inference — processes all articles in chunks of 32
    raw = finbert(texts, batch_size=32)

    records = []
    for result in raw:
        probs = {r["label"]: r["score"] for r in result}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        neu = probs.get("neutral",  0.0)
        records.append({
            "finbert_positive": pos,
            "finbert_negative": neg,
            "finbert_neutral":  neu,
            "sentiment":        pos - neg,   # compound [-1, 1]
        })

    probs_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, probs_df], axis=1)
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level FinBERT scores to daily features.
    """
    df = score_sentiment(df)

    # String-function aggregations (fast cython path)
    daily = df.groupby("date").agg(
        headline_count        = ("id",               "count"),
        sentiment_mean        = ("sentiment",         "mean"),
        sentiment_std         = ("sentiment",         "std"),
        sentiment_min         = ("sentiment",         "min"),
        sentiment_max         = ("sentiment",         "max"),
        finbert_positive_mean = ("finbert_positive",  "mean"),
        finbert_negative_mean = ("finbert_negative",  "mean"),
        finbert_neutral_mean  = ("finbert_neutral",   "mean"),
    ).reset_index()

    # Lambda aggregations computed separately to avoid pandas mixing limitation
    ratios = df.groupby("date").agg(
        negative_ratio = ("finbert_negative", lambda x: (x > 0.5).mean()),
        positive_ratio = ("finbert_positive", lambda x: (x > 0.5).mean()),
    ).reset_index()

    daily = daily.merge(ratios, on="date")
    daily["log_headline_count"] = np.log1p(daily["headline_count"])

    return daily


def merge_with_price_series(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join news features onto price series.
    Fills missing news days with 0 count and neutral sentiment.
    """
    merged = price_df.merge(news_df, on="date", how="left")
    merged["headline_count"]          = merged["headline_count"].fillna(0)
    merged["log_headline_count"]      = merged["log_headline_count"].fillna(0)
    merged["sentiment_mean"]          = merged["sentiment_mean"].fillna(0)
    merged["finbert_positive_mean"]   = merged["finbert_positive_mean"].fillna(0)
    merged["finbert_negative_mean"]   = merged["finbert_negative_mean"].fillna(0)
    merged["finbert_neutral_mean"]    = merged["finbert_neutral_mean"].fillna(0)
    merged["negative_ratio"]          = merged["negative_ratio"].fillna(0)
    merged["positive_ratio"]          = merged["positive_ratio"].fillna(0)
    return merged
