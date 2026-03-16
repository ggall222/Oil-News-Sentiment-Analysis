# src/features/event_flags.py
# Converts precision-tier articles into binary event flags for the model

import pandas as pd
import numpy as np


def build_opec_event_flags(opec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily OPEC event features.
    
    - opec_event_day: binary, 1 if any OPEC article published that day
    - opec_decision_day: binary, 1 if article contains decision-related terms
    - opec_event_window: 1 for event day ± 2 days (price impact window)
    """
    DECISION_TERMS = ["production cut", "production increase", "quota", "agreement", "deal"]

    daily = opec_df.groupby("date").agg(
        opec_article_count=("id", "count"),
        opec_sentiment=("sentiment", "mean"),
        opec_sentiment_std=("sentiment", "std"),
    ).reset_index()
    daily["opec_sentiment_std"] = daily["opec_sentiment_std"].fillna(0)

    daily["opec_event_day"] = 1

    # Flag days where decision-language appears in titles
    opec_df["is_decision"] = opec_df["title"].str.lower().apply(
        lambda t: int(any(term in t for term in DECISION_TERMS))
    )
    decision_days = opec_df[opec_df["is_decision"] == 1]["date"].unique()
    daily["opec_decision_day"] = daily["date"].isin(decision_days).astype(int)

    # ±2 day impact window around any OPEC event
    event_dates = set(daily["date"])
    window_dates = set()
    for d in event_dates:
        for delta in range(-2, 3):
            window_dates.add(d + pd.Timedelta(days=delta))
    daily["opec_event_window"] = daily["date"].isin(window_dates).astype(int)

    return daily


def build_disruption_flags(disruption_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily supply disruption features.
    
    - disruption_event_day: binary
    - disruption_intensity: article count (proxy for severity/coverage)
    - disruption_event_window: ±3 day window (disruptions have longer price tails)
    """
    daily = disruption_df.groupby("date").agg(
        disruption_intensity=("id", "count"),
        disruption_sentiment=("sentiment", "mean"),
        disruption_sentiment_std=("sentiment", "std"),
    ).reset_index()
    daily["disruption_sentiment_std"] = daily["disruption_sentiment_std"].fillna(0)

    daily["disruption_event_day"] = 1

    # Wider window for disruptions — physical supply impacts persist longer
    event_dates = set(daily["date"])
    window_dates = set()
    for d in event_dates:
        for delta in range(-1, 4):  # Asymmetric: 1 day before, 3 days after
            window_dates.add(d + pd.Timedelta(days=delta))
    daily["disruption_event_window"] = daily["date"].isin(window_dates).astype(int)

    return daily


def merge_all_news_features(
    price_df: pd.DataFrame,
    broad_daily: pd.DataFrame,
    opec_flags: pd.DataFrame,
    disruption_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all news feature layers onto the price series."""
    df = price_df.copy()
    df = df.merge(broad_daily, on="date", how="left")
    df = df.merge(opec_flags, on="date", how="left")
    df = df.merge(disruption_flags, on="date", how="left")

    # Fill non-event days with 0 (counts, flags, event indicators, decay/variance features)
    zero_fill_patterns = ["count", "flag", "day", "window", "intensity", "ratio",
                          "cs_di", "csi_v"]
    event_cols = [c for c in df.columns if any(x in c for x in zero_fill_patterns)]
    df[event_cols] = df[event_cols].fillna(0)

    # Neutral sentiment on days with no coverage
    sentiment_cols = [c for c in df.columns if "sentiment" in c]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    # FinBERT probability columns — fill with 0 on no-coverage weeks
    finbert_cols = [c for c in df.columns if c.startswith("finbert_")]
    df[finbert_cols] = df[finbert_cols].fillna(0)

    return df