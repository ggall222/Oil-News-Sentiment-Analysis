# src/features/sentiment_indicators.py
#
# Computes 4 weekly sentiment indicators for each of the 8 topic categories.
#
# For each (week, category) pair:
#
#   1. category_intensity   (CI)  = (1 / N_t) * Σ_j CW_{i,j}
#      where CW_{i,j}=1/N_j for multi-label article j that contains category i.
#
#   2. sentiment_intensity  (CSI) = mean sentiment of category i articles in week t.
#
#   3. sentiment_decay      (CS_DI) = CSI_{i,t} + Σ_{l=1}^{t-1} exp(-(t-l)/n) * CSI_{i,l}
#      implemented with decay_horizon_weeks = n and decay_lambda = 1/n by default.
#
#   4. sentiment_variance   (CSI_V) = CI_{i,t} * Var(sentiment in category i, week t)
#
# Output: wide DataFrame with columns named  {category}_{indicator},
#         indexed by week-ending Friday date, aligned to the WTI price series.

import numpy as np
import pandas as pd
from typing import Dict, List

from src.features.topic_classifier import CATEGORIES

DECAY_LAMBDA = 0.1   # exponential decay rate; tune for your data


def _week_end_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Return the Friday on or after `date`."""
    days_ahead = (4 - date.weekday()) % 7
    return date + pd.Timedelta(days=days_ahead)


def compute_weekly_indicators(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "lm_sentiment",
    topics_col: str = "lm_topics",
    decay_lambda: float = DECAY_LAMBDA,
    decay_horizon_weeks: int = 4,
) -> pd.DataFrame:
    """
    Compute 4 × 8 = 32 weekly sentiment indicators.

    Parameters
    ----------
    df           : article-level DataFrame with date, lm_sentiment, lm_topics
    date_col     : name of the date column (datetime or date)
    sentiment_col: name of the LM-S sentiment score column
    topics_col   : name of the comma-separated topics column
    decay_lambda : decay rate λ for sentiment_decay indicator

    Returns
    -------
    DataFrame with one row per week-ending Friday and 32 indicator columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Assign week-ending Friday for aggregation
    df["week_end"] = df[date_col].apply(_week_end_friday)

    # Explode topics so each article appears once per matched category
    df["topics_list"] = df[topics_col].apply(
        lambda x: x.split(",") if isinstance(x, str) and x else []
    )
    # Number of categories per article (N_j) and article weight within each category.
    df["num_categories"] = df["topics_list"].apply(len)
    df["category_weight"] = df["num_categories"].apply(lambda n: 1.0 / n if n > 0 else 0.0)

    exploded = df.explode("topics_list").rename(columns={"topics_list": "category"})
    exploded = exploded[exploded["category"].isin(CATEGORIES)]

    weeks = sorted(df["week_end"].unique())
    records = []

    # Precompute weekly total news volume N_t for CI normalization.
    weekly_total_news = df.groupby("week_end").size().to_dict()

    # Collect CSI history first so CS_DI can follow cross-week Eq. (12).
    csi_history: Dict[str, Dict[pd.Timestamp, float]] = {c: {} for c in CATEGORIES}

    # If user doesn't override lambda, align it with n from Eq. (12).
    if decay_lambda == DECAY_LAMBDA and decay_horizon_weeks > 0:
        decay_lambda = 1.0 / decay_horizon_weeks

    for week in weeks:
        row: Dict[str, object] = {"date": week}
        n_t = int(weekly_total_news.get(week, 0))

        for cat in CATEGORIES:
            cat_articles = exploded[
                (exploded["week_end"] == week) &
                (exploded["category"] == cat)
            ]

            n = len(cat_articles)
            scores = cat_articles[sentiment_col].dropna()

            # 1) CI_{i,t} = (1/N_t) * Σ CW_{i,j}
            if n_t > 0 and n > 0:
                ci_val = float(cat_articles["category_weight"].sum() / n_t)
            else:
                ci_val = 0.0
            row[f"{cat}_intensity"] = round(ci_val, 6)

            if n == 0 or scores.empty:
                row[f"{cat}_sentiment"] = 0.0
                row[f"{cat}_decay"] = 0.0
                row[f"{cat}_variance"] = 0.0
                csi_history[cat][week] = 0.0
                continue

            # 2) CSI_{i,t}
            csi_val = float(scores.mean())
            row[f"{cat}_sentiment"] = round(csi_val, 6)
            csi_history[cat][week] = csi_val

            # 4) CSI_V_{i,t} = CI_{i,t} * Var(sentiment in category/week)
            var_val = float(scores.var(ddof=1)) if n > 1 else 0.0
            row[f"{cat}_variance"] = round(ci_val * var_val, 6)

        records.append(row)

    # 3) CS_DI across weeks (Eq. 12): CSI_t + Σ exp(-(t-l)/n) * CSI_l
    week_index = {w: i for i, w in enumerate(weeks)}
    for row in records:
        week = row["date"]
        t_idx = week_index[week]
        for cat in CATEGORIES:
            csi_t = csi_history[cat].get(week, 0.0)
            decay_sum = csi_t
            for prev_week in weeks[:t_idx]:
                l_idx = week_index[prev_week]
                lag = t_idx - l_idx
                decay_sum += float(np.exp(-decay_lambda * lag) * csi_history[cat].get(prev_week, 0.0))
            row[f"{cat}_decay"] = round(decay_sum, 6)

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    return result


def build_indicator_column_names() -> List[str]:
    """Return the full list of 32 indicator column names in order."""
    cols = []
    for cat in CATEGORIES:
        for indicator in ["intensity", "sentiment", "decay", "variance"]:
            cols.append(f"{cat}_{indicator}")
    return cols
