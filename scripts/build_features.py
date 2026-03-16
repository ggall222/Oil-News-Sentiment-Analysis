# scripts/build_features.py
# Builds daily news features from the raw Parquet backfill.
#
# Usage (from project root):
#   python3 scripts/build_features.py
#
# Optional: set PRICE_CSV to a CSV with columns [date, close] to merge
# WTI price data and produce a fully merged feature matrix.

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.news_features import score_sentiment, build_daily_features
from src.features.event_flags import (
    build_opec_event_flags,
    build_disruption_flags,
    merge_all_news_features,
)
from src.features.regime_detection import add_vol_regime, add_hmm_regime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent.parent
RAW_DIR         = BASE_DIR / "data/raw/benzinga"
OILPRICE_DIR    = BASE_DIR / "data/raw/oilprice"
OUTPUT_DIR      = BASE_DIR / "data/features"

# Optional: path to a CSV with columns [date, close] for WTI price merging.
# Leave as None to output news features only.
PRICE_CSV   = BASE_DIR / "data/wti_prices.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_strategy(name: str) -> pd.DataFrame:
    """Load all monthly Parquet files for a strategy into one DataFrame."""
    files = sorted(glob(str(RAW_DIR / name / "*.parquet")))
    if not files:
        logging.warning(f"No parquet files found for strategy '{name}'")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="id")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logging.info(f"Loaded '{name}': {len(df):,} articles")
    return df


def load_oilprice() -> pd.DataFrame:
    """
    Load all OilPrice parquets from data/raw/oilprice/ and return in the
    same schema as load_strategy() — ready to concatenate with broad_df.
    Returns an empty DataFrame if no files exist yet.
    """
    files = sorted(glob(str(OILPRICE_DIR / "*.parquet")))
    if not files:
        logging.info("No OilPrice parquets found — skipping (run oilnewsscraper.py to populate)")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="id")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logging.info(f"Loaded OilPrice: {len(df):,} articles")
    return df


def compute_cs_di(series: pd.Series, n: int = 4) -> pd.Series:
    """
    Exponential decay sentiment impact (Eq 12, Bai et al. 2022).
    CS_DI_t = CSI_t + e^{-1/n} * CS_DI_{t-1}
    n = number of weeks over which news impact decays.
    """
    decay = np.exp(-1.0 / n)
    values = series.fillna(0).values.astype(float)
    result = np.zeros(len(values))
    result[0] = values[0]
    for t in range(1, len(values)):
        result[t] = values[t] + decay * result[t - 1]
    return pd.Series(result, index=series.index)


def resample_to_weekly(daily_df: pd.DataFrame, agg: dict) -> pd.DataFrame:
    """
    Resample daily news features to week-ending Friday to align with
    the EIA weekly WTI price series (PET.RWTC.W, always a Friday).
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("W-FRI").agg(agg).reset_index()
    df["date"] = df["date"].dt.date
    return df

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load each strategy tier
    broad_df       = load_strategy("broad")
    opec_df        = load_strategy("opec")
    disruption_df  = load_strategy("disruption")

    # 1b. Merge OilPrice articles into broad pool (same schema, deduplicated by id)
    oilprice_df = load_oilprice()
    if not oilprice_df.empty:
        broad_df = pd.concat([broad_df, oilprice_df], ignore_index=True).drop_duplicates(subset="id")
        logging.info(f"Broad pool after OilPrice merge: {len(broad_df):,} articles")

    # 2. Score sentiment on each tier (required before aggregation)
    logging.info("Scoring sentiment...")
    broad_df      = score_sentiment(broad_df)
    opec_df       = score_sentiment(opec_df)
    disruption_df = score_sentiment(disruption_df)

    # 3. Build broad daily features (headline count, sentiment stats, ratios)
    logging.info("Building broad daily features...")
    broad_daily = build_daily_features(broad_df)
    broad_daily.to_parquet(OUTPUT_DIR / "broad_daily.parquet", index=False)
    logging.info(f"  Saved {len(broad_daily)} daily rows → data/features/broad_daily.parquet")

    # 4. Build OPEC event flags
    logging.info("Building OPEC event flags...")
    opec_flags = build_opec_event_flags(opec_df)
    opec_flags.to_parquet(OUTPUT_DIR / "opec_flags.parquet", index=False)
    logging.info(f"  Saved {len(opec_flags)} daily rows → data/features/opec_flags.parquet")

    # 5. Build disruption event flags
    logging.info("Building disruption event flags...")
    disruption_flags = build_disruption_flags(disruption_df)
    disruption_flags.to_parquet(OUTPUT_DIR / "disruption_flags.parquet", index=False)
    logging.info(f"  Saved {len(disruption_flags)} daily rows → data/features/disruption_flags.parquet")

    # 6. Optionally merge all features onto WTI price series
    if PRICE_CSV:
        logging.info(f"Merging with price series: {PRICE_CSV}")
        price_df = pd.read_csv(PRICE_CSV, parse_dates=["date"])
        price_df["date"] = price_df["date"].dt.date

        # WTI series is weekly (Fridays) — resample daily news to week-ending Friday
        # so all articles within the week are captured, not just those published on Friday.
        logging.info("Resampling daily news features to week-ending Friday...")

        broad_weekly = resample_to_weekly(broad_daily, {
            "headline_count":         "sum",
            "sentiment_mean":         "mean",
            "sentiment_std":          "mean",
            "sentiment_min":          "min",
            "sentiment_max":          "max",
            "negative_ratio":         "mean",
            "positive_ratio":         "mean",
            "log_headline_count":     "sum",
            "finbert_positive_mean":  "mean",
            "finbert_negative_mean":  "mean",
            "finbert_neutral_mean":   "mean",
        })

        opec_weekly = resample_to_weekly(opec_flags, {
            "opec_article_count":   "sum",
            "opec_sentiment":       "mean",
            "opec_sentiment_std":   "mean",
            "opec_event_day":       "max",
            "opec_decision_day":    "max",
            "opec_event_window":    "max",
        })

        disruption_weekly = resample_to_weekly(disruption_flags, {
            "disruption_intensity":       "sum",
            "disruption_sentiment":       "mean",
            "disruption_sentiment_std":   "mean",
            "disruption_event_day":       "max",
            "disruption_event_window":    "max",
        })

        # ── Paper features (Bai et al. 2022): CS_DI and CSI_V ────────────────
        # CS_DI: exponential decay-weighted cumulative sentiment (n=4 weeks)
        broad_weekly["broad_cs_di"]       = compute_cs_di(broad_weekly["sentiment_mean"],       n=4)
        opec_weekly["opec_cs_di"]         = compute_cs_di(opec_weekly["opec_sentiment"],        n=4)
        disruption_weekly["disruption_cs_di"] = compute_cs_di(disruption_weekly["disruption_sentiment"], n=4)

        # CSI_V: intensity-weighted sentiment variance (mean × std)
        broad_weekly["broad_csi_v"]           = broad_weekly["sentiment_mean"] * broad_weekly["sentiment_std"]
        opec_weekly["opec_csi_v"]             = opec_weekly["opec_sentiment"]  * opec_weekly["opec_sentiment_std"]
        disruption_weekly["disruption_csi_v"] = disruption_weekly["disruption_sentiment"] * disruption_weekly["disruption_sentiment_std"]

        feature_matrix = merge_all_news_features(
            price_df, broad_weekly, opec_weekly, disruption_weekly
        )

        # ── Post-merge cleanup ─────────────────────────────────────────────────

        # 1. Proper datetime dtype
        feature_matrix["date"] = pd.to_datetime(feature_matrix["date"])

        # 2. Target: weekly return instead of absolute price (avoids non-stationarity)
        feature_matrix["close_pct_change"] = feature_matrix["close"].pct_change() * 100

        # 2b. Regime detection (computed on returns before lagging)
        logging.info("Adding volatility-based regime labels...")
        feature_matrix = add_vol_regime(feature_matrix)
        logging.info("Adding HMM-based regime labels...")
        feature_matrix = add_hmm_regime(feature_matrix)

        # 2c. Merge macro features (EIA + FRED) — aligned on week-ending Friday
        macro_path = OUTPUT_DIR / "macro_features.parquet"
        if macro_path.exists():
            logging.info(f"Merging macro features from {macro_path}...")
            macro_df = pd.read_parquet(macro_path)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            feature_matrix = feature_matrix.merge(macro_df, on="date", how="left")
            n_matched = feature_matrix[macro_df.columns[1]].notna().sum()
            logging.info(f"  Macro features matched: {n_matched}/{len(feature_matrix)} rows")
        else:
            logging.warning(
                f"macro_features.parquet not found at {macro_path} — "
                "run scripts/fetch_macro_features.py first to include EIA/FRED features."
            )

        # 3. Lag all news features by 1 week (use last week's news to predict this week's price)
        news_cols = [c for c in feature_matrix.columns if c not in ("date", "close", "close_pct_change")]
        feature_matrix[[f"{c}_lag1" for c in news_cols]] = feature_matrix[news_cols].shift(1)
        feature_matrix = feature_matrix.drop(columns=news_cols)

        # 4. Drop redundant / ordinal-encoded regime cols (keep one-hot versions only)
        feature_matrix = feature_matrix.drop(columns=[
            "opec_event_window_lag1",
            "disruption_event_window_lag1",
            "vol_regime_lag1",    # ordinal — one-hot lag1 cols carry the same info
            "hmm_regime_lag1",    # ordinal — one-hot lag1 cols carry the same info
        ], errors="ignore")

        # 5. Drop first row (NaN from lag + pct_change)
        feature_matrix = feature_matrix.dropna().reset_index(drop=True)

        feature_matrix.to_parquet(OUTPUT_DIR / "feature_matrix.parquet", index=False)
        logging.info(f"  Saved {len(feature_matrix)} rows → data/features/feature_matrix.parquet")

        matched = (feature_matrix["headline_count_lag1"] > 0).sum()
        logging.info(f"  Weeks with news coverage: {matched}/{len(feature_matrix)} ({matched/len(feature_matrix)*100:.1f}%)")
    else:
        logging.info("No PRICE_CSV set — skipping price merge. Set PRICE_CSV to produce feature_matrix.parquet.")

    print("\nFeature build complete:")
    print(f"  broad_daily:      {len(broad_daily):,} days")
    print(f"  opec_flags:       {len(opec_flags):,} days")
    print(f"  disruption_flags: {len(disruption_flags):,} days")
