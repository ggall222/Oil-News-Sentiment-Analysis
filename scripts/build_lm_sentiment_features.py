# scripts/build_lm_sentiment_features.py
#
# Builds LM-S weekly sentiment features across 8 oil-market topic categories.
#
# Pipeline:
#   1. Load Benzinga broad tier + OilPrice.com articles
#   2. Combine into unified text column (title + body)
#   3. Label price direction from headlines (rise / fall / None)
#   4. Build LM-S lexicon (base LM + oil extensions + corpus expansion)
#   5. Score all articles with LM-S lexicon
#   6. Classify articles into 8 topic categories
#   7. Compute 4 × 8 = 32 weekly indicators
#   8. Lag by 1 week + merge onto WTI price series → feature matrix
#   9. Save to data/features/lm_sentiment_weekly.parquet
#
# Usage (from WTI_News_API/ directory):
#   python3 scripts/build_lm_sentiment_features.py

import os
import sys
import logging
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.lm_sentiment import (
    label_price_direction,
    build_lm_s_lexicon,
    score_articles,
)
from src.features.paragraph_topic_classifier import assign_topics_from_paragraph_model
from src.features.sentiment_indicators import (
    compute_weekly_indicators,
    build_indicator_column_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.parent
RAW_DIR        = BASE_DIR / "data/raw/benzinga"
OILPRICE_CSV   = BASE_DIR / "data/raw/oilprice_relevant_articles.csv"
OUTPUT_DIR     = BASE_DIR / "data/features"
PRICE_CSV      = BASE_DIR / "data/wti_prices.csv"

DECAY_LAMBDA   = 0.1   # sentiment decay rate (higher = faster decay)
DECAY_HORIZON_WEEKS = 4
MIN_FREQ       = 3     # min word occurrences for lexicon expansion
MIN_RATIO      = 1.5   # min frequency ratio to add an expansion word

# ── Data loading ───────────────────────────────────────────────────────────────

def load_benzinga_broad() -> pd.DataFrame:
    files = sorted(glob(str(RAW_DIR / "broad" / "*.parquet")))
    if not files:
        logging.warning("No Benzinga broad parquets found — skipping")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="id")
    df["date"] = pd.to_datetime(df["date"])
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df["headline"] = df["title"].fillna("")
    logging.info(f"Benzinga broad: {len(df):,} articles")
    return df[["date", "headline", "text"]]


def load_oilprice() -> pd.DataFrame:
    if not OILPRICE_CSV.exists():
        logging.warning(f"OilPrice CSV not found: {OILPRICE_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(OILPRICE_CSV, parse_dates=["published_at"])
    df = df.rename(columns={"published_at": "date", "headline": "headline"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["text"] = (df["headline"].fillna("") + " " + df["body_text"].fillna("")).str.strip()
    df = df.dropna(subset=["date"])
    logging.info(f"OilPrice: {len(df):,} articles")
    return df[["date", "headline", "text"]]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and combine sources
    logging.info("Loading article data...")
    benzinga = load_benzinga_broad()
    oilprice = load_oilprice()

    parts = [df for df in [benzinga, oilprice] if not df.empty]
    if not parts:
        logging.error("No article data found. Exiting.")
        sys.exit(1)

    articles = pd.concat(parts, ignore_index=True)
    articles = articles.dropna(subset=["date", "text"])
    articles = articles[articles["text"].str.strip() != ""]
    logging.info(f"Combined corpus: {len(articles):,} articles")

    # 2. Label price direction from headlines
    logging.info("Labeling price direction from headlines...")
    articles["price_direction"] = articles["headline"].apply(label_price_direction)
    labeled_counts = articles["price_direction"].value_counts(dropna=False)
    logging.info(f"  Labels: {labeled_counts.to_dict()}")

    # 3. Build LM-S lexicon
    logging.info("Building LM-S lexicon...")
    lexicon = build_lm_s_lexicon(
        articles,
        text_col="text",
        label_col="price_direction",
        min_freq=MIN_FREQ,
        min_ratio=MIN_RATIO,
    )
    logging.info(f"  Lexicon size: {len(lexicon):,} terms "
                 f"(positive: {sum(1 for v in lexicon.values() if v > 0)}, "
                 f"negative: {sum(1 for v in lexicon.values() if v < 0)})")

    # 4. Score all articles
    logging.info("Scoring articles with LM-S lexicon...")
    articles = score_articles(articles, lexicon, text_col="text")
    logging.info(f"  Mean score: {articles['lm_sentiment'].mean():.4f}  "
                 f"Std: {articles['lm_sentiment'].std():.4f}")

    # 5. Paragraph-level ML classification into 8 topic categories
    logging.info("Classifying paragraph-level topics with ML model...")
    articles, clf_info = assign_topics_from_paragraph_model(
        articles,
        text_col="text",
        date_col="date",
        topics_col="lm_topics",
        paragraph_proba_threshold=0.40,
        min_chars=60,
    )
    logging.info(
        "  Paragraph classifier model=%s, paragraphs=%s, weak_labeled=%s",
        clf_info.get("model"), clf_info.get("paragraphs"), clf_info.get("labeled_paragraphs"),
    )
    articles_with_topic = (articles["lm_topics"] != "").sum()
    logging.info(f"  Articles with at least one topic: {articles_with_topic:,} "
                 f"({100 * articles_with_topic / len(articles):.1f}%)")

    # Category distribution
    for cat in build_indicator_column_names()[::4]:  # every 4th = intensity cols
        cat_name = cat.replace("_intensity", "")
        n = articles["lm_topics"].str.contains(cat_name, na=False).sum()
        logging.info(f"    {cat_name}: {n:,} articles")

    # 6. Compute weekly indicators
    logging.info("Computing weekly sentiment indicators (4 × 8 = 32 features)...")
    weekly = compute_weekly_indicators(
        articles,
        date_col="date",
        sentiment_col="lm_sentiment",
        topics_col="lm_topics",
        decay_lambda=DECAY_LAMBDA,
        decay_horizon_weeks=DECAY_HORIZON_WEEKS,
    )
    logging.info(f"  Weekly indicator shape: {weekly.shape}")

    # 7. Save standalone weekly indicators
    weekly.to_parquet(OUTPUT_DIR / "lm_sentiment_weekly.parquet", index=False)
    logging.info("  Saved → data/features/lm_sentiment_weekly.parquet")

    # 8. Optionally merge with WTI price series
    if PRICE_CSV.exists():
        logging.info(f"Merging with price series: {PRICE_CSV}")
        price_df = pd.read_csv(PRICE_CSV, parse_dates=["date"])
        price_df["date"] = price_df["date"].dt.date

        # 1-week lag: use last week's news to predict this week's price
        indicator_cols = build_indicator_column_names()
        weekly["date"] = pd.to_datetime(weekly["date"])

        weekly_lagged = weekly.copy()
        weekly_lagged["date"] = weekly_lagged["date"] + pd.Timedelta(weeks=1)
        weekly_lagged["date"] = weekly_lagged["date"].dt.date
        lagged_col_map = {c: f"{c}_lag1" for c in indicator_cols}
        weekly_lagged = weekly_lagged.rename(columns=lagged_col_map)

        merged = price_df.merge(weekly_lagged, on="date", how="left")

        # Fill weeks with no news coverage with 0
        lagged_cols = list(lagged_col_map.values())
        merged[lagged_cols] = merged[lagged_cols].fillna(0)

        # Weekly return target
        merged["close_pct_change"] = merged["close"].pct_change() * 100
        merged = merged.dropna(subset=["close_pct_change"]).reset_index(drop=True)

        merged.to_parquet(OUTPUT_DIR / "lm_feature_matrix.parquet", index=False)
        logging.info(f"  Saved {len(merged)} rows → data/features/lm_feature_matrix.parquet")

        covered = (merged[lagged_cols[0]] != 0).sum()
        logging.info(f"  Weeks with news coverage: {covered}/{len(merged)} "
                     f"({100 * covered / len(merged):.1f}%)")
    else:
        logging.warning(f"Price CSV not found: {PRICE_CSV} — skipping merge")

    print("\nLM-S sentiment build complete:")
    print(f"  Articles processed  : {len(articles):,}")
    print(f"  Lexicon terms       : {len(lexicon):,}")
    print(f"  Weekly indicator rows: {len(weekly):,}")
    print(f"  Output              : data/features/lm_sentiment_weekly.parquet")
