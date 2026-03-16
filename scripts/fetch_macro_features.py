# scripts/fetch_macro_features.py
# Fetches macro / fundamental features from EIA (v2 API) and FRED.
#
# EIA series (weekly, Fridays):
#   WCRSTUS1  — U.S. ending crude oil stocks (thousand barrels)
#   WCRFPUS2  — U.S. field production of crude oil (kbd)
#   WCRIMUS2  — U.S. imports of crude oil (kbd)
#   WPULEUS3  — U.S. refinery operable capacity utilisation (%)
#
# FRED series (daily → resampled to week-ending Friday):
#   DTWEXBGS  — Trade-weighted broad USD index
#   SP500     — S&P 500
#   DGS10     — 10-year Treasury yield
#
# Usage (from project root):
#   python3 scripts/fetch_macro_features.py
#
# Output: data/features/macro_features.parquet

import os
import sys
import time
import logging
import requests
from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "features"

START = "2015-01-01"   # back to 2015 so inventory_surprise_kb has 4+ years of same-week history before 2020
END   = date.today().isoformat()

EIA_BASE = "https://api.eia.gov/v2"

# (series_id, route, extra_facets)
# All series are fetched via the `series` facet on their respective v2 route.
EIA_SERIES = [
    ("WCRSTUS1", "petroleum/stoc/wstk", {}),   # Ending crude stocks (MBBL)
    ("WCRFPUS2", "petroleum/sum/sndw",  {}),   # Field production (MBBL/D)
    ("WCRIMUS2", "petroleum/sum/sndw",  {}),   # Crude imports (MBBL/D)
    ("WPULEUS3", "petroleum/pnp/wiup",  {}),   # Refinery utilisation (%)
]

FRED_SERIES = {
    "usd_index": "DTWEXBGS",   # Trade-weighted broad USD index
    "spx":       "SP500",      # S&P 500
    "yield_10y": "DGS10",      # 10-year Treasury yield (%)
}

# ── EIA v2 helpers ────────────────────────────────────────────────────────────

def _eia_fetch_page(route: str, params: dict, api_key: str) -> list:
    """Single page request to the EIA v2 data endpoint."""
    url = f"{EIA_BASE}/{route}/data/"
    resp = requests.get(url, params={**params, "api_key": api_key}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise ValueError(f"EIA API error for {route}: {body['error']}")
    return body["response"]["data"], int(body["response"].get("total", 0))


def fetch_eia_series(series_id: str, route: str, api_key: str,
                     start: str, end: str) -> pd.Series:
    """
    Fetch a complete weekly EIA v2 series by series ID.
    Handles pagination automatically (EIA default page size = 5000).
    Returns a date-indexed Series.
    """
    base_params = {
        "frequency": "weekly",
        "data[0]":   "value",
        "facets[series][]": series_id,
        "start":     start,
        "end":       end,
        "sort[0][column]":    "period",
        "sort[0][direction]": "asc",
        "length":    5000,
        "offset":    0,
    }

    all_rows = []
    while True:
        rows, total = _eia_fetch_page(route, base_params, api_key)
        all_rows.extend(rows)
        if len(all_rows) >= total:
            break
        base_params["offset"] += 5000
        time.sleep(0.2)

    if not all_rows:
        raise ValueError(f"No data returned for EIA series {series_id}")

    s = pd.Series(
        {pd.to_datetime(row["period"]): float(row["value"])
         for row in all_rows if row["value"] is not None}
    ).sort_index()
    s.name = series_id
    return s


# ── FRED helpers ──────────────────────────────────────────────────────────────

def fetch_fred_weekly(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch a FRED series and resample to week-ending Friday (last observation).
    """
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = s.dropna()
    s_weekly = s.resample("W-FRI").last().dropna()
    s_weekly.name = series_id
    return s_weekly


# ── Derived features ──────────────────────────────────────────────────────────

def build_eia_features(eia_df: pd.DataFrame) -> pd.DataFrame:
    df = eia_df.copy().sort_index()

    # Week-over-week changes (stationary transforms of level series)
    df["inventory_chg_kb"]      = df["WCRSTUS1"].diff()
    df["production_chg_kbd"]    = df["WCRFPUS2"].diff()
    df["imports_chg_kbd"]       = df["WCRIMUS2"].diff()
    df["refinery_util_chg_pct"] = df["WPULEUS3"].diff()

    # Inventory surprise vs expanding seasonal mean (same ISO week)
    # Captures the draw/build relative to the typical seasonal pattern.
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    seasonal_mean = (
        df[["week_of_year", "inventory_chg_kb"]]
        .groupby("week_of_year")["inventory_chg_kb"]
        .transform(lambda x: x.expanding(min_periods=4).mean())
    )
    df["inventory_surprise_kb"] = df["inventory_chg_kb"] - seasonal_mean
    df = df.drop(columns="week_of_year")

    return df


def build_fred_features(fred_df: pd.DataFrame) -> pd.DataFrame:
    df = fred_df.copy().sort_index()

    df["usd_ret_pct"]   = df["usd_index"].pct_change() * 100
    df["spx_ret_pct"]   = df["spx"].pct_change() * 100
    df["yield_10y_chg"] = df["yield_10y"].diff()

    # 4-week USD momentum — sustained dollar strength = headwind for oil
    df["usd_mom_4w"]    = df["usd_index"].pct_change(4) * 100

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    eia_key  = os.environ.get("EIA_API_KEY")
    fred_key = os.environ.get("FRED_API_KEY")

    if not eia_key:
        logging.error("EIA_API_KEY not set in .env"); sys.exit(1)
    if not fred_key:
        logging.error("FRED_API_KEY not set in .env"); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── EIA ────────────────────────────────────────────────────────────────────
    logging.info("Fetching EIA series (v2 API)...")
    eia_raw = {}
    for series_id, route, _ in EIA_SERIES:
        s = fetch_eia_series(series_id, route, eia_key, START, END)
        eia_raw[series_id] = s
        logging.info(f"  {series_id}  {len(s):4d} weeks  "
                     f"{s.index[0].date()} → {s.index[-1].date()}  "
                     f"({s.name})")

    eia_df = pd.DataFrame(eia_raw)

    # Snap EIA index to nearest Friday (it's already weekly, but confirm alignment)
    eia_df.index = eia_df.index.map(
        lambda d: d + pd.offsets.Week(weekday=4) if d.weekday() != 4 else d
    )
    eia_df = eia_df[~eia_df.index.duplicated(keep="last")]

    eia_features = build_eia_features(eia_df)

    # ── FRED ───────────────────────────────────────────────────────────────────
    logging.info("Fetching FRED series...")
    fred_client = Fred(api_key=fred_key)
    fred_raw = {}
    for col_name, series_id in FRED_SERIES.items():
        s = fetch_fred_weekly(fred_client, series_id, START, END)
        fred_raw[col_name] = s
        logging.info(f"  {col_name} ({series_id})  {len(s):4d} weeks  "
                     f"{s.index[0].date()} → {s.index[-1].date()}")

    fred_df = pd.DataFrame(fred_raw)
    fred_features = build_fred_features(fred_df)

    # ── Merge on week-ending Friday index ─────────────────────────────────────
    logging.info("Merging EIA + FRED on Friday index...")
    macro = eia_features.join(fred_features, how="outer").sort_index()
    macro = macro.loc[START:END]

    # ── Select final model-ready columns ──────────────────────────────────────
    FINAL_COLS = [
        # EIA — stationary features
        "inventory_chg_kb",       # WoW crude draw/build (primary catalyst)
        "inventory_surprise_kb",  # draw/build vs seasonal baseline
        "production_chg_kbd",     # WoW production change
        "imports_chg_kbd",        # WoW import change
        "WPULEUS3",               # refinery util level (mean-reverting)
        "refinery_util_chg_pct",  # WoW change in refinery util
        # FRED — returns / changes
        "usd_ret_pct",            # USD WoW return (inverse relationship)
        "usd_mom_4w",             # 4-week USD momentum
        "spx_ret_pct",            # S&P 500 WoW return (risk-on proxy)
        "yield_10y_chg",          # 10Y yield change (macro signal)
    ]
    macro_out = macro[FINAL_COLS].copy()
    macro_out.index.name = "date"
    macro_out = macro_out.reset_index()
    macro_out["date"] = pd.to_datetime(macro_out["date"])

    # ── Rename WPULEUS3 to a human-readable name ──────────────────────────────
    macro_out = macro_out.rename(columns={"WPULEUS3": "refinery_util_pct"})

    out_path = OUTPUT_DIR / "macro_features.parquet"
    macro_out.to_parquet(out_path, index=False)

    null_counts = macro_out.isnull().sum()
    logging.info(f"Saved {len(macro_out)} rows → {out_path}")
    logging.info(f"Columns: {list(macro_out.columns)}")
    if null_counts.any():
        logging.warning(f"Null counts:\n{null_counts[null_counts > 0]}")

    print(f"\nMacro feature build complete: {len(macro_out)} rows, "
          f"{macro_out['date'].min().date()} → {macro_out['date'].max().date()}")
