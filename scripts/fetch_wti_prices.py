# scripts/fetch_wti_prices.py
# Fetches weekly WTI spot price from the EIA API and saves to CSV.
#
# Series: PET.RWTC.W — Cushing, OK WTI Spot Price FOB (Dollars per Barrel, Weekly)
#
# Usage (from project root):
#   python3 scripts/fetch_wti_prices.py

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Configuration ──────────────────────────────────────────────────────────────

SERIES_ID  = "PET.RWTC.W"
START_DATE = "2019-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
OUTPUT_CSV = Path("data/wti_prices.csv")

EIA_BASE   = "https://api.eia.gov/v2/seriesid"

# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_wti_prices(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    url = f"{EIA_BASE}/{series_id}"
    params = {
        "api_key": api_key,
        "data[0]": "value",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code == 403:
        raise ValueError("EIA API auth failed — check your EIA_API_KEY.")
    resp.raise_for_status()

    data = resp.json()
    rows = data.get("response", {}).get("data", [])

    if not rows:
        raise ValueError(f"No data returned for series {series_id}. Check the series ID and date range.")

    df = pd.DataFrame(rows)[["period", "value"]].rename(columns={"period": "date", "value": "close"})
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    # Filter to requested date range (seriesid endpoint ignores start/end params)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    return df

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise EnvironmentError("EIA_API_KEY not found in environment. Check your .env file.")

    logging.info(f"Fetching {SERIES_ID}  ({START_DATE} → {END_DATE})...")
    df = fetch_wti_prices(SERIES_ID, START_DATE, END_DATE, api_key)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    logging.info(f"Saved {len(df)} weekly observations → {OUTPUT_CSV}")
    print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f} per barrel")
    print(df.tail())
