# scripts/run_backfill.py
# Main entry point for running the Benzinga news backfill pipeline.
#
# Usage (from project root):
#   python3 scripts/run_backfill.py

import os
import sys
import logging
from datetime import date
from dotenv import load_dotenv

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from src.data.benzinga_pipeline import BenzingaMultiStrategyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Configuration ──────────────────────────────────────────────────────────────

START_DATE = "2019-01-01"
END_DATE   = date.today().isoformat()

# Set to a subset to run only specific strategies, e.g. ["broad", "opec"]
# Leave as None to run all four: broad, opec, disruption, macro
STRATEGIES = None

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ["BENZINGA_API_KEY"]

    pipeline = BenzingaMultiStrategyPipeline(api_key=api_key)
    results = pipeline.run_backfill(
        start_date=START_DATE,
        end_date=END_DATE,
        strategies=STRATEGIES,
    )

    print("\nBackfill complete:")
    for name, df in results.items():
        print(f"  {name}: {len(df):,} articles")
