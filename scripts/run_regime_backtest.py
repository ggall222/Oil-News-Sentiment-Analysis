"""
Run regime-aware expanding-window backtest on weekly feature matrix.

Usage:
    python3 scripts/run_regime_backtest.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.regime_aware_forecast import (
    walk_forward_regime_backtest,
    overall_metrics,
    performance_by_regime,
    regime_distribution_by_fold,
    weak_regime_diagnostics,
    plot_inferred_regimes,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).parent.parent
FEATURE_PATH = BASE_DIR / "data/features/feature_matrix.parquet"
OUT_DIR = BASE_DIR / "data/features"


def parse_args():
    p = argparse.ArgumentParser(description="Run regime-aware walk-forward backtest")
    p.add_argument("--min-train-size", type=int, default=156)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--min-obs-per-regime", type=int, default=40)
    p.add_argument("--global-weight", type=float, default=0.20)
    p.add_argument("--nested-tuning", action="store_true")
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--inner-val-size", type=int, default=26)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature matrix not found: {FEATURE_PATH}")

    df = pd.read_parquet(FEATURE_PATH)
    logging.info("Loaded feature matrix: %d rows, %d cols", len(df), len(df.columns))

    bt = walk_forward_regime_backtest(
        df,
        target_col="close_pct_change",
        date_col="date",
        min_train_size=args.min_train_size,
        step=args.step,
        min_obs_per_regime=args.min_obs_per_regime,
        global_weight=args.global_weight,
        nested_tuning=args.nested_tuning,
        inner_splits=args.inner_splits,
        inner_val_size=args.inner_val_size,
    )

    if bt.empty:
        logging.warning("Backtest returned no rows.")
        sys.exit(0)

    metrics = overall_metrics(bt)
    by_regime = performance_by_regime(bt)
    fold_dist = regime_distribution_by_fold(bt, fold_size=52)
    weak = weak_regime_diagnostics(bt)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(OUT_DIR / "regime_backtest.parquet", index=False)
    by_regime.to_csv(OUT_DIR / "regime_performance.csv")
    fold_dist.to_csv(OUT_DIR / "regime_fold_distribution.csv", index=False)
    weak.to_csv(OUT_DIR / "regime_weakness.csv")

    fig = plot_inferred_regimes(bt)
    if fig is not None:
        fig.savefig(OUT_DIR / "regime_backtest_plot.png", dpi=140)

    print("\nRegime-aware backtest complete")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Rows: {len(bt)}")
    print(f"  Outputs: {OUT_DIR}")
