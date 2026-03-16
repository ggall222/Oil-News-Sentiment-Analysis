# src/features/regime_detection.py
# Regime detection features for the WTI feature matrix.
#
# Provides three complementary regime signals:
#   1. vol_regime      — percentile-based (0=calm, 1=normal, 2=high) from rolling vol
#   2. hmm_regime      — unsupervised HMM (3 states) fit on returns + vol
#   3. regime_0/1/2    — one-hot columns for vol_regime (ready to lag + feed into GRU)
#   4. hmm_regime_0/1/2 — one-hot columns for hmm_regime

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

VOL_WINDOW = 8  # rolling window for realized volatility (weeks)


def add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volatility-based regime label using rolling 8-week return std.

    Regime labels (quantile-based, so roughly equal class frequencies):
        0 = calm      (realized_vol <= 33rd percentile)
        1 = normal    (33rd < realized_vol <= 66th percentile)
        2 = high-vol  (realized_vol > 66th percentile)

    New columns: realized_vol, vol_regime, regime_0, regime_1, regime_2
    """
    df = df.copy()

    df["realized_vol"] = (
        df["close_pct_change"]
        .rolling(VOL_WINDOW, min_periods=4)
        .std()
    )

    # Compute thresholds on non-NaN rows only to avoid look-ahead bias
    # (using the full-sample quantiles is acceptable here because we're
    # building a static feature matrix; for live use, compute on train set only)
    vol_33 = df["realized_vol"].quantile(0.33)
    vol_66 = df["realized_vol"].quantile(0.66)

    df["vol_regime"] = pd.cut(
        df["realized_vol"],
        bins=[-np.inf, vol_33, vol_66, np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")  # nullable int so NaNs stay as NA, not -1

    # One-hot encode (fills NaN rows with 0)
    for label in [0, 1, 2]:
        df[f"regime_{label}"] = (df["vol_regime"] == label).fillna(False).astype(int)

    logger.info(
        f"vol_regime distribution — "
        f"calm={( df['vol_regime']==0).sum()} "
        f"normal={( df['vol_regime']==1).sum()} "
        f"high={(df['vol_regime']==2).sum()} "
        f"NaN={(df['vol_regime'].isna()).sum()}"
    )
    return df


def add_hmm_regime(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Adds HMM-based regime labels using GaussianHMM fit on
    [close_pct_change, realized_vol].

    Requires 'realized_vol' to already be present (call add_vol_regime first).

    New columns: hmm_regime, hmm_regime_0, hmm_regime_1, hmm_regime_2

    Notes:
      - HMM is fit on the full series (no train/test split here) because
        the Viterbi decoding is used as a feature, not a prediction.
      - The regime labels are arbitrary integers; the GRU will learn their
        meaning. One-hot encoding removes any ordinal assumption.
      - COVID weeks will naturally cluster into their own state due to the
        extreme returns + vol they produce.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed — skipping HMM regime. pip install hmmlearn")
        df["hmm_regime"] = np.nan
        for i in range(n_components):
            df[f"hmm_regime_{i}"] = 0
        return df

    df = df.copy()

    required = {"close_pct_change", "realized_vol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_hmm_regime requires columns: {missing}")

    obs_df = df[["close_pct_change", "realized_vol"]].copy()
    valid_mask = obs_df.notna().all(axis=1)
    obs = obs_df[valid_mask].values

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(obs)
    hidden_states = model.predict(obs)

    df["hmm_regime"] = np.nan
    df.loc[valid_mask, "hmm_regime"] = hidden_states.astype(float)
    df["hmm_regime"] = df["hmm_regime"].astype("Int64")

    # One-hot encode
    for i in range(n_components):
        df[f"hmm_regime_{i}"] = (df["hmm_regime"] == i).fillna(False).astype(int)

    # Log which state captured the extreme volatility (COVID regime)
    if "realized_vol" in df.columns:
        peak_vol_row = df["realized_vol"].idxmax()
        peak_state = df.loc[peak_vol_row, "hmm_regime"]
        logger.info(
            f"HMM converged ({model.monitor_.iter} iters). "
            f"Peak-vol state (COVID): {peak_state}. "
            f"State counts: { {i: (df['hmm_regime']==i).sum() for i in range(n_components)} }"
        )

    return df
