# src/features/oil_regime.py
"""
4-state oil market regime detector.

Regimes (economically motivated):
  supply   – OPEC / disruption activity dominates price moves
  demand   – global growth / inventory draws dominate
  macro    – USD / rates / cross-asset moves dominate
  risk_off – volatility spike, fear, flight-to-safety

No look-ahead: only lag-1 features (available at prediction time) are used.

Typical walk-forward usage
──────────────────────────
    det = OilRegimeDetector()
    det.fit(df_train)                         # train-window only
    labels = det.predict(df_test)             # str array: 'supply'/'demand'/...
    proba  = det.predict_proba(df_test)       # DataFrame, 4 regime_prob_* cols
    print(det.transition_matrix())            # persistence of each regime
    print(det.regime_summary(df_train))       # mean driver features per regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment


# ── Driver signals ─────────────────────────────────────────────────────────────
# Grouped by regime dimension they are most informative for.

DRIVER_COLS: list[str] = [
    # Supply dimension
    "opec_event_day_lag1",
    "disruption_event_day_lag1",
    "disruption_intensity_lag1",
    "opec_sentiment_lag1",
    # Demand dimension
    "inventory_chg_kb_lag1",   # negative = demand draw → bullish
    "spx_ret_pct_lag1",        # global risk-on proxy
    # Macro dimension
    "usd_ret_pct_lag1",
    "usd_mom_4w_lag1",
    "yield_10y_chg_lag1",
    # Risk / sentiment dimension
    "realized_vol_lag1",
    "negative_ratio_lag1",
    "finbert_negative_mean_lag1",
]

REGIME_NAMES: list[str] = ["supply", "demand", "macro", "risk_off"]

# Regime-appropriate CS_DI decay half-life (weeks).
# Smaller n → sentiment stales faster.
REGIME_DECAY_N: dict[str, int] = {
    "supply":   6,   # supply disruptions have persistent price tails
    "demand":   4,   # standard monthly decay
    "macro":    2,   # macro regimes rotate faster
    "risk_off": 1,   # crisis: old sentiment is immediately irrelevant
}


class OilRegimeDetector:
    """
    Fit-predict wrapper around GaussianHMM for oil market regimes.

    Parameters
    ----------
    n_states : int
        Number of HMM states (default 4, one per regime).
    n_iter : int
        Maximum EM iterations for HMM fitting.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.random_state = random_state
        self._hmm:       GaussianHMM | None           = None
        self._scaler:    StandardScaler | None         = None
        self._state_map: dict[int, str] | None         = None  # hmm_state → regime
        self._inv_map:   dict[str, int] | None         = None  # regime → hmm_state

    # ── Private helpers ────────────────────────────────────────────────────────

    def _avail_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in DRIVER_COLS if c in df.columns]

    def _driver_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = self._avail_cols(df)
        return df[cols].fillna(0.0).values.astype(np.float64)

    def _assign_labels(self, means: pd.DataFrame) -> dict[int, str]:
        """
        Score each HMM state on 4 regime dimensions, then use the
        Hungarian algorithm for an optimal 1-to-1 state→regime assignment.

        Score construction (all terms normalised to [-1, 1]):
          supply   ∝ OPEC + disruption activity
          demand   ∝ SPX return − inventory build
          macro    ∝ |USD momentum| + |yield change|
          risk_off ∝ realised vol + negative sentiment
        """
        n = self.n_states
        score = np.zeros((n, 4))

        def col(name: str) -> np.ndarray:
            if name not in means.columns:
                return np.zeros(n)
            v = means[name].values.astype(float)
            span = max(float(np.abs(v).max()), 1e-6)
            return v / span

        score[:, 0] = (col("opec_event_day_lag1")
                       + col("disruption_event_day_lag1")
                       + col("disruption_intensity_lag1"))

        score[:, 1] = col("spx_ret_pct_lag1") - col("inventory_chg_kb_lag1")

        score[:, 2] = np.abs(col("usd_mom_4w_lag1")) + np.abs(col("yield_10y_chg_lag1"))

        score[:, 3] = (col("realized_vol_lag1")
                       + col("negative_ratio_lag1")
                       + col("finbert_negative_mean_lag1"))

        row_ind, col_ind = linear_sum_assignment(-score)  # maximise
        return {int(s): REGIME_NAMES[r] for s, r in zip(row_ind, col_ind)}

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> "OilRegimeDetector":
        """
        Fit scaler + HMM on training data only.

        Parameters
        ----------
        df_train : DataFrame with lag-1 feature columns.
        """
        X_raw = self._driver_matrix(df_train)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw)

        self._hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",   # stable with small N; "full" needs N >> features²
            n_iter=self.n_iter,
            random_state=self.random_state,
            min_covar=1e-3,
        )
        self._hmm.fit(X)

        means_orig = pd.DataFrame(
            self._scaler.inverse_transform(self._hmm.means_),
            columns=self._avail_cols(df_train),
        )
        self._state_map = self._assign_labels(means_orig)
        self._inv_map   = {v: k for k, v in self._state_map.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return regime label string for each row (Viterbi decoding).

        Returns
        -------
        np.ndarray of str, shape (n,)
        """
        X = self._scaler.transform(self._driver_matrix(df))
        states = self._hmm.predict(X)
        return np.array([self._state_map[int(s)] for s in states])

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return posterior regime probabilities via forward-backward smoothing.

        Columns: regime_prob_supply, regime_prob_demand,
                 regime_prob_macro,  regime_prob_risk_off

        Falls back to one-hot Viterbi encoding if hmmlearn < 0.2.7.
        """
        X = self._scaler.transform(self._driver_matrix(df))
        try:
            proba = self._hmm.predict_proba(X)
        except AttributeError:
            states = self._hmm.predict(X)
            proba = np.zeros((len(X), self.n_states))
            proba[np.arange(len(X)), states] = 1.0

        result = {
            f"regime_prob_{self._state_map[i]}": proba[:, i]
            for i in range(self.n_states)
        }
        return pd.DataFrame(result, index=df.index)

    def regime_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean driver-feature values per detected regime (rows=regimes, cols=features).
        Useful for interpreting what each regime "looks like".
        """
        avail = self._avail_cols(df)
        tmp = df[avail].copy()
        tmp["regime"] = self.predict(df)
        return tmp.groupby("regime").mean().round(3)

    def regime_counts(self, df: pd.DataFrame) -> pd.Series:
        """Week counts per regime."""
        return pd.Series(self.predict(df)).value_counts().rename("n_weeks")

    def transition_matrix(self) -> pd.DataFrame:
        """
        HMM transition probability matrix with regime-name row/col labels.
        Diagonal = probability of staying in same regime next week.
        """
        A = self._hmm.transmat_
        names = [self._state_map[i] for i in range(self.n_states)]
        return pd.DataFrame(A, index=names, columns=names).round(3)

    def state_means(self) -> pd.DataFrame:
        """Raw (un-scaled) mean driver values per regime."""
        avail = self._avail_cols  # bound method — just call it with a dummy
        cols = DRIVER_COLS  # use full list; missing cols handled in fit
        cols_used = cols[:self._hmm.means_.shape[1]]
        means_orig = pd.DataFrame(
            self._scaler.inverse_transform(self._hmm.means_),
            columns=cols_used,
        )
        means_orig.index = [self._state_map[i] for i in range(self.n_states)]
        return means_orig.round(3)

    @property
    def decay_n(self) -> dict[str, int]:
        """Regime-appropriate CS_DI decay parameter n (weeks)."""
        return REGIME_DECAY_N.copy()
