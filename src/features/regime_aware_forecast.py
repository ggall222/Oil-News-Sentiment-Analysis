"""
Regime-aware weekly return forecasting utilities.

Design goals:
- Walk-forward safe (fit on train window only).
- Practical with weekly data.
- Minimal dependencies (numpy/pandas/sklearn/matplotlib).
- Regime-conditioned prediction logic + diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score


REGIME_NAMES = [
    "supply",
    "demand",
    "macro",
    "risk_off",
]

DEFAULT_SENTIMENT_MULTIPLIERS: Dict[str, float] = {
    "supply":   1.10,   # supply narratives are persistent — amplify CS_DI
    "demand":   1.00,   # standard
    "macro":    0.65,   # macro/tariff shocks — discount stale sentiment heavily
    "risk_off": 0.50,   # crisis — reset sentiment carry almost entirely
}

DEFAULT_MODEL_PARAM_GRID: List[Dict[str, object]] = [
    {"learning_rate": 0.03, "max_depth": 3, "max_iter": 220, "min_samples_leaf": 20, "l2_regularization": 0.10},
    {"learning_rate": 0.05, "max_depth": 4, "max_iter": 250, "min_samples_leaf": 15, "l2_regularization": 0.10},
    {"learning_rate": 0.07, "max_depth": 5, "max_iter": 300, "min_samples_leaf": 12, "l2_regularization": 0.05},
]


DEFAULT_REGIME_FEATURE_MAP: Dict[str, List[str]] = {
    "supply": [
        "opec_event_day_lag1",
        "disruption_event_day_lag1",
        "disruption_intensity_lag1",
        "opec_sentiment_lag1",
        "opec_cs_di_lag1",
        "disruption_cs_di_lag1",
        "opec_csi_v_lag1",
        "disruption_csi_v_lag1",
    ],
    "demand": [
        "inventory_chg_kb_lag1",
        "inventory_surprise_kb_lag1",
        "production_chg_kbd_lag1",
        "imports_chg_kbd_lag1",
        "refinery_util_pct_lag1",
        "refinery_util_chg_pct_lag1",
        "spx_ret_pct_lag1",
    ],
    "macro": [
        "usd_ret_pct_lag1",
        "usd_mom_4w_lag1",
        "yield_10y_chg_lag1",
        "spx_ret_pct_lag1",
        "realized_vol_lag1",
    ],
}


def _safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _robust_stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        med = float(np.nanmedian(s)) if s.notna().any() else 0.0
        q1 = float(np.nanpercentile(s, 25)) if s.notna().any() else 0.0
        q3 = float(np.nanpercentile(s, 75)) if s.notna().any() else 1.0
        iqr = max(q3 - q1, 1e-6)
        stats[c] = (med, iqr)
    return stats


def _z(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]], col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df))
    med, iqr = stats[col]
    vals = pd.to_numeric(df[col], errors="coerce").fillna(med).values.astype(float)
    return (vals - med) / iqr


def _softmax(x: np.ndarray) -> np.ndarray:
    x2 = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x2)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.clip(s, 1e-12, None)


@dataclass
class RuleRegimeModel:
    feature_stats: Dict[str, Tuple[float, float]]
    available_map: Dict[str, List[str]]
    normal_band: float = 0.50

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        score = np.zeros((n, len(REGIME_NAMES)), dtype=float)

        # 0) supply_shock
        for c in self.available_map["supply_shock"]:
            sign = 1.0
            if c == "opec_sentiment_lag1":
                sign = 0.4
            score[:, 0] += sign * _z(df, self.feature_stats, c)

        # 1) oversupply
        for c in self.available_map["oversupply"]:
            sign = 1.0
            if c in {"contango_1_3m_lag1", "contango_1_6m_lag1"}:
                sign = 1.3
            score[:, 1] += sign * _z(df, self.feature_stats, c)

        # 2) macro_risk_off
        for c in self.available_map["macro_risk_off"]:
            sign = 1.0
            if c == "spx_ret_pct_lag1":
                sign = -1.0
            score[:, 2] += sign * _z(df, self.feature_stats, c)

        # 3) normal_mixed: high when other regime signals are muted.
        score[:, 3] = -(
            np.abs(score[:, 0]) + np.abs(score[:, 1]) + np.abs(score[:, 2])
        ) * self.normal_band

        return pd.DataFrame(score, columns=[f"regime_score_{r}" for r in REGIME_NAMES], index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.score(df).values
        p = _softmax(s)
        return pd.DataFrame(
            p,
            columns=[f"regime_prob_{r}" for r in REGIME_NAMES],
            index=df.index,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        p = self.predict_proba(df).values
        idx = np.argmax(p, axis=1)
        return pd.Series([REGIME_NAMES[i] for i in idx], index=df.index, name="regime")


def fit_regime_model(
    df_train: pd.DataFrame,
    regime_feature_map: Optional[Dict[str, List[str]]] = None,
    normal_band: float = 0.50,
) -> RuleRegimeModel:
    """
    Fit a walk-forward-safe rule-based regime model on train window only.
    """
    fmap = regime_feature_map or DEFAULT_REGIME_FEATURE_MAP
    all_cols = sorted({c for cols in fmap.values() for c in cols})
    avail = _safe_cols(df_train, all_cols)
    stats = _robust_stats(df_train, avail)

    available_map = {
        r: _safe_cols(df_train, fmap.get(r, []))
        for r in ["supply_shock", "oversupply", "macro_risk_off"]
    }
    return RuleRegimeModel(
        feature_stats=stats,
        available_map=available_map,
        normal_band=normal_band,
    )


def fit_hmm_regime_model(df_train: pd.DataFrame):
    """
    Advanced option: fit HMM regime detector (from oil_regime.py) on train window.
    """
    from src.features.oil_regime import OilRegimeDetector

    det = OilRegimeDetector(n_states=4, n_iter=200, random_state=42)
    det.fit(df_train)
    return det


def infer_hmm_regime(detector, df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced option: infer HMM regime labels/probabilities.
    """
    label = pd.Series(detector.predict(df), index=df.index, name="regime")
    proba = detector.predict_proba(df)
    out = pd.concat([proba, label], axis=1)
    return out


def infer_regime(
    model: RuleRegimeModel,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return regime scores, probabilities, and hard label.
    """
    scores = model.score(df)
    probs = model.predict_proba(df)
    label = model.predict(df)
    out = pd.concat([scores, probs, label], axis=1)
    return out


def add_regime_features(
    df: pd.DataFrame,
    regime_info: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    for c in regime_info.columns:
        out[c] = regime_info[c]
    return out


def apply_regime_sentiment_adjustments(
    df: pd.DataFrame,
    regime_col: str = "regime",
    csdi_suffix: str = "_cs_di_lag1",
    base_sent_suffix: str = "_sentiment_lag1",
    reset_scale: float = 0.35,
    prev_regime: Optional[str] = None,
    multipliers: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Make sentiment persistence regime-dependent and reduce stale carry-over
    at regime switches.
    """
    out = df.copy()
    if regime_col not in out.columns:
        return out

    regime = out[regime_col].astype(str)
    switched_raw = regime.ne(regime.shift(1))
    if prev_regime is not None and len(switched_raw) > 0:
        switched_raw.iloc[0] = regime.iloc[0] != prev_regime
    switched = switched_raw.fillna(False).astype(float)
    out["regime_switched"] = switched

    multipliers = multipliers or DEFAULT_SENTIMENT_MULTIPLIERS
    mult = regime.map(multipliers).fillna(1.0).astype(float)
    out["regime_sent_mult"] = mult

    csdi_cols = [c for c in out.columns if c.endswith(csdi_suffix)]
    sent_cols = [c for c in out.columns if c.endswith(base_sent_suffix)]

    for c in csdi_cols:
        base = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        adjusted = base * mult * (1.0 - switched * reset_scale)
        out[f"{c}_regime_adj"] = adjusted

    for c in sent_cols:
        base = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[f"{c}_regime_adj"] = base * mult

    return out


def _build_regressor(
    random_state: int = 42,
    params: Optional[Dict[str, object]] = None,
) -> RegressorMixin:
    params = params or {}
    return HistGradientBoostingRegressor(
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 4)),
        max_iter=int(params.get("max_iter", 250)),
        min_samples_leaf=int(params.get("min_samples_leaf", 15)),
        l2_regularization=float(params.get("l2_regularization", 0.1)),
        random_state=random_state,
    )


def fit_regime_specific_models(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "close_pct_change",
    regime_col: str = "regime",
    min_obs_per_regime: int = 40,
    random_state: int = 42,
    global_params: Optional[Dict[str, object]] = None,
    expert_params_by_regime: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Fit one global model + one expert model per regime when sample size allows.
    """
    x = df_train[feature_cols].fillna(0.0)
    y = pd.to_numeric(df_train[target_col], errors="coerce").fillna(0.0)

    global_model = _build_regressor(random_state=random_state, params=global_params)
    global_model.fit(x, y)

    experts: Dict[str, Optional[RegressorMixin]] = {}
    if regime_col in df_train.columns:
        for r in REGIME_NAMES:
            subset = df_train[df_train[regime_col] == r]
            if len(subset) < min_obs_per_regime:
                experts[r] = None
                continue
            r_params = (expert_params_by_regime or {}).get(r)
            m = _build_regressor(random_state=random_state, params=r_params)
            m.fit(subset[feature_cols].fillna(0.0), subset[target_col].fillna(0.0))
            experts[r] = m
    else:
        experts = {r: None for r in REGIME_NAMES}

    return {"global": global_model, "experts": experts, "feature_cols": feature_cols}


def _compose_model_features(
    base_feature_cols: List[str],
    df_ref: pd.DataFrame,
) -> List[str]:
    extra = [
        c for c in df_ref.columns
        if c.endswith("_regime_adj")
        or c.startswith("regime_prob_")
        or c in {"regime_sent_mult", "regime_switched"}
    ]
    cols = []
    for c in base_feature_cols + extra:
        if c in df_ref.columns and np.issubdtype(df_ref[c].dtype, np.number):
            cols.append(c)
    return sorted(set(cols))


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def _inner_time_splits(
    n_rows: int,
    n_splits: int = 3,
    val_size: int = 26,
    min_train_size: int = 104,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if n_rows < (min_train_size + val_size):
        return splits

    start_points = np.linspace(min_train_size, n_rows - val_size, num=n_splits, dtype=int)
    seen = set()
    for end_train in start_points:
        end_train = int(end_train)
        if end_train in seen:
            continue
        seen.add(end_train)
        tr_idx = np.arange(0, end_train)
        va_idx = np.arange(end_train, min(end_train + val_size, n_rows))
        if len(va_idx) == 0:
            continue
        splits.append((tr_idx, va_idx))
    return splits


def _evaluate_inner_candidate(
    train_df: pd.DataFrame,
    base_feature_cols: List[str],
    target_col: str,
    min_obs_per_regime: int,
    random_state: int,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    reset_scale: float,
    multipliers: Dict[str, float],
    global_params: Optional[Dict[str, object]],
    expert_params_by_regime: Optional[Dict[str, Dict[str, object]]],
    score_regime: Optional[str] = None,
) -> float:
    scores = []
    for tr_idx, va_idx in inner_splits:
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()

        reg_model = fit_regime_model(tr)
        tr_reg = infer_regime(reg_model, tr)
        va_reg = infer_regime(reg_model, va)
        tr2 = add_regime_features(tr, tr_reg)
        va2 = add_regime_features(va, va_reg)

        tr2 = apply_regime_sentiment_adjustments(
            tr2, reset_scale=reset_scale, multipliers=multipliers
        )
        prev_reg = str(tr2["regime"].iloc[-1]) if not tr2.empty else None
        va2 = apply_regime_sentiment_adjustments(
            va2,
            reset_scale=reset_scale,
            prev_regime=prev_reg,
            multipliers=multipliers,
        )

        feat_cols = _compose_model_features(base_feature_cols, tr2)
        models = fit_regime_specific_models(
            tr2,
            feature_cols=feat_cols,
            target_col=target_col,
            regime_col="regime",
            min_obs_per_regime=min_obs_per_regime,
            random_state=random_state,
            global_params=global_params,
            expert_params_by_regime=expert_params_by_regime,
        )
        pred = predict_with_regime_logic(models, va2)

        if score_regime is None:
            y_true = va2[target_col].values.astype(float)
            scores.append(_directional_accuracy(y_true, pred))
        else:
            mask = (va2["regime"] == score_regime).values
            if mask.sum() == 0:
                continue
            y_true = va2.loc[mask, target_col].values.astype(float)
            y_pred = pred[mask]
            scores.append(_directional_accuracy(y_true, y_pred))

    if not scores:
        return -np.inf
    return float(np.mean(scores))


def tune_regime_hyperparameters(
    train_df: pd.DataFrame,
    base_feature_cols: List[str],
    target_col: str = "close_pct_change",
    min_obs_per_regime: int = 40,
    random_state: int = 42,
    inner_splits: int = 3,
    inner_val_size: int = 26,
    reset_scale_grid: Optional[List[float]] = None,
    multiplier_grid: Optional[List[float]] = None,
    model_param_grid: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Nested-CV tuning on the current walk-forward train window.
    Tunes:
      1) sentiment reset_scale (overall directional accuracy),
      2) per-regime multipliers (regime-specific directional accuracy),
      3) global params + per-regime expert params.
    """
    reset_scale_grid = reset_scale_grid or [0.20, 0.35, 0.50]
    multiplier_grid = multiplier_grid or [0.70, 0.85, 1.00, 1.15]
    model_param_grid = model_param_grid or DEFAULT_MODEL_PARAM_GRID
    baseline_params = model_param_grid[1] if len(model_param_grid) > 1 else model_param_grid[0]

    splits = _inner_time_splits(
        len(train_df),
        n_splits=inner_splits,
        val_size=inner_val_size,
        min_train_size=max(78, len(train_df) // 2),
    )
    if not splits:
        return {
            "reset_scale": 0.35,
            "multipliers": DEFAULT_SENTIMENT_MULTIPLIERS.copy(),
            "global_params": baseline_params,
            "expert_params_by_regime": {r: baseline_params for r in REGIME_NAMES},
        }

    # 1) tune reset scale
    best_reset, best_score = 0.35, -np.inf
    for rs in reset_scale_grid:
        sc = _evaluate_inner_candidate(
            train_df,
            base_feature_cols,
            target_col,
            min_obs_per_regime,
            random_state,
            splits,
                reset_scale=rs,
                multipliers=DEFAULT_SENTIMENT_MULTIPLIERS,
                global_params=baseline_params,
                expert_params_by_regime=None,
                score_regime=None,
            )
        if sc > best_score:
            best_score, best_reset = sc, rs

    # 2) tune per-regime multipliers
    multipliers = DEFAULT_SENTIMENT_MULTIPLIERS.copy()
    for r in REGIME_NAMES:
        best_m, best_m_score = multipliers[r], -np.inf
        for m in multiplier_grid:
            trial = multipliers.copy()
            trial[r] = float(m)
            sc = _evaluate_inner_candidate(
                train_df,
                base_feature_cols,
                target_col,
                min_obs_per_regime,
                random_state,
                splits,
                reset_scale=best_reset,
                multipliers=trial,
                global_params=baseline_params,
                expert_params_by_regime=None,
                score_regime=r,
            )
            if sc > best_m_score:
                best_m_score, best_m = sc, float(m)
        multipliers[r] = best_m

    # 3) tune global params on overall score
    best_global = baseline_params
    best_global_score = -np.inf
    for p in model_param_grid:
        sc = _evaluate_inner_candidate(
            train_df,
            base_feature_cols,
            target_col,
            min_obs_per_regime,
            random_state,
            splits,
            reset_scale=best_reset,
            multipliers=multipliers,
            global_params=p,
            expert_params_by_regime=None,
            score_regime=None,
        )
        if sc > best_global_score:
            best_global_score, best_global = sc, p

    # 4) tune per-regime expert params
    expert_params_by_regime: Dict[str, Dict[str, object]] = {}
    for r in REGIME_NAMES:
        best_p = best_global
        best_sc = -np.inf
        for p in model_param_grid:
            sc = _evaluate_inner_candidate(
                train_df,
                base_feature_cols,
                target_col,
                min_obs_per_regime,
                random_state,
                splits,
                reset_scale=best_reset,
                multipliers=multipliers,
                global_params=best_global,
                expert_params_by_regime={r: p},
                score_regime=r,
            )
            if sc > best_sc:
                best_sc, best_p = sc, p
        expert_params_by_regime[r] = dict(best_p)

    return {
        "reset_scale": float(best_reset),
        "multipliers": multipliers,
        "global_params": dict(best_global),
        "expert_params_by_regime": expert_params_by_regime,
    }


def predict_with_regime_logic(
    models: Dict[str, object],
    df_test: pd.DataFrame,
    regime_prob_cols: Optional[List[str]] = None,
    global_weight: float = 0.20,
) -> np.ndarray:
    """
    Mixture-of-experts prediction:
    yhat = w_global * y_global + (1-w_global) * Σ_r p(r|x) * y_r
    """
    feature_cols = models["feature_cols"]  # type: ignore[index]
    x = df_test[feature_cols].fillna(0.0)

    y_global = models["global"].predict(x)  # type: ignore[index]
    experts: Dict[str, Optional[RegressorMixin]] = models["experts"]  # type: ignore[index]

    if regime_prob_cols is None:
        regime_prob_cols = [f"regime_prob_{r}" for r in REGIME_NAMES if f"regime_prob_{r}" in df_test.columns]

    if not regime_prob_cols:
        return y_global

    probs = df_test[regime_prob_cols].fillna(0.0).values
    probs_ord = np.zeros((len(df_test), len(REGIME_NAMES)))
    for j, r in enumerate(REGIME_NAMES):
        c = f"regime_prob_{r}"
        if c in regime_prob_cols:
            probs_ord[:, j] = df_test[c].fillna(0.0).values
    row_sum = probs_ord.sum(axis=1, keepdims=True)
    probs_ord = np.divide(probs_ord, np.clip(row_sum, 1e-12, None))

    expert_pred = np.zeros(len(df_test))
    for j, r in enumerate(REGIME_NAMES):
        m = experts.get(r)
        if m is None:
            yr = y_global
        else:
            yr = m.predict(x)
        expert_pred += probs_ord[:, j] * yr

    return global_weight * y_global + (1.0 - global_weight) * expert_pred


def walk_forward_regime_backtest(
    df: pd.DataFrame,
    target_col: str = "close_pct_change",
    date_col: str = "date",
    feature_cols: Optional[List[str]] = None,
    min_train_size: int = 156,  # ~3 years weekly
    step: int = 1,
    min_obs_per_regime: int = 40,
    global_weight: float = 0.20,
    random_state: int = 42,
    nested_tuning: bool = True,
    inner_splits: int = 3,
    inner_val_size: int = 26,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest with regime-aware experts.
    Uses only information available up to t-1 for each prediction at t.
    """
    data = df.copy().reset_index(drop=True)
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    if feature_cols is None:
        excluded = {date_col, target_col, "close"}
        feature_cols = [
            c for c in data.columns
            if c not in excluded and np.issubdtype(data[c].dtype, np.number)
        ]

    rows = []
    for i in range(min_train_size, len(data), step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:i + step].copy()
        if test.empty:
            continue

        # 1) fit regime model on train only
        reg_model = fit_regime_model(train)
        train_reg = infer_regime(reg_model, train)
        test_reg = infer_regime(reg_model, test)

        train2 = add_regime_features(train, train_reg)
        test2 = add_regime_features(test, test_reg)

        # 2) nested tuning on train window only (optional)
        if nested_tuning:
            tuned = tune_regime_hyperparameters(
                train,
                base_feature_cols=feature_cols,
                target_col=target_col,
                min_obs_per_regime=min_obs_per_regime,
                random_state=random_state,
                inner_splits=inner_splits,
                inner_val_size=inner_val_size,
            )
        else:
            tuned = {
                "reset_scale": 0.35,
                "multipliers": DEFAULT_SENTIMENT_MULTIPLIERS.copy(),
                "global_params": DEFAULT_MODEL_PARAM_GRID[1],
                "expert_params_by_regime": {r: DEFAULT_MODEL_PARAM_GRID[1] for r in REGIME_NAMES},
            }

        # 3) regime-dependent sentiment transforms (train/test use inferred regime)
        train2 = apply_regime_sentiment_adjustments(
            train2,
            reset_scale=float(tuned["reset_scale"]),
            multipliers=tuned["multipliers"],  # type: ignore[arg-type]
        )
        prev_reg = str(train2["regime"].iloc[-1]) if not train2.empty else None
        test2 = apply_regime_sentiment_adjustments(
            test2,
            prev_regime=prev_reg,
            reset_scale=float(tuned["reset_scale"]),
            multipliers=tuned["multipliers"],  # type: ignore[arg-type]
        )

        model_features = _compose_model_features(feature_cols, train2)

        models = fit_regime_specific_models(
            train2,
            feature_cols=model_features,
            target_col=target_col,
            regime_col="regime",
            min_obs_per_regime=min_obs_per_regime,
            random_state=random_state,
            global_params=tuned["global_params"],  # type: ignore[arg-type]
            expert_params_by_regime=tuned["expert_params_by_regime"],  # type: ignore[arg-type]
        )
        yhat = predict_with_regime_logic(models, test2, global_weight=global_weight)

        for j in range(len(test2)):
            actual = float(test2.iloc[j][target_col])
            pred = float(yhat[j])
            rows.append(
                {
                    "date": test2.iloc[j][date_col],
                    "y_true": actual,
                    "y_pred": pred,
                    "regime": test2.iloc[j]["regime"],
                    "pred_dir": int(pred > 0),
                    "true_dir": int(actual > 0),
                    "strategy_ret": np.sign(pred) * actual,
                    "bh_ret": actual,
                }
            )

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["hit"] = (res["pred_dir"] == res["true_dir"]).astype(int)
    res["cum_strategy_ret"] = (1 + res["strategy_ret"] / 100.0).cumprod() - 1
    res["cum_bh_ret"] = (1 + res["bh_ret"] / 100.0).cumprod() - 1
    return res


def performance_by_regime(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame()
    grp = backtest_df.groupby("regime")
    out = grp.agg(
        n=("hit", "size"),
        directional_acc=("hit", "mean"),
        mean_strategy_ret=("strategy_ret", "mean"),
        mean_bh_ret=("bh_ret", "mean"),
    )
    return out.sort_values("directional_acc").round(4)


def overall_metrics(backtest_df: pd.DataFrame) -> Dict[str, float]:
    if backtest_df.empty:
        return {}
    return {
        "directional_accuracy": float(accuracy_score(backtest_df["true_dir"], backtest_df["pred_dir"])),
        "mean_strategy_ret": float(backtest_df["strategy_ret"].mean()),
        "mean_bh_ret": float(backtest_df["bh_ret"].mean()),
        "final_cum_strategy_ret": float(backtest_df["cum_strategy_ret"].iloc[-1]),
        "final_cum_bh_ret": float(backtest_df["cum_bh_ret"].iloc[-1]),
    }


def regime_distribution_by_fold(backtest_df: pd.DataFrame, fold_size: int = 52) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame()
    df = backtest_df.copy().reset_index(drop=True)
    df["fold"] = (np.arange(len(df)) // fold_size) + 1
    dist = (
        df.groupby(["fold", "regime"]).size()
        .rename("count")
        .reset_index()
    )
    total = dist.groupby("fold")["count"].transform("sum")
    dist["pct"] = dist["count"] / total
    return dist


def weak_regime_diagnostics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    stats = performance_by_regime(backtest_df)
    if stats.empty:
        return stats
    stats["below_50pct_acc"] = stats["directional_acc"] < 0.50
    return stats.sort_values(["below_50pct_acc", "directional_acc"])


def plot_inferred_regimes(backtest_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    if backtest_df.empty:
        return None

    d = backtest_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    color_map = {
        "supply_shock": "#d95f02",
        "oversupply": "#1b9e77",
        "macro_risk_off": "#7570b3",
        "normal_mixed": "#66a61e",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d["date"], d["cum_strategy_ret"], label="Strategy", color="#1f77b4")
    ax.plot(d["date"], d["cum_bh_ret"], label="Buy&Hold", color="#444444", alpha=0.8)

    for r, g in d.groupby("regime"):
        ax.scatter(g["date"], g["cum_strategy_ret"], s=12, alpha=0.5, color=color_map.get(r, "#999999"), label=f"Regime: {r}")

    ax.set_title("Walk-Forward Cumulative Return with Inferred Regimes")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
