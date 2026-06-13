#!/usr/bin/env python3
"""
threshold_optimizer.py — Per-token adaptive threshold, ATR & lookahead optimizer
=================================================================================
Generates thresholds INDEPENDENTLY from the production models:

  1. Loads 2 000 bars of OHLCV + features (same pipeline as retrain)
  2. Splits data: first 70% → train a fresh local XGBoost (200 rounds, no Optuna)
                  last  30% → out-of-sample evaluation (never seen by local model)
  3. Predicts directional probabilities on the eval window
  4. ATR multiplier sweep   → picks best SL distance (eval window)
  5. Lookahead window sweep → picks best trade-duration trade-off (eval window)
  6. Classifies eval bars into 9 regime buckets (3 vol x 3 vola)
  7. Per-regime + global: grid-search the best BUY / SELL thresholds on
     DIRECTIONAL PROBABILITY (p_buy for BUY proposals, p_sell for SELL)
  8. Saves data/token_params/<BASE>_params.json

Thresholds are on p_buy / p_sell (0-1 softmax output of the primary model).
predictor.py applies them directly against the production primary model's
class probabilities as an additional filter on top of the meta gate.

Consumed by:
  - scripts/retrain_model.py  — ATR multiplier + lookahead
  - src/ml/predictor.py       — regime-specific directional-prob thresholds

Usage
-----
    python scripts/threshold_optimizer.py                  # all fleet symbols
    python scripts/threshold_optimizer.py BTC/USDT ETH/USDT
    python scripts/threshold_optimizer.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ml.feature_engine import prepare_features, compute_atr
from src.ml.predictor import Predictor
from scripts.retrain_model import (
    FLEET_SYMBOLS,
    FEE_ROUNDTRIP,
    AdaptiveNormalizer,
    compute_soft_confluence_features,
    create_triple_barrier_labels,
    fetch_futures_data,
    fetch_fear_greed,
    get_atr_multiplier,
    sample_weights,
    MAX_LOOKAHEAD,
    CENSORED,
)

# ── output ────────────────────────────────────────────────────────────────────
PARAMS_DIR  = _ROOT / "data" / "token_params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_STORE = _ROOT / "src" / "ml" / "model_store"

# ── data ──────────────────────────────────────────────────────────────────────
HISTORY_HOURS = 4000
MIN_BARS      = 800     # need enough for a decent train/eval split

# ── independent local model ───────────────────────────────────────────────────
TRAIN_FRAC   = 0.70    # first 70% → fit local model
LOCAL_ROUNDS = 300     # quick fit; no Optuna

# ── regime grid  (3 × 3 × 3 = 27 buckets) ───────────────────────────────────
# Three axes capture the three most independent market dimensions:
#   vol  : liquidity / participation (rolling 24h volume)
#   vola : noise / bar size (ATR as % of price)
#   trend: directional momentum (24h price return vs its own distribution)
VOL_TIERS   = ["low", "med", "high"]   # volume tiers
VOLA_TIERS  = ["low", "med", "high"]   # volatility tiers
TREND_TIERS = ["down", "flat", "up"]   # momentum / trend tiers
REGIME_KEYS = [
    f"{v}_{va}_{tr}"
    for v  in VOL_TIERS
    for va in VOLA_TIERS
    for tr in TREND_TIERS
]  # 27 keys

# ── search grids ──────────────────────────────────────────────────────────────
# Thresholds are on p_buy / p_sell (softmax output).
# In a 3-class model the winning class typically sits in 0.35-0.70.
THRESHOLD_GRID = np.round(np.arange(0.30, 0.90, 0.02), 3).tolist()   # 30 pts
MARGIN_GRID    = np.round(np.arange(0.00, 0.40, 0.05), 3).tolist()   # 8 pts
MAX_HOLD_GRID  = [0.40, 0.45, 0.50, 0.60, 1.00]                      # 5 pts
ATR_MULT_GRID  = np.round(np.arange(0.75, 4.25, 0.25), 2).tolist()   # 14 pts
LOOKAHEAD_GRID = [12, 18, 24, 30, 36, 48, 60, 72]                     #  8 pts

# ── quality gates ─────────────────────────────────────────────────────────────
# 600 eval bars / 27 buckets ≈ 22 avg — keep floor low so rare combos aren't
# all skipped; noisy-but-present is better than falling back to the global.
MIN_REGIME_BARS = 10
MIN_SIGNALS     = 5
# Minimum viable precision: must exceed 55% to enforce higher quality trades.
BREAKEVEN       = 0.55                   # Hard floor for score
TARGET_PREC     = 0.60                   # "green" bar (local model is weaker than production)


# ─────────────────────────────────────────────────────────────────────────────
# Independent local model
# ─────────────────────────────────────────────────────────────────────────────

def _fit_local_model(
    feat_df: pd.DataFrame,
    labels:  np.ndarray,    # full barrier labels (including CENSORED)
    train_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a quick XGBoost on feat_df[:train_n] (CENSORED rows excluded).
    Predict on feat_df[train_n:] (the out-of-sample eval window).

    Returns (proposed, dir_conf) for the eval window:
      proposed  — 2=BUY or 0=SELL based on argmax of p_buy vs p_sell
      dir_conf  — p_buy when BUY proposed, p_sell when SELL proposed

    Thresholds found against dir_conf are directly comparable to the
    production primary model's p_buy / p_sell at inference time.
    """
    y_tr_raw = labels[:train_n]
    valid_tr = y_tr_raw != CENSORED
    if int(valid_tr.sum()) < 200:
        raise ValueError(f"Only {int(valid_tr.sum())} valid training bars.")

    cols = list(feat_df.columns)
    X_tr = feat_df.iloc[:train_n][valid_tr].values
    y_tr = y_tr_raw[valid_tr].astype(int)

    # Inverse-frequency class weights so HOLD majority doesn't swamp BUY/SELL
    w_tr = sample_weights(y_tr)
    n_valid_total = int(valid_tr.sum())

    # Adaptive hyperparameters: scale regularization to training set size.
    # Prior ultra-conservative values (max_depth=3, subsample=0.5, min_child_weight=10,
    # reg_lambda=2.0, gamma=1.0) produced near-uniform probabilities (~0.37 mean)
    # because the heavy regularization prevented any splits with only ~1400 valid bars.
    _max_depth   = 5 if n_valid_total > 800 else (4 if n_valid_total > 400 else 3)
    _min_child_w = max(3, min(8, n_valid_total // 100))
    _subsample   = 0.85 if n_valid_total > 500 else 0.90
    _colsample   = 0.75 if n_valid_total > 500 else 0.85

    params: Dict[str, Any] = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "max_depth":        _max_depth,
        "learning_rate":    0.05,
        "subsample":        _subsample,
        "colsample_bytree": _colsample,
        "min_child_weight": _min_child_w,
        "reg_lambda":       1.0,
        "gamma":            0.3,
        "seed":             42,
        "tree_method":      "hist",
        "missing":          np.nan,
        "verbosity":        0,
    }

    # Early stopping on last 20% of valid training rows to prevent overfitting.
    es_cut = max(20, int(len(X_tr) * 0.20))
    X_tr_es, X_val_es = X_tr[:-es_cut], X_tr[-es_cut:]
    y_tr_es, y_val_es = y_tr[:-es_cut], y_tr[-es_cut:]
    w_tr_es = w_tr[:-es_cut]

    dm_tr_es  = xgb.DMatrix(X_tr_es,  label=y_tr_es, weight=w_tr_es, feature_names=cols)
    dm_val_es = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=cols)
    try:
        model = xgb.train(
            params, dm_tr_es, num_boost_round=LOCAL_ROUNDS,
            evals=[(dm_val_es, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
    except Exception:
        dm_tr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=cols)
        model = xgb.train(params, dm_tr, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)

    X_ev  = feat_df.iloc[train_n:].values
    dm_ev = xgb.DMatrix(X_ev, feature_names=cols)
    probs = model.predict(dm_ev)   # (n_eval, 3): [p_sell, p_hold, p_buy]

    proposed = np.where(probs[:, 2] >= probs[:, 0], 2, 0).astype(int)
    dir_conf = np.where(proposed == 2, probs[:, 2], probs[:, 0]).astype(float)

    return proposed, dir_conf, probs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tier(val: float, p33: float, p67: float) -> str:
    if val <= p33:
        return "low"
    if val <= p67:
        return "med"
    return "high"


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────────────

def _trend_tier(val: float, p33: float, p67: float) -> str:
    """Map a momentum value to down / flat / up."""
    if val <= p33:
        return "down"
    if val <= p67:
        return "flat"
    return "up"


def classify_regimes(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Classify all bars into '<vol>_<vola>_<trend>' regime strings (27 buckets).

    Three axes:
      vol   — rolling 24-bar mean volume (liquidity / participation)
      vola  — ATR / close (bar-level noise)
      trend — 24-bar price return (directional momentum)

    Boundaries are computed from ALL passed data so percentiles are stable.
    Returns (series, boundaries_dict); boundaries saved to JSON for inference.
    """
    roll_vol = df["volume"].rolling(24, min_periods=1).mean()
    atr_vals = compute_atr(df, period=14)
    atr_pct  = (atr_vals / df["close"].replace(0, np.nan)).fillna(0)
    momentum = df["close"].pct_change(24).fillna(0)   # 24h return

    vp33 = float(roll_vol.quantile(0.33));  vp67 = float(roll_vol.quantile(0.67))
    ap33 = float(atr_pct.quantile(0.33));   ap67 = float(atr_pct.quantile(0.67))
    mp33 = float(momentum.quantile(0.33));  mp67 = float(momentum.quantile(0.67))

    boundaries: Dict[str, float] = {
        "vol_p33": vp33,  "vol_p67": vp67,
        "atr_pct_p33": ap33, "atr_pct_p67": ap67,
        "momentum_p33": mp33, "momentum_p67": mp67,
    }
    reg = pd.Series(
        [f"{_tier(float(roll_vol.iloc[i]), vp33, vp67)}"
         f"_{_tier(float(atr_pct.iloc[i]), ap33, ap67)}"
         f"_{_trend_tier(float(momentum.iloc[i]), mp33, mp67)}"
         for i in range(len(df))],
        index=df.index, dtype=str,
    )
    return reg, boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Threshold grid-search
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(
    probs:    np.ndarray,   # (N, 3): p_sell, p_hold, p_buy
    proposed: np.ndarray,
    labels:   np.ndarray,
    side:     int,          # 2=BUY, 0=SELL
    n_total:  int,
) -> Dict[str, Any]:
    """
    Sweep THRESHOLD_GRID, MARGIN_GRID and MAX_HOLD_GRID for one direction.

    Score = coverage × (2 × precision − 1 − FEE_ROUNDTRIP)
      — proportional to total expected PnL (1:1 R:R, minus fees).
      — Negative when precision ≤ 50%, so losing thresholds are rejected.

    When NO threshold clears the 50% floor (direction unprofitable in this
    regime/window), we return the highest-precision threshold as a reference
    so the user can see the best achievable result (ok=False, no live signals).
    """
    sm = proposed == side
    if int(sm.sum()) < MIN_SIGNALS:
        return {"threshold": 0.50, "margin": 0.0, "max_hold": 1.0, "precision": 0.0,
                "coverage": 0.0, "n_signals": 0, "ok": False}

    probs_s = probs[sm]
    lbl_s   = labels[sm]
    correct = side

    if side == 2:
        p_dir = probs_s[:, 2]
        p_opp = probs_s[:, 0]
    else:
        p_dir = probs_s[:, 0]
        p_opp = probs_s[:, 2]
        
    p_hold = probs_s[:, 1]
    margin = p_dir - p_opp

    all_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score = -np.inf

    for thr in THRESHOLD_GRID:
        for m in MARGIN_GRID:
            for h in MAX_HOLD_GRID:
                fired = (p_dir >= thr) & (margin >= m) & (p_hold <= h)
                n = int(fired.sum())
                if n < MIN_SIGNALS:
                    continue
                prec = float((lbl_s[fired] == correct).mean())
                cov  = n / max(n_total, 1)
                # We want to force precision higher (60-70%).
                # Using sqrt(cov) reduces the reward for pure volume, pushing the optimizer
                # to pick higher precision thresholds.
                score = (cov ** 0.5) * (prec - BREAKEVEN)
                
                row = {"threshold": float(thr), "margin": float(m), "max_hold": float(h), 
                       "precision": round(prec, 4), "coverage": round(cov, 4), 
                       "n_signals": n, "ok": prec >= TARGET_PREC, "_score": score}
                all_rows.append(row)
                if prec >= BREAKEVEN and score > best_score:
                    best_score = score
                    best = row

    if best is None:
        # No threshold cleared the 50% floor — direction is unprofitable here.
        # Return the highest-precision threshold as a reference (ok=False).
        if all_rows:
            ref = max(all_rows, key=lambda r: r["precision"])
            ref["ok"] = False
            return {k: v for k, v in ref.items() if k != "_score"}
        return {"threshold": 0.50, "margin": 0.0, "max_hold": 1.0, "precision": 0.0,
                "coverage": 0.0, "n_signals": 0, "ok": False}

    return {k: v for k, v in best.items() if k != "_score"}


# ─────────────────────────────────────────────────────────────────────────────
# ATR multiplier optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimize_atr(
    df:       pd.DataFrame,   # eval-window OHLCV + regime columns
    dir_conf: np.ndarray,     # local model directional confidence
    proposed: np.ndarray,
    base_mult: float,
) -> Dict[str, Any]:
    """Re-label eval window at each ATR candidate; score at top-10% dir_conf."""
    vr = df["volatility_regime"]   if "volatility_regime"   in df.columns else None
    er = df["efficiency_ratio_10"] if "efficiency_ratio_10" in df.columns else None
    tr = df["trend_regime"]        if "trend_regime"        in df.columns else None
    cs = df.get("macro_confluence_score")

    best_mult  = base_mult
    best_score = -np.inf
    scored: List[Dict[str, Any]] = []

    for mult in ATR_MULT_GRID:
        labs  = np.asarray(create_triple_barrier_labels(
            df, atr_multiplier=mult,
            volatility_regime=vr, efficiency_ratio=er,
            trend_regime=tr, macro_confluence_score=cs,
        ))
        valid = labs != CENSORED
        n = int(valid.sum())
        if n < MIN_REGIME_BARS:
            continue

        lbl  = np.asarray(labs[valid])
        conf = np.asarray(dir_conf[valid])
        prop = np.asarray(proposed[valid])

        parts: List[float] = []
        for side, correct in [(2, 2), (0, 0)]:
            sm = prop == side
            if int(sm.sum()) < MIN_SIGNALS:
                continue
            thr90 = float(np.quantile(conf[sm], 0.90))
            fired = (conf >= thr90) & sm
            nf    = int(fired.sum())
            if nf < 3:
                continue
            prec = float((lbl[fired] == correct).mean())
            cov  = nf / n
            if prec > BREAKEVEN:
                parts.append(prec * float(np.sqrt(cov)))

        score = float(np.mean(parts)) if parts else 0.0
        scored.append({"mult": round(float(mult), 2), "score": round(score, 5)})
        # Prefer larger multiplier on ties (wider barriers → cleaner labels).
        is_better  = score > best_score + 1e-6
        is_tied    = abs(score - best_score) <= 1e-6 and float(mult) > best_mult
        if is_better or is_tied:
            best_score = score
            best_mult  = float(mult)

    return {
        "atr_multiplier": round(best_mult, 2),
        "atr_score":      round(best_score, 5),
        "atr_top5":       sorted(scored, key=lambda x: -x["score"])[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimize_lookahead(
    df:       pd.DataFrame,
    dir_conf: np.ndarray,
    proposed: np.ndarray,
    atr_mult: float,
) -> Dict[str, Any]:
    """Sweep lookahead windows; 5% decay per step biases toward shorter windows."""
    vr = df["volatility_regime"]   if "volatility_regime"   in df.columns else None
    er = df["efficiency_ratio_10"] if "efficiency_ratio_10" in df.columns else None
    tr = df["trend_regime"]        if "trend_regime"        in df.columns else None
    cs = df.get("macro_confluence_score")

    best_lh    = MAX_LOOKAHEAD
    best_score = -np.inf
    scored: List[Dict[str, Any]] = []

    for lh in LOOKAHEAD_GRID:
        labs  = np.asarray(create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=lh,
            volatility_regime=vr, efficiency_ratio=er,
            trend_regime=tr, macro_confluence_score=cs,
        ))
        valid = labs != CENSORED
        n = int(valid.sum())
        if n < MIN_REGIME_BARS:
            continue

        lbl  = np.asarray(labs[valid])
        conf = np.asarray(dir_conf[valid])
        prop = np.asarray(proposed[valid])

        parts: List[float] = []
        for side, correct in [(2, 2), (0, 0)]:
            sm = prop == side
            if int(sm.sum()) < MIN_SIGNALS:
                continue
            thr90 = float(np.quantile(conf[sm], 0.90))
            fired = (conf >= thr90) & sm
            nf    = int(fired.sum())
            if nf < 3:
                continue
            prec = float((lbl[fired] == correct).mean())
            cov  = nf / n
            if prec > BREAKEVEN:
                parts.append(prec * float(np.sqrt(cov)))

        if not parts:
            continue

        decay = 1.0 - (LOOKAHEAD_GRID.index(lh) / len(LOOKAHEAD_GRID)) * 0.05
        score = float(np.mean(parts)) * decay
        scored.append({"lookahead": lh, "score": round(score, 5)})
        if score > best_score:
            best_score = score
            best_lh    = lh

    return {
        "lookahead_bars":  int(best_lh),
        "lookahead_score": round(best_score, 5),
        "lookahead_top5":  sorted(scored, key=lambda x: -x["score"])[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol entry point
# ─────────────────────────────────────────────────────────────────────────────

def optimize_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Full independent optimisation pipeline for one symbol."""
    base = symbol.replace("/", "_")

    if not (MODEL_STORE / f"{base}_model.json").exists():
        print(f"   [{symbol}] No trained model — skipping.")
        return None

    # Predictor used ONLY for data fetching + reading feature_cols / base ATR.
    # p.model and p.meta_model are NOT used for threshold search.
    try:
        p = Predictor(symbol)
    except Exception as exc:
        print(f"   [{symbol}] Predictor init failed: {exc}")
        return None

    feature_cols: Optional[List[str]] = p.meta.get("feature_cols")
    base_atr_mult = float(p.meta.get("atr_multiplier", get_atr_multiplier(symbol)))

    # ── fetch data ────────────────────────────────────────────────────────────
    try:
        df_1h = p.fetch_live_data(timeframe="1h", limit=HISTORY_HOURS)
        if df_1h is None or len(df_1h) < MIN_BARS:
            print(f"   [{symbol}] Not enough data — skipping.")
            return None
        btc_df  = p.fetch_btc_data(timeframe="1h", limit=HISTORY_HOURS)
        news_df = p.load_news_data()
        df_1d   = p.fetch_live_data(timeframe="1d", limit=300)
        fund_df, oi_df = fetch_futures_data(symbol, df_1h)
        fg_df   = fetch_fear_greed(days=700)
    except Exception as exc:
        print(f"   [{symbol}] Data fetch error: {exc}")
        return None

    # ── feature engineering ───────────────────────────────────────────────────
    try:
        features = prepare_features(
            df_1h, btc_df=btc_df, news_df=news_df, df_1d=df_1d,
            funding_df=fund_df, oi_df=oi_df, fg_df=fg_df,
        )
    except Exception as exc:
        print(f"   [{symbol}] Feature engineering failed: {exc}")
        return None

    if features is None or len(features) < MIN_BARS:
        print(f"   [{symbol}] Too few feature rows — skipping.")
        return None

    features = features.reset_index(drop=True)
    n_feat   = len(features)

    # ── soft confluence features (prc_total / macro_confluence_score are top-2 gain
    # features in production but absent without this call — matching MGO behaviour) ──
    try:
        _soft_feats = compute_soft_confluence_features(features)
        for _sc_col in _soft_feats.columns:
            features[_sc_col] = _soft_feats[_sc_col].values
        print(f"   [{symbol}] Soft confluence features added ({len(_soft_feats.columns)}): "
              f"{', '.join(list(_soft_feats.columns)[:5])}...")
    except Exception as _sc_err:
        print(f"   [{symbol}] Soft confluence skip ({_sc_err}) — model will lack prc_total/macro_conf")

    # ── align raw OHLCV to feature rows ──────────────────────────────────────
    df_raw = df_1h.iloc[-n_feat:].reset_index(drop=True).copy()
    for col in ("volatility_regime", "efficiency_ratio_10",
                "trend_regime", "macro_confluence_score"):
        if col in features.columns:
            df_raw[col] = features[col].values

    # ── feature matrix ────────────────────────────────────────────────────────
    if feature_cols:
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            features = pd.concat(
                [features, pd.DataFrame(0.0, index=features.index, columns=missing)],
                axis=1,
            )
        feat_df = features[feature_cols].copy()
    else:
        drop = [c for c in ("timestamp", "target") if c in features.columns]
        feat_df = features.drop(columns=drop, errors="ignore")
        feat_df = feat_df[[c for c in feat_df.columns if not c.startswith("_")]]
        feature_cols = list(feat_df.columns)

    # Drop non-numeric columns (e.g. volatility_regime='HIGH_VOL') before XGBoost
    _str_cols = feat_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if _str_cols:
        feat_df = feat_df.drop(columns=_str_cols)
        feature_cols = list(feat_df.columns)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── train / eval split ────────────────────────────────────────────────────
    train_n = int(n_feat * TRAIN_FRAC)
    eval_n  = n_feat - train_n

    # ── AdaptiveNormalizer: fit on train partition, transform all rows ─────────
    # Eliminates distribution shift between train and eval windows; the local
    # model sees z-scored features consistent with what the production model uses.
    _normalizer = AdaptiveNormalizer(window=1000, drift_threshold=0.2)
    _norm_cols = list(feat_df.select_dtypes(include=[np.number]).columns)
    _normalizer.fit_initial(feat_df.iloc[:train_n], _norm_cols)
    feat_df = _normalizer.transform(feat_df, _norm_cols)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── regime boundaries from ALL data (stable percentile anchors) ───────────
    regimes_all, boundaries = classify_regimes(df_raw)

    # ── barrier labels for all data (base ATR mult) → feeds local model ───────
    vr_all = df_raw["volatility_regime"]   if "volatility_regime"   in df_raw.columns else None
    er_all = df_raw["efficiency_ratio_10"] if "efficiency_ratio_10" in df_raw.columns else None
    tr_all = df_raw["trend_regime"]        if "trend_regime"        in df_raw.columns else None
    cs_all = df_raw.get("macro_confluence_score")

    labels_all = np.asarray(create_triple_barrier_labels(
        df_raw, atr_multiplier=base_atr_mult,
        volatility_regime=vr_all, efficiency_ratio=er_all,
        trend_regime=tr_all, macro_confluence_score=cs_all,
    ))

    # ── fit local model; get OOS directional predictions ─────────────────────
    print(f"   [{symbol}] Fitting local model "
          f"({train_n} train / {eval_n} eval)...", end=" ", flush=True)
    try:
        proposed_ev, dir_conf_ev, probs_ev = _fit_local_model(feat_df, np.asarray(labels_all), train_n)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print("done")

    # ── eval-window OHLCV + regime series ─────────────────────────────────────
    df_ev      = df_raw.iloc[train_n:].reset_index(drop=True).copy()
    regimes_ev = regimes_all.iloc[train_n:].reset_index(drop=True)

    # ── ATR optimisation on eval window ──────────────────────────────────────
    atr_res  = optimize_atr(df_ev, dir_conf_ev, proposed_ev, base_atr_mult)
    best_atr = atr_res["atr_multiplier"]

    # ── lookahead optimisation on eval window ─────────────────────────────────
    lh_res = optimize_lookahead(df_ev, dir_conf_ev, proposed_ev, best_atr)

    # ── recompute labels on eval window with best ATR ─────────────────────────
    vr_ev = df_ev["volatility_regime"]   if "volatility_regime"   in df_ev.columns else None
    er_ev = df_ev["efficiency_ratio_10"] if "efficiency_ratio_10" in df_ev.columns else None
    tr_ev = df_ev["trend_regime"]        if "trend_regime"        in df_ev.columns else None
    cs_ev = df_ev.get("macro_confluence_score")

    labels_ev = np.asarray(create_triple_barrier_labels(
        df_ev, atr_multiplier=best_atr,
        volatility_regime=vr_ev, efficiency_ratio=er_ev,
        trend_regime=tr_ev, macro_confluence_score=cs_ev,
    ))
    valid_ev = labels_ev != CENSORED

    # ── global threshold optimisation ─────────────────────────────────────────
    n_g     = int(valid_ev.sum())
    lbl_g   = np.asarray(labels_ev[valid_ev])
    probs_g = np.asarray(probs_ev[valid_ev])
    prop_g  = np.asarray(proposed_ev[valid_ev])

    g_buy  = best_threshold(probs_g, prop_g, lbl_g, side=2, n_total=n_g)
    g_sell = best_threshold(probs_g, prop_g, lbl_g, side=0, n_total=n_g)

    # ── per-regime threshold optimisation ─────────────────────────────────────
    regime_results: Dict[str, Any] = {}
    for rk in REGIME_KEYS:
        rmask = np.asarray(regimes_ev == rk) & valid_ev
        n_r   = int(rmask.sum())
        if n_r < MIN_REGIME_BARS:
            regime_results[rk] = {
                "n_bars": n_r, "skipped": True,
                "reason": f"too_few_bars (<{MIN_REGIME_BARS})",
            }
            continue

        lbl_r   = np.asarray(labels_ev[rmask])
        probs_r = np.asarray(probs_ev[rmask])
        prop_r  = np.asarray(proposed_ev[rmask])

        r_buy  = best_threshold(probs_r, prop_r, lbl_r, side=2, n_total=n_r)
        r_sell = best_threshold(probs_r, prop_r, lbl_r, side=0, n_total=n_r)

        regime_results[rk] = {
            "n_bars":         n_r,
            "skipped":        False,
            "buy_threshold":  r_buy["threshold"],
            "buy_margin":     r_buy["margin"],
            "buy_max_hold":   r_buy["max_hold"],
            "sell_threshold": r_sell["threshold"],
            "sell_margin":    r_sell["margin"],
            "sell_max_hold":  r_sell["max_hold"],
            "precision_buy":  r_buy["precision"],
            "precision_sell": r_sell["precision"],
            "coverage_buy":   r_buy["coverage"],
            "coverage_sell":  r_sell["coverage"],
            "n_signals_buy":  r_buy["n_signals"],
            "n_signals_sell": r_sell["n_signals"],
            "buy_ok":         r_buy["ok"],
            "sell_ok":        r_sell["ok"],
        }

    # ── assemble + persist ────────────────────────────────────────────────────
    params: Dict[str, Any] = {
        "symbol":          symbol,
        "updated_at":      datetime.now(timezone.utc).isoformat(),
        "n_bars":          n_feat,
        "n_train_bars":    train_n,
        "n_eval_bars":     eval_n,
        "threshold_type":  "directional_prob",   # p_buy / p_sell scale (0-1)
        "global": {
            "atr_multiplier": best_atr,
            "lookahead_bars": lh_res["lookahead_bars"],
            "buy_threshold":  g_buy["threshold"],
            "buy_margin":     g_buy["margin"],
            "buy_max_hold":   g_buy["max_hold"],
            "sell_threshold": g_sell["threshold"],
            "sell_margin":    g_sell["margin"],
            "sell_max_hold":  g_sell["max_hold"],
            "precision_buy":  g_buy["precision"],
            "precision_sell": g_sell["precision"],
            "coverage_buy":   g_buy["coverage"],
            "coverage_sell":  g_sell["coverage"],
            "buy_ok":         g_buy["ok"],
            "sell_ok":        g_sell["ok"],
        },
        "regime_boundaries": boundaries,
        "regimes":           regime_results,
        "diagnostics": {
            "atr_search":       atr_res,
            "lookahead_search": lh_res,
        },
    }

    out = PARAMS_DIR / f"{base}_params.json"
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2, default=str)

    n_live  = sum(1 for v in regime_results.values() if not v.get("skipped"))

    def _fmt(res: Dict[str, Any]) -> str:
        thr  = res["threshold"]
        prec = res["precision"]
        if res["ok"]:
            return f"[+] p>={thr:.2f} prec={prec:.0%}"
        if prec < BREAKEVEN:
            return f"[-] p>={thr:.2f} prec={prec:.0%}  (disabled: <{BREAKEVEN:.0%})"
        return f"[-] p>={thr:.2f} prec={prec:.0%}  (disabled: below {TARGET_PREC:.0%} target)"

    print(
        f"   [{symbol}] ATR x{best_atr:.2f} | lh={lh_res['lookahead_bars']}h | "
        f"BUY {_fmt(g_buy)} | SELL {_fmt(g_sell)} | "
        f"regimes {n_live}/{len(REGIME_KEYS)} active"
    )
    # Print the breakdown so the shifting thresholds across vol, vola, and trend can be monitored
    for rk, rres in regime_results.items():
        if not rres.get("skipped"):
            rb = {"threshold": rres["buy_threshold"], "precision": rres["precision_buy"], "ok": rres["buy_ok"]}
            rs = {"threshold": rres["sell_threshold"], "precision": rres["precision_sell"], "ok": rres["sell_ok"]}
            print(f"      -> [{rk:^13}] BUY {_fmt(rb)} | SELL {_fmt(rs)}")
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocess-safe wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _safe(symbol: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        return symbol, optimize_symbol(symbol)
    except Exception as exc:
        import traceback
        print(f"   [{symbol}] FATAL: {exc}\n{traceback.format_exc()}")
        return symbol, None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aegis per-token threshold + ATR + lookahead optimizer (independent)"
    )
    parser.add_argument("symbols", nargs="*",
                        help="Symbols to optimise (default: all FLEET_SYMBOLS with models)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1)")
    args = parser.parse_args()

    targets: List[str] = (
        [s.upper() for s in args.symbols] if args.symbols else list(FLEET_SYMBOLS)
    )
    targets = [s if "/" in s else s.replace("_", "/") for s in targets]

    print(f"\n{'='*72}")
    print("AEGIS — Per-Token Threshold + ATR + Lookahead Optimizer  [independent]")
    print(f"{'='*72}")
    print(f"Symbols      : {len(targets)}")
    print(f"History      : {HISTORY_HOURS} bars x 1h per symbol")
    print(f"Train / Eval : {int(TRAIN_FRAC*100)}% / {100-int(TRAIN_FRAC*100)}%  "
          f"(~{int(HISTORY_HOURS*TRAIN_FRAC)} / ~{HISTORY_HOURS - int(HISTORY_HOURS*TRAIN_FRAC)} bars)")
    print(f"Local model  : {LOCAL_ROUNDS} rounds, no Optuna, class-balanced weights")
    print(f"Thresholds   : on p_buy / p_sell  (directional prob, not meta_conf)")
    print(f"Grids        : ATR {len(ATR_MULT_GRID)} pts | "
          f"Thresh {len(THRESHOLD_GRID)} pts | Lookahead {len(LOOKAHEAD_GRID)} pts")
    print(f"Regimes      : {len(REGIME_KEYS)} buckets (3 vol x 3 vola x 3 trend)")
    print(f"Workers      : {args.workers}")
    print(f"Output       : {PARAMS_DIR}")
    print(f"{'='*72}\n")

    summary: Dict[str, Any] = {}
    t0 = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futs = {exe.submit(_safe, s): s for s in targets}
            for done_idx, fut in enumerate(as_completed(futs), 1):
                sym, result = fut.result()
                print(f"[{done_idx}/{len(targets)}] finished: {sym}")
                if result:
                    summary[sym] = result["global"]
    else:
        for idx, sym in enumerate(targets, 1):
            print(f"[{idx}/{len(targets)}] {sym}")
            _, result = _safe(sym)
            if result:
                summary[sym] = result["global"]

    summary_path = PARAMS_DIR / "_summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "n_symbols":       len(summary),
            "elapsed_seconds": round(time.time() - t0, 1),
            "symbols":         summary,
        }, fh, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"Done: {len(summary)}/{len(targets)} symbols in {elapsed / 60:.1f} min")
    print(f"Summary -> {summary_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
