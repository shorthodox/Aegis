#!/usr/bin/env python3
"""
threshold_optimizer.py — Per-token adaptive threshold, ATR & lookahead optimizer
=================================================================================
For each token with a trained model this script:

  1. Loads 2 000 bars of historical OHLCV through the same pipeline as retrain
  2. Gets batch predictions from the trained primary + meta XGBoost models
  3. Sweeps 14 ATR multiplier candidates → picks the one that maximises
     precision × sqrt(coverage) on fired signals
  4. Sweeps 8 lookahead windows → picks the best precision / trade-duration trade-off
  5. Classifies every bar into one of 9 regime buckets (3 volume × 3 volatility)
  6. Per-regime: grid-searches the best BUY / SELL confidence thresholds
  7. Saves everything to  data/token_params/<BASE>_params.json

These JSON files are consumed by:
  - scripts/retrain_model.py  — uses the optimised ATR multiplier instead of the
                                 static tier table
  - src/ml/predictor.py       — uses regime-specific thresholds at inference time

Usage
-----
    python scripts/threshold_optimizer.py                  # all fleet symbols
    python scripts/threshold_optimizer.py BTC/USDT ETH/USDT
    python scripts/threshold_optimizer.py --workers 4      # parallel processes
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
    create_triple_barrier_labels,
    fetch_futures_data,
    fetch_fear_greed,
    get_atr_multiplier,
    MAX_LOOKAHEAD,
    CENSORED,
)

# ── output directory ──────────────────────────────────────────────────────────
PARAMS_DIR  = _ROOT / "data" / "token_params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_STORE = _ROOT / "src" / "ml" / "model_store"

# ── data config ───────────────────────────────────────────────────────────────
HISTORY_HOURS = 2000    # bars per symbol — more history → better regime stats
MIN_BARS      = 600     # skip if fewer usable bars after feature engineering

# ── regime grid  (3 volume tiers × 3 volatility tiers = 9 buckets) ───────────
VOL_TIERS   = ["low", "med", "high"]
VOLA_TIERS  = ["low", "med", "high"]
REGIME_KEYS = [f"{v}_{va}" for v in VOL_TIERS for va in VOLA_TIERS]  # 9 keys

# ── search grids ──────────────────────────────────────────────────────────────
THRESHOLD_GRID = np.round(np.arange(0.48, 0.82, 0.02), 3).tolist()   # 17 pts
ATR_MULT_GRID  = np.round(np.arange(0.75, 4.25, 0.25), 2).tolist()   # 14 pts
LOOKAHEAD_GRID = [12, 18, 24, 30, 36, 48, 60, 72]                     #  8 pts

# ── quality gates ─────────────────────────────────────────────────────────────
MIN_REGIME_BARS = 50     # skip a regime bucket if it has fewer valid bars
MIN_SIGNALS     = 10     # minimum fired signals to trust a threshold result
BREAKEVEN       = FEE_ROUNDTRIP   # ~0.10 % round-trip — minimum viable precision
TARGET_PREC     = 0.60            # "green" precision target


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dm(X: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(X.values, feature_names=list(X.columns))


def _tier(val: float, p33: float, p67: float) -> str:
    if val <= p33:
        return "low"
    if val <= p67:
        return "med"
    return "high"


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(
    df: pd.DataFrame,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Classify each bar into one of 9 regime strings '<vol>_<vola>'.
    vol  axis: rolling 24-bar mean volume vs own historical percentiles.
    vola axis: ATR / close vs own historical percentiles.

    Returns (regime_series, boundaries_dict).
    boundaries_dict is saved to the JSON output so predictor.py can classify
    the current bar at inference time without the full historical distribution.
    """
    roll_vol = df["volume"].rolling(24, min_periods=1).mean()
    atr_vals = compute_atr(df, period=14)
    atr_pct  = (atr_vals / df["close"].replace(0, np.nan)).fillna(0)

    vp33 = float(roll_vol.quantile(0.33))
    vp67 = float(roll_vol.quantile(0.67))
    ap33 = float(atr_pct.quantile(0.33))
    ap67 = float(atr_pct.quantile(0.67))

    boundaries: Dict[str, float] = {
        "vol_p33": vp33, "vol_p67": vp67,
        "atr_pct_p33": ap33, "atr_pct_p67": ap67,
    }
    reg = pd.Series(
        [f"{_tier(float(roll_vol.iloc[i]), vp33, vp67)}"
         f"_{_tier(float(atr_pct.iloc[i]), ap33, ap67)}"
         for i in range(len(df))],
        index=df.index, dtype=str,
    )
    return reg, boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Threshold grid-search for one direction in one data slice
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(
    meta_conf: np.ndarray,
    proposed:  np.ndarray,
    labels:    np.ndarray,
    side:      int,        # 2 = BUY, 0 = SELL
    n_total:   int,
) -> Dict[str, Any]:
    """
    Sweep THRESHOLD_GRID for one direction.
    Score = precision × sqrt(coverage) balances signal quality vs trade count.

    Returns: {threshold, precision, coverage, n_signals, ok}
    'ok' = True only when precision >= TARGET_PREC.
    """
    sm = proposed == side
    if sm.sum() < MIN_SIGNALS:
        return {"threshold": 0.60, "precision": 0.0,
                "coverage": 0.0, "n_signals": 0, "ok": False}

    conf_s  = meta_conf[sm]
    lbl_s   = labels[sm]
    correct = side

    best: Optional[Dict[str, Any]] = None
    best_score = -np.inf

    for thr in THRESHOLD_GRID:
        fired = conf_s >= thr
        n = int(fired.sum())
        if n < MIN_SIGNALS:
            continue
        prec = float((lbl_s[fired] == correct).mean())
        cov  = n / max(n_total, 1)
        if prec < BREAKEVEN:
            continue
        score = prec * float(np.sqrt(cov))
        if score > best_score:
            best_score = score
            best = {
                "threshold": float(thr),
                "precision": round(prec, 4),
                "coverage":  round(cov, 4),
                "n_signals": n,
                "ok":        prec >= TARGET_PREC,
            }

    if best is None:
        # Nothing cleared the fee floor — return the best-precision entry
        # as a reference only (ok=False so callers won't act on it)
        rows = []
        for thr in THRESHOLD_GRID:
            fired = conf_s >= thr
            n = int(fired.sum())
            if n >= max(3, MIN_SIGNALS // 3):
                prec = float((lbl_s[fired] == correct).mean())
                rows.append({
                    "threshold": float(thr),
                    "precision": round(prec, 4),
                    "coverage":  round(n / max(n_total, 1), 4),
                    "n_signals": n,
                    "ok":        False,
                })
        if rows:
            return max(rows, key=lambda r: r["precision"])
        return {"threshold": 0.60, "precision": 0.0,
                "coverage": 0.0, "n_signals": 0, "ok": False}

    return best


# ─────────────────────────────────────────────────────────────────────────────
# ATR multiplier optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimize_atr(
    df:        pd.DataFrame,
    meta_conf: np.ndarray,
    proposed:  np.ndarray,
    base_mult: float,
) -> Dict[str, Any]:
    """
    Re-label data at each ATR candidate; score the model's precision on signals
    it would fire (top-10 % confidence per side).

    The multiplier that maximises mean(prec × sqrt(cov)) across BUY + SELL is
    returned.  This answers: "what SL distance gives the best quality/quantity
    trade-off for this token right now?"
    """
    vr = df["volatility_regime"]   if "volatility_regime"   in df.columns else None
    er = df["efficiency_ratio_10"] if "efficiency_ratio_10" in df.columns else None
    tr = df["trend_regime"]        if "trend_regime"        in df.columns else None
    cs = df.get("macro_confluence_score")

    best_mult  = base_mult
    best_score = -np.inf
    scored: List[Dict[str, Any]] = []

    for mult in ATR_MULT_GRID:
        labs  = create_triple_barrier_labels(
            df, atr_multiplier=mult,
            volatility_regime=vr, efficiency_ratio=er,
            trend_regime=tr, macro_confluence_score=cs,
        ).values
        valid = np.asarray(labs != CENSORED)
        n = int(valid.sum())
        if n < MIN_REGIME_BARS:
            continue

        lbl  = np.asarray(labs[valid])
        conf = np.asarray(meta_conf[valid])
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
        if score > best_score:
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
    df:        pd.DataFrame,
    meta_conf: np.ndarray,
    proposed:  np.ndarray,
    atr_mult:  float,
) -> Dict[str, Any]:
    """
    Sweep LOOKAHEAD_GRID and find the window that maximises signal quality.
    A 5 % rank-decay biases toward shorter windows (faster capital recycling
    when precision is equal).
    """
    vr = df["volatility_regime"]   if "volatility_regime"   in df.columns else None
    er = df["efficiency_ratio_10"] if "efficiency_ratio_10" in df.columns else None
    tr = df["trend_regime"]        if "trend_regime"        in df.columns else None
    cs = df.get("macro_confluence_score")

    best_lh    = MAX_LOOKAHEAD
    best_score = -np.inf
    scored: List[Dict[str, Any]] = []

    for lh in LOOKAHEAD_GRID:
        labs  = create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=lh,
            volatility_regime=vr, efficiency_ratio=er,
            trend_regime=tr, macro_confluence_score=cs,
        ).values
        valid = np.asarray(labs != CENSORED)
        n = int(valid.sum())
        if n < MIN_REGIME_BARS:
            continue

        lbl  = np.asarray(labs[valid])
        conf = np.asarray(meta_conf[valid])
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

        rank_decay = 1.0 - (LOOKAHEAD_GRID.index(lh) / len(LOOKAHEAD_GRID)) * 0.05
        score = float(np.mean(parts)) * rank_decay
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
    """Full optimisation pipeline for one symbol. Returns params dict or None."""
    base = symbol.replace("/", "_")

    if not (MODEL_STORE / f"{base}_model.json").exists():
        print(f"   [{symbol}] No trained model — skipping.")
        return None

    # ── load predictor (models + data-fetch methods) ──────────────────────────
    try:
        p = Predictor(symbol)
    except Exception as exc:
        print(f"   [{symbol}] Predictor init failed: {exc}")
        return None

    if p.model is None:
        print(f"   [{symbol}] Primary model missing — skipping.")
        return None

    feature_cols: Optional[List[str]] = p.meta.get("feature_cols")
    base_atr_mult = float(p.meta.get("atr_multiplier", get_atr_multiplier(symbol)))

    # ── fetch historical data ─────────────────────────────────────────────────
    try:
        df_1h = p.fetch_live_data(timeframe="1h", limit=HISTORY_HOURS)
        if df_1h is None or len(df_1h) < MIN_BARS:
            print(f"   [{symbol}] Not enough data — skipping.")
            return None

        btc_df  = (p.fetch_btc_data(timeframe="1h", limit=HISTORY_HOURS)
                   if hasattr(p, "fetch_btc_data") else None)
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

    # ── align raw OHLCV to feature rows ──────────────────────────────────────
    # prepare_features drops NaN-heavy leading rows, so we align from the tail.
    df_raw = df_1h.iloc[-n_feat:].reset_index(drop=True).copy()

    # Attach regime-auxiliary columns so create_triple_barrier_labels can use them.
    for col in ("volatility_regime", "efficiency_ratio_10",
                "trend_regime", "macro_confluence_score"):
        if col in features.columns:
            df_raw[col] = features[col].values

    # ── model input matrix ────────────────────────────────────────────────────
    if feature_cols:
        for c in feature_cols:
            if c not in features.columns:
                features[c] = 0.0
        feat_df = features[feature_cols].copy()
    else:
        drop = [c for c in ("timestamp", "target") if c in features.columns]
        feat_df = features.drop(columns=drop, errors="ignore")
        feat_df = feat_df[[c for c in feat_df.columns if not c.startswith("_")]]
        feature_cols = list(feat_df.columns)

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── batch predict ─────────────────────────────────────────────────────────
    try:
        dmat      = _dm(feat_df)
        raw_probs = p.model.predict(dmat)      # (n, 3) softmax
        if raw_probs.ndim != 2 or raw_probs.shape[1] != 3:
            print(f"   [{symbol}] Unexpected model output shape — skipping.")
            return None

        proposed  = np.where(raw_probs[:, 2] >= raw_probs[:, 0], 2, 0)
        meta_conf = (p.meta_model.predict(dmat).astype(float)
                     if p.meta_model is not None
                     else raw_probs.max(axis=1))
    except Exception as exc:
        print(f"   [{symbol}] Prediction error: {exc}")
        return None

    # ── regime classification ─────────────────────────────────────────────────
    regimes, boundaries = classify_regimes(df_raw)

    # ── optimise ATR multiplier ───────────────────────────────────────────────
    atr_res  = optimize_atr(df_raw, meta_conf, proposed, base_atr_mult)
    best_atr = atr_res["atr_multiplier"]

    # ── optimise lookahead ────────────────────────────────────────────────────
    lh_res = optimize_lookahead(df_raw, meta_conf, proposed, best_atr)

    # ── barrier labels with best ATR for threshold search ────────────────────
    vr = df_raw["volatility_regime"]   if "volatility_regime"   in df_raw.columns else None
    er = df_raw["efficiency_ratio_10"] if "efficiency_ratio_10" in df_raw.columns else None
    tr = df_raw["trend_regime"]        if "trend_regime"        in df_raw.columns else None
    cs = df_raw.get("macro_confluence_score")

    base_labels = create_triple_barrier_labels(
        df_raw, atr_multiplier=best_atr,
        volatility_regime=vr, efficiency_ratio=er,
        trend_regime=tr, macro_confluence_score=cs,
    ).values
    valid_mask = np.asarray(base_labels != CENSORED)

    # ── global threshold optimisation ─────────────────────────────────────────
    n_g    = int(valid_mask.sum())
    lbl_g  = np.asarray(base_labels[valid_mask])
    conf_g = np.asarray(meta_conf[valid_mask])
    prop_g = np.asarray(proposed[valid_mask])

    g_buy  = best_threshold(conf_g, prop_g, lbl_g, side=2, n_total=n_g)
    g_sell = best_threshold(conf_g, prop_g, lbl_g, side=0, n_total=n_g)

    # ── per-regime threshold optimisation ─────────────────────────────────────
    regime_results: Dict[str, Any] = {}
    for rk in REGIME_KEYS:
        rmask = np.asarray(regimes == rk) & valid_mask
        n_r   = int(rmask.sum())
        if n_r < MIN_REGIME_BARS:
            regime_results[rk] = {
                "n_bars": n_r, "skipped": True,
                "reason": f"too_few_bars (<{MIN_REGIME_BARS})",
            }
            continue

        lbl_r  = np.asarray(base_labels[rmask])
        conf_r = np.asarray(meta_conf[rmask])
        prop_r = np.asarray(proposed[rmask])

        r_buy  = best_threshold(conf_r, prop_r, lbl_r, side=2, n_total=n_r)
        r_sell = best_threshold(conf_r, prop_r, lbl_r, side=0, n_total=n_r)

        regime_results[rk] = {
            "n_bars":         n_r,
            "skipped":        False,
            "buy_threshold":  r_buy["threshold"],
            "sell_threshold": r_sell["threshold"],
            "precision_buy":  r_buy["precision"],
            "precision_sell": r_sell["precision"],
            "coverage_buy":   r_buy["coverage"],
            "coverage_sell":  r_sell["coverage"],
            "n_signals_buy":  r_buy["n_signals"],
            "n_signals_sell": r_sell["n_signals"],
            "buy_ok":         r_buy["ok"],
            "sell_ok":        r_sell["ok"],
        }

    # ── assemble JSON ─────────────────────────────────────────────────────────
    params: Dict[str, Any] = {
        "symbol":     symbol,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "n_bars":     n_feat,
        "n_valid":    n_g,
        "global": {
            "atr_multiplier": best_atr,
            "lookahead_bars": lh_res["lookahead_bars"],
            "buy_threshold":  g_buy["threshold"],
            "sell_threshold": g_sell["threshold"],
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
    buy_ok  = "+" if g_buy["ok"]  else "-"
    sell_ok = "+" if g_sell["ok"] else "-"
    print(
        f"   [{symbol}] ATR x{best_atr:.2f} | lh={lh_res['lookahead_bars']}h | "
        f"BUY [{buy_ok}] {g_buy['threshold']:.2f} ({g_buy['precision']:.0%}) | "
        f"SELL [{sell_ok}] {g_sell['threshold']:.2f} ({g_sell['precision']:.0%}) | "
        f"regimes {n_live}/9 active"
    )
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
        description="Aegis per-token threshold, ATR & lookahead optimizer"
    )
    parser.add_argument(
        "symbols", nargs="*",
        help="Symbols to optimise (default: all FLEET_SYMBOLS with trained models)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes (default 1; 2-4 recommended for speed)",
    )
    args = parser.parse_args()

    targets: List[str] = (
        [s.upper() for s in args.symbols] if args.symbols else list(FLEET_SYMBOLS)
    )
    targets = [s if "/" in s else s.replace("_", "/") for s in targets]

    print(f"\n{'='*72}")
    print("AEGIS — Per-Token Threshold + ATR + Lookahead Optimizer")
    print(f"{'='*72}")
    print(f"Symbols   : {len(targets)}")
    print(f"History   : {HISTORY_HOURS} bars x 1h per symbol")
    print(f"Grids     : ATR {len(ATR_MULT_GRID)} pts | "
          f"Thresh {len(THRESHOLD_GRID)} pts | Lookahead {len(LOOKAHEAD_GRID)} pts")
    print(f"Regimes   : {len(REGIME_KEYS)} buckets (3 vol x 3 vola)")
    print(f"Workers   : {args.workers}")
    print(f"Output    : {PARAMS_DIR}")
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
        json.dump(
            {
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "n_symbols":       len(summary),
                "elapsed_seconds": round(time.time() - t0, 1),
                "symbols":         summary,
            },
            fh, indent=2, default=str,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"Done: {len(summary)}/{len(targets)} symbols in {elapsed / 60:.1f} min")
    print(f"Summary -> {summary_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
