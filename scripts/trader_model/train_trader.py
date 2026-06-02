"""
train_trader.py — Universal Trader Model Training Pipeline

Trains 3 XGBoost models (scalping / intraday / swing) on 10 diverse tokens.
Each model uses 25 strategy features, percentile-ranked for token-agnostic inference.

Usage:
    python -m scripts.trader_model.train_trader
    python -m scripts.trader_model.train_trader --mode scalping
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trader_model.trader_config import (
    DEPLOYMENT_TOKENS, MODES, RISK_PROFILES, ALL_FEATURE_NAMES,
    TRADER_MODEL_STORE, TRAINING_BARS, TRAINING_SUMMARY, TRAINING_TOKENS,
)
from scripts.trader_model.strategy_features import (
    compute_all_features, percentile_rank_features,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

TRADER_MODEL_STORE.mkdir(parents=True, exist_ok=True)

# ── Exchange setup ─────────────────────────────────────────────────────────────
_exchange: Optional[ccxt.Exchange] = None

def get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})  # type: ignore[arg-type]
    return _exchange


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV data and return as DataFrame."""
    ex = get_exchange()
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(3):
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df  = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {symbol} {timeframe}: {last_exc}") from last_exc


# ── Label generation ───────────────────────────────────────────────────────────

def generate_labels(close: pd.Series, label_bars: int, threshold: float) -> pd.Series:
    """
    Forward-return triple-barrier labeling.

    Returns:
        pd.Series of int: 1 (BUY), -1 (SELL), 0 (HOLD)
    """
    fwd_return = close.shift(-label_bars) / close - 1
    labels = pd.Series(0.0, index=close.index, dtype=float)
    labels[fwd_return >  threshold] =  1
    labels[fwd_return < -threshold] = -1
    # Drop last label_bars rows (no future data)
    labels.iloc[-label_bars:] = np.nan
    return labels


# ── Token selection from training summary ─────────────────────────────────────

def select_training_tokens() -> List[str]:
    """
    Use TRAINING_TOKENS from config as the primary list.
    Validate against training_summary.json; log any missing entries.
    """
    if not TRAINING_SUMMARY.exists():
        log.warning("training_summary.json not found — using default TRAINING_TOKENS")
        return TRAINING_TOKENS

    with open(TRAINING_SUMMARY) as f:
        summary = json.load(f)

    trained_symbols = {r['symbol'].replace('/', '_') for r in summary.get('results', [])}
    selected = []
    for tok in TRAINING_TOKENS:
        key = tok.replace('/', '_')
        if key in trained_symbols:
            selected.append(tok)
            log.info(f"  Training token validated: {tok}")
        else:
            log.warning(f"  Token {tok} not in training_summary — skipping")

    if not selected:
        log.warning("No training tokens validated — using full TRAINING_TOKENS list")
        return TRAINING_TOKENS
    return selected


# ── Build dataset for one mode ─────────────────────────────────────────────────

def build_dataset(tokens: List[str], mode_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch data for all training tokens, compute strategy features, generate labels.
    Returns (X, y) as numpy arrays.
    """
    tf      = mode_cfg['timeframe']
    limit   = TRAINING_BARS[tf]
    bars    = mode_cfg['label_bars']
    thresh  = mode_cfg['label_threshold']

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for symbol in tokens:
        log.info(f"  Fetching {symbol} {tf} ({limit} bars)…")
        try:
            df = fetch_ohlcv(symbol, tf, limit)
            if len(df) < 200:
                log.warning(f"  {symbol}: only {len(df)} bars, skipping")
                continue

            # Strategy + feature-engine features (45 total)
            raw_features = compute_all_features(df)
            features     = percentile_rank_features(raw_features, lookback=100)

            # Forward-return labels
            labels = generate_labels(df['close'], bars, thresh)

            # Align and drop NaN rows
            combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
            if len(combined) < 100:
                log.warning(f"  {symbol}: too few clean rows ({len(combined)}), skipping")
                continue

            X = combined[ALL_FEATURE_NAMES].values.astype(np.float32)
            y = np.asarray(combined['label'].values, dtype=np.int64)

            all_X.append(X)
            all_y.append(y)
            uniq, cnts = np.unique(y, return_counts=True)
            log.info(f"  {symbol}: {len(X)} samples, label dist: {dict(zip(uniq.tolist(), cnts.tolist()))}")

        except Exception as e:
            log.error(f"  {symbol}: error — {e}")
            continue

        time.sleep(0.3)  # rate limit courtesy

    if not all_X:
        raise RuntimeError("No training data collected — check token list or API connection")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    log.info(f"  Total dataset: {X_all.shape[0]} samples, label dist: {dict(zip(*np.unique(y_all, return_counts=True)))}")
    return X_all, y_all


# ── Train one mode ─────────────────────────────────────────────────────────────

def train_mode(mode_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
    """Train XGBoost + CalibratedClassifierCV for one trading mode."""
    log.info(f"\nTraining {mode_name} model ({X.shape[0]} samples × {X.shape[1]} features)…")

    # Map labels -1/0/1 → 0/1/2 for XGBoost multiclass
    y_mapped = y + 1   # -1→0 (SELL), 0→1 (HOLD), 1→2 (BUY)

    # Class weights to handle imbalance
    classes, counts = np.unique(y_mapped, return_counts=True)
    total = len(y_mapped)
    weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weights[yi] for yi in y_mapped])

    xgb_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )

    # Time-series cross-val (5 folds, no shuffle) — evaluation only
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        fold_clf = XGBClassifier(**xgb_params)
        fold_clf.fit(
            X[train_idx], y_mapped[train_idx],
            sample_weight=sample_weights[train_idx],
            verbose=False,
        )
        preds = fold_clf.predict(X[val_idx])
        acc   = float((preds == y_mapped[val_idx]).mean())
        cv_scores.append(acc)
        log.info(f"  Fold {fold+1}: accuracy={acc:.3f}")

    cv_mean = float(np.mean(cv_scores))
    log.info(f"  CV accuracy: {cv_mean:.3f} ± {np.std(cv_scores):.3f}")

    # Final model: CalibratedClassifierCV with internal 5-fold CV
    # (sklearn 1.4+ removed cv='prefit'; internal CV is cleaner anyway)
    calibrated = CalibratedClassifierCV(
        XGBClassifier(**xgb_params),
        method='isotonic',
        cv=5,
        n_jobs=-1,
    )
    calibrated.fit(X, y_mapped, sample_weight=sample_weights)

    # Evaluate on held-out last 20%
    split_idx   = int(len(X) * 0.80)
    X_calib     = X[split_idx:]
    y_calib     = y_mapped[split_idx:]
    preds       = calibrated.predict(X_calib)
    proba       = calibrated.predict_proba(X_calib)
    max_conf    = proba.max(axis=1)
    high_conf   = max_conf >= 0.70
    if high_conf.any():
        high_conf_acc = float((preds[high_conf] == y_calib[high_conf]).mean())
        coverage      = float(high_conf.mean())
    else:
        high_conf_acc = 0.0
        coverage      = 0.0

    log.info(f"  Calibrated — 70%+ conf accuracy: {high_conf_acc:.3f}, coverage: {coverage:.3f}")
    log.info(f"\n{classification_report(y_calib, preds, target_names=['SELL','HOLD','BUY'])}")

    # Save model
    model_path = TRADER_MODEL_STORE / f"{mode_name}_model.pkl"
    joblib.dump(calibrated, model_path)
    log.info(f"  Saved → {model_path}")

    # Save metadata sidecar
    meta = {
        'mode':             mode_name,
        'cv_accuracy':      cv_mean,
        'high_conf_acc':    high_conf_acc,
        'high_conf_coverage': coverage,
        'n_samples':        int(X.shape[0]),
        'n_features':       int(X.shape[1]),
        'feature_names':    ALL_FEATURE_NAMES,
        'label_map':        {'0': 'SELL', '1': 'HOLD', '2': 'BUY'},
        'trained_tokens':   TRAINING_TOKENS,
    }
    meta_path = TRADER_MODEL_STORE / f"{mode_name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Universal Trader Models')
    parser.add_argument('--mode', choices=list(MODES.keys()) + ['all'], default='all',
                        help='Which mode to train (default: all)')
    args = parser.parse_args()

    modes_to_train = list(MODES.keys()) if args.mode == 'all' else [args.mode]

    log.info("═" * 60)
    log.info("  Universal Trader Model — Training Pipeline")
    log.info("═" * 60)

    tokens = select_training_tokens()
    log.info(f"Training on {len(tokens)} tokens: {tokens}")

    results = {}
    for mode_name in modes_to_train:
        mode_cfg = MODES[mode_name]
        log.info(f"\n{'─'*50}")
        log.info(f"Mode: {mode_name.upper()} | TF: {mode_cfg['timeframe']}")
        log.info(f"{'─'*50}")
        try:
            X, y = build_dataset(tokens, mode_cfg)
            meta  = train_mode(mode_name, X, y)
            results[mode_name] = meta
        except Exception as e:
            log.error(f"Failed to train {mode_name}: {e}")
            results[mode_name] = {'error': str(e)}

    # Save training summary
    summary_path = TRADER_MODEL_STORE / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nTraining complete. Summary saved → {summary_path}")


if __name__ == '__main__':
    main()
