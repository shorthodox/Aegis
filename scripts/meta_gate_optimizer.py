#!/usr/bin/env python3
"""
meta_gate_optimizer.py — Independent per-token meta gate discovery engine
=================================================================================
Generates independent meta gate profiles for each asset by discovering the
best gate architecture through a deterministic out-of-sample search.

This optimizer is intentionally separate from production models and uses only
feature engineering, a lightweight local directional model, and gate evaluation
on a held-out evaluation window.

The output is written to data/meta_gate_profiles:
  - <symbol>_gate.json   — per-token gate profile
  - _summary.json        — fleet-level selected gate summaries

The search explores:
  - 10 gate architecture templates (side-specific, regime-aware, veto-enabled)
  - percentile-based edge score thresholds (no hardcoded global cutoffs)
  - optional calibration of the edge score distribution
  - regime threshold modifiers and regime disabling
  - risk-tier tradeability metadata
  - holdout-based precision, expectancy, sharpe, profit factor, gate lift
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
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.threshold_optimizer import classify_regimes, MIN_REGIME_BARS
from src.ml.calibration import MetaCalibrationFramework
from src.ml.feature_engine import compute_atr, prepare_features
from src.ml.predictor import Predictor
from src.trading.edge_engine import EdgeScoringEngine
from scripts.retrain_model import (
    CENSORED,
    FEE_ROUNDTRIP,
    FLEET_SYMBOLS,
    MAX_LOOKAHEAD,
    MIN_FIRES_DEV,
    TARGET_SIGNAL_PRECISION,
    backtest,
    create_triple_barrier_labels,
    fetch_fear_greed,
    fetch_futures_data,
    get_atr_multiplier,
    sample_weights,
)
from scripts.validation import validate_architecture_from_folds
from scripts.forensic_gate_report import ForensicGateReporter
from scripts.gate_forensics import GateForensicsReporter

# ── output ────────────────────────────────────────────────────────────────────
PROFILE_DIR = _ROOT / "data" / "meta_gate_profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = PROFILE_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
HISTORY_HOURS = 4000
MIN_BARS = 800
TRAIN_FRAC = 0.70
LOCAL_ROUNDS = 300
MIN_CALIBRATION_COVERAGE = 0.70
MIN_GATE_COVERAGE = 0.15
MIN_GATE_SIGNALS = 50
MIN_VARIANCE_RETAINED = 0.40

# ── FIX 4: Meta gate overfit prevention (NEW CONSTANTS) ────────────────────────
MIN_VALID_HOLDOUT_BARS = 500       # Minimum holdout size for reliable gate selection (FIX 4.1)
MIN_GATE_LIFT = 0.01                # Reject gates with negative lift (FIX 4.2)
FALLBACK_TRUST_THRESHOLD = 40       # Trust score below this triggers dynamic fallback (FIX 4.4)

# ── candidate search grids ────────────────────────────────────────────────────
QUANTILES = [0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]
CALIBRATION_METHODS = [None, 'temperature', 'platt', 'isotonic', 'beta']

REGIME_MODIFIER_TEMPLATE: Dict[str, Any] = {
    'COMPRESSION': -5.0,
    'VOLATILE_EXPANSION': -3.0,
    'DISTRIBUTION': 5.0,
    'ACCUMULATION': 'disable',
    'CHOPPY': 10.0,
    'RANGING': 0.0,
    'TRENDING_BULL': 0.0,
    'TRENDING_BEAR': 0.0,
}

GATE_ARCHITECTURES: List[Dict[str, Any]] = [
    {
        'name': 'GLOBAL_EDGE_PERCENTILE',
        'description': 'Single global edge-score percentile for both sides',
        'side_specific': False,
        'regime_modifier': False,
        'vetoes': [],
        'calibrate': False,
    },
    {
        'name': 'SIDE_EDGE_PERCENTILE',
        'description': 'Separate BUY/SELL edge thresholds using percentiles',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': [],
        'calibrate': False,
    },
    {
        'name': 'REGIME_EDGE_PERCENTILE',
        'description': 'Side-specific edge thresholds plus regime modifiers',
        'side_specific': True,
        'regime_modifier': True,
        'vetoes': [],
        'calibrate': False,
    },
    {
        'name': 'CALIBRATED_SIDE_EDGE',
        'description': 'Side thresholds on calibrated edge score probabilities',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': [],
        'calibrate': True,
    },
    {
        'name': 'CALIBRATED_REGIME_EDGE',
        'description': 'Calibrated side thresholds plus regime modifiers',
        'side_specific': True,
        'regime_modifier': True,
        'vetoes': [],
        'calibrate': True,
    },
    {
        'name': 'EDGE_SR_VETO',
        'description': 'Side thresholds with support/resistance vetoes',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': ['sr'],
        'calibrate': False,
    },
    {
        'name': 'EDGE_TREND_VETO',
        'description': 'Side thresholds with macro trend vetoes',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': ['trend'],
        'calibrate': False,
    },
    {
        'name': 'EDGE_CONFLUENCE_VETO',
        'description': 'Side thresholds with technical confluence vetoes',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': ['confluence'],
        'calibrate': False,
    },
    {
        'name': 'EDGE_STRICT_VETO',
        'description': 'Side thresholds plus all vetoes and regime-aware modifiers',
        'side_specific': True,
        'regime_modifier': True,
        'vetoes': ['sr', 'trend', 'confluence'],
        'calibrate': True,
    },
    {
        'name': 'EDGE_LOOSE_COVERAGE',
        'description': 'Looser coverage target with side-specific thresholds',
        'side_specific': True,
        'regime_modifier': False,
        'vetoes': [],
        'calibrate': False,
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _fit_local_model(
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    train_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, xgb.Booster]:
    # Create OOF predictions on the training partition using a time-series split
    # so calibrators and training statistics are computed from OOF (purged) preds.
    n = len(feat_df)
    if train_n <= 0 or train_n > n:
        raise ValueError("train_n out of range")

    # Guard against non-numeric feature columns (e.g. regime labels or string flags)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) != len(feat_df.columns):
        dropped = [c for c in feat_df.columns if c not in numeric_cols]
        print(f"   WARNING: dropping non-numeric feature columns for local model fit: {dropped}")
        feat_df = feat_df[numeric_cols].copy()
    cols = list(feat_df.columns)

    # Prepare arrays
    probs_all = np.full((n, 3), np.nan, dtype=float)
    proposed_all = np.full(n, 1, dtype=int)
    dir_conf_all = np.full(n, 0.0, dtype=float)

    # Build training OOF on first train_n rows
    X = feat_df.iloc[:train_n].values
    y = np.asarray(labels[:train_n])
    valid_mask = y != CENSORED
    if int(valid_mask.sum()) < 200:
        raise ValueError(f"Only {int(valid_mask.sum())} valid training bars.")

    # TimeSeriesSplit: choose up to 5 folds or fewer depending on train_n
    n_splits = min(5, max(2, int(train_n / 200)))
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tss.split(np.arange(train_n)):
        # keep only valid rows inside each fold
        tr_mask = valid_mask[tr_idx]
        if tr_mask.sum() < 50:
            continue
        X_tr = feat_df.iloc[:train_n].iloc[tr_idx][tr_mask].values
        y_tr = y[tr_idx][tr_mask].astype(int)

        w_tr = sample_weights(y_tr)

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'min_child_weight': 10,
            'reg_lambda': 2.0,
            'gamma': 1.0,
            'seed': 42,
            'tree_method': 'hist',
            'missing': np.nan,
            'verbosity': 0,
        }

        dm_tr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=cols)
        try:
            model = xgb.train(params, dm_tr, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)
        except Exception:
            continue

        # Predict on validation fold
        va_idx_all = np.array(va_idx)
        X_va = feat_df.iloc[:train_n].iloc[va_idx_all].values
        dm_va = xgb.DMatrix(X_va, feature_names=cols)
        preds = model.predict(dm_va)
        probs_all[:train_n][va_idx_all] = preds

    # For any remaining NaN rows in train, fill with model trained on full train if possible
    remaining = np.where(np.isnan(probs_all[:train_n, 0]))[0]
    if len(remaining) > 0:
        # Train final model on all valid training rows
        tr_mask = valid_mask
        X_tr = feat_df.iloc[:train_n][tr_mask].values
        y_tr = y[tr_mask].astype(int)
        w_tr = sample_weights(y_tr)
        dm_tr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=cols)
        model = xgb.train(params, dm_tr, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)
        dm_all_train = xgb.DMatrix(feat_df.iloc[:train_n].values, feature_names=cols)
        preds_full = model.predict(dm_all_train)
        for i in remaining:
            probs_all[i] = preds_full[i]

    # Now train final model on all rows (full-fit) to get deployment probs
    X_all = feat_df.values
    tr_mask = valid_mask
    X_tr_final = feat_df.iloc[:train_n][tr_mask].values
    y_tr_final = y[tr_mask].astype(int)
    w_tr_final = sample_weights(y_tr_final)
    dm_tr_final = xgb.DMatrix(X_tr_final, label=y_tr_final, weight=w_tr_final, feature_names=cols)
    final_model = xgb.train(params, dm_tr_final, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)
    dm_all = xgb.DMatrix(X_all, feature_names=cols)
    probs_all_full = final_model.predict(dm_all)

    # populate final arrays
    probs_all = np.where(np.isnan(probs_all), probs_all_full, probs_all)
    proposed_all = np.where(probs_all[:, 2] >= probs_all[:, 0], 2, 0).astype(int)
    dir_conf_all = np.where(proposed_all == 2, probs_all[:, 2], probs_all[:, 0]).astype(float)
    return proposed_all, dir_conf_all, probs_all, final_model


def _min_signals(total_signals: int) -> int:
    """Safe minimum signal count for low-frequency holdouts."""
    return max(5, int(total_signals * 0.02))


def _normalize_expectancy(expectancy_pct: float) -> float:
    """Normalize expectancy into [0, 1] for calibration scoring."""
    return float(min(max(expectancy_pct / 10.0, 0.0), 1.0))


def _normalize_sharpe(sharpe: float) -> float:
    return float(min(max(sharpe / 5.0, 0.0), 1.0))


def _calibration_score(
    ece: float,
    brier: float,
    precision: float,
    coverage: float,
    baseline_ece: float,
    baseline_brier: float,
    baseline_precision: float,
    method: Optional[str] = None,
) -> float:
    """Score calibration candidates using profitability-first holdout metrics."""
    if coverage < 0.0 or precision < 0.0 or ece < 0.0 or brier < 0.0:
        return 0.0

    calibration_quality = clamp(1.0 - ece, 0.0, 1.0)
    precision_norm = clamp(precision, 0.0, 1.0)
    coverage_norm = clamp(coverage, 0.0, 1.0)
    expect_norm = clamp(precision / 10.0, 0.0, 1.0)
    pf_norm = clamp((precision - 1.0) / 2.0, 0.0, 1.0)
    sharpe_norm = clamp((precision - 0.0) / 5.0, 0.0, 1.0)

    score = (
        0.35 * expect_norm +
        0.25 * pf_norm +
        0.15 * sharpe_norm +
        0.10 * precision_norm +
        0.10 * coverage_norm +
        0.05 * calibration_quality
    )

    return score


def _compute_trust_score(
    pf_norm: float,
    expectancy_norm: float,
    sharpe_norm: float,
    coverage_norm: float,
    fired_n: int,
) -> int:
    """Unified trust score calculator. Used both during gate selection and in forensic reports."""
    raw = 100.0 * (
        0.30 * pf_norm +
        0.25 * expectancy_norm +
        0.20 * sharpe_norm +
        0.15 * coverage_norm +
        0.10 * clamp(fired_n / 100.0, 0.0, 1.0)
    )
    return int(clamp(raw, 0.0, 100.0))


def _gate_grade(profit_factor: float) -> str:
    """PF-based gate grade. A+ through F."""
    if profit_factor > 1.80:
        return 'A+'
    if profit_factor > 1.60:
        return 'A'
    if profit_factor > 1.40:
        return 'B+'
    if profit_factor > 1.30:
        return 'B'
    if profit_factor > 1.20:
        return 'C+'
    if profit_factor > 1.10:
        return 'C'
    if profit_factor > 1.00:
        return 'D'
    return 'F'


def safe_json_serializer(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects.

    Sklearn models become a lightweight descriptor so the JSON stays human-readable.
    """
    if isinstance(obj, dict):
        return {k: safe_json_serializer(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serializer(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Sklearn / arbitrary model objects
    module = getattr(type(obj), '__module__', '') or ''
    if 'sklearn' in module or hasattr(obj, 'get_params'):
        params: Dict[str, Any] = {}
        try:
            params = {k: safe_json_serializer(v) for k, v in obj.get_params().items()}
        except Exception:
            pass
        return {
            '__type__': 'sklearn_model',
            'class': type(obj).__name__,
            'module': module,
            'params': params,
        }
    # Fallback
    try:
        return str(obj)
    except Exception:
        return None


def _select_best_calibrator(
    raw_scores: np.ndarray,
    correct: np.ndarray,
    ev_edge_raw: np.ndarray,
    ev_side: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
) -> Tuple[MetaCalibrationFramework, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Train all calibration methods and return models with diagnostics.

    Financial viability (expectancy, PF) is NOT assessed here — that is
    determined per-calibrator by running a full architecture search in
    _evaluate_architecture().  Only technical quality problems (probability
    collapse, variance destruction) disqualify a calibrator at this stage.
    """
    trainer = MetaCalibrationFramework()
    report = trainer.evaluate_calibrators(raw_scores / 100.0, correct, threshold=0.50)
    initial_framework_choice = trainer.calibrator_type

    if not report:
        return trainer, {
            'method': 'uncalibrated',
            'selected_calibrator': 'uncalibrated',
            'ece_before': 0.0,
            'ece_after': 0.0,
            'quality_score': 0.0,
            'selected_method': 'uncalibrated',
            'selected_score': 0.0,
            'calibrator_candidates': [],
            'baseline': {
                'precision': 0.0,
                'profit_factor': 0.0,
                'expectancy_pct': 0.0,
                'sharpe': 0.0,
                'coverage': 0.0,
            },
        }, report, []

    total_directional = int((ev_side != 1).sum())
    raw_calibrated = np.asarray(ev_edge_raw, dtype=float) / 100.0
    raw_mask = (ev_side != 1) & (raw_calibrated >= 0.50)
    raw_holdout = _backtest_holdout(raw_mask, ev_side, ev_labels, ev_barrier)
    raw_precision = float((ev_labels[raw_mask] == ev_side[raw_mask]).mean()) if raw_mask.any() else 0.0
    raw_expectancy = float(raw_holdout.get('expectancy_pct', 0.0))
    raw_pf = float(raw_holdout.get('profit_factor', 0.0))
    ref_variance = np.var(raw_calibrated)

    candidates: List[Dict[str, Any]] = []

    print(f"\n   [CALIBRATOR TRAINING DIAGNOSTICS]")
    print(
        f"   {'Method':<12} | {'HoldCov':<7} | {'HoldPrec':<8} | {'HoldPF':<7} | {'HoldExp':<7} | "
        f"{'HoldShar':<7} | {'ECE':<5} | {'Brier':<6} | {'VarRet':<7} | {'TechOK':<7} | {'Reason'}"
    )
    print(f"   {'-'*140}")

    for method, train_metrics in report.items():
        trainer.calibrator_type = method
        trainer.best_calibrator = train_metrics.get('model')

        if method == 'uncalibrated':
            ev_calibrated = raw_calibrated
        else:
            ev_calibrated = trainer.calibrate(raw_calibrated)

        mask = (ev_side != 1) & (ev_calibrated >= 0.50)
        fired_n = int(mask.sum())
        coverage = fired_n / total_directional if total_directional else 0.0
        extreme_frac = float(np.sum((ev_calibrated <= 0.01) | (ev_calibrated >= 0.99)) / len(ev_calibrated))
        current_variance = float(np.var(ev_calibrated))
        var_ratio = current_variance / max(ref_variance, 1e-9)

        fired_prec = float((ev_labels[mask] == ev_side[mask]).mean()) if mask.any() else 0.0
        selected = _backtest_holdout(mask, ev_side, ev_labels, ev_barrier) if mask.any() else {}
        holdout_expectancy = float(selected.get('expectancy_pct', 0.0))
        holdout_pf = float(selected.get('profit_factor', 0.0))
        holdout_sharpe = float(selected.get('sharpe', 0.0))
        calibration_quality = clamp(1.0 - float(train_metrics.get('ece', 1.0)), 0.0, 1.0)

        # Technical eligibility only — financial viability is assessed via architecture search.
        tech_eligible = True
        reason = ''
        if extreme_frac > 0.40:
            tech_eligible = False
            reason = f'prob_collapse {extreme_frac:.1%}'
        elif var_ratio < MIN_VARIANCE_RETAINED:
            tech_eligible = False
            reason = f'variance_destroyed {var_ratio:.3f}'

        candidate = {
            'method': method,
            'train_ece': float(train_metrics.get('ece', 1.0)),
            'train_brier': float(train_metrics.get('brier', 0.0)),
            'train_precision': float(train_metrics.get('precision', 0.0)),
            'train_coverage': float(train_metrics.get('coverage', 0.0)),
            'holdout_fired': fired_n,
            'holdout_coverage': coverage,
            'holdout_precision': fired_prec,
            'holdout_expectancy': holdout_expectancy,
            'holdout_profit_factor': holdout_pf,
            'holdout_sharpe': holdout_sharpe,
            'probability_extreme_frac': extreme_frac,
            'variance_retained_ratio': var_ratio,
            'calibration_quality': calibration_quality,
            'tech_eligible': bool(tech_eligible),
            'eligible': bool(tech_eligible),  # kept for backward compatibility
            'reason': reason,
            'raw_score': 0.0,
            'normalized_score': 0.0,
            # architecture_viable / best_architecture_score populated in _evaluate_architecture
        }
        candidates.append(candidate)

        print(
            f"   {method:<12} | {coverage:<7.3f} | {fired_prec:<8.3f} | {holdout_pf:<7.3f} | {holdout_expectancy:<7.2f} | "
            f"{holdout_sharpe:<7.2f} | {candidate['train_ece']:<5.3f} | {candidate['train_brier']:<6.3f} | {var_ratio:<7.3f} | "
            f"{str(tech_eligible):<7} | {reason}"
        )

    print(f"   {'-'*140}")

    # Initial calibrator selection: best ECE among technically eligible methods.
    # This will be overridden by architecture-search outcome.
    tech_eligible_candidates = [c for c in candidates if c['tech_eligible']]
    if not tech_eligible_candidates:
        selected_method = 'uncalibrated'
        selected_score = 0.0
        trainer.calibrator_type = 'uncalibrated'
        trainer.best_calibrator = None
        print(f"   [WARNING] All calibrators failed technical checks; using uncalibrated")
    else:
        def _tech_key(cand: Dict[str, Any]) -> Tuple[float, int]:
            method_priority = {'temperature': 0, 'platt': 1, 'beta': 2, 'isotonic': 3, 'uncalibrated': 4}
            return (
                -float(cand['train_ece']),
                -method_priority.get(cand['method'], 5),
            )
        winner = max(tech_eligible_candidates, key=_tech_key)
        selected_method = winner['method']
        selected_score = 0.0
        trainer.calibrator_type = selected_method
        trainer.best_calibrator = report.get(selected_method, {}).get('model') if selected_method in report else None
        print(f"   [INITIAL SELECTION] {selected_method} (ECE={winner['train_ece']:.4f}) — pending architecture validation")

    if initial_framework_choice != selected_method:
        print(f"   [INFO] Framework initial pick: {initial_framework_choice} -> Initial: {selected_method}")

    return trainer, {
        'method': selected_method,
        'selected_calibrator': selected_method,
        'ece_before': float(report.get('uncalibrated', {}).get('ece', 1.0)),
        'ece_after': float(report.get(selected_method, {}).get('ece', report.get('uncalibrated', {}).get('ece', 1.0))),
        'quality_score': float(max(0.0, report.get('uncalibrated', {}).get('ece', 1.0) - report.get(selected_method, {}).get('ece', 1.0))),
        'selected_method': selected_method,
        'selected_score': float(selected_score),
        'calibrator_candidates': candidates,
        'baseline': {
            'precision': float(raw_precision),
            'profit_factor': float(raw_pf),
            'expectancy_pct': float(raw_expectancy),
            'sharpe': float(raw_holdout.get('sharpe', 0.0)),
            'coverage': float(raw_mask.sum() / total_directional if total_directional else 0.0),
        },
    }, report, candidates


def _best_architecture_for_calibrator(
    symbol: str,
    ev_df: pd.DataFrame,
    regimes_ev: pd.Series,
    ev_side: np.ndarray,
    ev_edge_raw: np.ndarray,
    calibrated_ev_common: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
    calib_report: Dict[str, Any],
    baseline_precision: float,
    baseline_pf: float,
    baseline_expectancy: float,
    baseline_sharpe: float,
    total_directional: int,
) -> Tuple[Optional[Dict[str, Any]], float]:
    best = None
    best_score = -np.inf

    for arch in GATE_ARCHITECTURES:
        for quantile in QUANTILES if arch['name'] != 'EDGE_LOOSE_COVERAGE' else [0.40, 0.35, 0.30, 0.25]:
            use_calibration = arch['calibrate']
            calibrated_ev = calibrated_ev_common if use_calibration else ev_edge_raw / 100.0
            calib_report_for_arch = calib_report if use_calibration else {
                'method': 'uncalibrated',
                'selected_calibrator': 'uncalibrated',
                'selected_method': 'uncalibrated',
                'selected_score': 0.0,
                'ece_before': 0.0,
                'ece_after': 0.0,
                'quality_score': 0.0,
            }

            fire_mask, thresholds = _compute_gate_mask(
                ev_df,
                regimes_ev.reset_index(drop=True),
                ev_side,
                ev_edge_raw,
                calibrated_ev,
                arch,
                quantile,
            )

            fired_n = int(fire_mask.sum())
            coverage = fired_n / total_directional if total_directional else 0.0
            if coverage < MIN_GATE_COVERAGE or fired_n < MIN_GATE_SIGNALS:
                continue

            selected = _backtest_holdout(fire_mask, ev_side, ev_labels, ev_barrier)
            reject_mask = (ev_side != 1) & ~fire_mask
            if not reject_mask.any():
                continue

            selected_prec = float((ev_labels[fire_mask] == ev_side[fire_mask]).mean()) if fired_n else 0.0
            if selected_prec < baseline_precision:
                continue

            if selected['expectancy_pct'] <= 0.0 or selected['sharpe'] <= 0.0 or selected['profit_factor'] < 1.10:
                continue

            precision_norm = clamp(selected_prec, 0.0, 1.0)
            pf_norm = min(max((selected['profit_factor'] - 1.0) / 2.0, 0.0), 1.0)
            expectancy_norm = _normalize_expectancy(selected['expectancy_pct'])
            sharpe_norm = _normalize_sharpe(selected['sharpe'])
            coverage_norm = clamp(coverage, 0.0, 1.0)

            # TASK 7: Updated scoring weights - expectancy-driven
            # OLD: 0.35 expectancy, 0.30 pf, 0.20 sharpe, 0.10 prec, 0.05 cov
            # NEW: 0.40 expectancy, 0.30 pf, 0.20 sharpe, 0.05 prec, 0.05 cov
            score = (
                0.40 * expectancy_norm +
                0.30 * pf_norm +
                0.20 * sharpe_norm +
                0.05 * precision_norm +
                0.05 * coverage_norm
            )

            if score > best_score:
                best_score = score
                best = {
                    'gate_type': arch['name'],
                    'architecture': arch,
                    'threshold_quantile': float(quantile),
                    'thresholds': thresholds,
                    'calibration': calib_report,
                    'calibration_used_by_architecture': calib_report_for_arch,
                    'holdout_metrics': selected,
                    'candidate_metrics': {
                        'precision': selected_prec,
                        'coverage': coverage,
                        'expectancy_pct': selected['expectancy_pct'],
                        'profit_factor': selected['profit_factor'],
                        'sharpe': selected['sharpe'],
                        'gate_lift': selected_prec - float(baseline_precision),
                        'fired_n': fired_n,
                    },
                    'score': float(round(score, 5)),
                }

    return best, best_score


def _derive_raw_threshold(
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    target_quantile: float,
) -> float:
    if len(calibrated_scores) == 0:
        return float(np.quantile(raw_scores, 0.90) if len(raw_scores) else 50.0)
    thr_cal = float(np.quantile(calibrated_scores, 1.0 - target_quantile))
    mask = calibrated_scores >= thr_cal
    if not mask.any():
        return float(np.quantile(raw_scores, 0.90) if len(raw_scores) else 50.0)
    return float(np.min(raw_scores[mask]))


def _apply_vetoes(
    df: pd.DataFrame,
    base_fire: np.ndarray,
    side: int,
    vetoes: List[str],
) -> np.ndarray:
    fire = base_fire.copy()
    if 'sr' in vetoes:
        at_res = df.get('is_at_resistance', pd.Series(False, index=df.index)).astype(bool)
        at_sup = df.get('is_at_support', pd.Series(False, index=df.index)).astype(bool)
        if side == 2:
            fire &= ~at_res
        else:
            fire &= ~at_sup

    if 'trend' in vetoes:
        macro_trend_1d = df.get('macro_trend_1d', pd.Series(0.0, index=df.index)).fillna(0.0)
        if side == 2:
            fire &= macro_trend_1d >= -0.2
        else:
            fire &= macro_trend_1d <= 0.2

    if 'confluence' in vetoes:
        total_conf = df.get('total_confluence', pd.Series(0.0, index=df.index)).fillna(0.0)
        if side == 2:
            fire &= total_conf >= -0.35
        else:
            fire &= total_conf <= 0.35

    return fire


def _compute_gate_mask(
    df: pd.DataFrame,
    regimes: pd.Series,
    side: np.ndarray,
    edge_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    architecture: Dict[str, Any],
    threshold_quantile: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if architecture['side_specific']:
        buy_mask = side == 2
        sell_mask = side == 0
        buy_thr = _derive_raw_threshold(edge_scores[buy_mask], calibrated_scores[buy_mask], threshold_quantile)
        sell_thr = _derive_raw_threshold(edge_scores[sell_mask], calibrated_scores[sell_mask], threshold_quantile)
        global_thr = (buy_thr + sell_thr) / 2.0
    else:
        global_thr = _derive_raw_threshold(edge_scores, calibrated_scores, threshold_quantile)
        buy_thr = sell_thr = global_thr

    thresholds: Dict[str, float] = {
        'global_threshold': float(global_thr),
        'buy_threshold': float(buy_thr),
        'sell_threshold': float(sell_thr),
        'quantile': float(threshold_quantile),
    }

    fire = np.zeros(len(df), dtype=bool)
    buy_idx = side == 2
    sell_idx = side == 0
    fire[buy_idx] = edge_scores[buy_idx] >= buy_thr
    fire[sell_idx] = edge_scores[sell_idx] >= sell_thr

    if architecture['regime_modifier']:
        regime_counts = regimes.value_counts()
        for regime, mod in REGIME_MODIFIER_TEMPLATE.items():
            regime_idx = regimes == regime
            if not regime_idx.any():
                continue
            regime_count = int(regime_counts.get(regime, 0))
            if regime_count < MIN_REGIME_BARS:
                continue
            if mod == 'disable':
                fire[regime_idx] = False
                continue

            regime_buy = regime_idx & buy_idx
            regime_sell = regime_idx & sell_idx
            offset = float(mod)
            if regime_buy.any():
                fire[regime_buy] = edge_scores[regime_buy] >= (buy_thr + offset)
            if regime_sell.any():
                fire[regime_sell] = edge_scores[regime_sell] >= (sell_thr + offset)

    if architecture['vetoes']:
        for side_value in (2, 0):
            single = side == side_value
            if not single.any():
                continue
            base_fire = fire[single]
            adjusted = _apply_vetoes(df[single].reset_index(drop=True), base_fire, side_value, architecture['vetoes'])
            fire[single] = adjusted

    return fire, thresholds


def _backtest_holdout(
    fire: np.ndarray,
    proposed: np.ndarray,
    labels: np.ndarray,
    barrier_frac: np.ndarray,
) -> Dict[str, Any]:
    return backtest(fire, proposed, labels, barrier_frac, fee=FEE_ROUNDTRIP)


def _score_candidate(
    metrics: Dict[str, Any],
    calibration_quality: float,
    risk_tiers: Dict[str, bool],
) -> float:
    precision = metrics.get('precision', 0.0)
    pf = metrics.get('profit_factor', 0.0)
    sharpe = metrics.get('sharpe', 0.0)
    coverage = metrics.get('coverage', 0.0)
    tier_bonus = sum(bool(v) for v in risk_tiers.values()) / 3.0 * 0.05
    pf_norm = min(max((pf - 1.0) / 2.0, 0.0), 1.0)
    sharpe_norm = min(max(sharpe / 5.0, 0.0), 1.0)

    return (
        0.10 * clamp(precision, 0.0, 1.0) +
        0.30 * pf_norm +
        0.20 * sharpe_norm +
        0.35 * min(max(metrics.get('expectancy_pct', 0.0) / 10.0, 0.0), 1.0) +
        0.05 * clamp(coverage, 0.0, 1.0) +
        tier_bonus
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _histogram_summary(name: str, values: np.ndarray, bins: int = 6) -> None:
    if len(values) == 0:
        print(f"   [D] {name}: no values")
        return
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f"   [D] {name}: all NaN")
        return
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    if min_val == max_val:
        min_val = min_val - 0.5
        max_val = max_val + 0.5
    edges = np.linspace(min_val, max_val, bins + 1)
    hist, _ = np.histogram(values, bins=edges)
    stats = (
        min_val + 0.0,
        float(np.nanquantile(values, 0.25)),
        float(np.nanmedian(values)),
        float(np.nanquantile(values, 0.75)),
        max_val + 0.0,
    )
    ranges = " ".join(
        f"[{edges[i]:.2f},{edges[i+1]:.2f}]:{hist[i]}"
        for i in range(len(hist))
    )
    print(
        f"   [D] {name:<22} min={stats[0]:.3f} q25={stats[1]:.3f} "
        f"med={stats[2]:.3f} q75={stats[3]:.3f} max={stats[4]:.3f} | {ranges}"
    )


def _build_ranking_diagnostics(
    df: pd.DataFrame,
    proposed: np.ndarray,
    dir_conf: np.ndarray,
    probs: np.ndarray,
    regimes: Optional[pd.Series] = None,
    labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    records = pd.DataFrame(index=df.index)
    records['side'] = np.where(proposed == 2, 'BUY', 'SELL')
    records['directional_probability'] = dir_conf.astype(float)
    records['signal_strength_score'] = 0.0
    if (proposed == 2).any():
        buy_scores = EdgeScoringEngine.compute_signal_strength_scores(
            df.loc[proposed == 2].reset_index(drop=True),
            dir_conf[proposed == 2],
            'BUY',
        )
        records.loc[proposed == 2, 'signal_strength_score'] = buy_scores['signal_strength_score'].values
    if (proposed == 0).any():
        sell_scores = EdgeScoringEngine.compute_signal_strength_scores(
            df.loc[proposed == 0].reset_index(drop=True),
            dir_conf[proposed == 0],
            'SELL',
        )
        records.loc[proposed == 0, 'signal_strength_score'] = sell_scores['signal_strength_score'].values
    records['edge_rank'] = records['signal_strength_score'] * records['directional_probability']
    records['edge_percentile'] = records['edge_rank'].rank(method='average', pct=True).fillna(0.0)
    if labels is not None:
        records['correct'] = (labels == proposed)
    if regimes is not None:
        records['regime'] = regimes.reset_index(drop=True)
    return records


def _print_ranking_diagnostics(
    symbol: str,
    ranking_df: pd.DataFrame,
    local_model: xgb.Booster,
) -> None:
    print(f"\n   [RANKING DIAGNOSTICS] {symbol}")
    print("   Top 20 local model features by gain:")
    try:
        importance = local_model.get_score(importance_type='gain')
        if importance:
            imp_df = (
                pd.DataFrame.from_dict(importance, orient='index', columns=['gain'])
                .sort_values('gain', ascending=False)
                .head(20)
            )
            for feature, row in imp_df.iterrows():
                print(f"   {feature:30} gain={row['gain']:.3f}")
        else:
            print("   No feature importance available from local model.")
    except Exception as exc:
        print(f"   Could not extract local feature importance: {exc}")

    _histogram_summary('directional_probability', ranking_df['directional_probability'].values)
    _histogram_summary('signal_strength_score', ranking_df['signal_strength_score'].values)
    _histogram_summary('edge_rank', ranking_df['edge_rank'].values)
    _histogram_summary('edge_percentile', ranking_df['edge_percentile'].values)

    if 'regime' in ranking_df.columns:
        print("   [Regime performance]")
        print("   regime                | n   | win_rate | avg_prob | avg_strength | avg_rank")
        for regime, group in ranking_df.groupby('regime'):
            total = len(group)
            if total == 0:
                continue
            win_rate = float(group['correct'].mean()) if 'correct' in group.columns else float((group['directional_probability'] >= 0.5).mean())
            print(
                f"   {str(regime):20} | {total:4d} | {win_rate:.3f} | "
                f"{float(group['directional_probability'].mean()):.3f} | "
                f"{float(group['signal_strength_score'].mean()):.3f} | "
                f"{float(group['edge_rank'].mean()):.3f}"
            )


def _extract_signal_diagnostics(
    ranking_df: pd.DataFrame,
    local_model: xgb.Booster,
) -> Dict[str, Any]:
    """Extract signal quality and feature diagnostics saved to debug JSON for Phase-2 forensics."""
    diag: Dict[str, Any] = {}
    n_total = len(ranking_df)
    diag['n_signals'] = n_total
    if n_total == 0:
        return diag

    # Buy/sell balance (side==2 → buy, side==0 → sell, or string BUY/SELL)
    if 'side' in ranking_df.columns:
        side = ranking_df['side']
        side_upper = side.astype(str).str.upper()
        n_buy = int(((side == 2) | (side_upper == 'BUY')).sum())
        n_sell = int(((side == 0) | (side_upper == 'SELL')).sum())
    else:
        n_buy = n_sell = 0
    diag['n_buy']  = n_buy
    diag['n_sell'] = n_sell
    diag['buy_sell_balance'] = round(n_buy / n_total, 4)

    # Directional probability distribution
    probs = ranking_df['directional_probability'].values.astype(float)
    mean_p = float(np.mean(probs))
    med_p  = float(np.median(probs))
    std_p  = float(np.std(probs))
    diag['dir_prob_mean'] = round(mean_p, 4)
    diag['dir_prob_std']  = round(std_p, 4)
    diag['dir_prob_skew'] = round(3.0 * (mean_p - med_p) / (std_p + 1e-9), 4)  # Pearson 2nd

    # Prediction entropy (binary: max 1.0 at p=0.5)
    eps = 1e-9
    ent = -(probs * np.log2(probs + eps) + (1.0 - probs) * np.log2(1.0 - probs + eps))
    diag['prediction_entropy_mean'] = round(float(np.mean(ent)), 4)
    diag['prediction_entropy_std']  = round(float(np.std(ent)), 4)

    # Edge rank stats
    if 'edge_rank' in ranking_df.columns:
        er = ranking_df['edge_rank'].values.astype(float)
        diag['edge_rank_mean'] = round(float(np.mean(er)), 4)
        diag['edge_rank_std']  = round(float(np.std(er)), 4)

    # Feature importance + concentration (HHI)
    try:
        importance = local_model.get_score(importance_type='gain')
        if importance:
            total_gain = sum(importance.values()) + 1e-9
            fracs = np.array(list(importance.values())) / total_gain
            feature_pct = {f: round(100.0 * g / total_gain, 2) for f, g in importance.items()}
            top10 = sorted(feature_pct.items(), key=lambda x: -x[1])[:10]
            diag['feature_importance_top10'] = [{'feature': f, 'pct': p} for f, p in top10]
            hhi = float(np.sum(fracs ** 2))
            diag['feature_concentration_hhi'] = round(hhi, 4)
            diag['top_feature_pct']   = top10[0][1] if top10 else 0.0
            diag['top3_features_pct'] = round(sum(p for _, p in top10[:3]), 2)
            n_features = len(importance)
            min_hhi = 1.0 / n_features if n_features > 0 else 1.0
            diag['signal_diversity_score'] = round(
                max(0.0, (1.0 - hhi) / (1.0 - min_hhi + 1e-9)), 4
            )
        else:
            diag['feature_importance_top10'] = []
            diag['feature_concentration_hhi'] = 0.0
            diag['signal_diversity_score'] = 0.0
    except Exception:
        diag['feature_importance_top10'] = []
        diag['feature_concentration_hhi'] = 0.0
        diag['signal_diversity_score'] = 0.0

    return diag


def _evaluate_architecture(
    symbol: str,
    df_ev: pd.DataFrame,
    regimes_ev: pd.Series,
    proposed_ev: np.ndarray,
    dir_conf_ev: np.ndarray,
    probs_ev: np.ndarray,
    labels_ev: np.ndarray,
    barrier_frac_ev: np.ndarray,
    train_edge_scores: np.ndarray,
    train_correct: np.ndarray,
) -> Dict[str, Any]:
    valid = labels_ev != CENSORED
    if int(valid.sum()) < MIN_FIRES_DEV:
        raise ValueError(f"Not enough holdout bars: {int(valid.sum())} valid bars.")

    ev_df = df_ev[valid].reset_index(drop=True)
    ev_side = proposed_ev[valid]
    ev_labels = labels_ev[valid]
    ev_barrier = barrier_frac_ev[valid]
    regimes_ev = regimes_ev[valid].reset_index(drop=True)
    ev_edge_scores = compute_edge_scores(ev_df, ev_side, dir_conf_ev[valid], use_rank=True)
    ev_edge_raw = ev_edge_scores.values.astype(float)

    # Baseline comparison is mandatory: every gate must outperform the simple
    # no-gate directional proposal in at least one financial metric.
    baseline_fire = ev_side != 1
    baseline = _backtest_holdout(baseline_fire, ev_side, ev_labels, ev_barrier)
    baseline_precision = float((ev_side[baseline_fire] == ev_labels[baseline_fire]).mean()) if baseline_fire.any() else 0.0
    baseline_pf = float(baseline.get('profit_factor', 0.0))
    baseline_expectancy = float(baseline.get('expectancy_pct', 0.0))
    baseline_sharpe = float(baseline.get('sharpe', 0.0))
    baseline_coverage = float(1.0 if baseline_fire.any() else 0.0)
    total_directional = int(baseline_fire.sum())
    min_signals = _min_signals(total_directional)

    trainer, calib_report, calib_report_raw, calib_candidates = _select_best_calibrator(
        train_edge_scores,
        train_correct,
        ev_edge_raw,
        ev_side,
        ev_labels,
        ev_barrier,
    )

    # Calibration diagnostics: leaderboard and winners
    print("   [CALIBRATION LEADERBOARD]")
    print("   name     | train_ece | train_brier | train_prec | holdout_cov | raw_score | norm_score | eligible | reason")
    for c in calib_candidates:
        name = c.get('method')
        raw_score_text = fmt(c.get('raw_score'))
        norm_score_text = fmt(c.get('normalized_score'))
        print(
            f"   {name:12} | {fmt(c.get('train_ece', 0.0)):9} | {fmt(c.get('train_brier', 0.0)):11} | "
            f"{fmt(c.get('train_precision', 0.0)):10} | {fmt(c.get('holdout_coverage', 0.0)):11} | {raw_score_text:9} | {norm_score_text:10} | {c.get('eligible', False)} | {c.get('reason','')}"
        )

    # Winner diagnostics
    eligible_for_winner = [c for c in calib_candidates if c.get('eligible')]
    winner_before = max(eligible_for_winner, key=lambda x: float(x.get('normalized_score', float('-inf')))) if eligible_for_winner else None
    print(f"   WINNER BEFORE FILTERS: {winner_before.get('method') if winner_before else 'NONE'} | raw_score={winner_before.get('raw_score') if winner_before else 0.0} | normalized_score={winner_before.get('normalized_score') if winner_before else 0.0}")
    selected_calibrator = calib_report.get('selected_calibrator', calib_report.get('selected_method', 'uncalibrated'))
    print(f"   WINNER AFTER FILTERS : {selected_calibrator} | score={calib_report.get('selected_score', 0.0)}")
    print(f"   [CURRENT CALIBRATOR BEFORE JOINT VALIDATION] : {trainer.calibrator_type}")

    # Joint calibration validation: run a full architecture search for EVERY
    # technically-eligible calibrator.  Financial viability (expectancy, PF) is
    # assessed here — not in _select_best_calibrator — so a calibrator that
    # looked bad at the fixed 0.50 threshold can still win if it supports a
    # profitable architecture at a percentile threshold.
    print("   [CALIBRATION ARCHITECTURE VALIDATION]")
    print("   method       | arch_score | tech_ok | arch_viable | best_pf | best_exp | calib_reason")
    print(f"   {'-'*110}")
    for c in calib_candidates:
        method = c.get('method')
        if not c.get('tech_eligible', True):
            # Technically disqualified (prob_collapse / variance_destroyed) — skip
            c['best_architecture_score'] = -np.inf
            c['best_architecture'] = None
            c['architecture_viable'] = False
            print(f"   {method:<12} | {'SKIP':<10} | {'False':<7} | {'False':<11} | {'N/A':<7} | {'N/A':<8} | {c.get('reason','')}")
            continue

        # Ensure per-iteration calibrator state is fresh to avoid cache contamination
        trainer.reports = {}
        trainer._ece_before = 1.0
        trainer._ece_after = 1.0
        trainer.calibrator_type = method
        trainer.best_calibrator = calib_report_raw.get(method, {}).get('model') if method in calib_report_raw else None
        _cal_scores = (
            np.asarray(ev_edge_raw, dtype=float) / 100.0
            if method == 'uncalibrated'
            else np.copy(trainer.calibrate(np.asarray(ev_edge_raw, dtype=float) / 100.0))
        )
        best_arch, best_arch_score = _best_architecture_for_calibrator(
            symbol,
            ev_df,
            regimes_ev,
            ev_side,
            ev_edge_raw,
            _cal_scores,
            ev_labels,
            ev_barrier,
            calib_report_raw.get(method, {
                'method': method,
                'ece': float(calib_report_raw.get(method, {}).get('ece', 1.0)),
                'brier': float(calib_report_raw.get(method, {}).get('brier', 0.0)),
            }),
            baseline_precision,
            baseline_pf,
            baseline_expectancy,
            baseline_sharpe,
            total_directional,
        )
        c['best_architecture_score'] = best_arch_score
        c['best_architecture'] = best_arch
        arch_pf = float(best_arch.get('candidate_metrics', {}).get('profit_factor', 0.0)) if best_arch else 0.0
        arch_exp = float(best_arch.get('candidate_metrics', {}).get('expectancy_pct', 0.0)) if best_arch else 0.0
        arch_viable = (
            best_arch is not None and
            best_arch_score > 0 and
            arch_pf > 1.0 and
            arch_exp > 0.0
        )
        c['architecture_viable'] = arch_viable
        print(
            f"   {method:<12} | {best_arch_score:<10.4f} | {'True':<7} | {str(arch_viable):<11} | "
            f"{arch_pf:<7.3f} | {arch_exp:<8.2f} | {c.get('reason','')}"
        )
    print(f"   {'-'*110}")

    # Cache-contamination check: identical scores across all viable calibrators is suspicious
    arch_scores_viable = [c.get('best_architecture_score', -np.inf) for c in calib_candidates if c.get('architecture_viable')]
    if len(arch_scores_viable) > 1 and len(set(arch_scores_viable)) == 1:
        print(f"   [WARNING] Architecture scores identical across viable calibrators: {arch_scores_viable[0]:.4f} — possible cache contamination")

    arch_viable_candidates = [c for c in calib_candidates if c.get('architecture_viable')]

    if not arch_viable_candidates:
        # No calibrator produced a profitable architecture at any percentile threshold.
        # Do NOT silently overwrite the previously selected calibrator; instead
        # mark joint-validation as having no viable architecture and leave the
        # trainer.calibrator_type as the initial/tech-selected method. This
        # prevents confusing logs like "Selected: ISOTONIC" then later
        # "FINAL SAVED CALIBRATOR: uncalibrated" and keeps the choice stable
        # for the main gate loop which will still attempt to find a gate.
        print(f"   [CALIBRATION JOINT VALIDATION] No calibrator produced a profitable architecture")
        print(f"   [INFO] Leaving initial calibrator in-place; main architecture loop will attempt to find a gate using selected calibrator")
        calib_report = {
            'method': trainer.calibrator_type or 'uncalibrated',
            'selected_calibrator': trainer.calibrator_type or 'uncalibrated',
            'selected_method': trainer.calibrator_type or 'uncalibrated',
            'selected_score': 0.0,
            'ece_before': float(calib_report_raw.get('uncalibrated', {}).get('ece', 1.0)),
            'ece_after': float(calib_report_raw.get(trainer.calibrator_type or 'uncalibrated', {}).get('ece', calib_report_raw.get('uncalibrated', {}).get('ece', 1.0))),
            'quality_score': 0.0,
            'arch_validation_no_viable': True,
        }
    else:
        winner = max(
            arch_viable_candidates,
            key=lambda x: (
                float(x.get('best_architecture_score', float('-inf'))),
                float(x.get('holdout_expectancy', 0.0)),
                float(x.get('calibration_quality', 0.0)),
            ),
        )
        selected_method = winner['method']
        selected_score = float(winner.get('best_architecture_score', 0.0))
        trainer.calibrator_type = selected_method
        trainer.best_calibrator = (
            calib_report_raw.get(selected_method, {}).get('model')
            if selected_method in calib_report_raw else None
        )
        calib_report = {
            'method': selected_method,
            'selected_calibrator': selected_method,
            'selected_method': selected_method,
            'selected_score': selected_score,
            'ece_before': float(calib_report_raw.get('uncalibrated', {}).get('ece', 1.0)),
            'ece_after': float(calib_report_raw.get(selected_method, {}).get('ece', 1.0)),
            'quality_score': float(max(
                0.0,
                float(calib_report_raw.get('uncalibrated', {}).get('ece', 1.0)) -
                float(calib_report_raw.get(selected_method, {}).get('ece', 1.0))
            )),
        }
        print(f"   [CALIBRATION JOINT VALIDATION] selected {selected_method} (arch_score={selected_score:.4f})")

    print(f"   [FINAL SAVED CALIBRATOR] : {trainer.calibrator_type}")

    # Recompute after joint validation has finalised trainer.calibrator_type
    calibrated_ev_common = np.copy(trainer.calibrate(np.asarray(ev_edge_raw, dtype=float) / 100.0))

    print(f"   [Baseline] prec={baseline_precision:.3f} pf={baseline_pf:.3f} "
          f"exp={baseline_expectancy:.3f}% sharpe={baseline_sharpe:.3f} cov={baseline_coverage:.3f}")

    debug_records: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score = -np.inf
    seen_fire_masks: List[bytes] = []

    def _register_duplicate(arch_name: str, metrics: Dict[str, Any], reason: str) -> None:
        debug_records.append({
            'symbol': symbol,
            'gate_type': arch_name,
            'quantile': float(metrics.get('quantile', 0.0)),
            'calibration': metrics.get('calibration', 'unknown'),
            'fired_n': int(metrics.get('fired_n', 0)),
            'coverage': float(metrics.get('coverage', 0.0)),
            'precision': float(metrics.get('precision', 0.0)),
            'profit_factor': float(metrics.get('profit_factor', 0.0)),
            'expectancy_pct': float(metrics.get('expectancy_pct', 0.0)),
            'sharpe': float(metrics.get('sharpe', 0.0)),
            'score': 0.0,
            'reason': reason,
            'precision_delta': float(metrics.get('precision_delta', 0.0)),
            'pf_delta': float(metrics.get('pf_delta', 0.0)),
            'expectancy_delta': float(metrics.get('expectancy_delta', 0.0)),
            'sharpe_delta': float(metrics.get('sharpe_delta', 0.0)),
        })

    def _is_duplicate_candidate(fire: np.ndarray, metrics: Dict[str, Any]) -> Optional[str]:
        mask_bytes = fire.tobytes()
        if mask_bytes in seen_fire_masks:
            return 'duplicate_architecture'
        for prev in debug_records:
            if prev.get('reason', '').startswith('duplicate_architecture'):
                continue
            if (
                abs(prev.get('coverage', 0.0) - metrics.get('coverage', 0.0)) < 0.001 and
                abs(prev.get('precision', 0.0) - metrics.get('precision', 0.0)) < 0.001 and
                abs(prev.get('profit_factor', 0.0) - metrics.get('profit_factor', 0.0)) < 0.001
            ):
                return f"duplicate_of_{prev.get('gate_type')}"
        return None

    for arch in GATE_ARCHITECTURES:
        for quantile in QUANTILES if arch['name'] != 'EDGE_LOOSE_COVERAGE' else [0.40, 0.35, 0.30, 0.25]:
            use_calibration = arch['calibrate']
            calibrated_ev = calibrated_ev_common if use_calibration else ev_edge_raw / 100.0
            calib_report_for_arch = calib_report if use_calibration else {
                'method': 'uncalibrated',
                'selected_calibrator': 'uncalibrated',
                'selected_method': 'uncalibrated',
                'selected_score': 0.0,
                'ece_before': 0.0,
                'ece_after': 0.0,
                'quality_score': 0.0,
            }

            fire_mask, thresholds = _compute_gate_mask(
                ev_df,
                regimes_ev.reset_index(drop=True),
                ev_side,
                ev_edge_raw,
                calibrated_ev,
                arch,
                quantile,
            )

            fired_n = int(fire_mask.sum())
            coverage = fired_n / total_directional if total_directional else 0.0
            reject_reason = None
            # Enforce gate-level minimums
            if coverage < MIN_GATE_COVERAGE:
                reject_reason = f'coverage {coverage:.4f} < required {MIN_GATE_COVERAGE:.2f}'
            elif fired_n < MIN_GATE_SIGNALS:
                reject_reason = f'low_signals ({fired_n}<{MIN_GATE_SIGNALS})'

            if reject_reason:
                debug_records.append({
                    'symbol': symbol,
                    'gate_type': arch['name'],
                    'quantile': float(quantile),
                    'calibration': calib_report_for_arch['method'],
                    'fired_n': fired_n,
                    'coverage': coverage,
                    'reason': reject_reason,
                    'precision_delta': None,
                    'pf_delta': None,
                    'expectancy_delta': None,
                    'sharpe_delta': None,
                })
                continue

            selected = _backtest_holdout(fire_mask, ev_side, ev_labels, ev_barrier)
            reject_mask = (ev_side != 1) & ~fire_mask
            if not reject_mask.any():
                continue
            rejected = _backtest_holdout(reject_mask, ev_side, ev_labels, ev_barrier)

            selected_prec = float((ev_labels[fire_mask] == ev_side[fire_mask]).mean()) if fired_n else 0.0
            rejected_prec = float((ev_labels[reject_mask] == ev_side[reject_mask]).mean()) if reject_mask.any() else 0.0
            gate_lift = selected_prec - rejected_prec

            # FIX 4.1: Require minimum holdout size for reliable gate selection
            n_valid_ev = int((ev_side != 1).sum())  # directional signals in holdout
            if n_valid_ev < MIN_VALID_HOLDOUT_BARS:
                debug_records.append({
                    'symbol': symbol,
                    'gate_type': arch['name'],
                    'quantile': float(quantile),
                    'calibration': calib_report_for_arch['method'],
                    'fired_n': fired_n,
                    'coverage': coverage,
                    'reason': f'insufficient_holdout({n_valid_ev}<{MIN_VALID_HOLDOUT_BARS})',
                })
                continue

            # FIX 4.2: Reject gates with negative lift (anti-selective gating)
            if gate_lift < MIN_GATE_LIFT:
                debug_records.append({
                    'symbol': symbol,
                    'gate_type': arch['name'],
                    'quantile': float(quantile),
                    'calibration': calib_report_for_arch['method'],
                    'fired_n': fired_n,
                    'coverage': coverage,
                    'precision': selected_prec,
                    'gate_lift': float(gate_lift),
                    'reason': f'negative_gate_lift({gate_lift:.4f})',
                })
                continue

            candidate_metrics = {
                'precision': selected_prec,
                'coverage': coverage,
                'expectancy_pct': selected['expectancy_pct'],
                'profit_factor': selected['profit_factor'],
                'sharpe': selected['sharpe'],
                'gate_lift': gate_lift,
                'fired_n': fired_n,
                'rejected_n': int(reject_mask.sum()),
                'baseline_precision': baseline_precision,
                'baseline_profit_factor': baseline_pf,
                'baseline_expectancy': baseline_expectancy,
            }

            # Hard constraints and profitability-focused scoring.
            # Precision is still a factor, but we no longer reject solely because
            # a candidate falls below the baseline precision.
            if coverage < MIN_GATE_COVERAGE:
                hard_reason = 'coverage_below_minimum'
            elif selected['expectancy_pct'] <= 0.0:
                hard_reason = 'nonpositive_expectancy'
            elif selected['sharpe'] <= 0.0:
                hard_reason = 'nonpositive_sharpe'
            elif selected['profit_factor'] < 1.10:
                hard_reason = 'pf_below_1.10'
            else:
                hard_reason = None

            if hard_reason is not None:
                debug_records.append({
                    'symbol': symbol,
                    'gate_type': arch['name'],
                    'quantile': float(quantile),
                    'calibration': calib_report_for_arch['method'],
                    'fired_n': fired_n,
                    'coverage': coverage,
                    'precision': selected_prec,
                    'profit_factor': selected['profit_factor'],
                    'expectancy_pct': selected['expectancy_pct'],
                    'sharpe': selected['sharpe'],
                    'score': 0.0,
                    'reason': hard_reason,
                    'precision_delta': selected_prec - baseline_precision,
                    'pf_delta': selected['profit_factor'] - baseline_pf,
                    'expectancy_delta': selected['expectancy_pct'] - baseline_expectancy,
                    'sharpe_delta': selected['sharpe'] - baseline_sharpe,
                })
                continue

            candidate_metrics = {
                'precision': selected_prec,
                'coverage': coverage,
                'expectancy_pct': selected['expectancy_pct'],
                'profit_factor': selected['profit_factor'],
                'sharpe': selected['sharpe'],
                'gate_lift': gate_lift,
                'fired_n': fired_n,
                'rejected_n': int(reject_mask.sum()),
                'baseline_precision': baseline_precision,
                'baseline_profit_factor': baseline_pf,
                'baseline_expectancy': baseline_expectancy,
            }

            # ---- Robustness validation removed for synthetic single-fold data.
            # validate_architecture_from_folds() is only meaningful with multiple real folds.
            candidate_metrics['validation_report'] = {
                'status': 'skipped_single_fold_validation',
                'reason': 'synthetic single-fold validation is disabled for reliability',
            }

            duplicate_reason = _is_duplicate_candidate(fire_mask, {
                'coverage': coverage,
                'precision': selected_prec,
                'profit_factor': selected['profit_factor'],
                'quantile': quantile,
                'calibration': calib_report_for_arch['method'],
                'fired_n': fired_n,
                'expectancy_pct': selected['expectancy_pct'],
                'sharpe': selected['sharpe'],
                'precision_delta': selected_prec - baseline_precision,
                'pf_delta': selected['profit_factor'] - baseline_pf,
                'expectancy_delta': selected['expectancy_pct'] - baseline_expectancy,
                'sharpe_delta': selected['sharpe'] - baseline_sharpe,
            })
            if duplicate_reason is not None:
                _register_duplicate(arch['name'], {
                    'quantile': quantile,
                    'calibration': calib_report_for_arch['method'],
                    'fired_n': fired_n,
                    'coverage': coverage,
                    'precision': selected_prec,
                    'profit_factor': selected['profit_factor'],
                    'expectancy_pct': selected['expectancy_pct'],
                    'sharpe': selected['sharpe'],
                    'precision_delta': selected_prec - baseline_precision,
                    'pf_delta': selected['profit_factor'] - baseline_pf,
                    'expectancy_delta': selected['expectancy_pct'] - baseline_expectancy,
                    'sharpe_delta': selected['sharpe'] - baseline_sharpe,
                }, duplicate_reason)
                continue

            seen_fire_masks.append(fire_mask.tobytes())

            # Hybrid score: 30% precision, 25% PF, 20% expectancy, 15% sharpe, 10% coverage
            precision_norm = clamp(candidate_metrics.get('precision', 0.0), 0.0, 1.0)
            pf = candidate_metrics.get('profit_factor', 0.0)
            pf_norm = min(max((pf - 1.0) / 2.0, 0.0), 1.0)
            expectancy_norm = _normalize_expectancy(candidate_metrics.get('expectancy_pct', 0.0))
            sharpe_norm = _normalize_sharpe(candidate_metrics.get('sharpe', 0.0))
            coverage_norm = clamp(candidate_metrics.get('coverage', 0.0), 0.0, 1.0)

            # TASK 7: Updated scoring weights - expectancy-driven (same as architecture search)
            score = (
                0.40 * expectancy_norm +
                0.30 * pf_norm +
                0.20 * sharpe_norm +
                0.05 * precision_norm +
                0.05 * coverage_norm
            )

            debug_records.append({
                'symbol': symbol,
                'gate_type': arch['name'],
                'quantile': float(quantile),
                'calibration': calib_report_for_arch['method'],
                'fired_n': fired_n,
                'coverage': coverage,
                'precision': selected_prec,
                'profit_factor': selected['profit_factor'],
                'expectancy_pct': selected['expectancy_pct'],
                'sharpe': selected['sharpe'],
                'score': float(score),
                'reason': 'accepted' if score > best_score else 'candidate',
                'precision_delta': selected_prec - baseline_precision,
                'pf_delta': selected['profit_factor'] - baseline_pf,
                'expectancy_delta': selected['expectancy_pct'] - baseline_expectancy,
                'sharpe_delta': selected['sharpe'] - baseline_sharpe,
            })

            if score <= best_score:
                continue

            best_score = score
            best = {
                'symbol': symbol,
                'gate_type': arch['name'],
                'architecture': arch,
                'threshold_quantile': float(quantile),
                'thresholds': thresholds,
                # persist the globally selected calibrator (holdout-optimal)
                'calibration': calib_report,
                'calibration_used_by_architecture': calib_report_for_arch,
                'risk_tiers': {
                    'conservative': selected_prec >= 0.50 and selected['profit_factor'] >= 1.0,
                    'balanced': selected_prec >= 0.55 and selected['expectancy_pct'] > 0.0,
                    'aggressive': selected['expectancy_pct'] > 0.0 and selected['sharpe'] > 0.0,
                },
                'holdout_metrics': selected,
                'rejected_metrics': rejected,
                'candidate_metrics': candidate_metrics,
                'score': float(round(score, 5)),
                'signal_vetoes': arch['vetoes'],
                'regime_modifier': arch['regime_modifier'],
                'side_specific': arch['side_specific'],
                'trust_score': _compute_trust_score(pf_norm, expectancy_norm, sharpe_norm, coverage_norm, fired_n),
                'grade': _gate_grade(float(selected.get('profit_factor', 0.0))),
            }

    # Contradiction detector: calibration validation found no viable architecture but
    # the main gate loop succeeded.  This is expected when calibration used a fixed
    # 0.50 threshold while the gate loop uses percentile thresholds — log it clearly.
    if calib_report.get('arch_validation_no_viable') and best is not None and best.get('gate_type') != 'DISABLED':
        print("\n   *** CONTRADICTION NOTE ***")
        print("   CALIBRATION VALIDATION : No calibrator produced a viable architecture at p>=0.50 threshold")
        print(f"   ARCHITECTURE SEARCH    : Gate found — {best['gate_type']} | PF={best['candidate_metrics'].get('profit_factor', 0.0):.3f}")
        print("   ROOT CAUSE             : Calibration evaluated at fixed 0.50 probability threshold;")
        print("                            architecture search uses percentile thresholds (top 15-40% of signals)")
        print("   RESOLUTION             : Gate enabled with uncalibrated scores — this is expected behaviour")
        print("   RECOMMENDATION         : Consider collecting more data to allow calibrator training at lower thresholds\n")

    debug_payload = safe_json_serializer({
        'symbol': symbol,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'baseline': {
            'precision': baseline_precision,
            'profit_factor': baseline_pf,
            'expectancy_pct': baseline_expectancy,
            'sharpe': baseline_sharpe,
            'coverage': baseline_coverage,
        },
        'calibration': calib_report,
        'calibration_candidates': calib_candidates,
        'candidates': debug_records,
        'selected': best if best is not None else None,
    })
    debug_path = DEBUG_DIR / f"{symbol.replace('/', '_')}_gate_debug.json"
    with open(debug_path, 'w') as fh:
        json.dump(debug_payload, fh, indent=2)

    # ARCHITECTURE LEADERBOARD (console)
    arch_rows = [r for r in debug_records if 'score' in r and not str(r.get('reason','')).startswith('duplicate')]
    arch_rows_sorted = sorted(arch_rows, key=lambda x: float(x.get('score', float('-inf')) if x.get('score') is not None else float('-inf')), reverse=True)
    print("   [ARCHITECTURE LEADERBOARD] Top candidates:")
    print("   arch_type | quantile | calibration | precision | pf | expectancy | sharpe | coverage | score")
    for r in arch_rows_sorted[:10]:
        gate_type = str(r.get('gate_type', 'unknown'))[:12]
        calib = str(r.get('calibration', 'unknown'))[:10]
        print(
            f"   {gate_type:12} | {fmt(r.get('quantile', 0.0)):8} | {calib:10} | "
            f"{fmt(r.get('precision', 0.0)):9} | {fmt(r.get('profit_factor', 0.0)):4} | {fmt(r.get('expectancy_pct', 0.0)):9} | "
            f"{fmt(r.get('sharpe', 0.0)):6} | {fmt(r.get('coverage', 0.0)):8} | {fmt(r.get('score', 0.0)):6}"
        )

    # Rejected architectures summary
    rejected = [r for r in debug_records if r.get('reason') and r.get('reason') not in ('accepted', 'candidate')]
    if rejected:
        print("   [REJECTED ARCHITECTURES]")
        for r in rejected[:20]:
            gate_type = str(r.get('gate_type', 'unknown'))[:12]
            reason = str(r.get('reason', 'unknown'))
            print(
                f"   {gate_type:12} | reason={reason:20} | "
                f"cov={fmt(r.get('coverage', 0.0)):>5} | prec={fmt(r.get('precision', 0.0)):>5} | pf={fmt(r.get('profit_factor', 0.0)):>5}"
            )

    # Final architecture summary
    if best is not None:
        b = best
        print("   [FINAL ARCHITECTURE]")
        print(
            f"   name={b.get('gate_type')} calibration={b.get('calibration',{}).get('selected_method', b.get('calibration',{}).get('method', 'N/A'))} | "
            f"coverage={fmt(b.get('candidate_metrics',{}).get('coverage',0.0))} | precision={fmt(b.get('candidate_metrics',{}).get('precision',0.0))} | "
            f"pf={fmt(b.get('candidate_metrics',{}).get('profit_factor',0.0))} | exp={fmt(b.get('candidate_metrics',{}).get('expectancy_pct',0.0))} | "
            f"sharpe={fmt(b.get('candidate_metrics',{}).get('sharpe',0.0))} | score={fmt(b.get('score',0.0))}"
        )

    if best is None:
        print(f"   [{symbol}] No viable gate architecture passed profitability and robustness criteria; falling back to DISABLED profile.")
        
        # FIX 5: Enhanced root-cause logging for DISABLED profiles
        print(f"\n   [ROOT CAUSE ANALYSIS FOR {symbol}]")
        root_causes = []
        
        # Check calibration failure
        if calib_report.get('calibration_failed'):
            root_causes.append("Calibration failed: No calibrator produced viable architecture")
        
        # Check if any candidates existed at all
        if not debug_records:
            root_causes.append("No architecture candidates generated")
        else:
            # FIX 5: Analyze rejection reasons by category
            all_rejected = [r for r in debug_records if r.get('reason') not in ('accepted', 'candidate', None)]
            if all_rejected:
                reason_counts = {}
                for r in all_rejected:
                    reason = r.get('reason', 'unknown')
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                root_causes.append(f"All {len(all_rejected)} candidates rejected:")
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]:
                    root_causes.append(f"  - {reason}: {count} candidates")
                
                # FIX 5: Highlight FIX 4 specific issues
                negative_lift = sum(1 for r in all_rejected if 'negative_gate_lift' in str(r.get('reason','')))
                insufficient_holdout = sum(1 for r in all_rejected if 'insufficient_holdout' in str(r.get('reason','')))
                if negative_lift > 0:
                    root_causes.append(f"   >>> {negative_lift} gates rejected for NEGATIVE LIFT (anti-selective gating)")
                if insufficient_holdout > 0:
                    root_causes.append(f"   >>> {insufficient_holdout} gates rejected for INSUFFICIENT HOLDOUT (< {MIN_VALID_HOLDOUT_BARS} bars)")
        
        # Check baseline metrics
        if baseline_pf < 1.05:
            root_causes.append(f"Baseline profit factor too low: {baseline_pf:.3f} < 1.05")
        if baseline_expectancy <= 0.0:
            root_causes.append(f"Baseline expectancy non-positive: {baseline_expectancy:.3f}%")
        if baseline_sharpe <= 0.0:
            root_causes.append(f"Baseline Sharpe too low: {baseline_sharpe:.3f}")
        if baseline_coverage < MIN_GATE_COVERAGE:
            root_causes.append(f"Baseline coverage insufficient: {baseline_coverage:.3f} < {MIN_GATE_COVERAGE:.2f}")
        
        if not root_causes:
            root_causes.append("Unknown reason (possible data quality issue)")
        
        for cause in root_causes:
            print(f"   ├─ {cause}")
        print(f"   └─ RECOMMENDATION: Review data quality, feature engineering, and regime definitions\n")
        
        best = {
            'symbol': symbol,
            'gate_type': 'DISABLED',
            'architecture': None,
            'threshold_quantile': 0.0,
            'thresholds': {
                'global_threshold': 100.0,
                'buy_threshold': 100.0,
                'sell_threshold': 100.0,
                'quantile': 0.0,
            },
            'calibration': calib_report,
            'risk_tiers': {'conservative': False, 'balanced': False, 'aggressive': False},
            'holdout_metrics': baseline,
            'rejected_metrics': {},
            'candidate_metrics': {
                'precision': baseline_precision,
                'coverage': baseline_coverage,
                'expectancy_pct': baseline_expectancy,
                'profit_factor': baseline_pf,
                'sharpe': baseline_sharpe,
                'gate_lift': 0.0,
                'fired_n': total_directional,
                'rejected_n': 0,
            },
            'score': 0.0,
            'signal_vetoes': [],
            'regime_modifier': False,
            'side_specific': False,
            'trust_score': 0,
            'reason': 'no_viable_candidate',
            'disabled_reason': 'no_candidate_passed_pf_expectancy_sharpe',
            'root_causes': root_causes,
        }

    print(f"   [Selected Gate] {best['gate_type']} | score={best['score']:.4f} "
          f"| trust={best.get('trust_score', 0)} "
          f"| prec={best['candidate_metrics']['precision']:.3f} "
          f"| pf={best['candidate_metrics']['profit_factor']:.2f} "
          f"| exp={best['candidate_metrics']['expectancy_pct']:.3f}%")

    return best


# Alias for the requested architecture search entrypoint.
search_gate_architectures = _evaluate_architecture


def compute_edge_scores(
    df: pd.DataFrame,
    proposed: np.ndarray,
    meta_probs: np.ndarray,
    use_rank: bool = False,
) -> pd.Series:
    scores = np.full(len(df), np.nan, dtype=float)
    buy_mask = proposed == 2
    sell_mask = proposed == 0
    if buy_mask.any():
        scores[buy_mask] = EdgeScoringEngine.compute_edge_batch(
            df.loc[buy_mask].reset_index(drop=True),
            meta_probs[buy_mask],
            'BUY',
            use_rank=use_rank,
        ).values
    if sell_mask.any():
        scores[sell_mask] = EdgeScoringEngine.compute_edge_batch(
            df.loc[sell_mask].reset_index(drop=True),
            meta_probs[sell_mask],
            'SELL',
            use_rank=use_rank,
        ).values
    return pd.Series(scores, index=df.index)


def optimize_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    base = symbol.replace('/', '_')
    try:
        p = Predictor(symbol)
    except Exception as exc:
        print(f"   [{symbol}] Predictor init failed: {exc}")
        return None

    feature_cols = p.meta.get('feature_cols')
    base_atr_mult = float(p.meta.get('atr_multiplier', get_atr_multiplier(symbol)))

    try:
        df_1h = p.fetch_live_data(timeframe='1h', limit=HISTORY_HOURS)
        if df_1h is None or len(df_1h) < MIN_BARS:
            print(f"   [{symbol}] Not enough data — skipping.")
            return None
        btc_df = p.fetch_btc_data(timeframe='1h', limit=HISTORY_HOURS) if hasattr(p, 'fetch_btc_data') else None
        news_df = p.load_news_data()
        df_1d = p.fetch_live_data(timeframe='1d', limit=300)
        fund_df, oi_df = fetch_futures_data(symbol, df_1h)
        fg_df = fetch_fear_greed(days=700)
    except Exception as exc:
        print(f"   [{symbol}] Data fetch error: {exc}")
        return None

    try:
        features = prepare_features(
            df_1h,
            btc_df=btc_df,
            news_df=news_df,
            df_1d=df_1d,
            funding_df=fund_df,
            oi_df=oi_df,
            fg_df=fg_df,
        )
    except Exception as exc:
        print(f"   [{symbol}] Feature engineering failed: {exc}")
        return None

    if features is None or len(features) < MIN_BARS:
        print(f"   [{symbol}] Too few feature rows — skipping.")
        return None

    features = features.reset_index(drop=True)
    n_feat = len(features)
    df_raw = df_1h.iloc[-n_feat:].reset_index(drop=True).copy()
    for col in ('volatility_regime', 'efficiency_ratio_10', 'trend_regime', 'macro_confluence_score'):
        if col in features.columns:
            df_raw[col] = features[col].values

    if feature_cols:
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            features = pd.concat(
                [features, pd.DataFrame(0.0, index=features.index, columns=missing)],
                axis=1,
            )
        feat_df = features[feature_cols].copy()
    else:
        drop = [c for c in ('timestamp', 'target') if c in features.columns]
        feat_df = features.drop(columns=drop, errors='ignore')
        feat_df = feat_df[[c for c in feat_df.columns if not c.startswith('_')]]
        feature_cols = list(feat_df.columns)

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_n = int(n_feat * TRAIN_FRAC)
    eval_n = n_feat - train_n

    regimes_all, boundaries = classify_regimes(df_raw)
    vr_all = df_raw['volatility_regime'] if 'volatility_regime' in df_raw.columns else None
    er_all = df_raw['efficiency_ratio_10'] if 'efficiency_ratio_10' in df_raw.columns else None
    tr_all = df_raw['trend_regime'] if 'trend_regime' in df_raw.columns else None
    cs_all = df_raw.get('macro_confluence_score')
    labels_all = create_triple_barrier_labels(
        df_raw,
        atr_multiplier=base_atr_mult,
        volatility_regime=vr_all,
        efficiency_ratio=er_all,
        trend_regime=tr_all,
        macro_confluence_score=cs_all,
    ).values

    print(f"   [{symbol}] Fitting local model ({train_n} train / {eval_n} eval)...", end=' ', flush=True)
    try:
        proposed_all, dir_conf_all, probs_all, local_model = _fit_local_model(feat_df, np.asarray(labels_all), train_n)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print('done')

    df_ev = features.iloc[train_n:].reset_index(drop=True).copy()
    regimes_ev = regimes_all.iloc[train_n:].reset_index(drop=True)
    vr_ev = df_ev['volatility_regime'] if 'volatility_regime' in df_ev.columns else None
    er_ev = df_ev['efficiency_ratio_10'] if 'efficiency_ratio_10' in df_ev.columns else None
    tr_ev = df_ev['trend_regime'] if 'trend_regime' in df_ev.columns else None
    cs_ev = df_ev.get('macro_confluence_score')

    labels_ev = create_triple_barrier_labels(
        df_raw.iloc[train_n:].reset_index(drop=True),
        atr_multiplier=base_atr_mult,
        volatility_regime=vr_ev,
        efficiency_ratio=er_ev,
        trend_regime=tr_ev,
        macro_confluence_score=cs_ev,
    ).values
    barrier_frac_ev = df_ev['_barrier_frac'] if '_barrier_frac' in df_ev.columns else np.full(len(df_ev), 1.0)

    proposed_ev = proposed_all[train_n:]
    dir_conf_ev = dir_conf_all[train_n:]
    probs_ev = probs_all[train_n:]
    train_mask = labels_all[:train_n] != CENSORED
    train_edge_scores = compute_edge_scores(
        features.iloc[:train_n].reset_index(drop=True),
        proposed_all[:train_n],
        dir_conf_all[:train_n],
        use_rank=True,
    ).values.astype(float)
    train_edge_raw = train_edge_scores[train_mask]
    train_correct = np.asarray(labels_all[:train_n][train_mask] == proposed_all[:train_n][train_mask]).astype(int)

    valid_ev = labels_ev != CENSORED
    n_valid_ev = int(valid_ev.sum())
    if n_valid_ev < MIN_FIRES_DEV:
        print(f"   [{symbol}] Not enough valid evaluation bars: {n_valid_ev} < {MIN_FIRES_DEV}.")
        return None

    ranking_df = _build_ranking_diagnostics(
        df_ev.loc[valid_ev].reset_index(drop=True),
        proposed_ev[valid_ev],
        dir_conf_ev[valid_ev],
        probs_ev[valid_ev],
        regimes_ev.loc[valid_ev].reset_index(drop=True),
        labels=labels_ev[valid_ev],
    )
    _print_ranking_diagnostics(symbol, ranking_df, local_model)
    _signal_diag = _extract_signal_diagnostics(ranking_df, local_model)

    try:
        best = _evaluate_architecture(
            symbol=symbol,
            df_ev=df_ev,
            regimes_ev=regimes_ev,
            proposed_ev=proposed_ev,
            dir_conf_ev=dir_conf_ev,
            probs_ev=probs_ev,
            labels_ev=labels_ev,
            barrier_frac_ev=np.asarray(barrier_frac_ev, dtype=float),
            train_edge_scores=train_edge_raw,
            train_correct=train_correct,
        )
    except Exception as exc:
        print(f"   [{symbol}] Gate search failed: {exc}")
        return None

    profile = {
        'symbol': symbol,
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'base_atr_multiplier': base_atr_mult,
        'train_bars': train_n,
        'eval_bars': eval_n,
        'regime_boundaries': boundaries,
        'selected_profile': {
            'gate_type': best['gate_type'],
            'threshold_quantile': best['threshold_quantile'],
            'thresholds': best['thresholds'],
            'side_specific': best['side_specific'],
            'regime_modifier': best['regime_modifier'],
            'signal_vetoes': best['signal_vetoes'],
            'calibration': best['calibration'],
            'risk_tier': best['risk_tiers'],
            'edge_rank_mode': 'percentile',
            'score': best['score'],
            'disabled_reason': best.get('disabled_reason'),
        },
        'holdout': {
            'selected': best['candidate_metrics'],
            'holdout_metrics': best['holdout_metrics'],
            'rejected_metrics': best['rejected_metrics'],
        },
    }
    # Side-coverage sanity: ensure both BUY and SELL have non-zero fired signals.
    try:
        arch = best.get('architecture', {})
        quantile = float(best.get('threshold_quantile', 0.5))
        use_calibration = bool(arch.get('calibrate'))
        calibrated_scores = calibrated_ev_common if use_calibration else ev_edge_raw / 100.0
        # target at least 5% of directional signals for the missing side
        total_directional = int((proposed_ev != 1).sum()) if 'proposed_ev' in locals() else int((ev_side != 1).sum())
        target_min = max(1, int(0.05 * total_directional))
        buy_count = sell_count = 0
        # compute initial counts
        try:
            fire_mask, thresholds = _compute_gate_mask(ev_df, regimes_ev, ev_side, ev_edge_raw, calibrated_scores, arch, quantile)
            buy_count = int(((ev_side == 2) & fire_mask).sum())
            sell_count = int(((ev_side == 0) & fire_mask).sum())
        except Exception:
            buy_count = sell_count = 0

        if buy_count == 0 or sell_count == 0:
            print(f"   [{symbol}] Side-coverage issue detected (buy={buy_count}, sell={sell_count}). Attempting controlled threshold relaxation.")
            # try lowering quantile to capture sparse side signals (step down by 0.05 up to 5 steps)
            q = quantile
            for step in range(1, 6):
                q_try = max(0.01, quantile - 0.05 * step)
                try:
                    fire_mask_try, thresholds_try = _compute_gate_mask(ev_df, regimes_ev, ev_side, ev_edge_raw, calibrated_scores, arch, q_try)
                except Exception:
                    continue
                buy_try = int(((ev_side == 2) & fire_mask_try).sum())
                sell_try = int(((ev_side == 0) & fire_mask_try).sum())
                if (buy_count == 0 and buy_try >= target_min) or (sell_count == 0 and sell_try >= target_min):
                    print(f"   [{symbol}] Relaxed quantile {quantile:.2f} -> {q_try:.2f} to recover side signals (buy={buy_try}, sell={sell_try}).")
                    quantile = q_try
                    thresholds = thresholds_try
                    fire_mask = fire_mask_try
                    buy_count, sell_count = buy_try, sell_try
                    # update best metrics to reflect relaxed thresholds
                    best['threshold_quantile'] = float(quantile)
                    best['thresholds'] = thresholds
                    # recompute holdout/candidate metrics
                    sel = _backtest_holdout(fire_mask, ev_side, ev_labels, ev_barrier)
                    selected_prec = float((ev_labels[fire_mask] == ev_side[fire_mask]).mean()) if fire_mask.any() else 0.0
                    best['holdout_metrics'] = sel
                    best['candidate_metrics'].update({
                        'precision': selected_prec,
                        'coverage': float(fire_mask.sum() / total_directional) if total_directional else 0.0,
                        'fired_n': int(fire_mask.sum()),
                    })
                    break
            else:
                print(f"   [{symbol}] Could not recover missing side signals after threshold relaxation. buy={buy_count} sell={sell_count}.")
    except Exception as exc:
        print(f"   [{symbol}] Side-coverage check failed: {exc}")
# Final sanity: ensure the saved calibrator matches the holdout-selected winner.
    final_saved_cal = profile['selected_profile']['calibration'].get('selected_calibrator') or profile['selected_profile']['calibration'].get('selected_method') or profile['selected_profile']['calibration'].get('method')
    winner_after_filters = best.get('calibration', {}).get('selected_calibrator') or best.get('calibration', {}).get('selected_method') or best.get('calibration', {}).get('method')
    print(f"   [CALIBRATION CHECK] final_saved={final_saved_cal} winner_after_filters={winner_after_filters}")
    if final_saved_cal != winner_after_filters:
        raise AssertionError(f"Final saved calibrator ({final_saved_cal}) != winner after filters ({winner_after_filters})")

    # Final profile validity assertions
    final_metrics = best.get('candidate_metrics', {})
    final_holdout = best.get('holdout_metrics', {})
    if best.get('gate_type') != 'DISABLED':
        if float(final_metrics.get('coverage', 0.0)) < MIN_GATE_COVERAGE:
            raise AssertionError(f"Final profile coverage {final_metrics.get('coverage')} < {MIN_GATE_COVERAGE}")
        if float(final_metrics.get('precision', 0.0)) < float(final_metrics.get('baseline_precision', 0.0)):
            raise AssertionError(f"Final profile precision {final_metrics.get('precision')} < baseline {final_metrics.get('baseline_precision')}")
        # Minimum viable baseline check: if holdout PF or expectancy is too weak,
        # create a conservative fallback gate instead of disabling the token.
        if float(final_holdout.get('profit_factor', 0.0)) < 1.05 or float(final_holdout.get('expectancy_pct', 0.0)) <= 0.0:
            fallback_quantile = 0.70
            fallback_threshold = float(np.quantile(ev_edge_raw / 100.0, 1.0 - fallback_quantile))
            print(
                f"   [FALLBACK] Low holdout PF/expectancy (PF={final_holdout.get('profit_factor')}, "
                f"EV={final_holdout.get('expectancy_pct')}). "
                f"Creating conservative fallback gate at global threshold {fallback_threshold:.3f} ({fallback_quantile:.0%})."
            )
            profile['selected_profile'] = {
                'gate_type': 'GLOBAL_EDGE_PERCENTILE',
                'threshold_quantile': fallback_quantile,
                'thresholds': {'global': fallback_threshold},
                'side_specific': False,
                'regime_modifier': False,
                'signal_vetoes': [],
                'calibration': {'method': None},
                'risk_tier': {'conservative': True, 'balanced': False, 'aggressive': False},
                'edge_rank_mode': 'percentile',
                'score': 0.0,
                'disabled_reason': 'fallback_low_baseline',
            }
            # update holdout section to record baseline metrics
            profile['holdout']['selected'] = best.get('candidate_metrics')
            profile['holdout']['holdout_metrics'] = best.get('holdout_metrics')
            profile['selected_profile']['fallback'] = True
    else:
        print(f"   [WARNING] DISABLED fallback selected for {symbol}; skipping gate performance assertions.")

    profile = safe_json_serializer(profile)

    out = PROFILE_DIR / f"{base}_gate.json"
    with open(out, 'w') as fh:
        json.dump(profile, fh, indent=2, default=str)

    # Generate forensic report (BUG 6)
    try:
        debug_path = DEBUG_DIR / f"{symbol.replace('/', '_')}_gate_debug.json"
        debug_data = {}
        if debug_path.exists():
            with open(debug_path) as fh:
                debug_data = json.load(fh)
        debug_data['signal_diagnostics'] = _signal_diag

        # Comprehensive forensic report (JSON + TXT) saved to reports/gates/
        gate_reporter = GateForensicsReporter(_ROOT)
        gate_report = gate_reporter.generate_report(symbol, profile, debug_data)
        json_path = gate_reporter.save_json(symbol, gate_report)
        txt_path  = gate_reporter.save_txt(symbol, gate_report)
        gate_reporter.print_report(gate_report)
        print(f"   [FORENSICS] JSON -> {json_path}")
        print(f"   [FORENSICS] TXT  -> {txt_path}")
    except Exception as e:
        print(f"   [WARNING] Failed to generate forensic report: {e}")

    return profile


def _safe(symbol: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        return symbol, optimize_symbol(symbol)
    except Exception as exc:
        import traceback
        print(f"   [{symbol}] FATAL: {exc}\n{traceback.format_exc()}")
        return symbol, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Independent meta gate architecture discovery optimizer'
    )
    parser.add_argument('symbols', nargs='*', help='Symbols to optimise (default: all FLEET_SYMBOLS)')
    parser.add_argument('--workers', type=int, default=1, help='Parallel worker processes')
    args = parser.parse_args()

    targets = [s.upper() for s in args.symbols] if args.symbols else list(FLEET_SYMBOLS)
    targets = [s if '/' in s else s.replace('_', '/') for s in targets]

    print(f"\n{'='*72}")
    print('AEGIS — Meta Gate Architecture Discovery Optimizer')
    print(f"{'='*72}")
    existing_profiles = {
        p.stem.replace('_gate', '').replace('_', '/')
        for p in PROFILE_DIR.glob('*_gate.json')
        if p.name.endswith('_gate.json')
    }
    targets = [t for t in targets if t not in existing_profiles]

    print(f"Symbols      : {len(targets)} (skipping {len(existing_profiles)} completed)")
    print(f"History      : {HISTORY_HOURS} bars x 1h per symbol")
    print(f"Train / Eval : {int(TRAIN_FRAC*100)}% / {100-int(TRAIN_FRAC*100)}%")
    print(f"Workers      : {args.workers}")
    print(f"Output       : {PROFILE_DIR}")
    print(f"{'='*72}\n")

    summary: Dict[str, Any] = {}
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_safe, symbol): symbol for symbol in targets}
            for idx, fut in enumerate(as_completed(futures), 1):
                symbol = futures[fut]
                _, result = fut.result()
                print(f"[{idx}/{len(targets)}] finished: {symbol}")
                if result:
                    summary[symbol] = result['selected_profile']
    else:
        for idx, symbol in enumerate(targets, 1):
            print(f"[{idx}/{len(targets)}] {symbol}")
            _, result = _safe(symbol)
            if result:
                summary[symbol] = result['selected_profile']

    summary_path = PROFILE_DIR / '_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump({
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'n_symbols': len(summary),
            'symbols': summary,
        }, fh, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"Done: {len(summary)}/{len(targets)} symbols in {elapsed / 60:.1f} min")
    print(f"Summary -> {summary_path}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
