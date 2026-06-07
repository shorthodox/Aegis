#!/usr/bin/env python3
"""
calibration_diagnostics.py — deep diagnostic of calibration collapse

Tests for:
- Probability collapse (all 0s/1s)
- Calibration overfitting (train vs holdout metrics divergence)
- Distribution shift (train vs holdout raw prob distribution)
- Threshold incompatibility (does calibrated distribution match expected percentiles)
- Signal erasure (coverage before/after calibration)

Outputs:
- Histograms of raw and calibrated probabilities
- Per-calibrator diagnostic table
- Overfitting tests (bootstrap CIs)
- Recommendation on whether to use calibration for ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        m = (binids == i)
        if m.any():
            ece += (m.sum() / len(y_prob)) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(np.mean((y_prob - y_true) ** 2))


def diagnose_calibrator(
    name: str,
    y_prob_train: np.ndarray,
    y_true_train: np.ndarray,
    y_prob_holdout: np.ndarray,
    y_true_holdout: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Comprehensive diagnostic for a single calibrator.
    
    Tests:
    1. Probability collapse: fraction of probs at 0/1 extremes
    2. Calibration overfitting: |train_ECE - holdout_ECE| and |train_Brier - holdout_Brier|
    3. Distribution shift: KS test train vs holdout
    4. Coverage destruction: what fraction of signals remain after threshold
    5. Precision degradation: does threshold-based signal lose precision
    """
    
    # Clean NaN
    valid_tr = ~np.isnan(y_prob_train)
    valid_ho = ~np.isnan(y_prob_holdout)
    p_tr = y_prob_train[valid_tr]
    y_tr = y_true_train[valid_tr].astype(int)
    p_ho = y_prob_holdout[valid_ho]
    y_ho = y_true_holdout[valid_ho].astype(int)
    
    if len(p_tr) < 10 or len(p_ho) < 10:
        return {'name': name, 'error': 'insufficient_data'}
    
    # 1. Probability collapse
    extreme_tr = np.sum((p_tr <= 0.01) | (p_tr >= 0.99)) / len(p_tr)
    extreme_ho = np.sum((p_ho <= 0.01) | (p_ho >= 0.99)) / len(p_ho)
    
    # 2. Calibration metrics
    ece_tr = _ece(y_tr, p_tr)
    ece_ho = _ece(y_ho, p_ho)
    brier_tr = _brier(y_tr, p_tr)
    brier_ho = _brier(y_ho, p_ho)
    ece_drift = abs(ece_tr - ece_ho)
    brier_drift = abs(brier_tr - brier_ho)
    
    # 3. Distribution shift: KS test
    ks_stat, ks_pval = stats.ks_2samp(p_tr, p_ho)
    
    # 4. Coverage after threshold
    fired_ho = p_ho >= threshold
    n_fired = int(fired_ho.sum())
    coverage_ho = n_fired / len(p_ho)
    
    # 5. Precision on fired signals
    precision_ho = float(y_ho[fired_ho].mean()) if fired_ho.any() else 0.0
    
    # 6. Bootstrap CI for holdout ECE (estimate variability)
    n_boot = 100
    ece_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(len(p_ho), len(p_ho), replace=True)
        ece_boot.append(_ece(y_ho[idx], p_ho[idx]))
    ece_ci_lower = np.percentile(ece_boot, 2.5)
    ece_ci_upper = np.percentile(ece_boot, 97.5)
    
    # 7. Overfitting risk: if train metrics much better than holdout
    overfitting_score = (ece_drift + brier_drift) / 2.0  # combined drift
    overfitting_risky = overfitting_score > 0.10  # heuristic threshold
    
    return {
        'name': name,
        'train_ece': float(ece_tr),
        'holdout_ece': float(ece_ho),
        'ece_drift': float(ece_drift),
        'ece_ci_lower': float(ece_ci_lower),
        'ece_ci_upper': float(ece_ci_upper),
        'train_brier': float(brier_tr),
        'holdout_brier': float(brier_ho),
        'brier_drift': float(brier_drift),
        'ks_stat': float(ks_stat),
        'ks_pval': float(ks_pval),
        'dist_shift_significant': ks_pval < 0.05,
        'extreme_pct_train': float(extreme_tr),
        'extreme_pct_holdout': float(extreme_ho),
        'coverage_at_threshold': float(coverage_ho),
        'precision_at_threshold': float(precision_ho),
        'overfitting_score': float(overfitting_score),
        'overfitting_risky': bool(overfitting_risky),
        'n_train': len(p_tr),
        'n_holdout': len(p_ho),
        'n_fired': n_fired,
    }


def plot_calibration_diagnostics(
    calibrator_results: Dict[str, Dict[str, Any]],
    y_prob_dict: Dict[str, np.ndarray],
    output_dir: Path,
    symbol: str,
) -> None:
    """
    Plot histograms and diagnostic charts for all calibrators.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Calibration Diagnostics — {symbol}')
    
    # 1. Probability distributions
    ax = axes[0, 0]
    for name, p_arr in y_prob_dict.items():
        valid = ~np.isnan(p_arr)
        if valid.sum() > 0:
            ax.hist(p_arr[valid], bins=30, alpha=0.5, label=name)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distributions (Holdout)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. ECE and Brier comparison
    ax = axes[0, 1]
    names = []
    eces = []
    briers = []
    for name, res in calibrator_results.items():
        if 'error' not in res:
            names.append(name)
            eces.append(res['holdout_ece'])
            briers.append(res['holdout_brier'])
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, eces, width, label='ECE', alpha=0.8)
    ax.bar(x + width/2, briers, width, label='Brier', alpha=0.8)
    ax.set_ylabel('Metric Value')
    ax.set_title('Holdout Calibration Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Coverage vs Precision
    ax = axes[1, 0]
    coverages = [res.get('coverage_at_threshold', 0.0) for res in calibrator_results.values()]
    precisions = [res.get('precision_at_threshold', 0.0) for res in calibrator_results.values()]
    colors = ['red' if res.get('overfitting_risky') else 'blue' for res in calibrator_results.values()]
    ax.scatter(coverages, precisions, s=100, c=colors, alpha=0.6)
    for i, name in enumerate(calibrator_results.keys()):
        ax.annotate(name, (coverages[i], precisions[i]), fontsize=9)
    ax.set_xlabel('Coverage (at threshold 0.50)')
    ax.set_ylabel('Precision (at threshold 0.50)')
    ax.set_title('Coverage vs Precision')
    ax.grid(True, alpha=0.3)
    
    # 4. ECE drift (train vs holdout)
    ax = axes[1, 1]
    drifts = [res.get('ece_drift', 0.0) for res in calibrator_results.values()]
    colors = ['red' if res.get('overfitting_risky') else 'green' for res in calibrator_results.values()]
    ax.bar(range(len(calibrator_results)), drifts, color=colors, alpha=0.6)
    ax.set_ylabel('ECE Drift (|train - holdout|)')
    ax.set_title('Overfitting Risk (ECE)')
    ax.set_xticks(range(len(calibrator_results)))
    ax.set_xticklabels(calibrator_results.keys(), rotation=45)
    ax.axhline(y=0.10, color='red', linestyle='--', label='Risk threshold (0.10)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = output_dir / f"{symbol.replace('/', '_')}_calibration_diagnostics.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"   Saved diagnostic plot to {out_path}")


def generate_diagnostic_report(
    symbol: str,
    y_prob_train_uncal: np.ndarray,
    y_true_train: np.ndarray,
    y_prob_holdout_uncal: np.ndarray,
    y_true_holdout: np.ndarray,
    calibrator_dict: Dict[str, np.ndarray],  # name -> calibrated probs
) -> Dict[str, Any]:
    """
    Generate comprehensive diagnostic report for a symbol.
    
    Args:
        symbol: asset name
        y_prob_train_uncal: raw (uncalibrated) train probabilities
        y_true_train: training labels
        y_prob_holdout_uncal: raw (uncalibrated) holdout probabilities
        y_true_holdout: holdout labels
        calibrator_dict: {name: calibrated_holdout_probs} for each calibrator
    
    Returns:
        Dictionary with all diagnostic results and recommendations.
    """
    
    results = {}
    y_prob_holdout_dict = {'uncalibrated': y_prob_holdout_uncal}
    
    # Diagnose each calibrator
    calibrator_diagnostics = {}
    for cal_name, y_prob_ho_cal in calibrator_dict.items():
        diag = diagnose_calibrator(
            cal_name,
            y_prob_train_uncal,
            y_true_train,
            y_prob_ho_cal,
            y_true_holdout,
        )
        calibrator_diagnostics[cal_name] = diag
        y_prob_holdout_dict[cal_name] = y_prob_ho_cal
    
    # Diagnose uncalibrated baseline
    baseline_diag = diagnose_calibrator(
        'uncalibrated',
        y_prob_train_uncal,
        y_true_train,
        y_prob_holdout_uncal,
        y_true_holdout,
    )
    calibrator_diagnostics['uncalibrated'] = baseline_diag
    
    # Produce recommendation
    recommendation = {
        'use_calibration': True,
        'recommended_calibrator': 'uncalibrated',
        'reason': '',
    }
    
    # Find best non-risky calibrator
    non_risky = {
        name: res for name, res in calibrator_diagnostics.items()
        if not res.get('overfitting_risky', False) and 'error' not in res
    }
    
    if non_risky:
        # Rank by: (1) coverage >= 0.10, (2) precision >= baseline, (3) lowest ECE
        baseline_prec = baseline_diag.get('precision_at_threshold', 0.0)
        candidates = [
            (name, res) for name, res in non_risky.items()
            if res.get('coverage_at_threshold', 0.0) >= 0.10
            and res.get('precision_at_threshold', 0.0) >= baseline_prec * 0.95
        ]
        
        if candidates:
            best = min(candidates, key=lambda x: x[1].get('holdout_ece', 1.0))
            recommendation['recommended_calibrator'] = best[0]
            recommendation['reason'] = (
                f"Selected {best[0]}: coverage={best[1].get('coverage_at_threshold', 0.0):.3f}, "
                f"precision={best[1].get('precision_at_threshold', 0.0):.3f}, "
                f"ece={best[1].get('holdout_ece', 0.0):.4f} (no overfitting risk)"
            )
        else:
            recommendation['use_calibration'] = False
            recommendation['reason'] = "No calibrator preserved coverage and precision; using uncalibrated."
    else:
        recommendation['use_calibration'] = False
        recommendation['reason'] = "All calibrators show overfitting risk; using uncalibrated."
    
    # Plot diagnostics
    plot_calibration_diagnostics(
        calibrator_diagnostics,
        y_prob_holdout_dict,
        _ROOT / 'data' / 'calibration_diagnostics',
        symbol,
    )
    
    return {
        'symbol': symbol,
        'calibrator_diagnostics': calibrator_diagnostics,
        'recommendation': recommendation,
    }


def print_diagnostic_table(diagnostics: Dict[str, Any]) -> None:
    """Pretty-print diagnostic results as a table."""
    print("\n" + "="*120)
    print(f"CALIBRATION DIAGNOSTICS — {diagnostics['symbol']}")
    print("="*120)
    
    print("\nDetailed Results:")
    print(f"{'Calibrator':<15} | {'Train ECE':<10} | {'Hold ECE':<10} | {'Drift':<8} | {'KS Pval':<8} | "
          f"{'Coverage':<10} | {'Precision':<10} | {'Overfit?':<10}")
    print("-"*120)
    
    for cal_name, res in diagnostics['calibrator_diagnostics'].items():
        if 'error' in res:
            print(f"{cal_name:<15} | ERROR: {res['error']}")
        else:
            print(f"{cal_name:<15} | {res['train_ece']:<10.4f} | {res['holdout_ece']:<10.4f} | "
                  f"{res['ece_drift']:<8.4f} | {res['ks_pval']:<8.4f} | "
                  f"{res['coverage_at_threshold']:<10.4f} | {res['precision_at_threshold']:<10.4f} | "
                  f"{'YES' if res['overfitting_risky'] else 'NO':<10}")
    
    rec = diagnostics['recommendation']
    print("\n" + "="*120)
    print(f"RECOMMENDATION:")
    print(f"  Use Calibration: {rec['use_calibration']}")
    print(f"  Selected Calibrator: {rec['recommended_calibrator']}")
    print(f"  Reason: {rec['reason']}")
    print("="*120 + "\n")


if __name__ == '__main__':
    # Example usage (standalone test)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTC/USDT', help='Symbol to diagnose')
    args = parser.parse_args()
    
    # Placeholder: in real usage, this would be called from the optimizer
    print(f"Calibration diagnostics framework ready for {args.symbol}")
