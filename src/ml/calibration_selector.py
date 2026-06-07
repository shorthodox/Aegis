from typing import Dict, Any
import numpy as np


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    probs = probs.flatten()
    y = y.flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        p_mean = probs[mask].mean()
        y_mean = y[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(p_mean - y_mean)
    return float(ece)


def evaluate_and_select(candidates: Dict[str, Dict[str, Any]], y_val: np.ndarray) -> Dict[str, Any]:
    """Given candidate calibrators mapping name->{'probs': array}, evaluate and
    select best calibrator while avoiding coverage collapse (isotonic overfit).
    Returns selected name and diagnostics.
    """
    best = None
    best_score = None
    details = {}
    for name, info in candidates.items():
        probs = np.asarray(info.get('probs')).flatten()
        if len(probs) == 0:
            continue
        ece = expected_calibration_error(probs, y_val)
        brier = float(np.mean((probs - y_val) ** 2))
        coverage = float((probs >= 0.5).mean())
        extreme_frac = float(((probs < 0.05) | (probs > 0.95)).mean())
        var_ratio = float(np.var(probs) / (np.var(info.get('raw_probs', probs)) + 1e-9)) if info.get('raw_probs') is not None else 1.0
        # eligibility rules: prevent collapsed calibrators
        eligible = True
        if coverage < 0.15:
            eligible = False
        if extreme_frac > 0.40:
            eligible = False
        if var_ratio < 0.05:
            eligible = False
        score = (-ece) + (0.5 * (1 - brier)) + (0.2 * coverage) + (0.1 * var_ratio)
        details[name] = {'ece': ece, 'brier': brier, 'coverage': coverage, 'extreme_frac': extreme_frac, 'var_ratio': var_ratio, 'eligible': eligible}
        if not eligible:
            continue
        if best is None or score > best_score:
            best = name
            best_score = score
    return {'selected': best, 'score': best_score, 'details': details}
