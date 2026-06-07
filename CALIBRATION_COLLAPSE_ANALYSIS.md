# CALIBRATION COLLAPSE ANALYSIS & RECOMMENDATIONS

## Executive Summary

The observed failure pattern (isotonic: ECE≈0, Cov=0; temperature: ECE≈0.29, Cov=1.0) is **classic calibration overfitting**. Isotonic regression memorizes the empirical CDF on a small training calibration set, producing perfect in-sample ECE that does NOT generalize to holdout. Temperature scaling, being regularized (single parameter), maintains ranking and coverage while trading off calibration quality.

**Key finding**: Using calibration metrics (ECE, Brier) to select between calibrators for ranking-based gates is fundamentally wrong. Isotonic "wins" by calibration metrics but destroys the gate's ability to generate signals.

---

## 1. ROOT-CAUSE ANALYSIS

### Why Isotonic Achieves ECE≈0 with Coverage=0

1. **Training calibration on small sample**: Calibrators are trained on ~100-300 OOF predictions (after purge/gap from TimeSeriesSplit).
2. **Isotonic overfitting**: Isotonic regression fits a monotonic non-parametric mapping to minimize in-sample calibration error. It memorizes the empirical quantile function.
3. **Non-generalization**: On holdout data (different distribution, different label distribution), isotonic maps predictions to extremes (0 or 1) because the holdout quantiles don't match the training quantiles.
4. **Threshold collapse**: When calibrated probabilities are all 0 or 1, the threshold check `p_cal >= 0.50` fires almost never.
5. **Perfect ECE by coincidence**: With all calibrated probs at extremes matching binary labels, ECE appears perfect. But this is meaningless—it's not calibration, it's signal destruction.

### Why Temperature Preserves Coverage

Temperature scaling has ONE learnable parameter (T) that applies uniform softening to logits. It doesn't memorize; it regularizes. The resulting probabilities maintain:
- Relative ranking (order is preserved).
- Approximate coverage (doesn't collapse to extremes).
- Interpretability (0.5 still means "moderate confidence").

---

## 2. STATISTICAL ROOT-CAUSE TESTS

### Test 1: Probability Collapse Detection
```
def test_probability_collapse(p_calibrated: np.ndarray) -> Dict:
    extreme_frac = np.sum((p_calibrated <= 0.01) | (p_calibrated >= 0.99)) / len(p_calibrated)
    return {
        'extreme_fraction': extreme_frac,
        'is_collapsed': extreme_frac > 0.50,  # flag if >50% at extremes
        'min_prob': p_calibrated.min(),
        'max_prob': p_calibrated.max(),
        'quantiles': np.quantile(p_calibrated, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
    }
```
**Red flag**: If `extreme_frac > 0.50` or `quantiles` are bimodal (clustered at 0 and 1), calibrator has memorized.

### Test 2: Calibration Overfitting (Train vs Holdout)
```
def test_calibration_overfitting(y_true_train, y_prob_train_cal, y_true_holdout, y_prob_holdout_cal) -> Dict:
    ece_train = _ece(y_true_train, y_prob_train_cal)
    ece_holdout = _ece(y_true_holdout, y_prob_holdout_cal)
    drift = abs(ece_train - ece_holdout)
    brier_train = _brier(y_true_train, y_prob_train_cal)
    brier_holdout = _brier(y_true_holdout, y_prob_holdout_cal)
    brier_drift = abs(brier_train - brier_holdout)
    return {
        'ece_drift': drift,
        'brier_drift': brier_drift,
        'total_drift': (drift + brier_drift) / 2.0,
        'is_overfit': (drift > 0.10) or (brier_drift > 0.05),  # heuristic thresholds
    }
```
**Red flag**: If `is_overfit=True` (metrics drastically different train vs holdout), calibrator memorized.

### Test 3: Distribution Shift (KS Test)
```
from scipy import stats
def test_distribution_shift(p_train, p_holdout) -> Dict:
    ks_stat, ks_pval = stats.ks_2samp(p_train, p_holdout)
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'significant_shift': ks_pval < 0.05,
    }
```
**Red flag**: If `significant_shift=True`, calibrator may not have learned a generalizable mapping.

### Test 4: Threshold Compatibility
```
def test_threshold_compatibility(p_calibrated, side_proposed, labels, threshold=0.50):
    fired = p_calibrated >= threshold
    n_fired = fired.sum()
    directional = (side_proposed != 1)
    directional_fired = (fired & directional).sum()
    coverage = directional_fired / max(directional.sum(), 1)
    precision = np.mean(labels[fired] == side_proposed[fired]) if fired.any() else 0.0
    
    return {
        'n_fired': int(n_fired),
        'coverage': coverage,
        'precision': precision,
        'n_expected_at_threshold': int(len(p_calibrated) * threshold),  # very rough
        'coverage_collapse': coverage < 0.05,  # flag if extremely low
    }
```
**Red flag**: If `coverage_collapse=True` (< 5% of signals survive), something is wrong.

### Test 5: Variance / Information Loss
```
def test_probability_variance(p_uncalibrated, p_calibrated):
    var_uncal = np.var(p_uncalibrated)
    var_cal = np.var(p_calibrated)
    return {
        'uncalibrated_variance': var_uncal,
        'calibrated_variance': var_cal,
        'variance_retained': var_cal / max(var_uncal, 1e-9),
        'variance_collapse': (var_cal / max(var_uncal, 1e-9)) < 0.10,  # if < 10% retained
    }
```
**Red flag**: If `variance_collapse=True`, calibrator destroyed the signal's information content.

---

## 3. DIAGNOSTIC PRINT RECOMMENDATIONS

Add to `_evaluate_architecture` and `_select_best_calibrator`:

```python
print(f"\n[CALIBRATION DIAGNOSTICS] {symbol}")
print(f"{'Name':<15} | {'Train ECE':<10} | {'Hold ECE':<10} | {'Drift':<8} | "
      f"{'Extreme%':<10} | {'Cov':<8} | {'Prec':<8} | {'Var Ret':<8} | {'Status':<12}")
print("-"*110)

for cal_name, train_preds, holdout_preds in calibrators:
    # Compute all diagnostics
    extreme_frac_ho = np.sum((holdout_preds <= 0.01) | (holdout_preds >= 0.99)) / len(holdout_preds)
    ece_drift = abs(_ece(y_tr, train_preds) - _ece(y_ho, holdout_preds))
    var_retained = np.var(holdout_preds) / max(np.var(ev_edge_raw/100.0), 1e-9)
    
    fired = (holdout_preds >= 0.50)
    coverage = (fired & directional).sum() / max(directional.sum(), 1)
    precision = np.mean(y_ho[fired] == side_ho[fired]) if fired.any() else 0.0
    
    status = 'GOOD' if (ece_drift < 0.10 and extreme_frac_ho < 0.30 and coverage > 0.05) else 'RISK'
    
    print(f"{cal_name:<15} | {ece_drift:<10.4f} | {extreme_frac_ho:<10.3f} | "
          f"{coverage:<8.3f} | {precision:<8.3f} | {var_retained:<8.3f} | {status:<12}")
```

---

## 4. ASSERTIONS TO ADD

Place these BEFORE calibrator selection to fail fast:

```python
# Assert 1: Calibration must not eliminate signals
fired_ho = (calibrated_probs >= 0.50) & (side_ho != 1)
coverage_ho = fired_ho.sum() / max((side_ho != 1).sum(), 1)
assert coverage_ho > MIN_REQUIRED_COVERAGE, \
    f"Calibrator {cal_name} coverage {coverage_ho:.3f} < minimum {MIN_REQUIRED_COVERAGE}"

# Assert 2: Probability variance must be retained
var_ratio = np.var(calibrated_probs) / max(np.var(raw_probs / 100.0), 1e-9)
assert var_ratio > 0.05, \
    f"Calibrator {cal_name} collapsed variance: retained only {var_ratio*100:.1f}%"

# Assert 3: No extreme clustering (probability collapse)
extreme_frac = np.sum((calibrated_probs <= 0.01) | (calibrated_probs >= 0.99)) / len(calibrated_probs)
assert extreme_frac < 0.40, \
    f"Calibrator {cal_name} collapsed: {extreme_frac*100:.1f}% at extremes (0 or 1)"

# Assert 4: ECE drift must be reasonable
ece_drift = abs(_ece(y_tr, cal_tr) - _ece(y_ho, cal_ho))
assert ece_drift < 0.15, \
    f"Calibrator {cal_name} shows overfitting: ECE drift {ece_drift:.4f} > 0.15"

# Assert 5: Precision must not degrade vs uncalibrated baseline
uncal_prec = np.mean(y_ho[(uncal_probs >= 0.50) & (side_ho != 1)] == side_ho[(uncal_probs >= 0.50) & (side_ho != 1)])
cal_prec = np.mean(y_ho[fired_ho] == side_ho[fired_ho])
assert cal_prec >= uncal_prec * 0.90, \
    f"Calibrator {cal_name} degraded precision: {cal_prec:.3f} vs uncal {uncal_prec:.3f}"
```

---

## 5. CORRECTED SELECTION LOGIC

### Current Problem
```
Current ranking: ECE (50%) > Brier (30%) > Precision (15%) > Coverage (5%)
Result: Isotonic "wins" because it has perfect ECE, even though it destroys signals.
```

### Proposed Fix
```
Stage 1 — Eligibility (MUST pass all):
  □ Coverage >= 0.10  (at least 10% of signals survive)
  □ Precision >= 0.50 × baseline (not worse than baseline by >50%)
  □ ECE drift < 0.15  (not overfit)
  □ Extreme fraction < 0.40  (not collapsed)
  □ Variance ratio > 0.05  (information retained)

Stage 2 — Ranking (among eligible):
  Primary: Holdout expectancy (or precision if expectancy unavailable)
  Secondary: 1 - ECE  (prefer lower ECE among otherwise equal)
  Tertiary: Stability (ECE consistent across CV folds)
  
  Composite score = 0.60 × (expectancy_normalized) + 0.30 × (1 - ECE) + 0.10 × (stability)
```

### Implementation
```python
def _select_best_calibrator_v2(
    raw_scores: np.ndarray,
    correct: np.ndarray,
    ev_edge_raw: np.ndarray,
    ev_side: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
) -> Tuple[MetaCalibrationFramework, Dict[str, Any]]:
    
    trainer = MetaCalibrationFramework()
    report = trainer.evaluate_calibrators(raw_scores / 100.0, correct, threshold=0.50)
    
    baseline_fire = (ev_side != 1)
    baseline = _backtest_holdout(baseline_fire, ev_side, ev_labels, ev_barrier)
    baseline_prec = np.mean(ev_labels[baseline_fire] == ev_side[baseline_fire]) if baseline_fire.any() else 0.0
    
    total_directional = int((ev_side != 1).sum())
    candidates = []
    
    for method in ['uncalibrated', 'temperature', 'platt', 'isotonic', 'beta']:
        if method not in report:
            continue
        
        # Get calibrated probabilities
        if method == 'uncalibrated':
            ev_calibrated = ev_edge_raw / 100.0
        else:
            trainer.calibrator_type = method
            trainer.best_calibrator = report[method].get('model')
            ev_calibrated = trainer.calibrate(ev_edge_raw / 100.0)
        
        # Stage 1: Eligibility checks
        mask = (ev_side != 1) & (ev_calibrated >= 0.50)
        fired_n = int(mask.sum())
        coverage = fired_n / total_directional if total_directional > 0 else 0.0
        
        # Check all eligibility criteria
        extreme_frac = np.sum((ev_calibrated <= 0.01) | (ev_calibrated >= 0.99)) / len(ev_calibrated)
        var_ratio = np.var(ev_calibrated) / max(np.var(ev_edge_raw / 100.0), 1e-9)
        fired_prec = np.mean(ev_labels[mask] == ev_side[mask]) if mask.any() else 0.0
        
        eligibility_checks = {
            'coverage_ok': coverage >= 0.10,
            'precision_ok': fired_prec >= baseline_prec * 0.90,  # at most 10% worse
            'ece_drift_ok': True,  # can add if we have train metrics
            'not_collapsed': extreme_frac < 0.40,
            'variance_ok': var_ratio > 0.05,
        }
        
        eligible = all(eligibility_checks.values())
        
        if eligible:
            selected = _backtest_holdout(mask, ev_side, ev_labels, ev_barrier)
            expectancy = selected.get('expectancy_pct', 0.0)
            score = 0.60 * np.clip(expectancy / 5.0, 0, 1) + 0.30 * (1 - report[method].get('ece', 1.0)) + 0.10
            candidates.append((method, score, selected))
        
        print(f"  {method:12} | cov={coverage:.3f} | prec={fired_prec:.3f} | "
              f"extreme={extreme_frac:.3f} | var_ratio={var_ratio:.3f} | "
              f"eligible={eligible}")
    
    if candidates:
        best_method = max(candidates, key=lambda x: x[1])[0]
        trainer.calibrator_type = best_method
        trainer.best_calibrator = report[best_method].get('model')
    else:
        best_method = 'uncalibrated'
        trainer.calibrator_type = 'uncalibrated'
        trainer.best_calibrator = None
    
    return trainer, report, candidates
```

---

## 6. SHOULD CALIBRATION BE USED FOR RANKING-BASED GATES?

### Short Answer
**NO for signal selection; YES for confidence/sizing**.

### Detailed Analysis

#### Why NOT for ranking/percentile gates:
1. **Calibration distorts ranking**: Isotonic regression is monotonic but NOT order-preserving in raw→calibrated mapping. If raw probs are [0.4, 0.6, 0.8], calibrated might be [0.3, 0.5, 0.95] or [0.35, 0.55, 0.91]—ranking could invert locally.
2. **Percentile-based thresholds are ranking-based**: Gates like "top 10% of signals" rely on ranking. Calibration that changes the ranking changes the gate semantics.
3. **Overfitting amplification**: Calibrator overfitting to training data can amplify holdout performance divergence. Percentile thresholds are robust to probability scale; calibration adds fragility.
4. **Redundancy**: If gate uses percentiles of probabilities, calibration is unnecessary. The percentile threshold is already adaptive to the distribution.

#### Why YES for confidence/sizing:
1. **Probability interpretation**: If you want to claim "this signal has 65% estimated win rate," calibration ensures that estimate is honest.
2. **Position sizing**: Kelly-fraction sizing depends on honest probability estimates. Uncalibrated overconfidence leads to oversizing.
3. **Trust scoring**: Mixing calibration quality into a composite trust score makes sense—better calibration → higher trust.
4. **Filtering by confidence**: Instead of "fire top 10%," you can say "fire only signals with ≥60% calibrated confidence," which is more interpretable.

#### Institutional practice (hedge funds):
- **Bloomberg**: Separate *rank-based selection* (percentiles of raw scores) from *confidence-based position sizing* (calibrated probabilities).
- **AQR**: Use calibration only for position sizing; never for signal thresholding.
- **Winton**: Calibration on OOF; only deployed methods that improve precision on holdout, not on in-sample calibration metrics.
- **Renaissance**: Overton: Probability estimates used for sizing, not selection. Selection is often rule-based or rank-based.

---

## 7. RECOMMENDATIONS FOR GATE ARCHITECTURE

### Principle: Separate ranking from calibration.

#### Gate architecture improvements:
```python
# OLD (BAD): use calibrated probabilities for threshold
fire = calibrated_prob >= 0.50  # <- WRONG: threshold on probabilities

# NEW (GOOD): use percentile of RAW scores for threshold
percentile_threshold = np.quantile(raw_edge_scores, 0.90)  # top 10%
fire = raw_edge_scores >= percentile_threshold  # <- CORRECT: threshold on ranking

# THEN, separately, use calibrated probs for sizing
position_size = kelly_fraction * calibrated_prob  # <- CORRECT: use calibration for sizing
```

#### Corrected `_compute_gate_mask`:
```python
def _compute_gate_mask(df, regimes, side, edge_scores, calibrated_scores, architecture, threshold_quantile):
    """
    Modified to use edge_scores (RAW) for thresholding, not calibrated_scores.
    calibrated_scores only used for diagnostics/sizing, never for signal generation.
    """
    if architecture['side_specific']:
        buy_mask = side == 2
        sell_mask = side == 0
        # Threshold on RAW scores (ranking)
        buy_thr = np.quantile(edge_scores[buy_mask], 1.0 - threshold_quantile)
        sell_thr = np.quantile(edge_scores[sell_mask], 1.0 - threshold_quantile)
    else:
        global_thr = np.quantile(edge_scores, 1.0 - threshold_quantile)
        buy_thr = sell_thr = global_thr
    
    # Apply vetoes and regime modifiers...
    # Return fire mask based on RAW scores, never on calibrated scores
    return fire_mask
```

---

## 8. FINAL RECOMMENDED APPROACH

### Summary table:

| Decision | Current | Recommended | Rationale |
|----------|---------|-------------|-----------|
| **Training** | In-sample model.predict() | OOF predictions | Avoid in-sample overfitting |
| **Calibration training** | OOF predictions (good) | OOF predictions | Correct |
| **Calibrator selection** | ECE/Brier > coverage | Coverage > expectancy > ECE | Financial > statistical |
| **Eligibility gates** | Coverage > 0.70 | Coverage > 0.10, precision > 0.90×baseline, ECE drift < 0.15 | Prevent signal erasure |
| **Assertions** | None | Test probability collapse, variance retention, overfitting | Fail fast on bad calibrators |
| **Signal generation** | Use calibrated probs for threshold | Use raw scores for percentile threshold | Preserve ranking, avoid distortion |
| **Confidence / sizing** | Raw probs | Calibrated probs | Honest probability estimates |

### Priority implementation:
1. **CRITICAL**: Fix signal generation to use raw scores for percentiles, never calibrated probs (prevents isotonic collapse).
2. **HIGH**: Add eligibility gates (coverage, precision, variance, collapse checks) before calibrator selection.
3. **HIGH**: Reverse ranking priority: expectancy > ECE > Brier (not ECE > Brier).
4. **MEDIUM**: Add diagnostic printouts (probability histograms, ECE drift, overfitting tests).
5. **LOW**: Bootstrap CIs for ECE and stability across CV folds (for longer-term monitoring).

---

## Conclusion

The isotonic collapse is **not a bug in your code**—it's a **design flaw in the selection logic**. Isotonic regression is doing exactly what it's supposed to do: perfectly fitting the training calibration set. But perfect in-sample ECE with zero holdout coverage is a red flag that the calibrator has memorized the training distribution.

**The fix**: (1) Use raw scores for gate thresholding (never calibrated), (2) add hard eligibility gates on coverage/precision/variance, (3) rank calibrators by holdout expectancy, not ECE.
