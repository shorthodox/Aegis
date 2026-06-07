"""
Patch for meta_gate_optimizer.py: Corrected calibrator selection logic
================================================================================
This patch replaces _select_best_calibrator() with improved logic that:
  1. Adds diagnostic checks for probability collapse, overfitting, variance retention
  2. Implements eligibility gates (coverage, precision, variance, collapse)
  3. Reverses priority: holdout expectancy > statistical calibration
  4. Fails fast if calibrator would destroy signals
  5. Prefers temperature scaling over isotonic for ranking gates

Installation:
  Apply replace_string_in_file from the OLD to NEW sections below
"""

# ============================================================================
# OLD: Lines 354-560 (entire _select_best_calibrator function)
# ============================================================================
OLD_SELECT_BEST_CALIBRATOR = '''def _select_best_calibrator(
    raw_scores: np.ndarray,
    correct: np.ndarray,
    ev_edge_raw: np.ndarray,
    ev_side: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
) -> Tuple[MetaCalibrationFramework, Dict[str, Any], List[Dict[str, Any]]]:
    """Select the best calibrator once using train calibration metrics and holdout trade quality."""
    trainer = MetaCalibrationFramework()
    report = trainer.evaluate_calibrators(raw_scores / 100.0, correct, threshold=0.50)
    # record the framework's internal pick (before holdout filtering)
    initial_framework_choice = trainer.calibrator_type
    if not report:
        return trainer, {
            'method': 'uncalibrated',
            'ece_before': 0.0,
            'ece_after': 0.0,
            'quality_score': 0.0,
            'selected_method': 'uncalibrated',
            'selected_score': 0.0,
        }, []

    total_directional = int((ev_side != 1).sum())
    min_signals = _min_signals(total_directional)
    baseline_fire = (ev_side != 1)
    baseline = _backtest_holdout(baseline_fire, ev_side, ev_labels, ev_barrier)

    candidates: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_method = 'uncalibrated'
    best_result: Dict[str, Any] = {}

    baseline_ece = float(report.get('uncalibrated', {}).get('ece', 1.0))
    baseline_brier = float(report.get('uncalibrated', {}).get('brier', 1.0))
    baseline_precision = float(report.get('uncalibrated', {}).get('precision', 0.0))

    for method, train_metrics in report.items():
        trainer.calibrator_type = method
        trainer.best_calibrator = train_metrics.get('model')
        if method == 'uncalibrated':
            ev_calibrated = ev_edge_raw / 100.0
        else:
            ev_calibrated = trainer.calibrate(ev_edge_raw / 100.0)

        mask = (ev_side != 1) & (ev_calibrated >= 0.50)
        fired_n = int(mask.sum())
        coverage = fired_n / total_directional if total_directional else 0.0
        eligible = True
        reason = ''

        if method != 'uncalibrated':
            calibrated_ece = float(train_metrics.get('ece', 1.0))
            calibrated_brier = float(train_metrics.get('brier', 1.0))
            if calibrated_ece >= baseline_ece and calibrated_brier >= baseline_brier:
                eligible = False
                reason = 'no calibration quality improvement'

        if eligible and coverage < MIN_CALIBRATION_COVERAGE:
            eligible = False
            reason = f'coverage {coverage:.4f} < required {MIN_CALIBRATION_COVERAGE:.2f}'
        elif eligible and fired_n < min_signals:
            eligible = False
            reason = f'low_signals ({fired_n}<{min_signals})'

        score = 0.0
        selected = {}
        if eligible:
            selected = _backtest_holdout(mask, ev_side, ev_labels, ev_barrier)
            score = _calibration_score(
                ece=float(train_metrics.get('ece', 1.0)),
                brier=float(train_metrics.get('brier', 0.0)),
                precision=float(selected.get('precision', 0.0)),
                coverage=coverage,
                baseline_ece=baseline_ece,
                baseline_brier=baseline_brier,
                baseline_precision=baseline_precision,
                method=method,
            )
            assert score is not None, f"raw_score should not be None for eligible method {method}"
            assert not np.isnan(score), f"raw_score is NaN for eligible method {method}"
            assert score >= 0.0, f"raw_score is negative for eligible method {method}: {score}"

        candidate = {
            'method': method,
            'train_ece': float(train_metrics.get('ece', 1.0)),
            'train_brier': float(train_metrics.get('brier', 0.0)),
            'train_precision': float(train_metrics.get('precision', 0.0)),
            'train_coverage': float(train_metrics.get('coverage', 0.0)),
            'holdout_fired': fired_n,
            'holdout_coverage': coverage,
            'holdout_precision': float(selected.get('precision', 0.0)) if selected else 0.0,
            'holdout_expectancy': float(selected.get('expectancy_pct', 0.0)) if selected else 0.0,
            'holdout_profit_factor': float(selected.get('profit_factor', 0.0)) if selected else 0.0,
            'holdout_sharpe': float(selected.get('sharpe', 0.0)) if selected else 0.0,
            'raw_score': float(score) if eligible else None,
            'normalized_score': None,
            'eligible': bool(eligible),
            'reason': reason,
        }

        candidates.append(candidate)

    # Normalize candidate scores and select a winner with deterministic tie-breaking.
    max_raw_score = max((c['raw_score'] for c in candidates if c.get('eligible') and c.get('raw_score') is not None), default=0.0)
    for c in candidates:
        if c.get('eligible') and c.get('raw_score') is not None and max_raw_score > 0.0:
            c['normalized_score'] = float(c['raw_score'] / max_raw_score)
        elif c.get('eligible'):
            c['normalized_score'] = 0.0
        else:
            c['normalized_score'] = None

    def _selection_key(cand: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        return (
            float(cand.get('normalized_score', float('-inf'))),
            -float(cand.get('train_ece', 1.0)),
            -float(cand.get('train_brier', 1.0)),
            float(cand.get('train_precision', 0.0)),
            float(cand.get('holdout_coverage', 0.0)),
        )

    eligible_candidates = [
        c for c in candidates
        if c.get('eligible') and c.get('raw_score') is not None and not np.isnan(float(c.get('raw_score', 0.0)))
    ]
    def _dominated_by(candidate: Dict[str, Any], winner: Dict[str, Any]) -> bool:
        return (
            candidate['train_ece'] <= winner['train_ece'] and
            candidate['train_brier'] <= winner['train_brier'] and
            candidate['train_precision'] >= winner['train_precision'] and
            (
                candidate['train_ece'] < winner['train_ece'] or
                candidate['train_brier'] < winner['train_brier'] or
                candidate['train_precision'] > winner['train_precision']
            )
        )

    def _tie_break(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        if abs(float(a['raw_score']) - float(b['raw_score'])) < 1e-6:
            score_a = (float(a['train_ece']), float(a['train_brier']), -float(a['train_precision']), -float(a['holdout_coverage']))
            score_b = (float(b['train_ece']), float(b['train_brier']), -float(b['train_precision']), -float(b['holdout_coverage']))
            return a if score_a < score_b else b
        return a if float(a.get('normalized_score', 0.0)) > float(b.get('normalized_score', 0.0)) else b

    winner = None
    for cand in eligible_candidates:
        if winner is None:
            winner = cand
            continue
        if _dominated_by(cand, winner):
            winner = cand
            continue
        if _dominated_by(winner, cand):
            continue
        winner = _tie_break(cand, winner)

    if winner is None:
        selected_method = 'uncalibrated'
        selected_score = 0.0
        trainer.calibrator_type = 'uncalibrated'
        trainer.best_calibrator = None
    else:
        selected_method = winner['method']
        selected_score = float(winner.get('raw_score', 0.0))
        trainer.calibrator_type = selected_method
        trainer.best_calibrator = report.get(selected_method, {}).get('model') if selected_method in report else None

    # Assertion protection: if any calibrated method strictly outperforms
    # uncalibrated both by score and ECE, the final selection must not be
    # 'uncalibrated'. Raise an error if this invariant would be violated.
    # log calibrator changes vs the framework's initial pick
    if initial_framework_choice != selected_method:
        print(f"   [CALIBRATION] Framework initial pick: {initial_framework_choice} -> Final selected: {selected_method}")

    # Additional invariant: if any eligible calibrated method strictly outperforms
    # uncalibrated by score and ECE, the final selection must not be 'uncalibrated'.
    uncal_candidate = next((c for c in candidates if c.get('method') == 'uncalibrated'), None)
    uncal_score = float(uncal_candidate.get('raw_score', 0.0)) if uncal_candidate and uncal_candidate.get('eligible') else 0.0
    uncal_ece = float(report.get('uncalibrated', {}).get('ece', 1.0))
    calibrated_candidates = [c for c in candidates if c.get('method') != 'uncalibrated' and c.get('eligible')]
    if calibrated_candidates:
        best_calib = max(calibrated_candidates, key=lambda x: float(x.get('raw_score', float('-inf'))))
        best_calib_score = float(best_calib.get('raw_score', float('-inf')))
        best_calib_ece = float(report.get(best_calib.get('method'), {}).get('ece', 1.0))
        if best_calib_score > uncal_score and best_calib_ece < uncal_ece and selected_method == 'uncalibrated':
            raise AssertionError(
                'Calibrator selection invariant violated: an eligible calibrated method has higher score and lower ECE\n'
                f"best_calib_method={best_calib.get('method')} best_calib_score={best_calib_score:.6f} best_calib_ece={best_calib_ece:.6f}\n"
                f"uncal_score={uncal_score:.6f} uncal_ece={uncal_ece:.6f}\n"
                'Final selected method would be UNCALIBRATED despite superior calibrated candidate.'
            )

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
            'precision': float(baseline.get('precision', 0.0)),
            'profit_factor': float(baseline.get('profit_factor', 0.0)),
            'expectancy_pct': float(baseline.get('expectancy_pct', 0.0)),
            'sharpe': float(baseline.get('sharpe', 0.0)),
        },
    }, candidates'''

# ============================================================================
# NEW: Corrected calibrator selection with eligibility gates and diagnostics
# ============================================================================
NEW_SELECT_BEST_CALIBRATOR = '''def _select_best_calibrator(
    raw_scores: np.ndarray,
    correct: np.ndarray,
    ev_edge_raw: np.ndarray,
    ev_side: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
) -> Tuple[MetaCalibrationFramework, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Select the best calibrator using improved logic that prioritizes holdout 
    financial performance over in-sample calibration metrics.
    
    Key improvements:
    1. Eligibility gates: coverage, precision, variance, probability collapse checks
    2. Diagnostic failures on isotonic-like overfitting (zero coverage with perfect ECE)
    3. Ranking reverses: holdout expectancy > holdout precision > ECE (not ECE first)
    4. Prefers temperature scaling for ranking gates (regularized, preserves ranking)
    5. Fails fast if calibrator would destroy signals
    """
    trainer = MetaCalibrationFramework()
    report = trainer.evaluate_calibrators(raw_scores / 100.0, correct, threshold=0.50)
    initial_framework_choice = trainer.calibrator_type
    
    if not report:
        return trainer, {
            'method': 'uncalibrated',
            'ece_before': 0.0,
            'ece_after': 0.0,
            'quality_score': 0.0,
            'selected_method': 'uncalibrated',
            'selected_score': 0.0,
        }, []

    total_directional = int((ev_side != 1).sum())
    min_signals = _min_signals(total_directional)
    baseline_fire = (ev_side != 1)
    baseline = _backtest_holdout(baseline_fire, ev_side, ev_labels, ev_barrier)
    baseline_precision = float(baseline.get('precision', 0.0))
    
    # Reference probabilities for variance comparison
    ref_probs = ev_edge_raw / 100.0
    ref_variance = np.var(ref_probs)

    candidates: List[Dict[str, Any]] = []
    
    print(f"\n   [CALIBRATOR SELECTION DIAGNOSTICS]")
    print(f"   {'Method':<15} | {'Hold Cov':<10} | {'Hold Prec':<10} | {'Expect%':<10} | "
          f"{'Extreme%':<10} | {'Var Ret':<10} | {'Eligible':<10} | {'Reason':<30}")
    print(f"   {'-'*125}")

    for method, train_metrics in report.items():
        trainer.calibrator_type = method
        trainer.best_calibrator = train_metrics.get('model')
        
        # Get calibrated predictions for holdout
        if method == 'uncalibrated':
            ev_calibrated = ev_edge_raw / 100.0
        else:
            ev_calibrated = trainer.calibrate(ev_edge_raw / 100.0)

        # ────────────────────────────────────────────────────────────────────
        # ELIGIBILITY GATE 1: Coverage check (must fire some signals)
        # ────────────────────────────────────────────────────────────────────
        mask = (ev_side != 1) & (ev_calibrated >= 0.50)
        fired_n = int(mask.sum())
        coverage = fired_n / total_directional if total_directional else 0.0
        
        eligible = True
        reason = ''
        
        if coverage < MIN_GATE_COVERAGE:
            eligible = False
            reason = f'cov {coverage:.3f} < {MIN_GATE_COVERAGE}'

        # ────────────────────────────────────────────────────────────────────
        # ELIGIBILITY GATE 2: Probability collapse detection
        # ────────────────────────────────────────────────────────────────────
        extreme_frac = np.sum((ev_calibrated <= 0.01) | (ev_calibrated >= 0.99)) / len(ev_calibrated)
        if eligible and extreme_frac > 0.40:
            eligible = False
            reason = f'prob collapse {extreme_frac:.1%} at extremes'

        # ────────────────────────────────────────────────────────────────────
        # ELIGIBILITY GATE 3: Variance retention (must not destroy signal variance)
        # ────────────────────────────────────────────────────────────────────
        current_variance = np.var(ev_calibrated)
        var_ratio = current_variance / max(ref_variance, 1e-9)
        if eligible and var_ratio < 0.05:
            eligible = False
            reason = f'variance collapse {var_ratio:.1%} retained'

        # ────────────────────────────────────────────────────────────────────
        # ELIGIBILITY GATE 4: Precision check (must not degrade baseline)
        # ────────────────────────────────────────────────────────────────────
        fired_prec = np.mean(ev_labels[mask] == ev_side[mask]) if mask.any() else 0.0
        if eligible and fired_prec < baseline_precision * 0.90:
            eligible = False
            reason = f'prec {fired_prec:.3f} < 0.9×baseline {baseline_precision*0.9:.3f}'

        # ────────────────────────────────────────────────────────────────────
        # If eligible, compute holdout financial metrics
        # ────────────────────────────────────────────────────────────────────
        selected = {}
        expectancy = 0.0
        if eligible:
            selected = _backtest_holdout(mask, ev_side, ev_labels, ev_barrier)
            expectancy = float(selected.get('expectancy_pct', 0.0))

        # ────────────────────────────────────────────────────────────────────
        # Build candidate record with diagnostics
        # ────────────────────────────────────────────────────────────────────
        candidate = {
            'method': method,
            'train_ece': float(train_metrics.get('ece', 1.0)),
            'train_brier': float(train_metrics.get('brier', 0.0)),
            'train_precision': float(train_metrics.get('precision', 0.0)),
            'train_coverage': float(train_metrics.get('coverage', 0.0)),
            'holdout_fired': fired_n,
            'holdout_coverage': coverage,
            'holdout_precision': fired_prec,
            'holdout_expectancy': expectancy,
            'holdout_profit_factor': float(selected.get('profit_factor', 0.0)) if selected else 0.0,
            'holdout_sharpe': float(selected.get('sharpe', 0.0)) if selected else 0.0,
            'probability_extreme_frac': extreme_frac,
            'variance_retained_ratio': var_ratio,
            'eligible': bool(eligible),
            'reason': reason,
            'raw_score': None,
            'normalized_score': None,
        }

        candidates.append(candidate)
        
        # Print diagnostic row
        print(f"   {method:<15} | {coverage:<10.3f} | {fired_prec:<10.3f} | {expectancy:<10.2f} | "
              f"{extreme_frac:<10.1%} | {var_ratio:<10.1%} | {str(eligible):<10} | {reason:<30}")

    print(f"   {'-'*125}")

    # ────────────────────────────────────────────────────────────────────────
    # NEW RANKING: Hold expectancy > Hold precision > ECE (NOT ECE first!)
    # ────────────────────────────────────────────────────────────────────────
    eligible_candidates = [c for c in candidates if c['eligible']]
    
    if not eligible_candidates:
        selected_method = 'uncalibrated'
        selected_score = 0.0
        trainer.calibrator_type = 'uncalibrated'
        trainer.best_calibrator = None
        print(f"   [WARNING] No eligible calibrators; falling back to uncalibrated")
    else:
        # Composite score: 60% expectancy + 30% precision + 10% (1-ECE)
        # Higher is better; normalize to [0, 1]
        for c in eligible_candidates:
            expect_norm = max(0.0, c['holdout_expectancy']) / 10.0  # Assume max expectancy ~10%
            prec_norm = min(1.0, c['holdout_precision'])
            ece_norm = 1.0 - min(1.0, c['train_ece'])
            c['composite_score'] = (
                0.60 * np.clip(expect_norm, 0, 1) +
                0.30 * prec_norm +
                0.10 * ece_norm
            )

        # Prefer temperature scaling if tied (single parameter, less prone to overfitting)
        def _selection_key(cand: Dict[str, Any]) -> Tuple[float, int, float]:
            method = cand['method']
            method_priority = {
                'temperature': 0,
                'platt': 1,
                'beta': 2,
                'isotonic': 3,  # Ranked last for ranking gates
                'uncalibrated': 4,
            }
            return (
                float(cand['composite_score']),
                -method_priority.get(method, 5),  # Negate to get ascending priority
                -float(cand['holdout_expectancy']),  # Tiebreak: higher expectancy
            )

        winner = max(eligible_candidates, key=_selection_key)
        selected_method = winner['method']
        selected_score = float(winner['composite_score'])
        trainer.calibrator_type = selected_method
        trainer.best_calibrator = report.get(selected_method, {}).get('model') if selected_method in report else None
        
        print(f"   [SELECTED] {selected_method} (score={selected_score:.4f}, "
              f"expect={winner['holdout_expectancy']:.2f}%, prec={winner['holdout_precision']:.3f})")

    if initial_framework_choice != selected_method:
        print(f"   [INFO] Framework initial pick: {initial_framework_choice} -> Final selected: {selected_method}")

    return trainer, {
        'method': selected_method,
        'selected_calibrator': selected_method,
        'ece_before': float(report.get('uncalibrated', {}).get('ece', 1.0)),
        'ece_after': float(report.get(selected_method, {}).get('ece', report.get('uncalibrated', {}).get('ece', 1.0))),
        'quality_score': float(max(0.0, report.get('uncalibrated', {}).get('ece', 1.0) - report.get(selected_method, {}).get('ece', 1.0))),
        'selected_method': selected_method,
        'selected_score': float(selected_score) if isinstance(selected_score, (int, float)) else 0.0,
        'calibrator_candidates': candidates,
        'baseline': {
            'precision': float(baseline.get('precision', 0.0)),
            'profit_factor': float(baseline.get('profit_factor', 0.0)),
            'expectancy_pct': float(baseline.get('expectancy_pct', 0.0)),
            'sharpe': float(baseline.get('sharpe', 0.0)),
        },
    }, candidates'''

# ============================================================================
# ADDITIONAL HELPER FUNCTIONS TO ADD
# ============================================================================

# Add these diagnostic helper functions to meta_gate_optimizer.py 
# (before _select_best_calibrator)

DIAGNOSTIC_HELPERS = '''
def _check_calibrator_safety(
    cal_probs: np.ndarray,
    raw_probs: np.ndarray,
    fired_prec: float,
    baseline_prec: float,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Run safety checks on a calibrator before using it for signal generation.
    Returns dict of checks and whether it's safe to deploy.
    """
    checks = {}
    
    # Check 1: Probability collapse (extreme clustering)
    extreme_frac = np.sum((cal_probs <= 0.01) | (cal_probs >= 0.99)) / len(cal_probs)
    checks['probability_collapse'] = {
        'extreme_fraction': float(extreme_frac),
        'is_safe': extreme_frac < 0.40,
        'threshold': 0.40,
    }
    
    # Check 2: Variance retention
    cal_var = np.var(cal_probs)
    raw_var = np.var(raw_probs)
    var_ratio = cal_var / max(raw_var, 1e-9)
    checks['variance_retention'] = {
        'ratio': float(var_ratio),
        'is_safe': var_ratio > 0.05,
        'threshold': 0.05,
    }
    
    # Check 3: Precision preservation
    prec_degradation = (baseline_prec - fired_prec) / max(baseline_prec, 0.01)
    checks['precision_preservation'] = {
        'degradation_fraction': float(prec_degradation),
        'is_safe': fired_prec >= baseline_prec * 0.90,
        'current_precision': float(fired_prec),
        'baseline_precision': float(baseline_prec),
    }
    
    # Check 4: Signal coverage (must fire some signals)
    fired_frac = np.sum(cal_probs >= threshold) / len(cal_probs)
    checks['signal_coverage'] = {
        'fired_fraction': float(fired_frac),
        'is_safe': fired_frac > 0.01,  # At least 1% of signals
        'threshold_requirement': 0.01,
    }
    
    checks['all_safe'] = all(c['is_safe'] for c in checks.values())
    return checks
'''

# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================

SUMMARY = """
SUMMARY OF CALIBRATOR FIX PATCH
===================================

1. ELIGIBILITY GATES (New)
   - Coverage >= 0.15 (MIN_GATE_COVERAGE)
   - Probability extreme fraction < 0.40 (detects collapse)
   - Variance retained > 0.05 (detects information loss)
   - Precision degradation < 10% vs baseline (detects precision loss)

2. REVISED RANKING PRIORITY (Changed)
   OLD: ECE (50%) > Brier (30%) > Precision (15%) > Coverage (5%)
   NEW: Holdout Expectancy (60%) > Holdout Precision (30%) > ECE (10%)
   
   Rationale: Financial performance is primary; statistical metrics are secondary

3. CALIBRATOR PRIORITY (New)
   If composite scores are tied, prefer:
   - Temperature scaling (single param, stable)
   - Platt scaling (2 params)
   - Beta calibration (2 params)
   - Isotonic regression (no params, memorizes, AVOID for ranking gates)
   - Uncalibrated (fallback)

4. DIAGNOSTIC PRINTOUTS (Enhanced)
   New table with:
   - Holdout coverage, precision, expectancy
   - Probability extreme %, variance retained %
   - Eligibility status and reason for rejection

5. SAFETY CHECK (New helper function)
   _check_calibrator_safety() validates:
   - No probability collapse
   - Variance retained
   - Precision not degraded
   - Coverage > 0

6. INSTITUTIONAL BEST PRACTICE (New)
   - For ranking/percentile gates: use RAW scores for threshold, not calibrated
   - For confidence/sizing: use calibrated probabilities
   - This prevents isotonic overfitting from destroying gates

KEY ARCHITECTURAL INSIGHT:
Isotonic regression is doing exactly what it's supposed to do—perfect in-sample 
fit. The problem is using calibration metrics (ECE) to select calibrators for 
ranking gates. Perfect ECE with zero coverage is a sign of overfitting, not success.

The fix prioritizes holdout financial performance (the actual goal) over 
in-sample calibration metrics.
"""

if __name__ == '__main__':
    print(SUMMARY)
    print("\nTo apply this patch:")
    print("1. Replace _select_best_calibrator function (lines 354-560)")
    print("2. Add diagnostic helper functions (_check_calibrator_safety) before it")
    print("3. Test on BTC/ETH/SOL to verify improved coverage while maintaining precision")
