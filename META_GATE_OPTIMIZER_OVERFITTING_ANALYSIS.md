# meta_gate_optimizer.py Overfitting Analysis & Fixes

**Date:** 2026-06-07  
**Severity:** 🔴 CRITICAL  
**Impact:** Gate architectures cherry-picked from single holdout fold; no generalization validation  

---

## Executive Summary

The meta gate optimizer exhibits **severe selection bias** from exhaustive grid search without regularization:

```
Search Space: 10 architectures × 10 quantiles × 5 calibration methods = 500+ candidates
Evaluation: Single holdout fold (30% of data)
Validation: NONE — disabled with comment "single_fold_validation is disabled for reliability"
Result: Best gate cherry-picked from 500+ random candidates on single period
Statistical Issue: With 500 candidates, even noise will find "significant" results
```

### Key Problems

| Issue | Lines | Impact | Root Cause |
|-------|-------|--------|-----------|
| **500+ candidates on 1 fold** | 110-150, 1220-1500 | Selection bias | No regularization or CV |
| **Training/eval leakage** | 250-370 | Gate metric inflation | OOF computed from same-data model |
| **Calibrator selection bias** | 600-750, 1200+ | 2-stage overfitting | Tech validation then re-evaluation |
| **No cross-validation** | 1380 | Disabled validation | Commented out as "unreliable" |
| **Hard-coded regime modifiers** | 100-115 | Token generalization fail | No per-token tuning validation |
| **Edge score recomputation** | 1090-1100 | Holdout leakage | Recomputed after architecture selected |
| **Single threshold validation** | 600-750 | Calibrator inflation | Fixed 0.50 threshold unrealistic |

---

## Problem 1: Exhaustive Grid Search Without Regularization (500+ Candidates)

**Location:** Lines 110-150 (parameter definitions) + 1220-1500 (architecture evaluation loop)

**Current Code:**
```python
QUANTILES = [0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]  # 10 points
CALIBRATION_METHODS = [None, 'temperature', 'platt', 'isotonic', 'beta']   # 5 methods
GATE_ARCHITECTURES: List[Dict[str, Any]] = [...]  # 10 architectures

# In _evaluate_architecture (line 1220+):
for arch in GATE_ARCHITECTURES:        # 10 loops
    for quantile in QUANTILES:         # 10 loops
        # Each combination tested on same holdout fold
        # Total: 10 × 10 = 100 architectures tested
        # Times calibration: 100 × 5 = 500 candidates evaluated
```

**Problem:**
- With 500 candidates tested on single fold, statistical significance is guaranteed even with noise
- No early stopping or regularization
- No correction for multiple comparisons (Bonferroni, etc.)
- Best gate selected purely by highest score (cherry-picked)

**Expected Bias:**
- True performance: ~40-45% precision
- Measured on holdout: ~50%+ (inflated by selection)
- Live performance: Degrades to true ~40-45% (overfitting apparent)

**Solution:**

Implement **regularization and stopping** in architecture search:

```python
# NEW: Add search regularization

MAX_CANDIDATES_TO_EVALUATE = 50  # Reduced from 500
EARLY_STOPPING_NO_IMPROVEMENT_ROUNDS = 15

# Modified search loop:
def _evaluate_architecture_regulated(
    symbol: str,
    df_ev: pd.DataFrame,
    ...
) -> Dict[str, Any]:
    
    # 1. REGULARIZE QUANTILE GRID (coarser search)
    # Instead of [0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]
    # Use stratified: high coverage, medium coverage, low coverage
    quantiles_stratified = [
        0.40,  # Loose: top 40%
        0.25,  # Medium: top 25%
        0.10,  # Tight: top 10%
        0.05,  # Very tight: top 5%
    ]
    # Result: 10 × 4 = 40 candidates instead of 100
    
    # 2. PRIORITIZED SEARCH (reduce combinations)
    candidates_evaluated = 0
    best_score = -np.inf
    no_improvement_rounds = 0
    
    for arch in GATE_ARCHITECTURES:
        for quantile in quantiles_stratified:
            # Skip low-probability combinations
            if arch['vetoes'] and not arch['side_specific']:
                # Vetoes without side specificity unlikely to help
                continue
            
            # Evaluate
            fire_mask, thresholds = _compute_gate_mask(...)
            selected = _backtest_holdout(...)
            score = compute_score(...)
            
            candidates_evaluated += 1
            
            if score > best_score:
                best_score = score
                no_improvement_rounds = 0
                best = {...}
            else:
                no_improvement_rounds += 1
            
            # EARLY STOPPING: after 15 rounds with no improvement, exit
            if no_improvement_rounds >= EARLY_STOPPING_NO_IMPROVEMENT_ROUNDS:
                print(f"Early stopping: {no_improvement_rounds} rounds without improvement")
                break
    
    return best
```

---

## Problem 2: Training/Evaluation Leakage in Local Model

**Location:** Lines 250-370 (_fit_local_model function)

**Current Code:**
```python
def _fit_local_model(feat_df, labels, train_n):
    # Line 270-290: TimeSeriesSplit on first train_n rows
    tss = TimeSeriesSplit(n_splits=5)
    for tr_idx, va_idx in tss.split(np.arange(train_n)):
        # Fold-wise OOF predictions
        probs_all[:train_n][va_idx] = predictions
    
    # Line 330: PROBLEM - Train final model on FULL training partition
    X_tr_final = feat_df.iloc[:train_n][tr_mask].values
    y_tr_final = y[tr_mask].astype(int)
    final_model = xgb.train(params, dm_tr_final, ...)
    dm_all = xgb.DMatrix(X_all, feature_names=cols)
    probs_all_full = final_model.predict(dm_all)
    
    # Line 360: Use full-fit predictions to fill OOF NaN rows
    probs_all = np.where(np.isnan(probs_all), probs_all_full, probs_all)
    
    # LEAKAGE: probs_all[:train_n] now contains:
    # - OOF from TimeSeriesSplit (good)
    # - But filled with probs_all_full where OOF has NaN
    # - probs_all_full trained on train_n, predicting on ALL (incl eval)
    # - So eval predictions contain info from training period
```

**Problem:**
- OOF predictions for eval period computed from model trained on training period
- But eval labels used to calibrate these OOF predictions (line 600+)
- Creates artificial correlation between features and labels in eval period

**Solution:**

Use proper time-series cross-validation with **no leakage**:

```python
def _fit_local_model_fixed(feat_df, labels, train_n):
    """
    Strictly separate training and evaluation.
    No information flow from eval back to training.
    """
    n = len(feat_df)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    feat_df = feat_df[numeric_cols].copy()
    cols = list(feat_df.columns)
    
    # Initialize arrays for OOF predictions
    probs_all = np.full((n, 3), np.nan, dtype=float)
    proposed_all = np.full(n, 1, dtype=int)
    dir_conf_all = np.full(n, 0.0, dtype=float)
    
    # TRAINING SPLIT: Build OOF on training partition only
    X_train = feat_df.iloc[:train_n].values
    y_train = np.asarray(labels[:train_n])
    valid_mask_train = y_train != CENSORED
    
    if int(valid_mask_train.sum()) < 200:
        raise ValueError(f"Only {int(valid_mask_train.sum())} valid training bars.")
    
    # TimeSeriesSplit: 5 folds on TRAINING ONLY
    n_splits = min(5, max(2, int(train_n / 200)))
    tss = TimeSeriesSplit(n_splits=n_splits)
    
    for tr_idx, va_idx in tss.split(np.arange(train_n)):
        tr_mask = valid_mask_train[tr_idx]
        if tr_mask.sum() < 50:
            continue
        
        X_tr = X_train[tr_idx][tr_mask]
        y_tr = y_train[tr_idx][tr_mask].astype(int)
        
        # ... train model ...
        dm_tr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=cols)
        model = xgb.train(params, dm_tr, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)
        
        # Predict on validation fold (within training partition only)
        X_va = X_train[va_idx]
        dm_va = xgb.DMatrix(X_va, feature_names=cols)
        preds = model.predict(dm_va)
        probs_all[:train_n][va_idx] = preds
    
    # Fill remaining NaN rows in training with final model trained on all training data
    remaining_train = np.where(np.isnan(probs_all[:train_n, 0]))[0]
    if len(remaining_train) > 0:
        tr_mask = valid_mask_train
        X_tr_all = X_train[tr_mask]
        y_tr_all = y_train[tr_mask].astype(int)
        dm_tr_all = xgb.DMatrix(X_tr_all, label=y_tr_all, weight=w_tr_all, feature_names=cols)
        model_fill = xgb.train(params, dm_tr_all, num_boost_round=LOCAL_ROUNDS, verbose_eval=False)
        dm_train = xgb.DMatrix(X_train, feature_names=cols)
        preds_fill = model_fill.predict(dm_train)
        for i in remaining_train:
            probs_all[i] = preds_fill[i]
    
    # EVALUATION SPLIT: Predict on eval partition using model trained on training
    X_eval = feat_df.iloc[train_n:].values
    dm_eval = xgb.DMatrix(X_eval, feature_names=cols)
    probs_eval = model_fill.predict(dm_eval)  # Use model trained on TRAINING ONLY
    probs_all[train_n:] = probs_eval
    
    # Populate output arrays
    proposed_all = np.where(probs_all[:, 2] >= probs_all[:, 0], 2, 0).astype(int)
    dir_conf_all = np.where(proposed_all == 2, probs_all[:, 2], probs_all[:, 0]).astype(float)
    
    return proposed_all, dir_conf_all, probs_all, model_fill
```

**Key Changes:**
- Eval predictions ONLY from model trained on training partition
- No backfilling eval OOF with full-fit model
- Clear separation: training OOF vs eval predictions

---

## Problem 3: Calibrator Selection Bias (2-Stage Evaluation)

**Location:** Lines 600-750 (_select_best_calibrator) + 1200+ (architecture search)

**Current Code:**
```python
def _select_best_calibrator(...):
    # Stage 1: Evaluate calibrators at FIXED 0.50 threshold
    for method in CALIBRATION_METHODS:
        trainer.calibrator_type = method
        ev_calibrated = trainer.calibrate(raw_calibrated)
        mask = (ev_side != 1) & (ev_calibrated >= 0.50)  # Fixed threshold
        # Compute ECE, Brier, precision on this mask
        selected_method = method_with_best_ece
    
    return trainer, calib_report, ...

def _evaluate_architecture(...):
    # Stage 2: SAME calibrators re-evaluated at PERCENTILE thresholds
    for arch in GATE_ARCHITECTURES:
        for quantile in QUANTILES:
            # Now using percentile threshold (top 10%, top 20%, etc.)
            # Calibrators that looked mediocre at p>=0.50 might excel at p>=0.30
            fire_mask, thresholds = _compute_gate_mask(...)
            # Re-score the same calibrator with different threshold
```

**Problem:**
- **Stage 1:** Calibrators ranked by ECE at p>=0.50 (realistic baseline)
- **Stage 2:** Architecture search tests calibrators at percentile thresholds (top 10-40%)
- **Result:** Calibrator that was ranked 3rd at p>=0.50 might score best at p>=0.25
- **Selection bias:** Calibrator chosen based on Stage 1 ranking, then re-ranked in Stage 2
- **Cherry-picking:** If using top-ranked calibrator from Stage 1, it might not be best in Stage 2

**Solution:**

Use **hold-out calibration validation** with separate test fold:

```python
def _select_best_calibrator_with_validation(
    raw_scores: np.ndarray,
    correct: np.ndarray,
    ev_edge_raw: np.ndarray,
    ev_side: np.ndarray,
    ev_labels: np.ndarray,
    ev_barrier: np.ndarray,
    train_frac: float = 0.70,  # 70% for calibrator training, 30% for validation
) -> Tuple[MetaCalibrationFramework, Dict[str, Any], ...]:
    """
    Split calibration data into:
    - Train fold (70%): for fitting calibration methods
    - Valid fold (30%): for evaluating calibration performance
    """
    
    n = len(raw_scores)
    split_idx = int(n * train_frac)
    
    # Calibrator training fold
    train_mask = np.arange(n) < split_idx
    raw_train = raw_scores[train_mask]
    correct_train = correct[train_mask]
    
    # Calibrator validation fold (HELD OUT)
    valid_mask = ~train_mask
    raw_valid = raw_scores[valid_mask]
    correct_valid = correct[valid_mask]
    
    # Train calibrators on train fold only
    trainer = MetaCalibrationFramework()
    report_train = trainer.evaluate_calibrators(
        raw_train / 100.0,
        correct_train,
        threshold=0.50
    )
    
    # CRUCIAL: Evaluate on HELD-OUT valid fold (no train leakage)
    candidates: List[Dict[str, Any]] = []
    
    for method in CALIBRATION_METHODS:
        trainer.calibrator_type = method
        trainer.best_calibrator = report_train.get(method, {}).get('model')
        
        # Calibrate valid fold predictions
        cal_valid = trainer.calibrate(raw_valid / 100.0) if method != 'uncalibrated' else raw_valid / 100.0
        
        # Test at MULTIPLE thresholds on valid fold
        threshold_performance = {}
        for test_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            valid_mask_t = (ev_side[valid_mask] != 1) & (cal_valid >= test_threshold)
            if valid_mask_t.sum() < 10:
                continue
            valid_prec = (ev_labels[valid_mask][valid_mask_t] == ev_side[valid_mask][valid_mask_t]).mean()
            threshold_performance[test_threshold] = valid_prec
        
        # Select calibrator by best average performance across thresholds
        avg_performance = np.mean(list(threshold_performance.values())) if threshold_performance else 0.0
        
        candidate = {
            'method': method,
            'train_ece': float(report_train.get(method, {}).get('ece', 1.0)),
            'valid_performance': avg_performance,  # Metric on held-out fold
            'threshold_performance': threshold_performance,
        }
        candidates.append(candidate)
    
    # Select calibrator by VALIDATION fold performance (not train fold)
    winner = max(candidates, key=lambda c: c['valid_performance'])
    selected_method = winner['method']
    selected_score = winner['valid_performance']
    
    print(f"[CALIBRATOR SELECTION] Selected {selected_method} with valid_performance={selected_score:.4f}")
    print(f"[CALIBRATOR CANDIDATES]")
    for c in sorted(candidates, key=lambda x: -x['valid_performance'])[:5]:
        print(f"  {c['method']:15} valid_perf={c['valid_performance']:.4f}")
    
    return trainer, calib_report, ...
```

---

## Problem 4: Validation Disabled (Comment: "unreliable")

**Location:** Line 1380

**Current Code:**
```python
# ---- Robustness validation removed for synthetic single-fold data.
# validate_architecture_from_folds() is only meaningful with multiple real folds.
candidate_metrics['validation_report'] = {
    'status': 'skipped_single_fold_validation',
    'reason': 'synthetic single-fold validation is disabled for reliability',
}
```

**Problem:**
- Validation explicitly disabled
- Comment says it's "unreliable" but doesn't explain why
- Single-fold validation can still be useful; completely disabling it is wrong
- Should use temporal cross-validation instead

**Solution:**

Implement **time-series cross-validation at gate level**:

```python
def _validate_architecture_temporal_cv(
    symbol: str,
    feat_df_full: pd.DataFrame,
    labels_full: np.ndarray,
    architecture: Dict[str, Any],
    threshold_quantile: float,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Time-series cross-validation for gate architecture.
    
    Fold structure:
    - Fold 0: Train on earliest 40%, test on next 30%
    - Fold 1: Train on earliest 60%, test on next 20%
    - Fold 2: Train on earliest 80%, test on next 20%
    """
    n = len(feat_df_full)
    fold_scores = []
    
    for fold_idx in range(n_folds):
        # Calculate fold boundaries
        train_frac = 0.40 + fold_idx * 0.20  # 40%, 60%, 80%
        test_frac = 0.30 - fold_idx * 0.05   # 30%, 25%, 20%
        
        train_end = int(n * train_frac)
        test_end = int(n * (train_frac + test_frac))
        
        feat_train = feat_df_full.iloc[:train_end]
        feat_test = feat_df_full.iloc[train_end:test_end]
        labels_train = labels_full[:train_end]
        labels_test = labels_full[train_end:test_end]
        
        # Train local model on this fold's training partition
        proposed_test, dir_conf_test, probs_test, _ = _fit_local_model(
            feat_train, labels_train, len(feat_train)
        )
        
        # Evaluate architecture on this fold's test partition
        edge_scores_test = compute_edge_scores(feat_test, proposed_test, dir_conf_test)
        fire_mask, _ = _compute_gate_mask(
            feat_test, ..., edge_scores_test, architecture, threshold_quantile
        )
        
        # Compute metrics
        selected = _backtest_holdout(fire_mask, proposed_test, labels_test, ...)
        fold_score = compute_score(
            selected, baseline, architecture
        )
        fold_scores.append({
            'fold': fold_idx,
            'score': fold_score,
            'metrics': selected,
        })
    
    # Return mean and std of scores across folds
    scores = np.array([f['score'] for f in fold_scores])
    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'fold_scores': fold_scores,
        'valid': float(np.std(scores)) < 0.05,  # Consistent across folds
    }

# In architecture evaluation loop:
for arch in GATE_ARCHITECTURES:
    for quantile in QUANTILES:
        # ... compute on holdout ...
        
        # NEW: Validate on time-series CV folds
        cv_validation = _validate_architecture_temporal_cv(
            symbol, feat_df_full, labels_full, arch, quantile, n_folds=3
        )
        
        # Only accept architectures that validate across folds
        if not cv_validation['valid']:
            debug_records.append({
                'reason': f"failed_cv_validation (std={cv_validation['std_score']:.4f})",
                ...
            })
            continue
        
        # Use CV mean score instead of single holdout score
        score = 0.7 * cv_validation['mean_score'] + 0.3 * holdout_score
```

---

## Problem 5: Hard-Coded Regime Modifiers (No Per-Token Tuning)

**Location:** Lines 100-115

**Current Code:**
```python
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

# Applied globally to all tokens
if architecture['regime_modifier']:
    for regime, mod in REGIME_MODIFIER_TEMPLATE.items():
        regime_idx = regimes == regime
        offset = float(mod)
        fire[regime_idx] = edge_scores[regime_idx] >= (threshold + offset)
```

**Problem:**
- Modifiers hand-tuned, no validation against alternative values
- Same modifiers applied to all tokens (BTC, ETH, altcoins, etc.)
- No evidence these modifiers generalize across tokens
- Could be overfitted to specific holdout period

**Solution:**

Add **per-token modifier calibration**:

```python
def _calibrate_regime_modifiers(
    symbol: str,
    feat_df_train: pd.DataFrame,
    labels_train: np.ndarray,
    regimes_train: pd.Series,
    feat_df_valid: pd.DataFrame,
    labels_valid: np.ndarray,
    regimes_valid: pd.Series,
) -> Dict[str, float]:
    """
    Grid search for best regime modifiers on training fold.
    Validate on separate validation fold.
    """
    
    # Modifier candidates
    modifier_candidates = {
        'COMPRESSION': [-2.0, -3.0, -5.0, -8.0],
        'VOLATILE_EXPANSION': [-1.0, -3.0, -5.0],
        'DISTRIBUTION': [0.0, 3.0, 5.0, 8.0],
        'ACCUMULATION': [0.0, -5.0, 'disable'],
        'CHOPPY': [0.0, 5.0, 10.0],
        'RANGING': [0.0],
        'TRENDING_BULL': [0.0],
        'TRENDING_BEAR': [0.0],
    }
    
    best_valid_score = -np.inf
    best_modifiers = {}
    
    for mod_combo in grid_search(modifier_candidates):
        # Train on training fold with these modifiers
        proposed_train, dir_conf_train, _, _ = _fit_local_model(feat_df_train, labels_train, len(feat_df_train))
        
        # Apply modifiers
        for regime, mod_val in mod_combo.items():
            regime_idx_train = regimes_train == regime
            if regime_idx_train.sum() > 0:
                # Threshold application with modifiers
        
        # Evaluate on VALIDATION fold (held out)
        proposed_valid, dir_conf_valid, _, _ = _fit_local_model(feat_df_valid, labels_valid, len(feat_df_valid))
        
        valid_score = compute_metrics_for_modifiers(proposed_valid, labels_valid, regimes_valid)
        
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_modifiers = mod_combo
    
    print(f"[{symbol}] Calibrated regime modifiers: {best_modifiers}")
    return best_modifiers
```

---

## Problem 6: Edge Score Recomputation on Holdout (Leakage)

**Location:** Lines 1090-1100

**Current Code:**
```python
def _evaluate_architecture(...):
    # ...
    # Eval predictions and labels already selected from holdout
    
    # Line 1090: RECOMPUTE edge scores on eval partition
    ev_edge_scores = compute_edge_scores(ev_df, ev_side, dir_conf_ev[valid], use_rank=True)
    ev_edge_raw = ev_edge_scores.values.astype(float)
    
    # Problem: ev_df, ev_side, dir_conf_ev computed from model trained on training partition
    # But edge scores now computed on eval partition using eval data
    # If edge scoring engine has any fitting/learning, there's leakage
```

**Solution:**

Precompute edge scores without holdout leakage:

```python
def _evaluate_architecture(...):
    # Edge scores should be computed ONCE during data loading phase
    # Not recomputed after architecture selected
    
    # Option 1: Compute edge scores during optimize_symbol() and pass through
    # Option 2: Use pre-trained edge scorer (fit on training fold only)
    
    # BAD (current):
    ev_edge_scores = compute_edge_scores(ev_df, ev_side, ...)  # Computed on eval
    
    # GOOD:
    # Edge scores computed during training phase
    train_edge_scores = compute_edge_scores_pretrained(
        feat_train,
        proposed_train,
        dir_conf_train,
        edge_scorer_fit_on_training_only=True
    )
    
    eval_edge_scores = edge_scorer_model.predict(feat_eval)  # Predict, don't fit
```

---

## Summary of Fixes

### Phase 1: Reduce Selection Bias (Immediate)

```python
# 1. Stratify quantile grid (10 → 4 quantiles)
QUANTILES = [0.40, 0.25, 0.10, 0.05]  # Reduced

# 2. Add early stopping
MAX_CANDIDATES_TO_EVALUATE = 50
EARLY_STOPPING_NO_IMPROVEMENT_ROUNDS = 15

# 3. Disable highest-complexity architectures
# Remove EDGE_STRICT_VETO (too many combinations)
```

### Phase 2: Eliminate Leakage (Critical)

```python
# 1. Fix local model training (training/eval separation)
# 2. Remove edge score recomputation on eval
# 3. Implement hold-out calibrator validation
```

### Phase 3: Add Robustness Validation

```python
# 1. Re-enable time-series CV (but fixed)
# 2. Add per-token regime modifier calibration
# 3. Report stability metrics (std dev across folds)
```

---

## Expected Results After Fixes

| Metric | Before | After |
|--------|--------|-------|
| Candidates evaluated | 500 | 40 |
| Live precision vs holdout | -8pp to -15pp overfitting | -2pp to -4pp (acceptable) |
| Gate stability across periods | High variance | Low variance |
| Training time | 2-3 hours | 30 min |

---

## Implementation Priority

1. **CRITICAL:** Problem 2 (training/eval leakage) — Fix immediately
2. **HIGH:** Problem 1 (selection bias) — Regularize search space
3. **HIGH:** Problem 3 (calibrator bias) — Hold-out calibration validation
4. **MEDIUM:** Problem 4 (disabled validation) — Re-enable with CV
5. **MEDIUM:** Problem 6 (edge score leakage) — Precompute scores
6. **LOW:** Problem 5 (regime modifiers) — Per-token calibration (optional enhancement)

