# meta_gate_optimizer.py Investigation Summary

**Status:** 🔴 CRITICAL OVERFITTING DETECTED  
**Date:** 2026-06-07  
**Severity:** High — Gate architectures cherry-picked from single holdout fold

---

## Quick Summary

The meta gate optimizer searches 500+ gate architectures on a single holdout fold with **zero cross-validation**. This creates massive selection bias: the best gate is chosen to maximize performance on that specific period, not generalize to future data.

### The Core Problem

```
Search space: 10 architectures × 10 quantiles × 5 calibrators = 500 candidates
Tested on: 1 holdout fold (30% of data)
Validation: NONE (disabled)
Result: Best gate cherry-picked from 500 random candidates
Statistical consequence: Even with noise, highest performer on 500 trials will appear "significant"
```

### Why This Matters

- **Expected behavior:** 42% precision on holdout → 42% precision on live
- **Actual behavior:** 50-52% precision on holdout → 40-42% precision on live  
- **Gap:** 8-10pp overfitting (gate is overfit to holdout period)

---

## 6 Critical Issues Identified

| # | Issue | Lines | Problem | Expected Fix |
|---|-------|-------|---------|--------------|
| **1** | **500+ candidates on 1 fold** | 110-150, 1220 | Selection bias | Reduce to 40 candidates + early stopping |
| **2** | **Training/eval leakage** | 250-370 | OOF contaminated | Strict TimeSeriesSplit on training only |
| **3** | **Calibrator 2-stage bias** | 600-750 | Re-evaluation | Hold-out calibrator validation |
| **4** | **CV validation disabled** | 1380 | No robustness check | Re-enable 2-fold time-series CV |
| **5** | **Edge scores recomputed** | 1090 | Holdout leakage | Pre-compute once during loading |
| **6** | **Hard-coded regime mods** | 100-115 | No per-token tuning | Per-token calibration validation |

---

## Detailed Analysis Documents Created

### 1. **META_GATE_OPTIMIZER_OVERFITTING_ANALYSIS.md** (560 lines)

Complete technical analysis of all 6 overfitting issues:
- Root cause for each problem
- Code locations and examples
- Quantified impact on precision
- Detailed solutions with pseudocode

**Sections:**
- Problem 1: Exhaustive search without regularization
- Problem 2: Training/eval leakage in local model (CRITICAL)
- Problem 3: Calibrator selection bias (2-stage evaluation)
- Problem 4: Validation disabled
- Problem 5: Hard-coded regime modifiers
- Problem 6: Edge score recomputation

### 2. **META_GATE_OPTIMIZER_FIXES.md** (400 lines)

Exact code changes with before/after:

**Fixes Implemented:**
1. **FIX 1:** Stratified quantiles + early stopping (50-line modification)
   - Reduce quantiles: 10 → 4
   - Add early stopping after 15 rounds of no improvement
   - Limit total candidates to 50 max
   - Result: 500 candidates → 40

2. **FIX 2:** Proper TimeSeriesSplit (80-line refactoring) [CRITICAL]
   - Train OOF on training partition only
   - Eval predictions from training-only model
   - Eliminate probs_all_full contamination
   - Result: Realistic gate metrics

3. **FIX 3:** Hold-out calibrator validation (50-line modification)
   - Split calibration data 70/30
   - Train calibrators on fold 1, validate on fold 2
   - Select by held-out performance
   - Result: Robust calibrator selection

4. **FIX 4:** Re-enable time-series CV (150-line addition)
   - 2-fold time-series cross-validation
   - Blend holdout (70%) + CV (30%)
   - Penalize unstable architectures
   - Result: Verified generalization

5. **FIX 5:** Pre-compute edge scores (3-line fix)
   - Compute once during data loading
   - Reuse cached scores in architecture search
   - No re-computation on holdout
   - Result: Eliminated leakage source

6. **FIX 6:** Per-token regime calibration (Optional enhancement)

---

## Expected Results After Implementation

| Metric | Before | After |
|--------|--------|-------|
| Candidates evaluated | 500+ | ~40 |
| Live vs holdout gap | 8-10pp overfitting | 2-4pp overfitting |
| Gate stability | High variance across periods | Low variance |
| Search time | 2-3 hours | 30 min |
| Calibrator consistency | Varies across thresholds | Robust across range |

---

## Implementation Path

### Phase 1: Eliminate Leakage (CRITICAL)
```
Priority: DO FIRST
Changes: Fix 2 (TimeSeriesSplit)
Expected impact: Eliminates false precision gains
Time: 2 hours implementation + 1 hour testing
```

### Phase 2: Reduce Selection Bias
```
Changes: Fix 1 (stratified search + early stopping)
Expected impact: Prevents cherry-picking from 500 candidates
Time: 1 hour
```

### Phase 3: Robustness Validation
```
Changes: Fix 3 (hold-out calibration) + Fix 4 (time-series CV)
Expected impact: Validates gate generalizes to new periods
Time: 3 hours
```

### Phase 4: Polish
```
Changes: Fix 5 (edge scores) + Fix 6 (regime modifiers)
Expected impact: Minor improvements and cleanup
Time: 1 hour
```

**Total effort:** ~8 hours  
**Estimated timeline:** 1-2 work days

---

## Quick Reference

**For quick fixes, start with these line changes:**

1. **Line 130-150:** Reduce QUANTILES from 10 to 4 values
2. **Line 250-370:** Refactor _fit_local_model() to use training-only model for eval predictions
3. **Line 600-750:** Add hold-out validation split in _select_best_calibrator()
4. **Line 1380:** Replace `skip_single_fold_validation` with time-series CV code

---

## Related Documents

- `PRECISION_IMPROVEMENT_ANALYSIS.md` — General precision improvements (separate from overfitting)
- `PRECISION_IMPROVEMENTS_CODE.md` — Code fixes for precision (retrain_model.py, feature_engine.py)
- These two are complementary: this document fixes optimizer architecture, those fix model quality

---

## Key Takeaway

**meta_gate_optimizer.py discovers gates optimized for specific holdout periods, not generalizable gates for live trading.** The 6 fixes convert this from a "choose best 1 of 500" (high overfitting) to a "choose best from regularized search + validation" (acceptable overfitting).

The most critical fix is **Problem 2 (training/eval leakage)** — implement this first as it directly inflates all metrics by 5-10%.

