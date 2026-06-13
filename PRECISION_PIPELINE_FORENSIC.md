# Precision Pipeline Forensic
## threshold_optimizer · meta_gate_optimizer · retrain_model

**Date:** 2026-06-09  
**Symptom:** holdout precision stalls ~35–40% after fleet retrain  
**Status:** Two root-cause fixes applied; one structural gap remains

---

## 1. Pipeline Architecture

```
retrain_model.py ──produces──► src/ml/model_store/<SYMBOL>_model.json
                                src/ml/model_store/<SYMBOL>_meta.json
                                data/meta/                      (OOF edge scores)

threshold_optimizer.py ──reads──► model_store   (ATR multiplier seed)
                       ──produces► data/token_params/<SYMBOL>_params.json
                                   • atr_multiplier, lookahead_bars
                                   • regime_boundaries  (vol/atr/momentum percentiles)
                                   • regimes[]           (per-regime buy_ok / sell_ok)

meta_gate_optimizer.py ──reads──► model_store, token_params
                       ──produces► data/meta_gate_profiles/<SYMBOL>_gate.json
                                   • gate_type, thresholds (absolute edge scores)
                                   • threshold_quantile
                                   • calibration, signal_vetoes, regime_modifier
                                   • risk_tier, profitability_bypass flag

retrain_model.py ──reads (second pass)──► token_params  (regime filter)
                                          meta_gate_profiles (gate thresholds)
                 ──evaluates──► holdout precision, PF, Sharpe, Exp
```

**Correct execution order:**
```
1. retrain_model.py   (trains new models, uses default fallbacks for old profiles)
2. threshold_optimizer.py  (refreshes token_params for the NEW model)
3. meta_gate_optimizer.py  (refreshes gate profiles for the NEW model)
4. retrain_model.py   (optional: re-evaluate with fresh profiles — only needed for
                        the sidecar JSON accuracy; live predictor uses fresh profiles)
```

---

## 2. What Each Script Does

### retrain_model.py

**Trains:** Primary XGBoost (3-class: SELL=0, HOLD=1, BUY=2) + Meta XGBoost (binary: correct direction vs not).

**Splits:**
- `train_pool`: first 75% of data, purged 15-fold time-series CV for primary + meta OOF
- `holdout`: last 25%, scored exactly once

**Key internal data flow:**
1. Primary model OOF predictions → triple-barrier labels → meta features (`mX`)
2. `META_HOLD_STRATEGY = "C_excluded"`: HOLD bars removed entirely from meta training; `mY` is binary (1=correct direction, 0=wrong)
3. Meta OOF probabilities → `EdgeScoringEngine.compute_edge_batch()` → per-bar edge scores (BUY side and SELL side, 0–100 scale)
4. Regime filter (`_opt["regimes"]` from `token_params`) applied to `prop_dev_filtered` on the dev set and to `fire` on holdout
5. Gate thresholds applied: profile path (from `meta_gate_profiles`) if valid; dynamic `pick_edge_threshold_by_side()` otherwise
6. Holdout evaluation: `fired_prec`, `PF`, `Sharpe`, `Exp` → tradeable decision → sidecar JSON

**Outputs read by other scripts:** model JSON, meta JSON, feature_cols list

---

### threshold_optimizer.py

**Trains:** A fresh local XGBoost (first 70% of 4000-bar history, `LOCAL_ROUNDS=300`, no Optuna). This is an **independent** model; the production primary model is NOT used.

**Evaluates:** Last 30% of history (out-of-sample for the local model).

**Produces** `data/token_params/<SYMBOL>_params.json`:
- `atr_multiplier`: best ATR multiplier from `ATR_MULT_GRID` (0.75–4.25)
- `lookahead_bars`: best lookahead from `LOOKAHEAD_GRID` (12–72)
- `regime_boundaries`: percentile boundaries for vol / atr_pct / momentum computed **from the eval window**
- `regimes[<vol>_<vola>_<trend>]`: per-27-bucket `{buy_ok, sell_ok, buy_threshold, sell_threshold, ...}` evaluated on the local model's eval window

**Critical point:** `regime_boundaries` encode the **percentile structure of the market during the eval window**. The `buy_ok` / `sell_ok` flags reflect whether the local model's directional confidence was above `TARGET_PREC=0.60` in that regime. These are regime policies for the **local model**, not the production model. They are used by `retrain_model.py` as a blunt direction-disable filter on the production model — which is a mismatch by design (the two models share the same feature space but have different calibration levels).

---

### meta_gate_optimizer.py

**Trains:** Same independent local model as `threshold_optimizer` (first 70% → fit, last 30% → eval).

**Searches** 10 gate architecture templates × 10 quantiles × 5 calibration methods = up to 500 candidates. For each candidate:
1. Applies the architecture (side-specific thresholds, regime modifiers, vetoes, calibration)
2. Computes `edge_score = EdgeScoringEngine.compute_edge_batch()` on the eval window
3. Fires signals above the quantile threshold
4. Evaluates precision, PF, Sharpe, Exp on the eval window
5. Scores: `0.40 × Exp + 0.30 × PF + 0.20 × Sharpe + 0.05 × Prec + 0.05 × Cov`

**Produces** `data/meta_gate_profiles/<SYMBOL>_gate.json`:
- `thresholds.buy_edge` / `thresholds.sell_edge`: **absolute** edge score values (e.g., 55.07, 55.21)
- `threshold_quantile`: the quantile that produced the best score (e.g., 0.10 = top 10%)
- `edge_rank_mode`: `'percentile'`

**Critical point:** These absolute thresholds are the **raw minimum edge score value** at the chosen quantile of the eval-window edge score distribution. The eval window comes from the **local model** (87% of train_n per fold for OOF, but the eval window is the 30% held-out from the local model fit). This is not the same distribution as the production meta model's OOF edge scores.

---

## 3. Where Precision Leaks

### Leak 1 — OOF-to-Deployment Distribution Shift (PRIMARY CAUSE)

**What happens:**

When `meta_gate_optimizer` records `thresholds.buy_edge = 55.07`, that value was the 90th percentile of edge scores from the **local model's eval window**. The local model was fit on 70% of 4000 bars (≈2800 bars). It has never seen the production model's data.

When `retrain_model.py` later applies that threshold against the **production meta model's OOF edge scores**, it is applying a value calibrated on one distribution to a completely different distribution. Concretely:

- Local model eval window edge scores: mean ≈ 40–50, 90th percentile ≈ 55
- Production meta OOF edge scores (OOF = 87% data per fold, conservative): mean ≈ 35–45, 90th percentile ≈ 55 *(still a good match when models were recently aligned)*
- Production meta **full-fit** edge scores (100% data, optimistic): mean ≈ 50–65, 90th percentile ≈ 70+

After retraining the production model, the OOF edge scores shift because the new model has different internal calibration. The absolute threshold `55.07` that used to fire on ~10% of dev bars now fires on **52%** of dev bars (observed in BTC forensic). The gate selects the **bottom half** of the distribution instead of the top 10%.

**Symptom:** dev_prec = 90%+ (only the top sliver of a massive over-fired set happens to be correct), holdout_prec = 35–36% (the gate fires everywhere on holdout and gets the wrong 52% of signals).

**Forensic signature:**
```
dev_prec = 90.3%  (suspiciously perfect — not a model achievement, a threshold artifact)
holdout_prec = 35.6%
coverage on holdout = 52% of all directional bars (should be ~10%)
selected precision 35.6% < rejected precision 53.8%  ← anti-selective gate
```

---

### Leak 2 — Stale Regime Filter (SECONDARY CAUSE)

**What happens:**

`threshold_optimizer` computes `regime_boundaries` percentiles from its eval window (last 30% of a 4000-bar history). After retrain, the market may have moved significantly. The percentile boundaries `vol_p33/p67`, `atr_pct_p33/p67`, `momentum_p33/p67` may now classify many recent bars into `skipped=True` or `buy_ok=False` regimes.

`retrain_model.py` applies these stale regime policies at two points:
1. **Dev set** (lines 2615–2648): Suppresses `prop_dev_filtered` bars to HOLD
2. **Holdout** (lines 3168–3206): Suppresses fired signals to zero

In the BTC forensic case: 608 out of 873 directional signals (70%) were suppressed by regime filter. This means the edge threshold fires on 265 of the remaining 310 bars, a 86% fire rate on the non-suppressed subset — clearly still the wrong percentile.

**Important:** Regime suppression is applied **before** the edge threshold in the dev path (line 2614 suppresses `prop_dev_filtered`, which is passed to `get_profile_edge_thresholds`), but **after** edge threshold on holdout (line 3168 comes after `fire = edge_buy >= thr_buy | edge_sell >= thr_sell` at ≈line 3140). This asymmetry means the dev evaluation is measuring a different filtered population than the holdout evaluation.

---

### Leak 3 — Regime Filter Applied Asymmetrically Between Dev and Holdout

**Dev path (lines 2614–2648):**
```python
prop_dev_filtered = prop_v.copy()   # start from raw OOF proposals
# Apply regime filter to prop_dev_filtered (suppresses to HOLD)
...
# Then compute edge thresholds against prop_dev_filtered
profile_thresholds = get_profile_edge_thresholds(
    meta_gate_profile, edge_buy, edge_sell, prop_dev_filtered, y_v
)
```

**Holdout path (lines 3100–3206):**
```python
fire = (edge_buy_h >= thr_buy) | (edge_sell_h >= thr_sell)  # threshold first
...
# Then apply regime filter to fire (suppresses already-fired signals)
regime_suppress[i] = True  # if regime is bad
fire = fire & ~regime_suppress
```

On dev, regime suppression changes which bars are eligible for threshold selection (modifying `prop_dev_filtered` before the threshold is applied). On holdout, regime suppression removes signals after the threshold has already fired. When `prop_dev_filtered` is heavily suppressed (70%), the threshold is calibrated on a tiny high-precision subset; the holdout then fires broadly (before regime suppression) and the subsequent regime suppression may remove a different and smaller fraction.

---

### Leak 4 — Three-Class Precision vs Financial Profitability Mismatch

**What happens:**

`fired_prec = (prop_h[fire] == y_test[fire]).mean()` includes HOLD-labeled bars as "wrong" signals. A bar labeled HOLD in triple-barrier means price moved in the proposed direction but didn't reach the ATR barrier within `max_lookahead` bars. That signal loses only `-FEE_ROUNDTRIP` (0.1%), not the full barrier distance.

When `MIN_TRADEABLE_PRECISION = 0.60`, a token with 58% 3-class precision, PF=2.1, Sharpe=12 gets disabled. This is correct for signals that hit the wrong barrier, but incorrect for signals that time out as HOLD. With `C_excluded` meta strategy, the meta model never learned to filter HOLD timeouts — it only learned correct vs incorrect directional calls, so HOLD-labeled holdout bars appear as precision misses.

**Impact magnitude:** Approximately 5–15 percentage points of "fake" precision drop depending on the HOLD rate of the token's label distribution.

---

## 4. Fixes Applied

### Fix 1 — Two-Phase Stale Threshold Guard (retrain_model.py, lines 2661–2708)

**Phase 1:** Before using profile thresholds, check whether they produce anomalous dev statistics:
```python
if (
    _side_prec_max > 0.85 or   # top-1% slice — likely massively over-firing
    _dev_cov_total < 0.01 or   # < 1% coverage — threshold too tight, fires nothing
    _dev_cov_total > 0.50      # > 50% coverage — threshold too loose, fires everything
):
    profile_thresholds = None  # force dynamic selection
```

**Phase 2:** Clean `if / else` so dynamic selection always runs when `profile_thresholds is None`:
```python
if profile_thresholds is not None:
    thr_buy, thr_sell, ... = profile_thresholds
else:
    thr_buy, ... = pick_edge_threshold_by_side(edge_buy, ...)
    thr_sell, ... = pick_edge_threshold_by_side(edge_sell, ...)
```

**What it fixes:** Catches the anti-selective gate at the dev stage; falls back to dynamic percentile sweep on the current model's OOF edge scores.

**What it does NOT fix:** It is a defensive fallback, not a true fix. After the stale threshold is detected, the dynamic threshold is calibrated on OOF edge scores, but the holdout uses `full-fit meta` edge scores. The OOF→full-fit distribution shift still exists; dynamic selection just avoids the worst case.

---

### Fix 2 — Profitability Bypass (retrain_model.py, lines 3289–3307; meta_gate_optimizer.py, lines 700–706 and 1440–1444)

When 3-class precision is below `MIN_TRADEABLE_PRECISION = 0.60` but financial metrics are strong, the token qualifies at AGGRESSIVE tier only:

**retrain_model.py:**
```python
_profitability_bypass_candidate = (
    fired_n >= 10 and coverage >= MIN_COVERAGE and
    bt['profit_factor'] >= 1.50 and
    bt['sharpe'] >= 5.0 and
    bt['expectancy_pct'] >= 0.10
)
```

**meta_gate_optimizer.py:**
```python
_profitability_bypass = (
    selected.get('profit_factor', 0.0) >= 1.50 and
    selected.get('sharpe', 0.0) >= 5.0 and
    selected.get('expectancy_pct', 0.0) >= 0.15
)
```

Risk tier capping: bypass tokens get `tier_conservative = False, tier_balanced = False`; only `tier_aggressive` is set.

---

### Fix 3 — Anti-Selective Gate Veto (retrain_model.py, lines 3382–3384)

```python
if selected_prec <= rejected_prec:
    passes_validation = False
    print("[VETO] Selected trades did not outperform rejected trades!")
```

This catches the anti-selective gate condition directly: if the gate's selected signals have lower precision than the blocked signals, the gate is disabled regardless of absolute precision.

---

### Fix 4 — Defaults Block for UnboundLocalError (retrain_model.py, lines 3232–3239)

```python
selected_n = 0; rejected_n = 0
selected_prec = rejected_prec = 0.0
selected_exp_pct = rejected_exp_pct = 0.0
selected_sharpe = rejected_sharpe = 0.0
meta_gate_lift = meta_gate_lift_exp = 0.0
per_side_approved = False
_via_profitability_bypass = False
```

Prevents `UnboundLocalError` when `fired_n == 0`.

---

## 5. Remaining Structural Gap — The Core Problem Is Not Fully Fixed

The stale threshold guard (Fix 1) is a **detection-and-fallback** mechanism. The dynamic fallback `pick_edge_threshold_by_side()` sweeps quantiles on the **OOF edge score distribution** of the production meta model. But the production meta model at inference time uses **full-fit meta** (trained on 100% of the training data), which produces systematically higher confidence scores than OOF meta (trained on 87% per fold).

This means even a correctly calibrated OOF threshold will fire on a different fraction of bars at inference than it did during dev evaluation. The gap is inherent to the purged-CV training paradigm and cannot be fully closed within `retrain_model.py` alone.

**The real fix is to run `meta_gate_optimizer.py` after every retrain.** `meta_gate_optimizer` evaluates its thresholds on the **eval window's edge scores**, which come from its own local model — a different model than the production meta, but the key is that both the threshold search and the final evaluation use the **same model**. There is no OOF-vs-full-fit split within `meta_gate_optimizer`; it fits on 70% and evaluates on 30% with the same model. This means its thresholds are internally consistent and do not suffer from the OOF→full-fit shift.

When `retrain_model.py` loads a **fresh** `meta_gate_profile` (from a just-run `meta_gate_optimizer`), the absolute thresholds are already at the right percentile of the local model's eval distribution. The question is whether the local model's eval distribution is close enough to the production meta's distribution. It is close because both use the same feature columns and the same market data window; the main difference is model capacity (local model has `max_depth=3`, production has `max_depth=5` with Optuna-tuned params). The directional ranking of bars tends to be preserved across model capacity differences, so the percentile-rank of an edge score is more stable than its absolute value.

---

## 6. What "Precision ≈ 40%" Actually Means Right Now

After Fix 1 (stale threshold guard) but before re-running `meta_gate_optimizer`:

| Stage | What's happening |
|-------|-----------------|
| Profile thresholds | Rejected by stale guard → dynamic fallback used |
| Dynamic `thr_buy / thr_sell` | Calibrated on OOF edge scores to hit `TARGET_SIGNAL_PRECISION = 0.62` on dev |
| Dev precision | ≈ 62% OOF (as designed by `pick_edge_threshold_by_side`) |
| Holdout precision | ≈ 40% because full-fit meta edge scores are higher than OOF → same absolute threshold fires on a different (larger) fraction of holdout bars |
| Regime filter | Stale token_params causing over-suppression; reduces holdout sample size, increases variance |

**The ~22pp dev→holdout gap is the residual OOF→full-fit shift operating on the dynamic threshold.**

---

## 7. Resolution Checklist

### Immediate (unblocked now)
- [x] Stale threshold guard active — worst-case anti-selective gate prevented
- [x] Profitability bypass active — financially strong tokens not incorrectly disabled
- [x] Anti-selective gate veto active — gates where selected < rejected precision are auto-disabled
- [x] `UnboundLocalError` on `selected_n` fixed

### Required to reach ≥ 60% holdout precision
- [ ] **Run `threshold_optimizer.py`** after full fleet retrain completes
  - Refreshes `regime_boundaries` with current market structure
  - Refreshes `buy_ok / sell_ok` regime policies on the NEW model's eval window
  - Fixes 70% over-suppression in holdout regime filter
- [ ] **Run `meta_gate_optimizer.py`** after `threshold_optimizer.py` completes
  - Generates fresh absolute thresholds from the local model's current eval window
  - These thresholds are internally consistent (no OOF-vs-full-fit split within MGO)
  - Refreshes gate architecture (architecture type, calibration method, vetoes)
- [ ] **Verify** that `meta_gate_optimizer` eval-window precision ≥ 60% for all tradeable tokens
  - Tokens where local model eval precision < 55% will likely fall to profitability bypass
  - Expected: BTC, ETH, SOL should reach 60%+; mid-cap alts may need bypass

### Optional (reduces residual OOF→full-fit gap)
- [ ] **Holdout-percentile calibration in `retrain_model.py`:** Instead of applying the raw OOF-derived threshold against holdout edge scores directly, re-compute the threshold as a percentile of the **holdout's own edge score distribution**. This is the same quantile found on dev, applied to holdout's distribution. Cost: the holdout edge scores must be available before the firing decision.
- [ ] **Edge score normalization:** Normalize edge scores by their per-bar percentile rank within the current distribution (already available when `edge_rank_mode='percentile'` in `meta_gate_optimizer`). Percentile-ranked scores are distribution-invariant by construction.

---

## 8. Data Flow Diagram for a Single Token

```
retrain_model.py (train pass)
  raw OHLCV → features → primary XGBoost (OOF via 15-fold purged CV)
  → OOF proposals (prop_v) + OOF probs (raw_probs)
  → triple-barrier labels (y_v)
  → mX (meta features) = OOF probs + confluence + regime features
  → meta XGBoost (binary: correct direction vs wrong) → meta_oof
  → MetaCalibrationFramework.calibrate() → meta_oof_cal
  → EdgeScoringEngine.compute_edge_batch(dev_df, meta_oof_cal, 'BUY') → edge_buy  [OOF distribution]
  → EdgeScoringEngine.compute_edge_batch(dev_df, meta_oof_cal, 'SELL') → edge_sell [OOF distribution]

  LOAD token_params → regime filter on prop_dev_filtered
  LOAD meta_gate_profile → get_profile_edge_thresholds() → check stale guard
    if stale: pick_edge_threshold_by_side(edge_buy, prop_dev_filtered, y_v, 2)
    else: use thr_buy from profile

  holdout evaluation
  → primary.predict(holdout) → prop_h (full-fit primary, higher confidence)
  → meta.predict(holdout meta features) → meta_h (full-fit meta, higher confidence)
  → EdgeScoringEngine.compute_edge_batch(holdout_df, meta_h, 'BUY') → edge_buy_h  [FULL-FIT dist]
  → fire = (edge_buy_h >= thr_buy) | (edge_sell_h >= thr_sell)
    ← thr_buy was calibrated on OOF distribution, applied to full-fit distribution
    ← THIS IS THE SOURCE OF THE GAP
  → regime_suppress using stale token_params
  → fired_prec, PF, Sharpe, Exp → tradeable decision
```

---

## 9. Quick Diagnostic Reference

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| dev_prec 85-95%, holdout_prec 35-45% | Stale threshold fires on top-1% of new distribution | `[STALE THRESHOLD GUARD]` in output; coverage > 50% on dev |
| dev_prec 62%, holdout_prec 40-45% | OOF→full-fit shift on dynamic threshold | Normal until fresh MGO run; check dev→holdout gap |
| selected_prec < rejected_prec on holdout | Anti-selective gate | `[VETO]` message; re-run threshold_optimizer + meta_gate_optimizer |
| 608/873 regime-suppressed signals | Stale token_params regime boundaries | Re-run threshold_optimizer |
| PF=2.1, Sharpe=12, prec=37%, DISABLED | Profitability bypass not triggering | Check `bt['sharpe'] >= 5.0` and `bt['profit_factor'] >= 1.50` in output |
| No signals fired at all | Regime filter suppressing everything | `_opt["regimes"]` all `buy_ok=False` — stale token_params |
| Coverage < 1% or > 50% on dev | Threshold at wrong tail of distribution | Stale guard should catch this; check output for `[STALE THRESHOLD GUARD]` |

---

## 10. File Reference

| File | Purpose | Stale after retrain? |
|------|---------|----------------------|
| `data/token_params/<SYM>_params.json` | ATR mult, lookahead, regime policies | YES — re-run threshold_optimizer |
| `data/meta_gate_profiles/<SYM>_gate.json` | Absolute edge thresholds, architecture | YES — re-run meta_gate_optimizer |
| `src/ml/model_store/<SYM>_model.json` | Primary XGBoost | Updated by retrain |
| `src/ml/model_store/<SYM>_meta.json` | Meta XGBoost | Updated by retrain |
| `src/ml/model_store/<SYM>_meta_model.json` | Meta XGBoost (alternate path) | Updated by retrain |

---

*Generated from code audit of scripts/retrain_model.py, scripts/threshold_optimizer.py, scripts/meta_gate_optimizer.py as of 2026-06-09.*
