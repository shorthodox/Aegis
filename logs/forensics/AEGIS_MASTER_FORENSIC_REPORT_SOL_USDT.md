# AEGIS-1 Master Forensic Report

**Symbol:** SOL/USDT  |  **Generated:** 2026-06-14 02:00:53

---

> **Audit scope:** 15 sections covering Training Pipeline, Meta Model,
> HMM Regime Engine, LSTM Temporal Engine, Live Signal Engine,
> Risk Engine, Execution Filters, Portfolio Engine, Drift Monitor,
> and Post-Trade Performance.
>
> Every conclusion cites supporting metrics.
> Where evidence is weak, uncertainty is explicitly stated.

---


## ❓ The Three Questions

> These three questions answer what matters for P&L.
> Every finding cites the exact file and line where the problem lives.

### Q1 — Why Is Precision Lower Than It Should Be?

> Measured precision is 44.7% — estimated 31.4pp below achievable ceiling of ~76.1%. Primary contributors: CV accuracy near random — primary learned nothing (−14.7pp), Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), Barrier skew suppresses BUY labels (−2.6pp)

#### Precision Waterfall

```
Measured holdout precision :  44.7%

  [✅ FIXED       ]  −14.7pp prec    CV accuracy near random — primary learned nothing
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −2.6pp prec    Barrier skew suppresses BUY labels
  [✅ FIXED       ]  −2.9pp prec    HOLD over-representation in labels
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [✅ FIXED       ]  −2.1pp prec    Anti-selective gate — precision below chance
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.044 — overconfidence

Achievable precision       :  ~76.1%  (+31.4pp precision)
Recall deficit (BUY side)  :  −8.2pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.044 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.044 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **6 issues already applied** in the codebase (expected gain: +30.9pp once retrained).

> Meta gate removed entirely. Primary-only calibrated gate with signal_prec >= token_breakeven tradeable check. FIXED.

#### Signal Gate Trace (Primary-Only Calibrated Gate)

🔴 **Signal gate DISABLED — root cause at Gate: 3. Holdout signal precision ≥ token breakeven**

✅ **1. Primary-only mode active + directional skill ≥ 45%**
   - `scripts/retrain_model.py — primary_only_gate block`
   - Check: `primary_only_mode=True AND primary dir_prec ≥ 45% (else veto → None)`
   - Value: primary_only_mode=True, calibrator=Y
   - PASS — primary-only mode active. Calibrated LR maps raw probs to confidence.

✅ **2. Val sweep finds threshold with ≥50 fires**
   - `scripts/retrain_model.py — val sweep (0.50→0.95, min_fires=50)`
   - Check: `max(signal_prec over thresholds with ≥50 val fires)`
   - Value: primary_confidence_threshold=0.560
   - PASS — val sweep selected threshold 0.560.

❌ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (51.1%)`
   - Value: signal_prec=44.7% vs breakeven=51.1% (691 fired, dir_prec≈76.9%)
   - FAIL — signal_prec=44.7% < breakeven=51.1% (gap=6.4pp). Primary model needs stronger directional skill.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=41.6%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +15.9pp)  ❌ signal_prec < breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 44.7% |
| Rejected signal precision | 28.8% |
| Gate lift (precision) | +15.9pp |
| Primary conf. threshold | 0.560 |
| Token breakeven | 51.1% |
| Selected signals | 691 |
| Rejected signals | 971 |

> Primary gate adds 15.9pp of precision. Selected signals (44.7%) beat rejected (28.8%). Threshold=0.560. | DISABLED: selected signal_prec=44.7% < breakeven=51.1%.


---

## Section 15 — Executive Summary

**Symbol:** SOL/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=44.7%  Sharpe=34.55
**Expected after fixes:** Precision≈74.5%  (+29.8pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🔴 **Anti-Selective Gate (precision < 50%)** — Score: 90/100
   > holdout_prec=44.7% — gate is selecting WRONG signals. Worse than random for directional trading.

3. 🔴 **Signal Precision Below Token Breakeven** — Score: 85/100
   > signal_prec=44.7% < breakeven=51.1% (gap=6.4pp). Primary model has insufficient directional skill to clear the fee breakeven after HOLD-timeout dilution.

4. 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** — Score: 85/100
   > cv_accuracy=60.7% vs majority_baseline=69.1%. binary_dual SPW inflation compresses hold_residual → argmax always BUY/SELL. Check sidecar for bayes_prior_correction key — if absent, Bayes fix not applied.

5. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 69/100
   > HOLD=69.1% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **Anti-Selective Gate (precision < 50%)**
   → Fix primary OOF quality (soft confluence features + local model hyperparams). Directional precision veto (_fired_dir_prec<0.50) disables gate and outputs tradeable=False.

3. **Signal Precision Below Token Breakeven**
   → Increase primary model AUPRC (Optuna aucpr objective). Target dir_prec >= 65% so fired signals clear breakeven even at 30% HOLD-timeout rate.

4. **Primary Model CV Near Random (NEAR_RANDOM)**
   → Apply Bayes prior correction to OOF and holdout raw_probs: corrects SPW-inflated probabilities back to true class posterior scale so hold_residual is meaningful. bayes_prior_correction key must appear in sidecar after retrain.

5. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 60.7% | ⚠ |
| Dev OOF Precision | 26.4% | ✗ |
| Holdout Precision | 44.7% | ✗ |
| Holdout Coverage | 41.6% | ✓ |
| 95% CI Precision | [41.0%, 48.4%] | — |
| OOF→Holdout Gap | +18.3% | ✓ holdout beat OOF |
| Holdout Fired | 691 trades | ✓ |
| SELL Win Rate | 72.5% (287 trades) | ✓ |
| BUY Win Rate | 87.8% (115 trades) | ✓ |
| Sharpe (annualised) | 34.55 | ✓ |
| Max Drawdown | 18.88% | ✗ |
| Profit Factor | 3.94 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.5691% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.560 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | -6.4pp (be=51.1%) | ✗ below breakeven |
| Statistical Sig. | p=0.0055 (z=-2.78) | ✓ significant |

### Class Distribution
- HOLD: **69.1%** — ⚠ severe imbalance
- SELL: **19.7%**
- BUY:  **11.3%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 69.1% HOLD labels biases model toward neutrality.
- **CRITICAL** — Anti-selective gate: holdout precision=44.7% < 50%. Gate is selecting the worst signals. Rebuild with fixed local model.
- **CRITICAL** — Precision below random chance (44.7%). Model is directionally anti-predictive on holdout data.
- **CRITICAL** — CV accuracy=60.7% ≈ random baseline=69.1%. Primary model learned nothing from features.
- **CRITICAL** — Signal precision -6.4pp below breakeven (51.1%). Token correctly DISABLED. Increase primary model directional precision to clear breakeven gate.


---

## Section 2 — Feature Forensics

**Feature health summary:** 45 HEALTHY | 24 WARNING | 7 DEGRADED | 8 CRITICAL  (of 84 total)

**Estimated total precision gain if top drifters removed:** +18.6 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `close` | **CRITICAL** | 22.772 | 0.945 | 0.467 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `high` | **CRITICAL** | 22.704 | 0.945 | 0.466 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 3 | `vwap_decay_mean_24` | **CRITICAL** | 21.349 | 0.935 | 11.206 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 4 | `low` | **CRITICAL** | 20.437 | 0.945 | 0.467 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `vwap_decay_std_24` | **CRITICAL** | 4.455 | 0.583 | 0.932 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `vwap_delta_12` | **CRITICAL** | 3.099 | 0.590 | 4.350 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 7 | `atr_14` | **CRITICAL** | 1.323 | 0.523 | 0.388 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 8 | `funding_rate_ma8` | **CRITICAL** | 0.921 | 0.455 | 7.994 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 9 | `fib_range_pct` | **DEGRADED** | 0.706 | 0.202 | 0.272 | 0.50 | MONITOR | +0.9pp |
| 10 | `btc_corr_24h` | **DEGRADED** | 0.595 | 0.283 | 0.174 | 0.50 | MONITOR | +0.9pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `atr_14`
- `vwap_decay_std_24`
- `funding_rate_ma8`
- `close`
- `high`
- `vwap_delta_12`
- `vwap_decay_mean_24`
- `low`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 49  | 98.0%  |
| SELL | 64 | 84.4% |
| HOLD | 8,623 | — |

### Signal Rejection Funnel

```
Generated (directional):       113  (100%)
Below Primary Conf. Thr:   -    0  (0%)
Blocked by Quality (<55):  -    9
Blocked by HMM:            -    5
Blocked by Confluence:     -    4
Blocked by Fake Breakout:  -    3
Blocked by Portfolio Guard:-    2
Blocked by Safe Mode:      -    1
Blocked by Drift:          -    2
Blocked by Cooldown:       -    1
─────────────────────────────────────
Estimated Executed:            86  (76.1%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.5386%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 62.4% |
| 12h | 86.1% |
| 18h | 96.3% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 0 | 0 | 0 | 0.0% | +0.00% |
| quality | 9 | 5 | 4 | 61.5% | +13.85% ⚠ |
| hmm | 5 | 2 | 3 | 57.6% | +7.69% ⚠ |
| confluence | 4 | 2 | 2 | 54.5% | +6.15% |
| fake_breakout | 3 | 1 | 2 | 60.0% | +4.62% ⚠ |
| portfolio | 2 | 1 | 1 | 57.6% | +3.08% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 2.0444 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.044) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 44.7% | -0.30 | ⚠ overconfident |
| 80-90% | 54.7% | -0.30 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 691 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

⚠ HMM model not available.



---

## Section 7 — LSTM Forensics

⚠ LSTM meta not found



---

## Section 8 — Quality Engine Forensics

**Verdict:** VALID — higher quality scores predict better outcomes

| Quality Bucket | Trades | Precision | Expectancy | P.Factor |
|---------------|--------|-----------|------------|---------|
| 0-20 | 0 | 0.0% | +0.0000% | 0.00 |
| 20-40 | 113 | 77.0% | +0.0000% | 2.79 |
| 40-60 | 191 | 81.2% | +0.0000% | 4.06 |
| 60-80 | 172 | 81.4% | +0.0000% | 4.18 |
| 80-100 | 116 | 89.7% | +0.0000% | 8.05 |

**Monotone precision:** ✓ YES — quality is predictive
**Paper trading:** 24 trades, 79.2% WR


---

## Section 9 — Drift Monitor Forensics

**Overall Drift Status:** 🔴 **CRITICAL**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🟢 OK | 8 CRITICAL (3 active in model, 5 ✅ blacklisted/FIXED) / 7 DEGRADED / 84 total |
| Confidence Drift | 🔴 CRITICAL | T=2.044 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +44.72pp |

**Estimated precision loss from feature drift:** ~2.1pp


---

## Section 10 — Portfolio Forensics

**Open positions:** 1/6  |  **Effective leverage:** 0.50×  |  **HHI (concentration):** 0.050

| Symbol | Capital Allocation |
|--------|-------------------|
| SEI/USDT | 8.0% |
| VET/USDT | 8.0% |
| ATOM/USDT | 8.0% |
| FLOW/USDT | 4.0% |
| BAT/USDT | 4.0% |
| IMX/USDT | 4.0% |
| DOGE/USDT | 4.0% |
| ALGO/USDT | 4.0% |
| ADA/USDT | 4.0% |
| SUI/USDT | 4.0% |
| UNI/USDT | 4.0% |
| FIL/USDT | 4.0% |
| KAVA/USDT | 4.0% |
| SAND/USDT | 4.0% |
| XLM/USDT | 4.0% |
| ARB/USDT | 4.0% |
| STX/USDT | 4.0% |
| DOT/USDT | 4.0% |
| ENA/USDT | 4.0% |
| NEAR/USDT | 4.0% |
| ZEC/USDT | 4.0% |
| THETA/USDT | 4.0% |

**Hidden leverage:** ✓ NO  |  **Over-concentration:** ✓ NO


---

## Section 11 — Risk Engine Forensics

| Metric | Value | Assessment |
|--------|-------|------------|
| ATR Multiplier | 2.35× | ✓ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 34.55 | ✓ |
| Holdout Max DD | 18.88% | ✗ |
| Stop Assessment | TARGETS TOO CLOSE | — |


---

## Section 12 — Live Execution Forensics

**Closed:** 24  |  **Open:** 1  |  **Avg hold:** 0.8h  |  **Avg PnL:** +1.154%

**Confidence discriminates wins from losses:** ✗ NO (WIN conf=0.627 vs LOSS conf=0.636)

### Best Trades
- **ZEC/USDT** BUY  PnL=+10.31%  conf=0.785  exit=TP1_HIT
- **THETA/USDT** BUY  PnL=+3.87%  conf=0.701  exit=TP1_HIT
- **NEAR/USDT** BUY  PnL=+3.15%  conf=0.713  exit=TP1_HIT

### Worst Trades
- **ENA/USDT** SELL  PnL=-3.40%  conf=0.749  exit=SL_HIT
- **VET/USDT** BUY  PnL=-0.79%  conf=0.620  exit=SL_HIT
- **ADA/USDT** BUY  PnL=-0.75%  conf=0.595  exit=SL_HIT

### Exit Reasons
| Reason | Count |
|--------|-------|
| TP1_HIT | 19 |
| SL_HIT | 5 |


---

## Section 13 — Root Cause Engine

**7 root causes identified.**  Combined top-5 impact score: **428/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🔴 **Anti-Selective Gate (precision < 50%)** | Gate Failure | 90/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | holdout_prec=44.7% — gate is selecting WRONG signals. Worse than rando… |
| 3 | 🔴 **Signal Precision Below Token Breakeven** | Gate Architecture | 85/100 | `scripts/retrain_model.py:_signal_prec_h >= token_breakeven check in primary_only_gate` 🔴 ACTIVE | signal_prec=44.7% < breakeven=51.1% (gap=6.4pp). Primary model has ins… |
| 4 | 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** | Model Quality | 85/100 | `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` ✅ FIXED | cv_accuracy=60.7% vs majority_baseline=69.1%. binary_dual SPW inflatio… |
| 5 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 69/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=69.1% of labels. Meta model sees 60% zero-labels → calibration di… |
| 6 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 7 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.044>1.0. Model overestimates confidence.… |

### Fixes

**1. Meta Model Calibration Failure**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

**2. Anti-Selective Gate (precision < 50%)**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta gate fully removed — primary-only calibrated `
> Fix primary OOF quality (soft confluence features + local model hyperparams). Directional precision veto (_fired_dir_prec<0.50) disables gate and outputs tradeable=False.

**3. Signal Precision Below Token Breakeven**
> 📍 `scripts/retrain_model.py:_signal_prec_h >= token_breakeven check in primary_only_gate` — `signal_prec_h < token_breakeven → tradeable=False`
> Increase primary model AUPRC (Optuna aucpr objective). Target dir_prec >= 65% so fired signals clear breakeven even at 30% HOLD-timeout rate.

**4. Primary Model CV Near Random (NEAR_RANDOM)**
> 📍 `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` — `cv_accuracy below majority-class baseline (HOLD%) `
> Apply Bayes prior correction to OOF and holdout raw_probs: corrects SPW-inflated probabilities back to true class posterior scale so hold_residual is meaningful. bayes_prior_correction key must appear in sidecar after retrain.

**5. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 44.7%  →  **Expected precision (all fixes):** 74.5%  (+29.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Fix local model quality (highest ROI) | +15.0pp | +3.0pp | +12.0pp | HIGH | LOW |
| 2 | Increase primary model AUPRC to clear signal_prec breakeven (gap=6.4pp) | +7.7pp | +5.0pp | +6.4pp | MEDIUM | MEDIUM |
| 3 | Remove / normalise 8 CRITICAL drifted features | +18.6pp | +1.5pp | +14.9pp | MEDIUM | LOW |
| 4 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 5 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 6 | Full retrain with all pipeline fixes applied | +13.7pp | +2.0pp | +11.5pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 691 |
| Rejected signals | 971 |
| Selected precision | 44.7% |
| Rejected precision | 28.8% |
| Meta gate lift (precision) | +15.9% |
| Selected expectancy | +0.569% |
| Rejected expectancy | +0.165% |
| Selected Sharpe | +43.49 |
| Rejected Sharpe | +8.19 |

**Verdict:** ✅ HELPFUL — Gate selects higher-precision signals than rejected


---

## Section 17 — Hold Pollution Audit

| Strategy | Hold Weight | Brier | PF | Sharpe | Prec | Lift | Notes |
|----------|-------------|-------|----|----|------|------|-------|
| A_current  | 1.00 | 0.330 | 1.20 | 0.45 | 59.0% | -0.02 | Baseline — no mitigation |
| B_reduced  | 0.15 | 0.310 | 1.35 | 0.58 | 62.0% | +0.03 | Partial HOLD downweight — recommended |
| C_excluded ✅ BEST | 0.00 | 0.300 | 1.40 | 0.62 | 64.0% | +0.05 | Total HOLD exclusion — most aggressive |

**Current Strategy Score:** 0.820
**Best Strategy Score:** 0.820
**Potential Improvement:** +0.000

**Recommendation:** Current strategy C_excluded is near-optimal


---

## Section 18 — Regime Threshold Audit

⚠ No regime policies found in metadata



---

## Section 19 — Deep SOL/USDT vs BTC/ETH Comparison

| Metric | SOL | BTC | ETH | SOL vs BTC |
|--------|-----|-----|-----|-----------|
| Primary Threshold | 0.795 | 0.620 | 0.820 | +0.175 |
| Tradeable BUY | False | False | False | — |
| Tradeable SELL | False | False | False | — |
| Holdout Precision | 37.4% | 48.0% | 26.0% | -10.6% |
| Win Rate (PnL) | 48.0% | 48.0% | 26.0% | — |
| Regime Disability | 50% | 20% | 60% | +30% |
| Calibration T | 0.888 | 0.920 | 0.950 | — |

### Top Discriminators (SOL vs BTC)

1. **Regime disability** — gap: +30.00
2. **Meta threshold** — gap: +0.18
3. **Holdout precision** — gap: -0.11

**Root Cause Hypothesis:** SOL fails on Regime disability (gap: 30.00)


---

## Section 20 — AEGIS Gate Lift Engine

**Gate Lift = Selected Precision − Rejected Precision**

| Metric | Value |
|--------|-------|
| Gate Lift (pp) | +15.9% |
| Selected signals | 691 |
| Rejected signals | 971 |
| Gate coverage | 41.6% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 65/100 |
| Recommended Action | USE_META_GATE |

**Recommendation:** Unknown status


---

## Section 22 — AEGIS BTC Difference Engine (Phase 7)

Comparing this token against BTC baseline:

| Metric | BTC | This Token | Gap |
|--------|-----|-----------|-----|
| Gate Lift Difference | — | — | TBD (requires BTC comparison data) |
| Precision Gap | — | — | TBD |
| Profit Factor Gap | — | — | TBD |


---

## Section 23 — AEGIS Token Profile (Phase 3)

| Metric | Value |
|--------|-------|
| Precision Target | 56.1% |
| Actual Precision | 44.7% |
| Gap | -11.4% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 65/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
