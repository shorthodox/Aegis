# AEGIS-1 Master Forensic Report

**Symbol:** BTC/USDT  |  **Generated:** 2026-06-14 02:00:52

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

> Measured precision is 54.1% — estimated 26.7pp below achievable ceiling of ~80.8%. Primary contributors: CV accuracy near random — primary learned nothing (−12.8pp), Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), HOLD over-representation in labels (−2.6pp)

#### Precision Waterfall

```
Measured holdout precision :  54.1%

  [✅ FIXED       ]  −12.8pp prec    CV accuracy near random — primary learned nothing
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −2.6pp prec    HOLD over-representation in labels
  [✅ FIXED       ]  −2.2pp prec    Barrier skew suppresses BUY labels
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.248 — overconfidence

Achievable precision       :  ~80.8%  (+26.7pp precision)
Recall deficit (BUY side)  :  −7.1pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.248 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.248 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **5 issues already applied** in the codebase (expected gain: +25.8pp once retrained).

> Meta gate removed entirely. Primary-only calibrated gate with signal_prec >= token_breakeven tradeable check. FIXED.

#### Signal Gate Trace (Primary-Only Calibrated Gate)

✅ **Signal gate ENABLED — all 4 criteria met.**

✅ **1. Primary-only mode active + directional skill ≥ 45%**
   - `scripts/retrain_model.py — primary_only_gate block`
   - Check: `primary_only_mode=True AND primary dir_prec ≥ 45% (else veto → None)`
   - Value: primary_only_mode=True, calibrator=Y
   - PASS — primary-only mode active. Calibrated LR maps raw probs to confidence.

✅ **2. Val sweep finds threshold with ≥50 fires**
   - `scripts/retrain_model.py — val sweep (0.50→0.95, min_fires=50)`
   - Check: `max(signal_prec over thresholds with ≥50 val fires)`
   - Value: primary_confidence_threshold=0.600
   - PASS — val sweep selected threshold 0.600.

✅ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (53.9%)`
   - Value: signal_prec=54.1% vs breakeven=53.9% (501 fired, dir_prec≈96.8%)
   - PASS — signal_prec=54.1% ≥ breakeven=53.9%.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=29.9%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +18.2pp)  ✅ signal_prec ≥ breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 54.1% |
| Rejected signal precision | 35.9% |
| Gate lift (precision) | +18.2pp |
| Primary conf. threshold | 0.600 |
| Token breakeven | 53.9% |
| Selected signals | 501 |
| Rejected signals | 1175 |

> Primary gate adds 18.2pp of precision. Selected signals (54.1%) beat rejected (35.9%). Threshold=0.600.


---

## Section 15 — Executive Summary

**Symbol:** BTC/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=54.1%  Sharpe=33.34
**Expected after fixes:** Precision≈80.4%  (+26.3pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** — Score: 85/100
   > cv_accuracy=61.2% vs majority_baseline=67.1%. binary_dual SPW inflation compresses hold_residual → argmax always BUY/SELL. Check sidecar for bayes_prior_correction key — if absent, Bayes fix not applied.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 67/100
   > HOLD=67.1% of labels. Meta model sees 60% zero-labels → calibration distorted.

4. 🟡 **Critical Feature Drift** — Score: 64/100
   > 16 features CRITICAL. Top: low PSI=20.86.

5. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **Primary Model CV Near Random (NEAR_RANDOM)**
   → Apply Bayes prior correction to OOF and holdout raw_probs: corrects SPW-inflated probabilities back to true class posterior scale so hold_residual is meaningful. bayes_prior_correction key must appear in sidecar after retrain.

3. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

4. **Critical Feature Drift**
   → FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.

5. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 61.2% | ⚠ |
| Dev OOF Precision | 42.5% | ✗ |
| Holdout Precision | 54.1% | ✗ |
| Holdout Coverage | 29.9% | ✓ |
| 95% CI Precision | [49.7%, 58.4%] | — |
| OOF→Holdout Gap | +11.6% | ✓ holdout beat OOF |
| Holdout Fired | 501 trades | ✓ |
| SELL Win Rate | 95.0% (160 trades) | ✓ |
| BUY Win Rate | 99.2% (120 trades) | ✓ |
| Sharpe (annualised) | 33.34 | ✓ |
| Max Drawdown | 2.50% | ✓ |
| Profit Factor | 4.47 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.1914% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.600 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | +0.2pp (be=53.9%) | ✓ above breakeven |
| Statistical Sig. | p=0.0670 (z=1.83) | ⚠ insufficient data |

### Class Distribution
- HOLD: **67.1%** — ⚠ severe imbalance
- SELL: **20.2%**
- BUY:  **12.6%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 67.1% HOLD labels biases model toward neutrality.
- **CRITICAL** — CV accuracy=61.2% ≈ random baseline=67.1%. Primary model learned nothing from features.


---

## Section 2 — Feature Forensics

**Feature health summary:** 31 HEALTHY | 11 WARNING | 9 DEGRADED | 16 CRITICAL  (of 67 total)

**Estimated total precision gain if top drifters removed:** +21.0 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `low` | **CRITICAL** | 20.855 | 0.938 | 0.326 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `close` | **CRITICAL** | 20.552 | 0.938 | 0.325 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 3 | `se_mid` | **CRITICAL** | 20.515 | 0.935 | 0.325 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 4 | `vwap_decay_mean_24` | **CRITICAL** | 17.391 | 0.906 | 6.531 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `returns_1h_decay_std_24` | **CRITICAL** | 3.355 | 0.536 | 0.529 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `vwap_decay_std_24` | **CRITICAL** | 3.236 | 0.709 | 1.812 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 7 | `funding_rate_ma8` | **CRITICAL** | 2.368 | 0.541 | 1.072 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 8 | `funding_rate` | **CRITICAL** | 2.257 | 0.488 | 1.075 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 9 | `gk_vol` | **CRITICAL** | 1.911 | 0.576 | 0.667 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 10 | `volume_decay_mean_24` | **CRITICAL** | 1.739 | 0.503 | 0.560 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `returns_1h_decay_std_24`
- `vwap_decay_mean_24`
- `donchian_width`
- `gk_vol`
- `vwap_decay_std_24`
- `funding_rate_ma8`
- `volume_decay_std_24`
- `close_decay_std_24`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 88  | 87.5%  |
| SELL | 352 | 81.0% |
| HOLD | 8,296 | — |

### Signal Rejection Funnel

```
Generated (directional):       440  (100%)
Below Primary Conf. Thr:   -   45  (10%)
Blocked by Quality (<55):  -   35
Blocked by HMM:            -   22
Blocked by Confluence:     -   17
Blocked by Fake Breakout:  -   13
Blocked by Portfolio Guard:-    8
Blocked by Safe Mode:      -    4
Blocked by Drift:          -    8
Blocked by Cooldown:       -    4
─────────────────────────────────────
Estimated Executed:           284  (64.5%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **0.6881%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 74.7% |
| 12h | 91.2% |
| 18h | 97.9% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **3 bars** (3h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 45 | 24 | 21 | 53.8% | +30.96% |
| quality | 35 | 21 | 14 | 61.5% | +24.08% ⚠ |
| hmm | 22 | 12 | 10 | 57.6% | +15.14% ⚠ |
| confluence | 17 | 9 | 8 | 54.5% | +11.70% |
| fake_breakout | 13 | 7 | 6 | 60.0% | +8.95% ⚠ |
| portfolio | 8 | 4 | 4 | 57.6% | +5.50% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 2.2476 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.248) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 54.1% | -0.21 | ⚠ overconfident |
| 80-90% | 64.1% | -0.21 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 501 dev samples):** `isotonic`


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
| Feature Drift | 🟡 WARNING | 16 CRITICAL (8 active in model, 8 ✅ blacklisted/FIXED) / 9 DEGRADED / 67 total |
| Confidence Drift | 🔴 CRITICAL | T=2.248 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +54.09pp |

**Estimated precision loss from feature drift:** ~4.5pp


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
| ATR Multiplier | 0.9× | ⚠ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 33.34 | ✓ |
| Holdout Max DD | 2.50% | ✓ |
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

**6 root causes identified.**  Combined top-5 impact score: **377/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** | Model Quality | 85/100 | `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` ✅ FIXED | cv_accuracy=61.2% vs majority_baseline=67.1%. binary_dual SPW inflatio… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 67/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=67.1% of labels. Meta model sees 60% zero-labels → calibration di… |
| 4 | 🟡 **Critical Feature Drift** | Feature Drift | 64/100 | `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` ✅ FIXED | 16 features CRITICAL. Top: low PSI=20.86.… |
| 5 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 6 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.248>1.0. Model overestimates confidence.… |

### Fixes

**1. Meta Model Calibration Failure**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

**2. Primary Model CV Near Random (NEAR_RANDOM)**
> 📍 `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` — `cv_accuracy below majority-class baseline (HOLD%) `
> Apply Bayes prior correction to OOF and holdout raw_probs: corrects SPW-inflated probabilities back to true class posterior scale so hold_residual is meaningful. bayes_prior_correction key must appear in sidecar after retrain.

**3. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

**4. Critical Feature Drift**
> 📍 `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` — `ema_9/21/50/100/200, vwap, avwap_*, ichimoku_senko`
> FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.

**5. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 54.1%  →  **Expected precision (all fixes):** 80.4%  (+26.3pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Fix local model quality (highest ROI) | +15.0pp | +3.0pp | +12.0pp | HIGH | LOW |
| 2 | Remove / normalise 16 CRITICAL drifted features | +21.0pp | +1.5pp | +16.8pp | MEDIUM | LOW |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 5 | Full retrain with all pipeline fixes applied | +12.2pp | +2.0pp | +10.1pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 501 |
| Rejected signals | 1175 |
| Selected precision | 54.1% |
| Rejected precision | 35.9% |
| Meta gate lift (precision) | +18.2% |
| Selected expectancy | +0.191% |
| Rejected expectancy | -0.021% |
| Selected Sharpe | +49.28 |
| Rejected Sharpe | -3.93 |

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

## Section 19 — Deep BTC/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +18.2% |
| Selected signals | 501 |
| Rejected signals | 1175 |
| Gate coverage | 29.9% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 68/100 |
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
| Precision Target | 58.9% |
| Actual Precision | 54.1% |
| Gap | -4.8% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 68/100 |
| Verdict | ⚠️ BELOW TARGET |


---
