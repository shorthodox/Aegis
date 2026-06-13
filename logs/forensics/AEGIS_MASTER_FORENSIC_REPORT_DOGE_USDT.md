# AEGIS-1 Master Forensic Report

**Symbol:** DOGE/USDT  |  **Generated:** 2026-06-14 02:00:52

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

> Measured precision is 56.3% — estimated 15.8pp below achievable ceiling of ~72.1%. Primary contributors: Absolute price features in model (7 critical) (−8.0pp), Other drifted features (3 CRITICAL/DEGRADED) (−3.7pp), Brier score above target (−2.0pp)

#### Precision Waterfall

```
Measured holdout precision :  56.3%

  [✅ FIXED       ]  −8.0pp prec    Absolute price features in model (7 critical)
  [✅ FIXED       ]  −3.7pp prec    Other drifted features (3 CRITICAL/DEGRADED)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.1pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.040 — overconfidence

Achievable precision       :  ~72.1%  (+15.8pp precision)
Recall deficit (BUY side)  :  −1.1pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.040 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.040 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **3 issues already applied** in the codebase (expected gain: +13.1pp once retrained).

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
   - Value: primary_confidence_threshold=0.620
   - PASS — val sweep selected threshold 0.620.

✅ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (52.4%)`
   - Value: signal_prec=56.3% vs breakeven=52.4% (410 fired, dir_prec≈91.3%)
   - PASS — signal_prec=56.3% ≥ breakeven=52.4%.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=24.6%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +22.9pp)  ✅ signal_prec ≥ breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 56.3% |
| Rejected signal precision | 33.4% |
| Gate lift (precision) | +22.9pp |
| Primary conf. threshold | 0.620 |
| Token breakeven | 52.4% |
| Selected signals | 410 |
| Rejected signals | 1258 |

> Primary gate adds 22.9pp of precision. Selected signals (56.3%) beat rejected (33.4%). Threshold=0.620.


---

## Section 15 — Executive Summary

**Symbol:** DOGE/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=56.3%  Sharpe=30.14
**Expected after fixes:** Precision≈67.0%  (+10.7pp)

### Top 5 Problems

1. 🔴 **Critical Feature Drift** — Score: 100/100
   > 26 features CRITICAL. Top: vwap_decay_mean_24 PSI=20.62.

2. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

3. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.

4. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 58/100
   > HOLD=58.7% of labels. Meta model sees 60% zero-labels → calibration distorted.

5. 🟡 **Confidence Inflation** — Score: 55/100
   > T=2.040>1.0. Model overestimates confidence.


### Top 5 Fixes

1. **Critical Feature Drift**
   → FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.

2. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

3. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

4. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

5. **Confidence Inflation**
   → Temperature scaling already applied. Verify aegis_state.pkl is loaded at inference.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 65.3% | ✓ |
| Dev OOF Precision | 40.6% | ✗ |
| Holdout Precision | 56.3% | ✗ |
| Holdout Coverage | 24.6% | ✓ |
| 95% CI Precision | [51.5%, 61.1%] | — |
| OOF→Holdout Gap | +15.7% | ✓ holdout beat OOF |
| Holdout Fired | 410 trades | ✓ |
| SELL Win Rate | 88.8% (197 trades) | ✓ |
| BUY Win Rate | 100.0% (56 trades) | ✓ |
| Sharpe (annualised) | 30.14 | ✓ |
| Max Drawdown | 6.18% | ✓ |
| Profit Factor | 4.50 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.2859% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.620 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | +4.0pp (be=52.4%) | ✓ above breakeven |
| Statistical Sig. | p=0.0102 (z=2.57) | ✓ significant |

### Class Distribution
- HOLD: **58.7%** — ⚠ severe imbalance
- SELL: **25.0%**
- BUY:  **16.2%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 58.7% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

**Feature health summary:** 38 HEALTHY | 14 WARNING | 12 DEGRADED | 26 CRITICAL  (of 90 total)

**Estimated total precision gain if top drifters removed:** +11.9 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `vwap_decay_mean_24` | **CRITICAL** | 20.623 | 1.000 | 0.181 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `vwap` | **CRITICAL** | 20.612 | 1.000 | 0.182 | 0.15 | NORMALISE | +0.8pp |
| 3 | `ema_200` | **CRITICAL** | 12.832 | 0.964 | 0.450 | 0.15 | NORMALISE | +0.8pp |
| 4 | `avwap_200` | **CRITICAL** | 12.688 | 0.913 | 0.443 | 0.15 | NORMALISE | +0.8pp |
| 5 | `r2` | **CRITICAL** | 12.685 | 0.913 | 0.451 | 0.15 | NORMALISE | +0.8pp |
| 6 | `s1` | **CRITICAL** | 12.634 | 0.913 | 0.441 | 0.15 | NORMALISE | +0.8pp |
| 7 | `rolling_support` | **CRITICAL** | 12.567 | 0.905 | 0.440 | 0.15 | NORMALISE | +0.8pp |
| 8 | `ema_100` | **CRITICAL** | 12.462 | 0.947 | 0.447 | 0.15 | NORMALISE | +0.8pp |
| 9 | `se_mid` | **CRITICAL** | 12.410 | 0.903 | 0.445 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 10 | `ichimoku_senkou_a` | **CRITICAL** | 12.396 | 0.928 | 0.445 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `vwap_decay_std_24`
- `keltner_lower`
- `kama_10`
- `keltner_upper`
- `dist_vwap`
- `vwap_decay_mean_24`
- `vwap_delta_12`
- `atr_14`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 39  | 94.9%  |
| SELL | 68 | 94.1% |
| HOLD | 8,629 | — |

### Signal Rejection Funnel

```
Generated (directional):       107  (100%)
Below Primary Conf. Thr:   -   28  (26%)
Blocked by Quality (<55):  -    8
Blocked by HMM:            -    5
Blocked by Confluence:     -    4
Blocked by Fake Breakout:  -    3
Blocked by Portfolio Guard:-    2
Blocked by Safe Mode:      -    1
Blocked by Drift:          -    2
Blocked by Cooldown:       -    1
─────────────────────────────────────
Estimated Executed:            53  (49.5%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.6664%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 63.5% |
| 12h | 86.7% |
| 18h | 96.2% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 28 | 15 | 13 | 53.8% | +46.66% |
| quality | 8 | 4 | 4 | 61.5% | +13.33% ⚠ |
| hmm | 5 | 2 | 3 | 57.6% | +8.33% ⚠ |
| confluence | 4 | 2 | 2 | 54.5% | +6.67% |
| fake_breakout | 3 | 1 | 2 | 60.0% | +5.00% ⚠ |
| portfolio | 2 | 1 | 1 | 57.6% | +3.33% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 2.0397 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.040) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 56.3% | -0.19 | ⚠ overconfident |
| 80-90% | 66.3% | -0.19 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 410 dev samples):** `isotonic`


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
| Feature Drift | 🟡 WARNING | 26 CRITICAL (5 active in model, 21 ✅ blacklisted/FIXED) / 12 DEGRADED / 90 total |
| Confidence Drift | 🔴 CRITICAL | T=2.040 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +56.34pp |

**Estimated precision loss from feature drift:** ~5.1pp


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
| ATR Multiplier | 0.75× | ⚠ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 30.14 | ✓ |
| Holdout Max DD | 6.18% | ✓ |
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

**5 root causes identified.**  Combined top-5 impact score: **374/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Critical Feature Drift** | Feature Drift | 100/100 | `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` ✅ FIXED | 26 features CRITICAL. Top: vwap_decay_mean_24 PSI=20.62.… |
| 2 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 3 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 4 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 58/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=58.7% of labels. Meta model sees 60% zero-labels → calibration di… |
| 5 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.040>1.0. Model overestimates confidence.… |

### Fixes

**1. Critical Feature Drift**
> 📍 `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` — `ema_9/21/50/100/200, vwap, avwap_*, ichimoku_senko`
> FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.

**2. Meta Model Calibration Failure**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

**3. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

**4. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

**5. Confidence Inflation**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Temperature scaling already applied. Verify aegis_state.pkl is loaded at inference.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 56.3%  →  **Expected precision (all fixes):** 67.0%  (+10.7pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Remove / normalise 26 CRITICAL drifted features | +11.9pp | +1.5pp | +9.5pp | MEDIUM | LOW |
| 2 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 3 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 4 | Full retrain with all pipeline fixes applied | +4.9pp | +2.0pp | +4.1pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 410 |
| Rejected signals | 1258 |
| Selected precision | 56.3% |
| Rejected precision | 33.4% |
| Meta gate lift (precision) | +23.0% |
| Selected expectancy | +0.286% |
| Rejected expectancy | -0.056% |
| Selected Sharpe | +49.26 |
| Rejected Sharpe | -7.87 |

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

## Section 19 — Deep DOGE/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +23.0% |
| Selected signals | 410 |
| Rejected signals | 1258 |
| Gate coverage | 24.6% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 72/100 |
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
| Precision Target | 57.4% |
| Actual Precision | 56.3% |
| Gap | -1.0% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 72/100 |
| Verdict | ⚠️ BELOW TARGET |


---
