# AEGIS-1 Master Forensic Report

**Symbol:** ADA/USDT  |  **Generated:** 2026-06-14 02:00:51

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

> Measured precision is 54.9% — estimated 17.6pp below achievable ceiling of ~72.5%. Primary contributors: Absolute price features in model (6 critical) (−7.4pp), Other drifted features (4 CRITICAL/DEGRADED) (−4.0pp), Barrier skew suppresses BUY labels (−1.6pp)

#### Precision Waterfall

```
Measured holdout precision :  54.9%

  [✅ FIXED       ]  −7.4pp prec    Absolute price features in model (6 critical)
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (4 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −1.6pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.6pp prec    HOLD over-representation in labels
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.173 — overconfidence

Achievable precision       :  ~72.5%  (+17.6pp precision)
Recall deficit (BUY side)  :  −2.2pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.173 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.173 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **4 issues already applied** in the codebase (expected gain: +15.3pp once retrained).

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
   - Check: `signal_prec_h >= token_breakeven (52.0%)`
   - Value: signal_prec=54.9% vs breakeven=52.0% (266 fired, dir_prec≈99.3%)
   - PASS — signal_prec=54.9% ≥ breakeven=52.0%.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=15.9%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +17.4pp)  ✅ signal_prec ≥ breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 54.9% |
| Rejected signal precision | 37.5% |
| Gate lift (precision) | +17.4pp |
| Primary conf. threshold | 0.620 |
| Token breakeven | 52.0% |
| Selected signals | 266 |
| Rejected signals | 1402 |

> Primary gate adds 17.4pp of precision. Selected signals (54.9%) beat rejected (37.5%). Threshold=0.620.


---

## Section 15 — Executive Summary

**Symbol:** ADA/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=54.9%  Sharpe=25.38
**Expected after fixes:** Precision≈67.2%  (+12.3pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 60/100
   > HOLD=60.9% of labels. Meta model sees 60% zero-labels → calibration distorted.

4. 🟡 **Confidence Inflation** — Score: 55/100
   > T=2.173>1.0. Model overestimates confidence.

5. 🟡 **Critical Feature Drift** — Score: 48/100
   > 12 features CRITICAL. Top: avwap_50 PSI=22.69.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

3. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

4. **Confidence Inflation**
   → Temperature scaling already applied. Verify aegis_state.pkl is loaded at inference.

5. **Critical Feature Drift**
   → FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 66.3% | ✓ |
| Dev OOF Precision | 26.1% | ✗ |
| Holdout Precision | 54.9% | ✗ |
| Holdout Coverage | 16.0% | ✓ |
| 95% CI Precision | [48.9%, 60.8%] | — |
| OOF→Holdout Gap | +28.8% | ✓ holdout beat OOF |
| Holdout Fired | 266 trades | ✓ |
| SELL Win Rate | 99.0% (99 trades) | ✓ |
| BUY Win Rate | 100.0% (48 trades) | ✓ |
| Sharpe (annualised) | 25.38 | ✓ |
| Max Drawdown | 1.13% | ✓ |
| Profit Factor | 5.25 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.1959% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.620 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | +2.9pp (be=52.0%) | ✓ above breakeven |
| Statistical Sig. | p=0.1109 (z=1.59) | ⚠ insufficient data |

### Class Distribution
- HOLD: **60.9%** — ⚠ severe imbalance
- SELL: **24.4%**
- BUY:  **14.7%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 60.9% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

**Feature health summary:** 40 HEALTHY | 13 WARNING | 2 DEGRADED | 12 CRITICAL  (of 67 total)

**Estimated total precision gain if top drifters removed:** +14.5 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `avwap_50` | **CRITICAL** | 22.686 | 0.976 | 0.556 | 0.15 | NORMALISE | +0.8pp |
| 2 | `avwap_200` | **CRITICAL** | 22.674 | 0.984 | 0.552 | 0.15 | NORMALISE | +0.8pp |
| 3 | `avwap_100` | **CRITICAL** | 22.606 | 0.998 | 0.554 | 0.15 | NORMALISE | +0.8pp |
| 4 | `ichimoku_senkou_a` | **CRITICAL** | 22.535 | 0.961 | 0.555 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `dist_vwap` | **CRITICAL** | 16.503 | 0.931 | 3.192 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `rolling_support` | **CRITICAL** | 15.399 | 0.945 | 0.557 | 0.15 | NORMALISE | +0.8pp |
| 7 | `vwap` | **CRITICAL** | 11.370 | 0.849 | 0.145 | 0.15 | NORMALISE | +0.8pp |
| 8 | `vwap_decay_mean_24` | **CRITICAL** | 11.297 | 0.846 | 0.142 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 9 | `close_decay_std_24` | **CRITICAL** | 3.222 | 0.597 | 0.563 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 10 | `atr_14` | **CRITICAL** | 2.142 | 0.649 | 0.550 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `vwap_decay_mean_24`
- `vwap_decay_std_24`
- `close_decay_std_24`
- `atr_14`
- `vwap_delta_12`
- `dist_vwap`
- `ichimoku_senkou_a`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 1  | 100.0%  |
| SELL | 10 | 90.0% |
| HOLD | 8,725 | — |

### Signal Rejection Funnel

```
Generated (directional):        11  (100%)
Below Primary Conf. Thr:   -    5  (45%)
Blocked by Quality (<55):  -    0
Blocked by HMM:            -    0
Blocked by Confluence:     -    0
Blocked by Fake Breakout:  -    0
Blocked by Portfolio Guard:-    0
Blocked by Safe Mode:      -    0
Blocked by Drift:          -    0
Blocked by Cooldown:       -    0
─────────────────────────────────────
Estimated Executed:             6  (54.5%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.6299%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 64.7% |
| 12h | 88.4% |
| 18h | 96.5% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 5 | 2 | 3 | 53.8% | +8.15% |
| quality | 0 | 0 | 0 | 0.0% | +0.00% |
| hmm | 0 | 0 | 0 | 0.0% | +0.00% |
| confluence | 0 | 0 | 0 | 0.0% | +0.00% |
| fake_breakout | 0 | 0 | 0 | 0.0% | +0.00% |
| portfolio | 0 | 0 | 0 | 0.0% | +0.00% |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 2.1729 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.173) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 54.9% | -0.20 | ⚠ overconfident |
| 80-90% | 64.9% | -0.20 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 266 dev samples):** `isotonic`


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
| Feature Drift | 🟢 OK | 12 CRITICAL (3 active in model, 9 ✅ blacklisted/FIXED) / 2 DEGRADED / 67 total |
| Confidence Drift | 🔴 CRITICAL | T=2.173 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +54.89pp |

**Estimated precision loss from feature drift:** ~2.5pp


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
| ATR Multiplier | 0.6× | ⚠ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 25.38 | ✓ |
| Holdout Max DD | 1.13% | ✓ |
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

**5 root causes identified.**  Combined top-5 impact score: **324/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 60/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=60.9% of labels. Meta model sees 60% zero-labels → calibration di… |
| 4 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.173>1.0. Model overestimates confidence.… |
| 5 | 🟡 **Critical Feature Drift** | Feature Drift | 48/100 | `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` ✅ FIXED | 12 features CRITICAL. Top: avwap_50 PSI=22.69.… |

### Fixes

**1. Meta Model Calibration Failure**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

**2. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

**3. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

**4. Confidence Inflation**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Temperature scaling already applied. Verify aegis_state.pkl is loaded at inference.

**5. Critical Feature Drift**
> 📍 `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` — `ema_9/21/50/100/200, vwap, avwap_*, ichimoku_senko`
> FEATURE_BLACKLIST (31 features: raw OHLCV, se_mid, EMA/VWAP levels, decay means) + OBV/PVT z-score. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 54.9%  →  **Expected precision (all fixes):** 67.2%  (+12.3pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Remove / normalise 12 CRITICAL drifted features | +14.5pp | +1.5pp | +11.6pp | MEDIUM | LOW |
| 2 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 3 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 4 | Full retrain with all pipeline fixes applied | +5.7pp | +2.0pp | +4.8pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 266 |
| Rejected signals | 1402 |
| Selected precision | 54.9% |
| Rejected precision | 37.5% |
| Meta gate lift (precision) | +17.4% |
| Selected expectancy | +0.196% |
| Rejected expectancy | +0.011% |
| Selected Sharpe | +51.50 |
| Rejected Sharpe | +1.77 |

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

## Section 19 — Deep ADA/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +17.4% |
| Selected signals | 266 |
| Rejected signals | 1402 |
| Gate coverage | 15.9% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 67/100 |
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
| Precision Target | 57.0% |
| Actual Precision | 54.9% |
| Gap | -2.1% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 67/100 |
| Verdict | ⚠️ BELOW TARGET |


---
