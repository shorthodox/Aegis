# AEGIS-1 Master Forensic Report

**Symbol:** BNB/USDT  |  **Generated:** 2026-06-14 02:00:52

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

> Measured precision is 53.9% — estimated 15.5pp below achievable ceiling of ~69.4%. Primary contributors: CV accuracy near random — primary learned nothing (−6.4pp), Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), Absolute price features in model (1 critical) (−2.1pp)

#### Precision Waterfall

```
Measured holdout precision :  53.9%

  [✅ FIXED       ]  −6.4pp prec    CV accuracy near random — primary learned nothing
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.208 — overconfidence

Achievable precision       :  ~69.4%  (+15.5pp precision)
Recall deficit (BUY side)  :  −1.9pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.208 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.208 — overconfidence**
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
   - Value: primary_confidence_threshold=0.660
   - PASS — val sweep selected threshold 0.660.

✅ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (51.9%)`
   - Value: signal_prec=53.9% vs breakeven=51.9% (360 fired, dir_prec≈98.5%)
   - PASS — signal_prec=53.9% ≥ breakeven=51.9%.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=21.7%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +11.0pp)  ✅ signal_prec ≥ breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 53.9% |
| Rejected signal precision | 42.9% |
| Gate lift (precision) | +11.0pp |
| Primary conf. threshold | 0.660 |
| Token breakeven | 51.9% |
| Selected signals | 360 |
| Rejected signals | 1302 |

> Primary gate adds 11.0pp of precision. Selected signals (53.9%) beat rejected (42.9%). Threshold=0.660.


---

## Section 15 — Executive Summary

**Symbol:** BNB/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=53.9%  Sharpe=33.81
**Expected after fixes:** Precision≈67.3%  (+13.5pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 55/100
   > HOLD=55.1% of labels. Meta model sees 60% zero-labels → calibration distorted.

4. 🟡 **Confidence Inflation** — Score: 55/100
   > T=2.208>1.0. Model overestimates confidence.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

3. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

4. **Confidence Inflation**
   → Temperature scaling already applied. Verify aegis_state.pkl is loaded at inference.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 57.1% | ⚠ |
| Dev OOF Precision | 44.2% | ✗ |
| Holdout Precision | 53.9% | ✗ |
| Holdout Coverage | 21.7% | ✓ |
| 95% CI Precision | [48.7%, 59.0%] | — |
| OOF→Holdout Gap | +9.7% | ✓ holdout beat OOF |
| Holdout Fired | 360 trades | ✓ |
| SELL Win Rate | 98.2% (114 trades) | ✓ |
| BUY Win Rate | 98.8% (83 trades) | ✓ |
| Sharpe (annualised) | 33.81 | ✓ |
| Max Drawdown | 2.55% | ✓ |
| Profit Factor | 9.42 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.4487% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.660 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | +2.0pp (be=51.9%) | ✓ above breakeven |
| Statistical Sig. | p=0.1400 (z=1.48) | ⚠ insufficient data |

### Class Distribution
- HOLD: **55.1%** — ⚠ severe imbalance
- SELL: **24.7%**
- BUY:  **20.2%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 55.1% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

**Feature health summary:** 27 HEALTHY | 15 WARNING | 9 DEGRADED | 6 CRITICAL  (of 57 total)

**Estimated total precision gain if top drifters removed:** +16.2 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `vwap_decay_mean_24` | **CRITICAL** | 19.650 | 0.894 | 10.316 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `low` | **CRITICAL** | 4.776 | 0.695 | 0.247 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 3 | `close` | **CRITICAL** | 4.458 | 0.695 | 0.247 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 4 | `high` | **CRITICAL** | 4.365 | 0.692 | 0.247 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `vwap_decay_std_24` | **CRITICAL** | 1.379 | 0.268 | 0.412 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `volume_decay_mean_24` | **CRITICAL** | 0.939 | 0.357 | 0.382 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 7 | `fib_range_pct` | **DEGRADED** | 0.654 | 0.258 | 0.177 | 0.50 | MONITOR | +0.9pp |
| 8 | `price_zscore_200` | **DEGRADED** | 0.598 | 0.291 | 2.686 | 0.50 | MONITOR | +0.9pp |
| 9 | `returns_1h_decay_std_24` | **DEGRADED** | 0.516 | 0.173 | 0.065 | 0.50 | MONITOR | +0.9pp |
| 10 | `volume_decay_std_24` | **DEGRADED** | 0.440 | 0.289 | 0.371 | 0.50 | MONITOR | +0.9pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `vwap_decay_std_24`
- `vwap_decay_mean_24`
- `low`
- `high`
- `volume_decay_mean_24`
- `close`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 0  | 0.0%  |
| SELL | 0 | 0.0% |
| HOLD | 8,736 | — |

### Signal Rejection Funnel

```
Generated (directional):         0  (100%)
Below Primary Conf. Thr:   -    0  (0%)
Blocked by Quality (<55):  -    0
Blocked by HMM:            -    0
Blocked by Confluence:     -    0
Blocked by Fake Breakout:  -    0
Blocked by Portfolio Guard:-    0
Blocked by Safe Mode:      -    0
Blocked by Drift:          -    0
Blocked by Cooldown:       -    0
─────────────────────────────────────
Estimated Executed:             0  (0.0%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **0.8552%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 75.0% |
| 12h | 92.3% |
| 18h | 98.2% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **4 bars** (4h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 0 | 0 | 0 | 0.0% | +0.00% |
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
| Cal. Temperature | 2.2078 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.208) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 53.9% | -0.21 | ⚠ overconfident |
| 80-90% | 63.9% | -0.21 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 360 dev samples):** `isotonic`


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
| Feature Drift | 🟢 OK | 6 CRITICAL (0 active in model, 6 ✅ blacklisted/FIXED) / 9 DEGRADED / 57 total |
| Confidence Drift | 🔴 CRITICAL | T=2.208 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +53.89pp |

**Estimated precision loss from feature drift:** ~3.2pp


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
| ATR Multiplier | 1.5× | ✓ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 33.81 | ✓ |
| Holdout Max DD | 2.55% | ✓ |
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

**4 root causes identified.**  Combined top-5 impact score: **271/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 55/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=55.1% of labels. Meta model sees 60% zero-labels → calibration di… |
| 4 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.208>1.0. Model overestimates confidence.… |

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



---

## Section 14 — Automated Improvement Engine

**Base precision:** 53.9%  →  **Expected precision (all fixes):** 67.3%  (+13.5pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Remove / normalise 6 CRITICAL drifted features | +16.2pp | +1.5pp | +13.0pp | MEDIUM | LOW |
| 2 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 3 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 4 | Full retrain with all pipeline fixes applied | +6.2pp | +2.0pp | +5.2pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 360 |
| Rejected signals | 1302 |
| Selected precision | 53.9% |
| Rejected precision | 42.9% |
| Meta gate lift (precision) | +10.9% |
| Selected expectancy | +0.449% |
| Rejected expectancy | +0.137% |
| Selected Sharpe | +58.97 |
| Rejected Sharpe | +14.91 |

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

## Section 19 — Deep BNB/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +10.9% |
| Selected signals | 360 |
| Rejected signals | 1302 |
| Gate coverage | 21.7% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 60/100 |
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
| Precision Target | 56.9% |
| Actual Precision | 53.9% |
| Gap | -3.0% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 60/100 |
| Verdict | ⚠️ BELOW TARGET |


---
