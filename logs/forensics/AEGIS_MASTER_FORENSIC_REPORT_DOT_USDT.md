# AEGIS-1 Master Forensic Report

**Symbol:** DOT/USDT  |  **Generated:** 2026-06-14 02:00:52

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

> Measured precision is 51.2% — estimated 6.4pp below achievable ceiling of ~57.6%. Primary contributors: Barrier skew suppresses BUY labels (−1.9pp), Brier score above target (−2.0pp), HOLD over-representation in labels (−1.5pp)

#### Precision Waterfall

```
Measured holdout precision :  51.2%

  [✅ FIXED       ]  −1.9pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.5pp prec    HOLD over-representation in labels
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=1.912 — overconfidence

Achievable precision       :  ~57.6%  (+6.4pp precision)
Recall deficit (BUY side)  :  −2.5pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=1.912 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=1.912 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **2 issues already applied** in the codebase (expected gain: +4.1pp once retrained).

> Meta gate removed entirely. Primary-only calibrated gate with signal_prec >= token_breakeven tradeable check. FIXED.

#### Signal Gate Trace (Primary-Only Calibrated Gate)

🔴 **Signal gate DISABLED — root cause at Gate: 2. Val sweep finds threshold with ≥50 fires**

✅ **1. Primary-only mode active + directional skill ≥ 45%**
   - `scripts/retrain_model.py — primary_only_gate block`
   - Check: `primary_only_mode=True AND primary dir_prec ≥ 45% (else veto → None)`
   - Value: primary_only_mode=True, calibrator=Y
   - PASS — primary-only mode active. Calibrated LR maps raw probs to confidence.

❌ **2. Val sweep finds threshold with ≥50 fires**
   - `scripts/retrain_model.py — val sweep (0.50→0.95, min_fires=50)`
   - Check: `max(signal_prec over thresholds with ≥50 val fires)`
   - Value: primary_confidence_threshold=0.500
   - FAIL — no threshold with ≥50 val fires found. Defaulted to 0.85.

❌ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (52.3%)`
   - Value: signal_prec=51.2% vs breakeven=52.3% (767 fired, dir_prec≈82.4%)
   - FAIL — signal_prec=51.2% < breakeven=52.3% (gap=1.0pp). Primary model needs stronger directional skill.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=46.0%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +19.1pp)  ❌ signal_prec < breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 51.2% |
| Rejected signal precision | 32.2% |
| Gate lift (precision) | +19.1pp |
| Primary conf. threshold | 0.500 |
| Token breakeven | 52.3% |
| Selected signals | 767 |
| Rejected signals | 901 |

> Primary gate adds 19.1pp of precision. Selected signals (51.2%) beat rejected (32.2%). Threshold=0.500. | DISABLED: selected signal_prec=51.2% < breakeven=52.3%.


---

## Section 15 — Executive Summary

**Symbol:** DOT/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=51.2%  Sharpe=29.91
**Expected after fixes:** Precision≈54.2%  (+2.9pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 60/100
   > HOLD=60.2% of labels. Meta model sees 60% zero-labels → calibration distorted.

4. 🟡 **Confidence Inflation** — Score: 55/100
   > T=1.912>1.0. Model overestimates confidence.


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
| CV Accuracy (OOF) | 68.6% | ✓ |
| Dev OOF Precision | 36.1% | ✗ |
| Holdout Precision | 51.2% | ✗ |
| Holdout Coverage | 46.0% | ✓ |
| 95% CI Precision | [47.7%, 54.8%] | — |
| OOF→Holdout Gap | +15.1% | ✓ holdout beat OOF |
| Holdout Fired | 767 trades | ✓ |
| SELL Win Rate | 83.7% (257 trades) | ✓ |
| BUY Win Rate | 80.9% (220 trades) | ✓ |
| Sharpe (annualised) | 29.91 | ✓ |
| Max Drawdown | 3.12% | ✓ |
| Profit Factor | 2.88 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.1943% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.500 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | -1.0pp (be=52.3%) | ✗ below breakeven |
| Statistical Sig. | p=0.4927 (z=0.69) | ⚠ insufficient data |

### Class Distribution
- HOLD: **60.2%** — ⚠ severe imbalance
- SELL: **26.1%**
- BUY:  **13.7%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 60.2% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

⚠ feature_health.json not found



---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 39  | 82.0%  |
| SELL | 25 | 92.0% |
| HOLD | 8,672 | — |

### Signal Rejection Funnel

```
Generated (directional):        64  (100%)
Below Primary Conf. Thr:   -    0  (0%)
Blocked by Quality (<55):  -    5
Blocked by HMM:            -    3
Blocked by Confluence:     -    2
Blocked by Fake Breakout:  -    1
Blocked by Portfolio Guard:-    1
Blocked by Safe Mode:      -    0
Blocked by Drift:          -    1
Blocked by Cooldown:       -    0
─────────────────────────────────────
Estimated Executed:            51  (79.7%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.8915%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 51.9% |
| 12h | 79.8% |
| 18h | 94.3% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **6 bars** (6h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 0 | 0 | 0 | 0.0% | +0.00% |
| quality | 5 | 3 | 2 | 61.5% | +9.46% ⚠ |
| hmm | 3 | 1 | 2 | 57.6% | +5.67% ⚠ |
| confluence | 2 | 1 | 1 | 54.5% | +3.78% |
| fake_breakout | 1 | 0 | 1 | 60.0% | +1.89% ⚠ |
| portfolio | 1 | 0 | 1 | 57.6% | +1.89% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.9117 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=1.912) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 51.2% | -0.24 | ⚠ overconfident |
| 80-90% | 61.2% | -0.24 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 767 dev samples):** `isotonic`


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
| Feature Drift | 🟢 OK | 0 CRITICAL (0 active in model, 0 ✅ blacklisted/FIXED) / 0 DEGRADED / 0 total |
| Confidence Drift | 🔴 CRITICAL | T=1.912 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +51.24pp |

**Estimated precision loss from feature drift:** ~0.0pp


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
| Holdout Sharpe | 29.91 | ✓ |
| Holdout Max DD | 3.12% | ✓ |
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

**4 root causes identified.**  Combined top-5 impact score: **276/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 60/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=60.2% of labels. Meta model sees 60% zero-labels → calibration di… |
| 4 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=1.912>1.0. Model overestimates confidence.… |

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

**Base precision:** 51.2%  →  **Expected precision (all fixes):** 54.2%  (+2.9pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 2 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 3 | Full retrain with all pipeline fixes applied | +1.3pp | +2.0pp | +1.1pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 767 |
| Rejected signals | 901 |
| Selected precision | 51.2% |
| Rejected precision | 32.2% |
| Meta gate lift (precision) | +19.1% |
| Selected expectancy | +0.194% |
| Rejected expectancy | -0.025% |
| Selected Sharpe | +35.74 |
| Rejected Sharpe | -3.74 |

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

## Section 19 — Deep DOT/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +19.1% |
| Selected signals | 767 |
| Rejected signals | 901 |
| Gate coverage | 46.0% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 69/100 |
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
| Precision Target | 57.3% |
| Actual Precision | 51.2% |
| Gap | -6.0% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 69/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
