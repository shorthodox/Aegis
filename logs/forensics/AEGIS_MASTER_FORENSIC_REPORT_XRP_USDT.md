# AEGIS-1 Master Forensic Report

**Symbol:** XRP/USDT  |  **Generated:** 2026-06-14 02:00:54

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

> Measured precision is 48.7% — estimated 10.7pp below achievable ceiling of ~59.4%. Primary contributors: Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), Absolute price features in model (1 critical) (−2.1pp), Brier score above target (−2.0pp)

#### Precision Waterfall

```
Measured holdout precision :  48.7%

  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.1pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=1.911 — overconfidence
  [✅ FIXED       ]  −0.5pp prec    Anti-selective gate — precision below chance

Achievable precision       :  ~59.4%  (+10.7pp precision)
Recall deficit (BUY side)  :  −1.1pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=1.911 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=1.911 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **4 issues already applied** in the codebase (expected gain: +8.0pp once retrained).

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
   - Check: `signal_prec_h >= token_breakeven (52.9%)`
   - Value: signal_prec=48.7% vs breakeven=52.9% (977 fired, dir_prec≈72.7%)
   - FAIL — signal_prec=48.7% < breakeven=52.9% (gap=4.2pp). Primary model needs stronger directional skill.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=58.8%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +24.9pp)  ❌ signal_prec < breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 48.7% |
| Rejected signal precision | 23.8% |
| Gate lift (precision) | +24.9pp |
| Primary conf. threshold | 0.500 |
| Token breakeven | 52.9% |
| Selected signals | 977 |
| Rejected signals | 685 |

> Primary gate adds 24.9pp of precision. Selected signals (48.7%) beat rejected (23.8%). Threshold=0.500. | DISABLED: selected signal_prec=48.7% < breakeven=52.9%.


---

## Section 15 — Executive Summary

**Symbol:** XRP/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=48.7%  Sharpe=26.45
**Expected after fixes:** Precision≈77.5%  (+28.8pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🔴 **Anti-Selective Gate (precision < 50%)** — Score: 90/100
   > holdout_prec=48.7% — gate is selecting WRONG signals. Worse than random for directional trading.

3. 🔴 **Signal Precision Below Token Breakeven** — Score: 85/100
   > signal_prec=48.7% < breakeven=52.9% (gap=4.2pp). Primary model has insufficient directional skill to clear the fee breakeven after HOLD-timeout dilution.

4. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.

5. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 57/100
   > HOLD=57.1% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **Anti-Selective Gate (precision < 50%)**
   → Fix primary OOF quality (soft confluence features + local model hyperparams). Directional precision veto (_fired_dir_prec<0.50) disables gate and outputs tradeable=False.

3. **Signal Precision Below Token Breakeven**
   → Increase primary model AUPRC (Optuna aucpr objective). Target dir_prec >= 65% so fired signals clear breakeven even at 30% HOLD-timeout rate.

4. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

5. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 67.0% | ✓ |
| Dev OOF Precision | 30.5% | ✗ |
| Holdout Precision | 48.7% | ✗ |
| Holdout Coverage | 58.8% | ✓ |
| 95% CI Precision | [45.6%, 51.8%] | — |
| OOF→Holdout Gap | +18.2% | ✓ holdout beat OOF |
| Holdout Fired | 977 trades | ✓ |
| SELL Win Rate | 66.7% (487 trades) | ✓ |
| BUY Win Rate | 89.9% (168 trades) | ✓ |
| Sharpe (annualised) | 26.45 | ✓ |
| Max Drawdown | 12.08% | ✗ |
| Profit Factor | 2.25 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.2437% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.500 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | -4.2pp (be=52.9%) | ✗ below breakeven |
| Statistical Sig. | p=0.4238 (z=-0.80) | ⚠ insufficient data |

### Class Distribution
- HOLD: **57.1%** — ⚠ severe imbalance
- SELL: **26.4%**
- BUY:  **16.5%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 57.1% HOLD labels biases model toward neutrality.
- **CRITICAL** — Anti-selective gate: holdout precision=48.7% < 50%. Gate is selecting the worst signals. Rebuild with fixed local model.
- **CRITICAL** — Precision below random chance (48.7%). Model is directionally anti-predictive on holdout data.


---

## Section 2 — Feature Forensics

**Feature health summary:** 34 HEALTHY | 15 WARNING | 5 DEGRADED | 9 CRITICAL  (of 63 total)

**Estimated total precision gain if top drifters removed:** +19.8 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `high` | **CRITICAL** | 21.125 | 0.934 | 0.415 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `vwap_decay_mean_24` | **CRITICAL** | 20.940 | 0.928 | 6.054 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 3 | `open` | **CRITICAL** | 20.571 | 0.934 | 0.416 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 4 | `se_mid` | **CRITICAL** | 20.553 | 0.933 | 0.416 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `low` | **CRITICAL** | 19.590 | 0.934 | 0.416 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `vwap_decay_std_24` | **CRITICAL** | 2.428 | 0.648 | 0.785 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 7 | `vwap_delta_4` | **CRITICAL** | 2.100 | 0.549 | 3.520 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 8 | `funding_rate` | **CRITICAL** | 1.235 | 0.446 | 2.205 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 9 | `fib_range_pct` | **CRITICAL** | 0.847 | 0.150 | 0.149 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 10 | `atr_14` | **DEGRADED** | 0.607 | 0.403 | 0.324 | 0.50 | MONITOR | +0.9pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `high`
- `open`
- `vwap_decay_mean_24`
- `fib_range_pct`
- `vwap_delta_4`
- `low`
- `funding_rate`
- `se_mid`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 59  | 96.6%  |
| SELL | 80 | 86.2% |
| HOLD | 8,597 | — |

### Signal Rejection Funnel

```
Generated (directional):       139  (100%)
Below Primary Conf. Thr:   -    0  (0%)
Blocked by Quality (<55):  -   11
Blocked by HMM:            -    6
Blocked by Confluence:     -    5
Blocked by Fake Breakout:  -    4
Blocked by Portfolio Guard:-    2
Blocked by Safe Mode:      -    1
Blocked by Drift:          -    2
Blocked by Cooldown:       -    1
─────────────────────────────────────
Estimated Executed:           107  (77.0%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.3512%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 64.1% |
| 12h | 86.8% |
| 18h | 96.4% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 0 | 0 | 0 | 0.0% | +0.00% |
| quality | 11 | 6 | 5 | 61.5% | +14.86% ⚠ |
| hmm | 6 | 3 | 3 | 57.6% | +8.11% ⚠ |
| confluence | 5 | 2 | 3 | 54.5% | +6.76% |
| fake_breakout | 4 | 2 | 2 | 60.0% | +5.40% ⚠ |
| portfolio | 2 | 1 | 1 | 57.6% | +2.70% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.9105 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=1.910) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 48.7% | -0.26 | ⚠ overconfident |
| 80-90% | 58.7% | -0.26 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 977 dev samples):** `isotonic`


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
| Feature Drift | 🟢 OK | 9 CRITICAL (3 active in model, 6 ✅ blacklisted/FIXED) / 5 DEGRADED / 63 total |
| Confidence Drift | 🔴 CRITICAL | T=1.911 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +48.72pp |

**Estimated precision loss from feature drift:** ~2.7pp


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
| ATR Multiplier | 1.35× | ✓ |
| Win Rate | 76.0% | ✓ |
| Avg Win / Avg Loss | 1.768% / 1.178% | — |
| R:R Ratio | 1.50 | ✓ favourable |
| Kelly Fraction | 60.0% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 26.45 | ✓ |
| Holdout Max DD | 12.08% | ✗ |
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

**6 root causes identified.**  Combined top-5 impact score: **393/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🔴 **Anti-Selective Gate (precision < 50%)** | Gate Failure | 90/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | holdout_prec=48.7% — gate is selecting WRONG signals. Worse than rando… |
| 3 | 🔴 **Signal Precision Below Token Breakeven** | Gate Architecture | 85/100 | `scripts/retrain_model.py:_signal_prec_h >= token_breakeven check in primary_only_gate` 🔴 ACTIVE | signal_prec=48.7% < breakeven=52.9% (gap=4.2pp). Primary model has ins… |
| 4 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 5 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 57/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=57.1% of labels. Meta model sees 60% zero-labels → calibration di… |
| 6 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=1.911>1.0. Model overestimates confidence.… |

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

**4. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

**5. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 48.7%  →  **Expected precision (all fixes):** 77.5%  (+28.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Fix local model quality (highest ROI) | +15.0pp | +3.0pp | +12.0pp | HIGH | LOW |
| 2 | Increase primary model AUPRC to clear signal_prec breakeven (gap=4.2pp) | +5.0pp | +5.0pp | +4.2pp | MEDIUM | MEDIUM |
| 3 | Remove / normalise 9 CRITICAL drifted features | +19.8pp | +1.5pp | +15.8pp | MEDIUM | LOW |
| 4 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 5 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 6 | Full retrain with all pipeline fixes applied | +13.3pp | +2.0pp | +11.1pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 977 |
| Rejected signals | 685 |
| Selected precision | 48.7% |
| Rejected precision | 23.8% |
| Meta gate lift (precision) | +24.9% |
| Selected expectancy | +0.244% |
| Rejected expectancy | -0.167% |
| Selected Sharpe | +28.00 |
| Rejected Sharpe | -14.79 |

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

## Section 19 — Deep XRP/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +24.9% |
| Selected signals | 977 |
| Rejected signals | 685 |
| Gate coverage | 58.8% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 74/100 |
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
| Precision Target | 57.9% |
| Actual Precision | 48.7% |
| Gap | -9.2% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 74/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
