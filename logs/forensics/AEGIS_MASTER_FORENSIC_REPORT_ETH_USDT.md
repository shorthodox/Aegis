# AEGIS-1 Master Forensic Report

**Symbol:** ETH/USDT  |  **Generated:** 2026-06-14 02:00:53

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

> Measured precision is 40.9% — estimated 31.4pp below achievable ceiling of ~72.3%. Primary contributors: CV accuracy near random — primary learned nothing (−13.3pp), Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), Anti-selective gate — precision below chance (−3.6pp)

#### Precision Waterfall

```
Measured holdout precision :  40.9%

  [✅ FIXED       ]  −13.3pp prec    CV accuracy near random — primary learned nothing
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −3.6pp prec    Anti-selective gate — precision below chance
  [✅ FIXED       ]  −2.4pp prec    Barrier skew suppresses BUY labels
  [✅ FIXED       ]  −2.6pp prec    HOLD over-representation in labels
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=2.012 — overconfidence
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~72.3%  (+31.4pp precision)
Recall deficit (BUY side)  :  −7.4pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=2.012 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |
| Gate blocking signals that would have won | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 0.4pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=2.012 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**3. Gate blocking signals that would have won**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **6 issues already applied** in the codebase (expected gain: +30.2pp once retrained).

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
   - Value: primary_confidence_threshold=0.780
   - PASS — val sweep selected threshold 0.780.

❌ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (52.8%)`
   - Value: signal_prec=40.9% vs breakeven=52.8% (66 fired, dir_prec≈90.0%)
   - FAIL — signal_prec=40.9% < breakeven=52.8% (gap=11.9pp). Primary model needs stronger directional skill.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=4.0%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

🟡 **Gate status: NEUTRAL**  (lift: +2.1pp)  ❌ signal_prec < breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 40.9% |
| Rejected signal precision | 38.8% |
| Gate lift (precision) | +2.1pp |
| Primary conf. threshold | 0.780 |
| Token breakeven | 52.8% |
| Selected signals | 66 |
| Rejected signals | 1596 |

> Primary gate minimal discrimination (40.9% selected ≈ 38.8% rejected). Threshold=0.780. | DISABLED: selected signal_prec=40.9% < breakeven=52.8%.


---

## Section 15 — Executive Summary

**Symbol:** ETH/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=40.9%  Sharpe=6.57
**Expected after fixes:** Precision≈75.8%  (+34.8pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🔴 **Anti-Selective Gate (precision < 50%)** — Score: 90/100
   > holdout_prec=40.9% — gate is selecting WRONG signals. Worse than random for directional trading.

3. 🔴 **Signal Precision Below Token Breakeven** — Score: 85/100
   > signal_prec=40.9% < breakeven=52.8% (gap=11.9pp). Primary model has insufficient directional skill to clear the fee breakeven after HOLD-timeout dilution.

4. 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** — Score: 85/100
   > cv_accuracy=60.8% vs majority_baseline=67.4%. binary_dual SPW inflation compresses hold_residual → argmax always BUY/SELL. Check sidecar for bayes_prior_correction key — if absent, Bayes fix not applied.

5. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 67/100
   > HOLD=67.4% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply platt calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

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
| CV Accuracy (OOF) | 60.8% | ⚠ |
| Dev OOF Precision | 27.3% | ✗ |
| Holdout Precision | 40.9% | ✗ |
| Holdout Coverage | 4.0% | ✓ |
| 95% CI Precision | [29.9%, 52.9%] | — |
| OOF→Holdout Gap | +13.6% | ✓ holdout beat OOF |
| Holdout Fired | 66 trades | ✓ |
| SELL Win Rate | 100.0% (12 trades) | ✓ |
| BUY Win Rate | 83.3% (18 trades) | ✓ |
| Sharpe (annualised) | 6.57 | ✓ |
| Max Drawdown | 2.06% | ✓ |
| Profit Factor | 2.19 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.0998% | ✓ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.780 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | -11.9pp (be=52.8%) | ✗ below breakeven |
| Statistical Sig. | p=0.1396 (z=-1.48) | ⚠ insufficient data |

### Class Distribution
- HOLD: **67.4%** — ⚠ severe imbalance
- SELL: **20.8%**
- BUY:  **11.9%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 67.4% HOLD labels biases model toward neutrality.
- **CRITICAL** — Anti-selective gate: holdout precision=40.9% < 50%. Gate is selecting the worst signals. Rebuild with fixed local model.
- **CRITICAL** — Precision below random chance (40.9%). Model is directionally anti-predictive on holdout data.
- **CRITICAL** — CV accuracy=60.8% ≈ random baseline=67.4%. Primary model learned nothing from features.
- **CRITICAL** — Signal precision -11.9pp below breakeven (52.8%). Token correctly DISABLED. Increase primary model directional precision to clear breakeven gate.


---

## Section 2 — Feature Forensics

**Feature health summary:** 35 HEALTHY | 15 WARNING | 14 DEGRADED | 9 CRITICAL  (of 73 total)

**Estimated total precision gain if top drifters removed:** +19.8 pp

### Top 25 Drifting Features

| Rank | Feature | State | PSI | KS | Mean Drift | Penalty | Rec. | Est. Gain |
|------|---------|-------|-----|----|------------|---------|------|-----------|
| 1 | `vwap_decay_mean_24` | **CRITICAL** | 20.758 | 0.915 | 22.876 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 2 | `se_mid` | **CRITICAL** | 18.159 | 0.888 | 0.398 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 3 | `close` | **CRITICAL** | 18.047 | 0.886 | 0.398 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 4 | `open` | **CRITICAL** | 18.042 | 0.885 | 0.398 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 5 | `low` | **CRITICAL** | 15.596 | 0.888 | 0.399 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 6 | `vwap_delta_12` | **CRITICAL** | 5.331 | 0.681 | 2.276 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 7 | `funding_rate_ma8` | **CRITICAL** | 1.958 | 0.515 | 1.418 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 8 | `funding_rate` | **CRITICAL** | 1.681 | 0.468 | 1.419 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 9 | `vwap_decay_std_24` | **CRITICAL** | 1.399 | 0.466 | 0.400 | 0.15 | REMOVE_OR_NORMALISE | +2.1pp |
| 10 | `gk_vol` | **DEGRADED** | 0.676 | 0.316 | 0.285 | 0.50 | MONITOR | +0.9pp |

### Critical Non-Price Indicators (highest priority for retraining)
- `se_mid`
- `low`
- `vwap_decay_std_24`
- `open`
- `close`
- `funding_rate_ma8`
- `funding_rate`
- `vwap_delta_12`


---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 0  | 0.0%  |
| SELL | 392 | 82.1% |
| HOLD | 8,344 | — |

### Signal Rejection Funnel

```
Generated (directional):       392  (100%)
Below Primary Conf. Thr:   -  345  (88%)
Blocked by Quality (<55):  -   31
Blocked by HMM:            -   19
Blocked by Confluence:     -   15
Blocked by Fake Breakout:  -   11
Blocked by Portfolio Guard:-    7
Blocked by Safe Mode:      -    3
Blocked by Drift:          -    7
Blocked by Cooldown:       -    3
─────────────────────────────────────
Estimated Executed:             0  (0.0%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.1120%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 73.3% |
| 12h | 90.6% |
| 18h | 98.0% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **3 bars** (3h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 345 | 185 | 160 | 53.8% | +383.63% |
| quality | 31 | 19 | 12 | 61.5% | +34.47% ⚠ |
| hmm | 19 | 10 | 9 | 57.6% | +21.13% ⚠ |
| confluence | 15 | 8 | 7 | 54.5% | +16.68% |
| fake_breakout | 11 | 6 | 5 | 60.0% | +12.23% ⚠ |
| portfolio | 7 | 4 | 3 | 57.6% | +7.78% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 2.0124 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=2.012) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 40.9% | -0.34 | ⚠ overconfident |
| 80-90% | 50.9% | -0.34 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 66 dev samples):** `platt`


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
| Feature Drift | 🟢 OK | 9 CRITICAL (3 active in model, 6 ✅ blacklisted/FIXED) / 14 DEGRADED / 73 total |
| Confidence Drift | 🔴 CRITICAL | T=2.012 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +40.91pp |

**Estimated precision loss from feature drift:** ~3.8pp


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
| Holdout Sharpe | 6.57 | ✓ |
| Holdout Max DD | 2.06% | ✓ |
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

**7 root causes identified.**  Combined top-5 impact score: **426/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🔴 **Anti-Selective Gate (precision < 50%)** | Gate Failure | 90/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | holdout_prec=40.9% — gate is selecting WRONG signals. Worse than rando… |
| 3 | 🔴 **Signal Precision Below Token Breakeven** | Gate Architecture | 85/100 | `scripts/retrain_model.py:_signal_prec_h >= token_breakeven check in primary_only_gate` 🔴 ACTIVE | signal_prec=40.9% < breakeven=52.8% (gap=11.9pp). Primary model has in… |
| 4 | 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** | Model Quality | 85/100 | `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` ✅ FIXED | cv_accuracy=60.8% vs majority_baseline=67.4%. binary_dual SPW inflatio… |
| 5 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 67/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=67.4% of labels. Meta model sees 60% zero-labels → calibration di… |
| 6 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 7 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=2.012>1.0. Model overestimates confidence.… |

### Fixes

**1. Meta Model Calibration Failure**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrate`
> Apply platt calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

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

**Base precision:** 40.9%  →  **Expected precision (all fixes):** 75.8%  (+34.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Fix local model quality (highest ROI) | +15.0pp | +3.0pp | +12.0pp | HIGH | LOW |
| 2 | Increase primary model AUPRC to clear signal_prec breakeven (gap=11.9pp) | +14.3pp | +5.0pp | +11.9pp | MEDIUM | MEDIUM |
| 3 | Remove / normalise 9 CRITICAL drifted features | +19.8pp | +1.5pp | +15.8pp | MEDIUM | LOW |
| 4 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 5 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 6 | Full retrain with all pipeline fixes applied | +16.1pp | +2.0pp | +13.4pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 66 |
| Rejected signals | 1596 |
| Selected precision | 40.9% |
| Rejected precision | 38.8% |
| Meta gate lift (precision) | +2.1% |
| Selected expectancy | +0.100% |
| Rejected expectancy | +0.063% |
| Selected Sharpe | +26.78 |
| Rejected Sharpe | +8.97 |

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

## Section 19 — Deep ETH/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +2.1% |
| Selected signals | 66 |
| Rejected signals | 1596 |
| Gate coverage | 4.0% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 52/100 |
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
| Precision Target | 57.8% |
| Actual Precision | 40.9% |
| Gap | -16.9% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 52/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
