# AEGIS-1 Master Forensic Report

**Symbol:** TRX/USDT  |  **Generated:** 2026-06-14 02:00:54

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

> Measured precision is 49.1% — estimated 20.4pp below achievable ceiling of ~69.5%. Primary contributors: CV accuracy near random — primary learned nothing (−13.7pp), HOLD over-representation in labels (−1.8pp), Brier score above target (−2.0pp)

#### Precision Waterfall

```
Measured holdout precision :  49.1%

  [✅ FIXED       ]  −13.7pp prec    CV accuracy near random — primary learned nothing
  [✅ FIXED       ]  −1.8pp prec    HOLD over-representation in labels
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.1pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=1.898 — overconfidence
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won
  [✅ FIXED       ]  −0.4pp prec    Anti-selective gate — precision below chance

Achievable precision       :  ~69.5%  (+20.4pp precision)
Recall deficit (BUY side)  :  −5.9pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 2.0pp | 0.0pp |
| Temperature T=1.898 — overconfidence | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 1.0pp | 0.0pp |
| Gate blocking signals that would have won | `scripts/retrain_model.py` | meta gate block (removed) | `meta_full LR gate removed — primary` | 0.4pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**2. Temperature T=1.898 — overconfidence**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

**3. Gate blocking signals that would have won**
> 📍 `scripts/retrain_model.py:meta gate block (removed)` — `meta_full LR gate removed — primary-only calibrated confidence gate`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). The entire 663-line LR OOF/edge engine/regime bloc…

### Q3 — What Is The Best Fix Right Now?

✅ **4 issues already applied** in the codebase (expected gain: +18.8pp once retrained).

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
   - Value: primary_confidence_threshold=0.660
   - PASS — val sweep selected threshold 0.660.

❌ **3. Holdout signal precision ≥ token breakeven**
   - `scripts/retrain_model.py — tradeable_final check`
   - Check: `signal_prec_h >= token_breakeven (58.2%)`
   - Value: signal_prec=49.1% vs breakeven=58.2% (332 fired, dir_prec≈96.4%)
   - FAIL — signal_prec=49.1% < breakeven=58.2% (gap=9.2pp). Primary model needs stronger directional skill.

✅ **4. tradeable_final (all criteria: signal_prec, dir_prec ≥55%, cov ≥5%)**
   - `scripts/retrain_model.py — tradeable_final condition`
   - Check: `fired_n >= MIN_FIRES AND dir_prec >= 55% AND coverage_dir >= 5% AND signal_prec >= breakeven`
   - Value: tradeable=True, coverage=20.0%
   - PASS — token ENABLED. Signal gate cleared all criteria.

#### Primary Confidence Gate Audit

✅ **Gate status: HELPING**  (lift: +10.2pp)  ❌ signal_prec < breakeven

| Metric | Value |
|--------|-------|
| Selected signal precision | 49.1% |
| Rejected signal precision | 38.9% |
| Gate lift (precision) | +10.2pp |
| Primary conf. threshold | 0.660 |
| Token breakeven | 58.2% |
| Selected signals | 332 |
| Rejected signals | 1324 |

> Primary gate adds 10.2pp of precision. Selected signals (49.1%) beat rejected (38.9%). Threshold=0.660. | DISABLED: selected signal_prec=49.1% < breakeven=58.2%.


---

## Section 15 — Executive Summary

**Symbol:** TRX/USDT  |  **Audit:** 2026-06-14 02:00  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=49.1%  Sharpe=-5.37
**Expected after fixes:** Precision≈68.9%  (+19.8pp)

### Top 5 Problems

1. 🔴 **Meta Model Calibration Failure** — Score: 99/100
   > ECE=0.2496 (target <0.10). Confidence does not reflect true win probability.

2. 🔴 **Anti-Selective Gate (precision < 50%)** — Score: 90/100
   > holdout_prec=49.1% — gate is selecting WRONG signals. Worse than random for directional trading.

3. 🔴 **Signal Precision Below Token Breakeven** — Score: 85/100
   > signal_prec=49.1% < breakeven=58.2% (gap=9.2pp). Primary model has insufficient directional skill to clear the fee breakeven after HOLD-timeout dilution.

4. 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** — Score: 85/100
   > cv_accuracy=54.6% vs majority_baseline=61.7%. binary_dual SPW inflation compresses hold_residual → argmax always BUY/SELL. Check sidecar for bayes_prior_correction key — if absent, Bayes fix not applied.

5. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.


### Top 5 Fixes

1. **Meta Model Calibration Failure**
   → Apply isotonic calibration. Use C_excluded meta (LR trains only on directional bars, class_weight='balanced').

2. **Anti-Selective Gate (precision < 50%)**
   → Fix primary OOF quality (soft confluence features + local model hyperparams). Directional precision veto (_fired_dir_prec<0.50) disables gate and outputs tradeable=False.

3. **Signal Precision Below Token Breakeven**
   → Increase primary model AUPRC (Optuna aucpr objective). Target dir_prec >= 65% so fired signals clear breakeven even at 30% HOLD-timeout rate.

4. **Primary Model CV Near Random (NEAR_RANDOM)**
   → Apply Bayes prior correction to OOF and holdout raw_probs: corrects SPW-inflated probabilities back to true class posterior scale so hold_residual is meaningful. bayes_prior_correction key must appear in sidecar after retrain.

5. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 54.6% | ⚠ |
| Dev OOF Precision | 20.9% | ✗ |
| Holdout Precision | 49.1% | ✗ |
| Holdout Coverage | 20.1% | ✓ |
| 95% CI Precision | [43.8%, 54.4%] | — |
| OOF→Holdout Gap | +28.1% | ✓ holdout beat OOF |
| Holdout Fired | 332 trades | ✓ |
| SELL Win Rate | 100.0% (45 trades) | ✓ |
| BUY Win Rate | 95.2% (124 trades) | ✓ |
| Sharpe (annualised) | -5.37 | ✗ |
| Max Drawdown | 6.20% | ✓ |
| Profit Factor | 0.77 | ✗ |
| Kelly Fraction | 0.0% | — |
| Expectancy/Trade | -0.0126% | ✗ |
| Gate mode | PRIMARY-ONLY (calibrated) | ✓ |
| Primary conf. threshold | 0.660 | ✓ |
| Primary calibrator | present (primary_only) | ✓ |
| Signal prec vs breakeven | -9.2pp (be=58.2%) | ✗ below breakeven |
| Statistical Sig. | p=0.7419 (z=-0.33) | ⚠ insufficient data |

### Class Distribution
- HOLD: **61.7%** — ⚠ severe imbalance
- SELL: **22.0%**
- BUY:  **16.3%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 61.7% HOLD labels biases model toward neutrality.
- **CRITICAL** — Anti-selective gate: holdout precision=49.1% < 50%. Gate is selecting the worst signals. Rebuild with fixed local model.
- **CRITICAL** — Precision below random chance (49.1%). Model is directionally anti-predictive on holdout data.
- **CRITICAL** — CV accuracy=54.6% ≈ random baseline=61.7%. Primary model learned nothing from features.
- **CRITICAL** — Signal precision -9.2pp below breakeven (58.2%). Token correctly DISABLED. Increase primary model directional precision to clear breakeven gate.


---

## Section 2 — Feature Forensics

⚠ feature_health.json not found



---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 73  | 91.8%  |
| SELL | 141 | 90.8% |
| HOLD | 8,522 | — |

### Signal Rejection Funnel

```
Generated (directional):       214  (100%)
Below Primary Conf. Thr:   -   85  (40%)
Blocked by Quality (<55):  -   17
Blocked by HMM:            -   10
Blocked by Confluence:     -    8
Blocked by Fake Breakout:  -    6
Blocked by Portfolio Guard:-    4
Blocked by Safe Mode:      -    2
Blocked by Drift:          -    4
Blocked by Cooldown:       -    2
─────────────────────────────────────
Estimated Executed:            76  (35.5%)
```

**BUY side:** ✓ ENABLED  |  **SELL side:** ✓ ENABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **0.5527%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 68.0% |
| 12h | 89.6% |
| 18h | 97.2% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **4 bars** (4h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 85 | 45 | 40 | 53.8% | +46.98% |
| quality | 17 | 10 | 7 | 61.5% | +9.40% ⚠ |
| hmm | 10 | 5 | 5 | 57.6% | +5.53% ⚠ |
| confluence | 8 | 4 | 4 | 54.5% | +4.42% |
| fake_breakout | 6 | 3 | 3 | 60.0% | +3.32% ⚠ |
| portfolio | 4 | 2 | 2 | 57.6% | +2.21% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.2496 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.2496 | <0.10 | ✗ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.8984 | ~1.0 | ⚠ model overconfident |
| Calibration Type | temperature (T=1.898) | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 49.1% | -0.26 | ⚠ overconfident |
| 80-90% | 59.1% | -0.26 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** YES — model claims higher confidence than earned
**Recommended calibrator (for 332 dev samples):** `isotonic`


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
| Confidence Drift | 🔴 CRITICAL | T=1.898 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +49.10pp |

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
| Holdout Sharpe | -5.37 | ✗ |
| Holdout Max DD | 6.20% | ✓ |
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

**7 root causes identified.**  Combined top-5 impact score: **421/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Meta Model Calibration Failure** | Calibration | 99/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | ECE=0.2496 (target <0.10). Confidence does not reflect true win probab… |
| 2 | 🔴 **Anti-Selective Gate (precision < 50%)** | Gate Failure | 90/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | holdout_prec=49.1% — gate is selecting WRONG signals. Worse than rando… |
| 3 | 🔴 **Signal Precision Below Token Breakeven** | Gate Architecture | 85/100 | `scripts/retrain_model.py:_signal_prec_h >= token_breakeven check in primary_only_gate` 🔴 ACTIVE | signal_prec=49.1% < breakeven=58.2% (gap=9.2pp). Primary model has ins… |
| 4 | 🔴 **Primary Model CV Near Random (NEAR_RANDOM)** | Model Quality | 85/100 | `scripts/retrain_model.py:binary_dual OOF → Bayes prior correction → 3-class accuracy` ✅ FIXED | cv_accuracy=54.6% vs majority_baseline=61.7%. binary_dual SPW inflatio… |
| 5 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |
| 6 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 61/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=61.7% of labels. Meta model sees 60% zero-labels → calibration di… |
| 7 | 🟡 **Confidence Inflation** | Calibration | 55/100 | `scripts/retrain_model.py:meta gate block (removed)` ✅ FIXED | T=1.898>1.0. Model overestimates confidence.… |

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

**5. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 49.1%  →  **Expected precision (all fixes):** 68.9%  (+19.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Fix local model quality (highest ROI) | +15.0pp | +3.0pp | +12.0pp | HIGH | LOW |
| 2 | Increase primary model AUPRC to clear signal_prec breakeven (gap=9.2pp) | +11.0pp | +5.0pp | +9.2pp | MEDIUM | MEDIUM |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Reduce HOLD% in training labels | +2.0pp | +4.0pp | +3.0pp | MEDIUM | LOW |
| 5 | Full retrain with all pipeline fixes applied | +9.2pp | +2.0pp | +7.6pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 332 |
| Rejected signals | 1324 |
| Selected precision | 49.1% |
| Rejected precision | 38.9% |
| Meta gate lift (precision) | +10.2% |
| Selected expectancy | -0.013% |
| Rejected expectancy | -0.071% |
| Selected Sharpe | -9.75 |
| Rejected Sharpe | -36.19 |

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

## Section 19 — Deep TRX/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +10.2% |
| Selected signals | 332 |
| Rejected signals | 1324 |
| Gate coverage | 20.0% |
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
| Precision Target | 63.3% |
| Actual Precision | 49.1% |
| Gap | -14.2% |
| Coverage | 0.0% |
| Gating Strategy | GLOBAL_THRESHOLD |
| Gate Trust Score | 60/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
