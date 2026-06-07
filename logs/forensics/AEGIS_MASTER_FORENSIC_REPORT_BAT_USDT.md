# AEGIS-1 Master Forensic Report

**Symbol:** BAT/USDT  |  **Generated:** 2026-06-07 16:35:51

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

> Measured precision is 53.8% — estimated 3.3pp below achievable ceiling of ~57.1%. Primary contributors: Brier score above target (−2.0pp), Barrier skew suppresses BUY labels (−0.9pp)

#### Precision Waterfall

```
Measured holdout precision :  53.8%

  [✅ FIXED       ]  −8.0pp recall  BUY side completely disabled (min_fires deadlock)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −0.9pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~57.1%  (+3.3pp precision)
Recall deficit (BUY side)  :  −8.9pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 2.0pp | 0.0pp |
| Gate blocking signals that would have won | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 0.4pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

**2. Gate blocking signals that would have won**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

### Q3 — What Is The Best Fix Right Now?

✅ **2 issues already applied** in the codebase (expected gain: +3.6pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 2. pick_threshold_by_side(BUY) can qualify**

✅ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 88 BUY proposals / 440 total directional
   - PASS — primary fires BUY on some bars.

❌ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(70)=24 ≥ min_fires=35`
   - Value: 24 max fires vs 35 required
   - FAIL (FIXED) — 24 < 35. Deadlock: every quantile rejected before precision is checked. Fix: MAX_SIDE_COVERAGE→0.35 + adaptive effective_min_fires.

❌ **3. hit_buy=True (OOF precision clears target)**
   - `scripts/retrain_model.py:1996-2004`
   - Check: `pick_threshold_by_side(side=2).hit_target → stored as tradeable_buy`
   - Value: tradeable_buy in sidecar = False
   - FAIL — hit_buy=False because Gate 2 deadlock blocked threshold qualification.

❌ **4. buy_fire mask fires BUY holdout signals**
   - `scripts/retrain_model.py:2169-2174`
   - Check: `buy_fire = (meta_prob_h ≥ max(thr_buy, rank_thr)) & (prop_h==2)`
   - Value: buy_h_n = 0 holdout BUY signals fired
   - FAIL — 0 BUY holdout trades. Even when hit_buy=True, SELL-biased rank gate may crowd out BUY signals. File: retrain_model.py:2097-2108.

❌ **5. tradeable_buy_holdout = True**
   - `scripts/retrain_model.py:2288-2292`
   - Check: `hit_buy AND buy_h_n > 0 AND buy_h_prec ≥ 0.50`
   - Value: buy_h_n=0, buy_win_rate=0.0%
   - FAIL — `buy_h_n > 0` is a hard requirement. If Gate 4 fails (zero holdout BUY fires), this is permanently False.

#### Meta Gate Audit

🟡 **Gate status: NEUTRAL**  (lift: -0.0pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 53.8% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -0.0pp |
| OOF → Holdout gap | +46.2pp |
| thr_buy / thr_sell | 70.171 / 70.130 |
| Blocked signals | 0 (218 would-win / 188 would-lose) |

> Gate provides minimal discrimination (53.8% gated ≈ 53.8% blocked). | OOF overfit warning: dev_prec (100.0%) exceeds holdout (53.8%) by 46.2pp.


---

## Section 15 — Executive Summary

**Symbol:** BAT/USDT  |  **Audit:** 2026-06-07 16:35  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=53.8%  Sharpe=26.80
**Expected after fixes:** Precision≈59.6%  (+5.8pp)

### Top 5 Problems

1. 🔴 **BUY Side Fully Disabled** — Score: 88/100
   > buy_n=0 in holdout. Model fires SELL-only. Halves available signal universe.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=52.6%. HMM assigning most bars to one state.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 58/100
   > HOLD=58.0% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **BUY Side Fully Disabled**
   → Raise MAX_SIDE_COVERAGE→0.35, adaptive effective_min_fires. Already applied.

2. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

3. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 55.2% | ⚠ |
| Dev OOF Precision | 100.0% | ✓ |
| Holdout Precision | 53.8% | ✗ |
| Holdout Coverage | 14.8% | ✓ |
| 95% CI Precision | [46.8%, 60.6%] | — |
| OOF→Holdout Gap | -46.2% | ⚠ degradation |
| Holdout Fired | 197 trades | ✓ |
| SELL Win Rate | 92.2% (115 trades) | ✓ |
| BUY Win Rate | 0.0% (0 trades) | ✗ disabled |
| Sharpe (annualised) | 26.80 | ✓ |
| Max Drawdown | 13.15% | ✗ |
| Profit Factor | 6.63 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.7520% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | EDGE_CONFLUENCE_VETO | ✓ |
| Optimizer threshold match | NO | ⚠ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.2852 (z=1.07) | ⚠ insufficient data |

### Class Distribution
- HOLD: **58.0%** — ⚠ severe imbalance
- SELL: **25.1%**
- BUY:  **16.9%** — OK

### Issues Detected
- **CRITICAL** — BUY side fully disabled (buy_n=0). Model is SELL-only. Directional asymmetry score: 80/100.
- **WARNING** — Class imbalance: 58.0% HOLD labels biases model toward neutrality.
- **WARNING** — Model meta thresholds do not match optimizer-selected gate thresholds. Investigate whether the optimizer output is stale or not fully applied.


---

## Section 2 — Feature Forensics

⚠ feature_health.json not found



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
Blocked by Meta Gate:      -  406  (92%)
Blocked by Quality (<55):  -   35
Blocked by HMM:            -   22
Blocked by Confluence:     -   17
Blocked by Fake Breakout:  -   13
Blocked by Portfolio Guard:-    8
Blocked by Safe Mode:      -    4
Blocked by Drift:          -    8
Blocked by Cooldown:       -    4
─────────────────────────────────────
Estimated Executed:             0  (0.0%)
```

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


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
| meta_gate | 406 | 218 | 188 | 53.8% | +279.36% |
| quality | 35 | 21 | 14 | 61.5% | +24.08% ⚠ |
| hmm | 22 | 12 | 10 | 57.6% | +15.14% ⚠ |
| confluence | 17 | 9 | 8 | 54.5% | +11.70% |
| fake_breakout | 13 | 7 | 6 | 60.0% | +8.95% ⚠ |
| portfolio | 8 | 4 | 4 | 57.6% | +5.50% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.0797 | <0.10 | ✓ |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.0128 | ~1.0 | ✓ |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 53.8% | -0.21 | ⚠ overconfident |
| 80-90% | 63.8% | -0.21 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 53 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

**States:** 6  |  **Global precision:** 66.0%  |  **Max state concentration:** 52.6% ⚠ REGIME COLLAPSE

### Per-Regime Performance

| Regime | Trades | Precision | Sell Prec | Expectancy | P.Factor | Modifier | Rec. |
|--------|--------|-----------|-----------|------------|----------|----------|------|
| TRENDING_BULL | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| TRENDING_BEAR | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| ACCUMULATION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| DISTRIBUTION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| COMPRESSION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| VOLATILE_EXPANSION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| CHOPPY | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |


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
**Paper trading:** 19 trades, 73.7% WR


---

## Section 9 — Drift Monitor Forensics

**Overall Drift Status:** 🟢 **OK**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🟢 OK | 0 CRITICAL / 0 DEGRADED / 0 total |
| Confidence Drift | 🟢 OK | T=1.013 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +46.19pp |

**Estimated precision loss from feature drift:** ~0.0pp


---

## Section 10 — Portfolio Forensics

**Open positions:** 0/6  |  **Effective leverage:** 0.38×  |  **HHI (concentration):** 0.064

| Symbol | Capital Allocation |
|--------|-------------------|
| VET/USDT | 10.5% |
| ATOM/USDT | 10.5% |
| ADA/USDT | 5.3% |
| SEI/USDT | 5.3% |
| SUI/USDT | 5.3% |
| UNI/USDT | 5.3% |
| FIL/USDT | 5.3% |
| KAVA/USDT | 5.3% |
| SAND/USDT | 5.3% |
| XLM/USDT | 5.3% |
| ARB/USDT | 5.3% |
| STX/USDT | 5.2% |
| DOT/USDT | 5.2% |
| ENA/USDT | 5.2% |
| NEAR/USDT | 5.2% |
| ZEC/USDT | 5.2% |
| THETA/USDT | 5.2% |

**Hidden leverage:** ✓ NO  |  **Over-concentration:** ✓ NO


---

## Section 11 — Risk Engine Forensics

| Metric | Value | Assessment |
|--------|-------|------------|
| ATR Multiplier | 2.0× | ✓ |
| Win Rate | 73.7% | ✓ |
| Avg Win / Avg Loss | 2.078% / 1.178% | — |
| R:R Ratio | 1.76 | ✓ favourable |
| Kelly Fraction | 58.8% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 26.80 | ✓ |
| Holdout Max DD | 13.15% | ✗ |
| Stop Assessment | TARGETS TOO CLOSE | — |


---

## Section 12 — Live Execution Forensics

**Closed:** 19  |  **Open:** 0  |  **Avg hold:** 0.6h  |  **Avg PnL:** +1.221%

**Confidence discriminates wins from losses:** ✗ NO (WIN conf=0.645 vs LOSS conf=0.636)

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
| TP1_HIT | 14 |
| SL_HIT | 5 |


---

## Section 13 — Root Cause Engine

**3 root causes identified.**  Combined top-5 impact score: **208/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **BUY Side Fully Disabled** | Model Failure | 88/100 | `scripts/retrain_model.py:1363-1397` ✅ FIXED | buy_n=0 in holdout. Model fires SELL-only. Halves available signal uni… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=52.6%. HMM assigning most bars to one state.… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 58/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=58.0% of labels. Meta model sees 60% zero-labels → calibration di… |

### Fixes

**1. BUY Side Fully Disabled**
> 📍 `scripts/retrain_model.py:1363-1397` — `MAX_SIDE_COVERAGE=0.25, _SIDE_MIN_FIRES=35`
> Raise MAX_SIDE_COVERAGE→0.35, adaptive effective_min_fires. Already applied.

**2. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

**3. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 53.8%  →  **Expected precision (all fixes):** 59.6%  (+5.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Enable BUY side (fix directional asymmetry) | +0.0pp | +8.0pp | +5.0pp | HIGH | MEDIUM |
| 2 | Remove / normalise top-10 drifted features | +3.0pp | +1.5pp | +2.4pp | MEDIUM | LOW |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Redesign triple-barrier labels (reduce HOLD%) | +1.5pp | +4.0pp | +3.0pp | MEDIUM | MEDIUM |
| 5 | Extend lookahead for low-ER tokens | +1.0pp | +2.0pp | +1.5pp | MEDIUM | LOW |
| 6 | Regime-specific meta thresholds | +1.5pp | +1.0pp | +2.5pp | MEDIUM | LOW |
| 7 | Retrain meta model on 60-symbol fleet data | +2.0pp | +0.5pp | +2.5pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 197 |
| Rejected signals | 1137 |
| Selected precision | 53.8% |
| Rejected precision | 25.6% |
| Meta gate lift (precision) | +28.2% |
| Selected expectancy | +0.752% |
| Rejected expectancy | +0.168% |
| Selected Sharpe | +63.17 |
| Rejected Sharpe | +9.29 |

**Verdict:** ✅ HELPFUL — Gate selects higher-precision signals than rejected


---

## Section 17 — Hold Pollution Audit

| Strategy | Hold Weight | Brier | PF | Sharpe | Prec | Lift | Notes |
|----------|-------------|-------|----|----|------|------|-------|
| A_current 🔴 CURRENT | 1.00 | 0.330 | 1.20 | 0.45 | 59.0% | -0.02 | Baseline — no mitigation |
| B_reduced  | 0.15 | 0.310 | 1.35 | 0.58 | 62.0% | +0.03 | Partial HOLD downweight — recommended |
| C_excluded ✅ BEST | 0.00 | 0.300 | 1.40 | 0.62 | 64.0% | +0.05 | Total HOLD exclusion — most aggressive |

**Current Strategy Score:** 0.190
**Best Strategy Score:** 0.820
**Potential Improvement:** +0.630

**Recommendation:** Switch from A_current to C_excluded (+0.630 score)


---

## Section 18 — Regime Threshold Audit

**Regime Summary:** 7 enabled, 0 disabled (0.0% disability rate)

| Regime | BUY OK | SELL OK | BUY Thr | SELL Thr | Status | Est. Prec | Est. PF |
|--------|--------|---------|---------|----------|--------|-----------|---------|
| ACCUMULATION         | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 70.2 | 70.1 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep BAT/USDT vs BTC/ETH Comparison

| Metric | SOL | BTC | ETH | SOL vs BTC |
|--------|-----|-----|-----|-----------|
| Meta Threshold | 79.5 | 82.4 | 82.9 | -2.9 |
| Tradeable BUY | False | True | False | — |
| Tradeable SELL | False | True | False | — |
| Holdout Precision | 37.4% | 66.0% | 45.0% | -28.6% |
| Win Rate (PnL) | 48.0% | 72.0% | 52.0% | — |
| Regime Disability | 50% | 20% | 60% | +30% |
| Calibration T | 0.888 | 0.920 | 0.950 | — |

### Top Discriminators (SOL vs BTC)

1. **Regime disability** — gap: +30.00
2. **Meta threshold** — gap: -2.90
3. **Holdout precision** — gap: -0.29

**Root Cause Hypothesis:** SOL fails on Regime disability (gap: 30.00)


---

## Section 20 — AEGIS Gate Lift Engine

**Gate Lift = Selected Precision − Rejected Precision**

| Metric | Value |
|--------|-------|
| Gate Lift (pp) | +28.2% |
| Selected signals | 197 |
| Rejected signals | 1137 |
| Gate coverage | 14.8% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 78/100 |
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
| Precision Target | 56.6% |
| Actual Precision | 53.8% |
| Gap | -2.8% |
| Coverage | 2.6% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 78/100 |
| Verdict | ⚠️ BELOW TARGET |


---
