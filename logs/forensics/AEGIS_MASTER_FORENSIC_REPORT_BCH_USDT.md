# AEGIS-1 Master Forensic Report

**Symbol:** BCH/USDT  |  **Generated:** 2026-06-07 16:35:52

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

> Measured precision is 33.7% — estimated 6.5pp below achievable ceiling of ~40.2%. Primary contributors: Barrier skew suppresses BUY labels (−2.0pp), HOLD over-representation in labels (−2.1pp)

#### Precision Waterfall

```
Measured holdout precision :  33.7%

  [✅ FIXED       ]  −2.0pp prec    Barrier skew suppresses BUY labels
  [✅ FIXED       ]  −8.0pp recall  BUY side completely disabled (min_fires deadlock)
  [✅ FIXED       ]  −2.1pp prec    HOLD over-representation in labels
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~40.2%  (+6.5pp precision)
Recall deficit (BUY side)  :  −10.8pp  (signals not firing)
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

✅ **3 issues already applied** in the codebase (expected gain: +7.4pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 2. pick_threshold_by_side(BUY) can qualify**

✅ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 7 BUY proposals / 154 total directional
   - PASS — primary fires BUY on some bars.

❌ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(5)=1 ≥ min_fires=35`
   - Value: 1 max fires vs 35 required
   - FAIL (FIXED) — 1 < 35. Deadlock: every quantile rejected before precision is checked. Fix: MAX_SIDE_COVERAGE→0.35 + adaptive effective_min_fires.

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

🔴 **Gate status: HURTING**  (lift: -20.1pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 33.7% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -20.1pp |
| OOF → Holdout gap | +50.9pp |
| thr_buy / thr_sell | 73.333 / 46.667 |
| Blocked signals | 0 (58 would-win / 50 would-lose) |

> ⚠ Gate is DESTROYING 20.1pp of precision. Blocked signals (53.8%) would have beaten gated (33.7%). Meta model is anti-selective. | OOF overfit warning: dev_prec (84.6%) exceeds holdout (33.7%) by 50.9pp.


---

## Section 15 — Executive Summary

**Symbol:** BCH/USDT  |  **Audit:** 2026-06-07 16:35  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=33.7%  Sharpe=10.39
**Expected after fixes:** Precision≈39.5%  (+5.8pp)

### Top 5 Problems

1. 🔴 **BUY Side Fully Disabled** — Score: 88/100
   > buy_n=0 in holdout. Model fires SELL-only. Halves available signal universe.

2. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 63/100
   > HOLD=63.7% of labels. Meta model sees 60% zero-labels → calibration distorted.

3. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.


### Top 5 Fixes

1. **BUY Side Fully Disabled**
   → Raise MAX_SIDE_COVERAGE→0.35, adaptive effective_min_fires. Already applied.

2. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

3. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 58.4% | ⚠ |
| Dev OOF Precision | 84.6% | ✓ |
| Holdout Precision | 33.7% | ✗ |
| Holdout Coverage | 33.0% | ✓ |
| 95% CI Precision | [29.5%, 38.2%] | — |
| OOF→Holdout Gap | -50.9% | ⚠ degradation |
| Holdout Fired | 442 trades | ✓ |
| SELL Win Rate | 79.3% (188 trades) | ✓ |
| BUY Win Rate | 0.0% (0 trades) | ✗ disabled |
| Sharpe (annualised) | 10.39 | ✓ |
| Max Drawdown | 153.94% | ✗ |
| Profit Factor | 1.71 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.0960% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | DISABLED | ✓ |
| Optimizer threshold match | NO | ⚠ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.0000 (z=-6.85) | ✓ significant |

### Class Distribution
- HOLD: **63.7%** — ⚠ severe imbalance
- SELL: **23.0%**
- BUY:  **13.3%** — ⚠ minority class

### Issues Detected
- **CRITICAL** — BUY side fully disabled (buy_n=0). Model is SELL-only. Directional asymmetry score: 80/100.
- **WARNING** — Class imbalance: 63.7% HOLD labels biases model toward neutrality.
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
| BUY  | 7  | 71.4%  |
| SELL | 147 | 79.6% |
| HOLD | 8,582 | — |

### Signal Rejection Funnel

```
Generated (directional):       154  (100%)
Blocked by Meta Gate:      -  108  (70%)
Blocked by Quality (<55):  -   12
Blocked by HMM:            -    7
Blocked by Confluence:     -    6
Blocked by Fake Breakout:  -    4
Blocked by Portfolio Guard:-    3
Blocked by Safe Mode:      -    1
Blocked by Drift:          -    3
Blocked by Cooldown:       -    1
─────────────────────────────────────
Estimated Executed:             9  (5.8%)
```

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.3684%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 65.8% |
| 12h | 88.4% |
| 18h | 96.3% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 108 | 58 | 50 | 53.8% | +147.79% |
| quality | 12 | 7 | 5 | 61.5% | +16.42% ⚠ |
| hmm | 7 | 4 | 3 | 57.6% | +9.58% ⚠ |
| confluence | 6 | 3 | 3 | 54.5% | +8.21% |
| fake_breakout | 4 | 2 | 2 | 60.0% | +5.47% ⚠ |
| portfolio | 3 | 1 | 2 | 57.6% | +4.11% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.1087 | <0.10 | ✗ overcalibrated |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.1517 | ~1.0 | ⚠ model overconfident |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 33.7% | -0.41 | ⚠ overconfident |
| 80-90% | 43.7% | -0.41 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 175 dev samples):** `isotonic`


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
**Paper trading:** 19 trades, 73.7% WR


---

## Section 9 — Drift Monitor Forensics

**Overall Drift Status:** 🟢 **OK**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🟢 OK | 0 CRITICAL / 0 DEGRADED / 0 total |
| Confidence Drift | 🟡 WARNING | T=1.152 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +50.86pp |

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
| ATR Multiplier | 1.5× | ✓ |
| Win Rate | 73.7% | ✓ |
| Avg Win / Avg Loss | 2.078% / 1.178% | — |
| R:R Ratio | 1.76 | ✓ favourable |
| Kelly Fraction | 58.8% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 10.39 | ✓ |
| Holdout Max DD | 153.94% | ✗ |
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

**3 root causes identified.**  Combined top-5 impact score: **213/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **BUY Side Fully Disabled** | Model Failure | 88/100 | `scripts/retrain_model.py:1363-1397` ✅ FIXED | buy_n=0 in holdout. Model fires SELL-only. Halves available signal uni… |
| 2 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 63/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=63.7% of labels. Meta model sees 60% zero-labels → calibration di… |
| 3 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |

### Fixes

**1. BUY Side Fully Disabled**
> 📍 `scripts/retrain_model.py:1363-1397` — `MAX_SIDE_COVERAGE=0.25, _SIDE_MIN_FIRES=35`
> Raise MAX_SIDE_COVERAGE→0.35, adaptive effective_min_fires. Already applied.

**2. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

**3. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 33.7%  →  **Expected precision (all fixes):** 39.5%  (+5.8pp)

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
| Selected signals | 442 |
| Rejected signals | 898 |
| Selected precision | 33.7% |
| Rejected precision | 23.2% |
| Meta gate lift (precision) | +10.5% |
| Selected expectancy | +0.096% |
| Rejected expectancy | +0.158% |
| Selected Sharpe | +16.36 |
| Rejected Sharpe | +13.53 |

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
| ACCUMULATION         | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 100.0 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep BCH/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +10.5% |
| Selected signals | 442 |
| Rejected signals | 898 |
| Gate coverage | 33.0% |
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
| Precision Target | 57.9% |
| Actual Precision | 33.7% |
| Gap | -24.2% |
| Coverage | 9.9% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 60/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
