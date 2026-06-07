# AEGIS-1 Master Forensic Report

**Symbol:** DOGE/USDT  |  **Generated:** 2026-06-07 16:35:56

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

> Measured precision is 40.6% — estimated 15.2pp below achievable ceiling of ~55.8%. Primary contributors: Absolute price features in model (7 critical) (−8.0pp), Other drifted features (3 CRITICAL/DEGRADED) (−3.7pp), Brier score above target (−2.0pp)

#### Precision Waterfall

```
Measured holdout precision :  40.6%

  [✅ FIXED       ]  −8.0pp prec    Absolute price features in model (7 critical)
  [✅ FIXED       ]  −3.7pp prec    Other drifted features (3 CRITICAL/DEGRADED)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.1pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~55.8%  (+15.2pp precision)
Recall deficit (BUY side)  :  −1.1pp  (signals not firing)
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

✅ **3 issues already applied** in the codebase (expected gain: +13.1pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 2. pick_threshold_by_side(BUY) can qualify**

✅ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 39 BUY proposals / 107 total directional
   - PASS — primary fires BUY on some bars.

❌ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(31)=10 ≥ min_fires=35`
   - Value: 10 max fires vs 35 required
   - FAIL (FIXED) — 10 < 35. Deadlock: every quantile rejected before precision is checked. Fix: MAX_SIDE_COVERAGE→0.35 + adaptive effective_min_fires.

❌ **3. hit_buy=True (OOF precision clears target)**
   - `scripts/retrain_model.py:1996-2004`
   - Check: `pick_threshold_by_side(side=2).hit_target → stored as tradeable_buy`
   - Value: tradeable_buy in sidecar = False
   - FAIL — hit_buy=False because Gate 2 deadlock blocked threshold qualification.

✅ **4. buy_fire mask fires BUY holdout signals**
   - `scripts/retrain_model.py:2169-2174`
   - Check: `buy_fire = (meta_prob_h ≥ max(thr_buy, rank_thr)) & (prop_h==2)`
   - Value: buy_h_n = 24 holdout BUY signals fired
   - PASS — 24 BUY holdout trades.

✅ **5. tradeable_buy_holdout = True**
   - `scripts/retrain_model.py:2288-2292`
   - Check: `hit_buy AND buy_h_n > 0 AND buy_h_prec ≥ 0.50`
   - Value: buy_h_n=24, buy_win_rate=87.5%
   - PASS — 24 trades, 87.5% WR.

#### Meta Gate Audit

🔴 **Gate status: HURTING**  (lift: -13.2pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 40.6% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -13.2pp |
| OOF → Holdout gap | +38.6pp |
| thr_buy / thr_sell | 46.429 / 80.042 |
| Blocked signals | 0 (55 would-win / 48 would-lose) |

> ⚠ Gate is DESTROYING 13.2pp of precision. Blocked signals (53.8%) would have beaten gated (40.6%). Meta model is anti-selective. | OOF overfit warning: dev_prec (79.2%) exceeds holdout (40.6%) by 38.6pp.


---

## Section 15 — Executive Summary

**Symbol:** DOGE/USDT  |  **Audit:** 2026-06-07 16:35  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=40.6%  Sharpe=13.51
**Expected after fixes:** Precision≈50.8%  (+10.2pp)

### Top 5 Problems

1. 🔴 **Critical Feature Drift** — Score: 100/100
   > 26 features CRITICAL. Top: vwap_decay_mean_24 PSI=20.62.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=50.6%. HMM assigning most bars to one state.

3. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 58/100
   > HOLD=58.7% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **Critical Feature Drift**
   → FEATURE_BLACKLIST (25 features) + OBV/PVT z-score + decay mean normalization. Already applied.

2. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

3. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 65.3% | ✓ |
| Dev OOF Precision | 40.6% | ✗ |
| Holdout Precision | 40.6% | ✗ |
| Holdout Coverage | 9.9% | ✓ |
| 95% CI Precision | [32.6%, 49.1%] | — |
| OOF→Holdout Gap | +0.0% | ⚠ degradation |
| Holdout Fired | 133 trades | ✓ |
| SELL Win Rate | 86.8% (38 trades) | ✓ |
| BUY Win Rate | 87.5% (24 trades) | ✓ |
| Sharpe (annualised) | 13.51 | ✓ |
| Max Drawdown | 114.67% | ✗ |
| Profit Factor | 3.42 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.2784% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | EDGE_SR_VETO | ✓ |
| Optimizer threshold match | NO | ⚠ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.0302 (z=-2.17) | ✓ significant |

### Class Distribution
- HOLD: **58.7%** — ⚠ severe imbalance
- SELL: **25.0%**
- BUY:  **16.2%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 58.7% HOLD labels biases model toward neutrality.
- **WARNING** — Model meta thresholds do not match optimizer-selected gate thresholds. Investigate whether the optimizer output is stale or not fully applied.


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
Blocked by Meta Gate:      -  103  (96%)
Blocked by Quality (<55):  -    8
Blocked by HMM:            -    5
Blocked by Confluence:     -    4
Blocked by Fake Breakout:  -    3
Blocked by Portfolio Guard:-    2
Blocked by Safe Mode:      -    1
Blocked by Drift:          -    2
Blocked by Cooldown:       -    1
─────────────────────────────────────
Estimated Executed:             0  (0.0%)
```

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


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
| meta_gate | 103 | 55 | 48 | 53.8% | +171.64% |
| quality | 8 | 4 | 4 | 61.5% | +13.33% ⚠ |
| hmm | 5 | 2 | 3 | 57.6% | +8.33% ⚠ |
| confluence | 4 | 2 | 2 | 54.5% | +6.67% |
| fake_breakout | 3 | 1 | 2 | 60.0% | +5.00% ⚠ |
| portfolio | 2 | 1 | 1 | 57.6% | +3.33% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.0636 | <0.10 | ✓ |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.1936 | ~1.0 | ⚠ model overconfident |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 40.6% | -0.34 | ⚠ overconfident |
| 80-90% | 50.6% | -0.34 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 24 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

**States:** 5  |  **Global precision:** 66.0%  |  **Max state concentration:** 50.6% ⚠ REGIME COLLAPSE

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

**Overall Drift Status:** 🔴 **CRITICAL**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🔴 CRITICAL | 26 CRITICAL / 12 DEGRADED / 90 total |
| Confidence Drift | 🟡 WARNING | T=1.194 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +38.57pp |

**Estimated precision loss from feature drift:** ~5.1pp


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
| Holdout Sharpe | 13.51 | ✓ |
| Holdout Max DD | 114.67% | ✗ |
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

**3 root causes identified.**  Combined top-5 impact score: **220/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🔴 **Critical Feature Drift** | Feature Drift | 100/100 | `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` ✅ FIXED | 26 features CRITICAL. Top: vwap_decay_mean_24 PSI=20.62.… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=50.6%. HMM assigning most bars to one state.… |
| 3 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 58/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=58.7% of labels. Meta model sees 60% zero-labels → calibration di… |

### Fixes

**1. Critical Feature Drift**
> 📍 `scripts/retrain_model.py:165-168 (FEATURE_ADDONS) + 1838-1840 (feature_cols)` — `ema_9/21/50/100/200, vwap, avwap_*, ichimoku_senko`
> FEATURE_BLACKLIST (25 features) + OBV/PVT z-score + decay mean normalization. Already applied.

**2. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.

**3. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 40.6%  →  **Expected precision (all fixes):** 50.8%  (+10.2pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Enable BUY side (fix directional asymmetry) | +0.0pp | +8.0pp | +5.0pp | HIGH | MEDIUM |
| 2 | Remove / normalise top-10 drifted features | +11.9pp | +1.5pp | +9.5pp | MEDIUM | LOW |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Redesign triple-barrier labels (reduce HOLD%) | +1.5pp | +4.0pp | +3.0pp | MEDIUM | MEDIUM |
| 5 | Extend lookahead for low-ER tokens | +1.0pp | +2.0pp | +1.5pp | MEDIUM | LOW |
| 6 | Regime-specific meta thresholds | +1.5pp | +1.0pp | +2.5pp | MEDIUM | LOW |
| 7 | Retrain meta model on 60-symbol fleet data | +2.0pp | +0.5pp | +2.5pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 133 |
| Rejected signals | 1205 |
| Selected precision | 40.6% |
| Rejected precision | 27.7% |
| Meta gate lift (precision) | +12.9% |
| Selected expectancy | +0.278% |
| Rejected expectancy | +0.164% |
| Selected Sharpe | +38.78 |
| Rejected Sharpe | +14.38 |

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
| ACCUMULATION         | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 46.4 | 80.0 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep DOGE/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +12.9% |
| Selected signals | 133 |
| Rejected signals | 1205 |
| Gate coverage | 9.9% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 62/100 |
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
| Actual Precision | 40.6% |
| Gap | -16.8% |
| Coverage | 1.2% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 62/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
