# AEGIS-1 Master Forensic Report

**Symbol:** ETH/USDT  |  **Generated:** 2026-06-07 00:47:45

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

> Measured precision is 27.4% — estimated 14.5pp below achievable ceiling of ~41.9%. Primary contributors: Barrier skew suppresses BUY labels (−6.0pp), Other drifted features (9 CRITICAL/DEGRADED) (−4.0pp), Absolute price features in model (1 critical) (−2.1pp)

#### Precision Waterfall

```
Measured holdout precision :  27.4%

  [✅ FIXED       ]  −6.0pp prec    Barrier skew suppresses BUY labels
  [✅ FIXED       ]  −4.0pp prec    Other drifted features (9 CRITICAL/DEGRADED)
  [✅ FIXED       ]  −2.1pp prec    Absolute price features in model (1 critical)
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~41.9%  (+14.5pp precision)
Recall deficit (BUY side)  :  −6.0pp  (signals not firing)
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

✅ **3 issues already applied** in the codebase (expected gain: +13.9pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 1. Primary model generates BUY labels**

❌ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 0 BUY proposals / 392 total directional
   - FAIL — zero BUY labels. vol_threshold or barrier too restrictive.

❌ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(0)=0 ≥ min_fires=35`
   - Value: 0 max fires vs 35 required
   - FAIL (FIXED) — 0 < 35. Deadlock: every quantile rejected before precision is checked. Fix: MAX_SIDE_COVERAGE→0.35 + adaptive effective_min_fires.

❌ **3. hit_buy=True (OOF precision clears target)**
   - `scripts/retrain_model.py:1996-2004`
   - Check: `pick_threshold_by_side(side=2).hit_target → stored as tradeable_buy`
   - Value: tradeable_buy in sidecar = False
   - FAIL — hit_buy=False because Gate 2 deadlock blocked threshold qualification.

✅ **4. buy_fire mask fires BUY holdout signals**
   - `scripts/retrain_model.py:2169-2174`
   - Check: `buy_fire = (meta_prob_h ≥ max(thr_buy, rank_thr)) & (prop_h==2)`
   - Value: buy_h_n = 83 holdout BUY signals fired
   - PASS — 83 BUY holdout trades.

✅ **5. tradeable_buy_holdout = True**
   - `scripts/retrain_model.py:2288-2292`
   - Check: `hit_buy AND buy_h_n > 0 AND buy_h_prec ≥ 0.50`
   - Value: buy_h_n=83, buy_win_rate=63.9%
   - PASS — 83 trades, 63.9% WR.

#### Meta Gate Audit

🔴 **Gate status: HURTING**  (lift: -26.5pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 27.4% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -26.5pp |
| OOF → Holdout gap | +51.3pp |
| thr_buy / thr_sell | 46.667 / 46.667 |
| Blocked signals | 0 (170 would-win / 147 would-lose) |

> ⚠ Gate is DESTROYING 26.5pp of precision. Blocked signals (53.8%) would have beaten gated (27.4%). Meta model is anti-selective. | OOF overfit warning: dev_prec (78.6%) exceeds holdout (27.4%) by 51.3pp.


---

## Section 15 — Executive Summary

**Symbol:** ETH/USDT  |  **Audit:** 2026-06-07 00:47  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=27.4%  Sharpe=7.01
**Expected after fixes:** Precision≈41.5%  (+14.2pp)

### Top 5 Problems


### Top 5 Fixes


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 0.0% | ⚠ |
| Dev OOF Precision | 78.6% | ✓ |
| Holdout Precision | 27.4% | ✗ |
| Holdout Coverage | 27.0% | ✓ |
| 95% CI Precision | [23.0%, 32.2%] | — |
| OOF→Holdout Gap | -51.3% | ⚠ degradation |
| Holdout Fired | 362 trades | ✓ |
| SELL Win Rate | 86.8% (53 trades) | ✓ |
| BUY Win Rate | 63.9% (83 trades) | ✓ |
| Sharpe (annualised) | 7.01 | ✓ |
| Max Drawdown | 48.16% | ✗ |
| Profit Factor | 1.48 | ✗ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.0657% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | DISABLED | ✓ |
| Optimizer threshold match | NO | ⚠ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.0000 (z=-8.62) | ✓ significant |

### Class Distribution
- HOLD: **0.0%** — OK
- SELL: **0.0%**
- BUY:  **0.0%** — ⚠ minority class

### Issues Detected
- **WARNING** — Model meta thresholds do not match optimizer-selected gate thresholds. Investigate whether the optimizer output is stale or not fully applied.


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
Blocked by Meta Gate:      -  317  (81%)
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

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


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
| meta_gate | 317 | 170 | 147 | 53.8% | +352.50% |
| quality | 31 | 19 | 12 | 61.5% | +34.47% ⚠ |
| hmm | 19 | 10 | 9 | 57.6% | +21.13% ⚠ |
| confluence | 15 | 8 | 7 | 54.5% | +16.68% |
| fake_breakout | 11 | 6 | 5 | 60.0% | +12.23% ⚠ |
| portfolio | 7 | 4 | 3 | 57.6% | +7.78% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.0711 | <0.10 | ✓ |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.1328 | ~1.0 | ⚠ model overconfident |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 27.3% | -0.48 | ⚠ overconfident |
| 80-90% | 37.3% | -0.48 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 103 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

**States:** 5  |  **Global precision:** 29.1%  |  **Max state concentration:** 44.2%

### Per-Regime Performance

| Regime | Trades | Precision | Sell Prec | Expectancy | P.Factor | Modifier | Rec. |
|--------|--------|-----------|-----------|------------|----------|----------|------|
| TRENDING_BULL | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| TRENDING_BEAR | 634 | 27.4% | 27.3% | +0.164% | 1.30 | -0.016 | NEUTRAL |
| ACCUMULATION | 310 | 35.5% | 45.1% | +0.238% | 1.41 | +0.064 | BOOST (+threshold reduction) 🟢 |
| DISTRIBUTION | 817 | 28.6% | 32.5% | +0.130% | 1.29 | -0.004 | NEUTRAL |
| COMPRESSION | 195 | 19.0% | 13.5% | +0.037% | 1.17 | -0.101 | SUPPRESS (+threshold increase) |
| VOLATILE_EXPANSION | 441 | 32.2% | 27.0% | +0.287% | 1.58 | +0.031 | NEUTRAL |
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

**Overall Drift Status:** 🟡 **WARNING**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🟡 WARNING | 9 CRITICAL / 14 DEGRADED / 73 total |
| Confidence Drift | 🟡 WARNING | T=1.133 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +51.29pp |

**Estimated precision loss from feature drift:** ~3.8pp


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
| Holdout Sharpe | 7.01 | ✓ |
| Holdout Max DD | 48.16% | ✗ |
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

**0 root causes identified.**  Combined top-5 impact score: **0/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|

### Fixes



---

## Section 14 — Automated Improvement Engine

**Base precision:** 27.4%  →  **Expected precision (all fixes):** 41.5%  (+14.2pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Enable BUY side (fix directional asymmetry) | +0.0pp | +8.0pp | +5.0pp | HIGH | MEDIUM |
| 2 | Remove / normalise top-10 drifted features | +19.8pp | +1.5pp | +15.8pp | MEDIUM | LOW |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Redesign triple-barrier labels (reduce HOLD%) | +1.5pp | +4.0pp | +3.0pp | MEDIUM | MEDIUM |
| 5 | Extend lookahead for low-ER tokens | +1.0pp | +2.0pp | +1.5pp | MEDIUM | LOW |
| 6 | Regime-specific meta thresholds | +1.5pp | +1.0pp | +2.5pp | MEDIUM | LOW |
| 7 | Retrain meta model on 60-symbol fleet data | +2.0pp | +0.5pp | +2.5pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 362 |
| Rejected signals | 978 |
| Selected precision | 27.4% |
| Rejected precision | 22.4% |
| Meta gate lift (precision) | +5.0% |
| Selected expectancy | +0.066% |
| Rejected expectancy | +0.130% |
| Selected Sharpe | +12.19 |
| Rejected Sharpe | +13.75 |

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
| ACCUMULATION         | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 46.7 | 46.7 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep ETH/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +5.0% |
| Selected signals | 362 |
| Rejected signals | 978 |
| Gate coverage | 27.0% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 54/100 |
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
| Actual Precision | 27.3% |
| Gap | -30.5% |
| Coverage | 6.4% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 54/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
