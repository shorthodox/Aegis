# AEGIS-2 ROOT CAUSE FORENSIC REPORT

> Executive Summary: This framework statistically isolates why BUY precision degrades out-of-sample.

## SECTION 1 — HOLDOUT-ONLY ANALYSIS
- **Train Accuracy:** 39.30% -> **Holdout Accuracy:** 31.17%
- **Train BUY Precision:** 74.09% -> **Holdout BUY Precision:** 71.05%
- **BUY Precision Degradation:** 3.04%

## SECTION 2 — BUY CLASS SEPARABILITY
- **PCA BUY/HOLD Centroid Distance:** 55735.5645
- **PCA BUY/SELL Centroid Distance:** 28434.8923
> Conclusion: BUY labels show moderate separability.

## SECTION 3 — FEATURE DRIFT ANALYSIS
- `vwap_decay_std_24`: 4.537 (High)
- `dist_vwap`: 3.549 (High)
- `returns_1h_decay_std_24`: 1.541 (High)
- `gk_vol`: 0.851 (High)
- `fib_range_pct`: 0.823 (High)

## SECTION 4 — HMM REGIME FORENSICS
- **Regime Score 0.0:** BUY Precision = 71.05% (Fired = 152)

## SECTION 5 — SHAP BUY ANALYSIS

## SECTION 6 — META MODEL FORENSICS
- **Brier Score:** 0.3328
- **Expected Calibration Error (ECE):** 0.2496

## SECTION 7 — LABEL NOISE INVESTIGATION
- **Mean 18h Return for BUY Labels:** 1.22%
- **Median 18h Return for BUY Labels:** 1.08%
- **Volatility of Returns:** 1.75%

## SECTION 8 — HORIZON INVESTIGATION
- Horizon 6h: BUY density 18.48% (1241 labels)
- Horizon 12h: BUY density 28.48% (1911 labels)
- Horizon 18h: BUY density 33.70% (2259 labels)
- Horizon 24h: BUY density 36.43% (2440 labels)
- Horizon 36h: BUY density 38.60% (2581 labels)
- Horizon 48h: BUY density 40.25% (2686 labels)

## SECTION 9 — CONTINUATION LSTM FORENSICS
> Continuation LSTM AUC (0.506) reveals the model is failing to learn sequential time dependencies. Likely causes: the sequences are too short (12-24 bars is not enough to capture macro shifts) and targets are too noisy.

## SECTION 10 — ROOT CAUSE SCORING ENGINE
- **Feature Drift:** 100/100
- **Meta Model Failure:** 100/100
- **Class Imbalance:** 40/100
- **Label Noise:** 38/100
- **Buy Non Separability:** 20/100

## SECTION 11 — ESTIMATED IMPROVEMENT ANALYSIS
- **Increase BUY Separability (Label Tuning):** +6–9% Precision
- **Drop Drifting Features:** +2–4% Precision
- **Meta Isotonic Calibration:** +3–5% Precision