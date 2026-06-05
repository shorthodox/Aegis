# Confidence Reliability Report — BNB/USDT

**Generated:** 2026-06-05 00:37:09

---

## Overview

This report answers: **when the model predicts X% confidence, how often does it actually win?**

A perfectly calibrated model would have `actual_win_rate ≈ predicted_confidence`.

Positive gap = overconfident. Negative gap = underconfident.


## Per-Bucket Reliability

| Confidence Bucket | Samples | Predicted | Actual Win Rate | Gap | Status |
|-------------------|---------|-----------|-----------------|-----|--------|
| 0.50–0.55 | 110 | 52% | 53.6% | -0.011 | ✓ calibrated |
| 0.55–0.60 | 89 | 58% | 58.4% | -0.009 | ✓ calibrated |
| 0.60–0.65 | 26 | 63% | 61.5% | +0.010 | ✓ calibrated |
| 0.65–0.70 | 19 | 68% | 68.4% | -0.009 | ✓ calibrated |
| 0.70–0.75 | 0 | 73% | — | — | no data |
| 0.75–0.80 | 0 | 78% | — | — | no data |
| 0.80–0.85 | 0 | 83% | — | — | no data |
| 0.85–0.90 | 0 | 88% | — | — | no data |
| 0.90–0.95 | 0 | 93% | — | — | no data |
| 0.95–1.00 | 3 | 98% | 100.0% | -0.025 | ⚡ low sample |

## Key Questions

- **When model predicts ~90%:** insufficient data
- **When model predicts ~80%:** insufficient data
- **When model predicts ~70%:** insufficient data
- **When model predicts ~60%:** actual win rate = **61.5%** (n=26, gap=+0.010)

**Mean absolute calibration gap:** 0.0099 (✓ well calibrated)
