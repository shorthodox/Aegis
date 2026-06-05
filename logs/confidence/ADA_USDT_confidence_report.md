# Confidence Reliability Report — ADA/USDT

**Generated:** 2026-06-04 19:12:17

---

## Overview

This report answers: **when the model predicts X% confidence, how often does it actually win?**

A perfectly calibrated model would have `actual_win_rate ≈ predicted_confidence`.

Positive gap = overconfident. Negative gap = underconfident.


## Per-Bucket Reliability

| Confidence Bucket | Samples | Predicted | Actual Win Rate | Gap | Status |
|-------------------|---------|-----------|-----------------|-----|--------|
| 0.50–0.55 | 2 | 52% | 50.0% | +0.025 | ⚡ low sample |
| 0.55–0.60 | 0 | 58% | — | — | no data |
| 0.60–0.65 | 48 | 63% | 60.4% | +0.021 | ✓ calibrated |
| 0.65–0.70 | 0 | 68% | — | — | no data |
| 0.70–0.75 | 0 | 73% | — | — | no data |
| 0.75–0.80 | 0 | 78% | — | — | no data |
| 0.80–0.85 | 0 | 83% | — | — | no data |
| 0.85–0.90 | 0 | 88% | — | — | no data |
| 0.90–0.95 | 0 | 93% | — | — | no data |
| 0.95–1.00 | 10 | 98% | 100.0% | -0.025 | ✓ calibrated |

## Key Questions

- **When model predicts ~90%:** insufficient data
- **When model predicts ~80%:** insufficient data
- **When model predicts ~70%:** insufficient data
- **When model predicts ~60%:** actual win rate = **60.4%** (n=48, gap=+0.021)

**Mean absolute calibration gap:** 0.0229 (✓ well calibrated)
