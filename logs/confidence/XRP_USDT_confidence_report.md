# Confidence Reliability Report — XRP/USDT

**Generated:** 2026-06-04 23:36:19

---

## Overview

This report answers: **when the model predicts X% confidence, how often does it actually win?**

A perfectly calibrated model would have `actual_win_rate ≈ predicted_confidence`.

Positive gap = overconfident. Negative gap = underconfident.


## Per-Bucket Reliability

| Confidence Bucket | Samples | Predicted | Actual Win Rate | Gap | Status |
|-------------------|---------|-----------|-----------------|-----|--------|
| 0.50–0.55 | 127 | 52% | 54.3% | -0.018 | ✓ calibrated |
| 0.55–0.60 | 0 | 58% | — | — | no data |
| 0.60–0.65 | 0 | 63% | — | — | no data |
| 0.65–0.70 | 24 | 68% | 66.7% | +0.008 | ✓ calibrated |
| 0.70–0.75 | 0 | 73% | — | — | no data |
| 0.75–0.80 | 0 | 78% | — | — | no data |
| 0.80–0.85 | 0 | 83% | — | — | no data |
| 0.85–0.90 | 0 | 88% | — | — | no data |
| 0.90–0.95 | 0 | 93% | — | — | no data |
| 0.95–1.00 | 0 | 98% | — | — | no data |

## Key Questions

- **When model predicts ~90%:** insufficient data
- **When model predicts ~80%:** insufficient data
- **When model predicts ~70%:** insufficient data
- **When model predicts ~60%:** insufficient data

**Mean absolute calibration gap:** 0.0133 (✓ well calibrated)
