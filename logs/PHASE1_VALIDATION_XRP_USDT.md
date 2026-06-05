# AEGIS Phase-1 Validation — XRP/USDT

**Generated:** 2026-06-04 23:36:19

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.0711 | 0.0000 | -0.0711 ✓ improved |
| Brier  | 0.1669 | 0.1578 | -0.0091 |
| Precision | 0.4717 | 0.5629 | +0.0912 |
| Coverage | 0.2134 | 0.0264 | -0.1870 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.3721 | 0.0000 | -0.3721 |
| Coverage  | 0.1803  | 0.0000  | -0.1803 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 35 |
| WARNING | 14 |
| DEGRADED | 3 |
| CRITICAL | 22 |

**Top drifting features:**

- `avwap_200`: score=11.7679 [CRITICAL] PSI=22.691 KS=0.928
- `avwap_100`: score=11.6920 [CRITICAL] PSI=22.537 KS=0.938
- `avwap_50`: score=11.6806 [CRITICAL] PSI=22.516 KS=0.934
- `pivot`: score=11.6091 [CRITICAL] PSI=22.374 KS=0.933
- `ichimoku_senkou_a`: score=11.5662 [CRITICAL] PSI=22.291 KS=0.927

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 340 | 0.476 | +0.159% | 0.450 | +0.007 |
| CHOPPY | 810 | 0.499 | +0.436% | 0.561 | +0.030 |
| COMPRESSION | 26 | 0.192 | -0.350% | 0.450 | -0.277 |
| DISTRIBUTION | 208 | 0.447 | +0.090% | 0.468 | -0.022 |
| TRENDING_BEAR | 179 | 0.441 | +0.649% | 0.468 | -0.028 |
| VOLATILE_EXPANSION | 74 | 0.338 | +0.453% | 0.450 | -0.131 |