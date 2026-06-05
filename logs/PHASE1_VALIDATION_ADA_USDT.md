# AEGIS Phase-1 Validation — ADA/USDT

**Generated:** 2026-06-04 19:12:17

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.0900 | 0.0000 | -0.0900 ✓ improved |
| Brier  | 0.1660 | 0.1527 | -0.0133 |
| Precision | 0.4232 | 0.6667 | +0.2434 |
| Coverage | 0.2021 | 0.0133 | -0.1888 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.4677 | 0.0000 | -0.4677 |
| Coverage  | 0.2006  | 0.0000  | -0.2006 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 38 |
| WARNING | 20 |
| DEGRADED | 4 |
| CRITICAL | 20 |

**Top drifting features:**

- `vwap`: score=11.1303 [CRITICAL] PSI=21.395 KS=1.000
- `vwap_decay_mean_24`: score=11.0958 [CRITICAL] PSI=21.325 KS=1.000
- `ema_200`: score=6.8274 [CRITICAL] PSI=12.839 KS=0.944
- `s2`: score=6.6376 [CRITICAL] PSI=12.495 KS=0.878
- `ema_100`: score=6.6077 [CRITICAL] PSI=12.420 KS=0.913

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 369 | 0.404 | +0.171% | 0.350 | -0.020 |
| COMPRESSION | 16 | 0.688 | +0.646% | 0.350 | +0.200 |
| DISTRIBUTION | 52 | 0.385 | +0.339% | 0.439 | -0.039 |
| TRENDING_BULL | 134 | 0.351 | +0.220% | 0.439 | -0.073 |
| VOLATILE_EXPANSION | 435 | 0.458 | +0.546% | 0.439 | +0.034 |