# AEGIS Phase-1 Validation — BTC/USDT

**Generated:** 2026-06-05 00:28:24

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.0857 | 0.0000 | -0.0857 ✓ improved |
| Brier  | 0.1451 | 0.1331 | -0.0120 |
| Precision | 0.3803 | 0.7500 | +0.3697 |
| Coverage | 0.1807 | 0.0014 | -0.1793 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.4371 | 0.0000 | -0.4371 |
| Coverage  | 0.0683  | 0.0000  | -0.0683 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 33 |
| WARNING | 12 |
| DEGRADED | 7 |
| CRITICAL | 13 |

**Top drifting features:**

- `low`: score=10.8414 [CRITICAL] PSI=20.843 KS=0.937
- `close`: score=10.6900 [CRITICAL] PSI=20.543 KS=0.938
- `vwap_decay_mean_24`: score=9.1000 [CRITICAL] PSI=17.385 KS=0.905
- `returns_1h_decay_std_24`: score=1.9168 [CRITICAL] PSI=3.348 KS=0.535
- `vwap_decay_std_24`: score=1.9141 [CRITICAL] PSI=3.227 KS=0.709

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 24 | 0.167 | -0.416% | 0.408 | -0.257 |
| COMPRESSION | 7 | 0.571 | +0.304% | 0.408 | +0.147 |
| DISTRIBUTION | 150 | 0.387 | +0.096% | 0.408 | -0.037 |
| TRENDING_BEAR | 120 | 0.425 | +0.119% | 0.408 | +0.001 |
| VOLATILE_EXPANSION | 41 | 0.683 | +0.583% | 0.400 | +0.200 |