# AEGIS Phase-1 Validation — BNB/USDT

**Generated:** 2026-06-05 00:37:09

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.0456 | 0.0000 | -0.0456 ✓ improved |
| Brier  | 0.1573 | 0.1526 | -0.0047 |
| Precision | 0.4741 | 0.5789 | +0.1048 |
| Coverage | 0.1845 | 0.0421 | -0.1424 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.5606 | 0.3529 | -0.2077 |
| Coverage  | 0.0334  | 0.0202  | -0.0132 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 27 |
| WARNING | 15 |
| DEGRADED | 9 |
| CRITICAL | 6 |

**Top drifting features:**

- `vwap_decay_mean_24`: score=10.2303 [CRITICAL] PSI=19.650 KS=0.894
- `low`: score=2.6957 [CRITICAL] PSI=4.776 KS=0.695
- `close`: score=2.5332 [CRITICAL] PSI=4.458 KS=0.695
- `high`: score=2.4840 [CRITICAL] PSI=4.365 KS=0.692
- `vwap_decay_std_24`: score=0.8150 [CRITICAL] PSI=1.379 KS=0.268

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 83 | 0.506 | +0.257% | 0.594 | -0.085 |
| CHOPPY | 65 | 0.723 | +0.816% | 0.450 | +0.132 |
| COMPRESSION | 15 | 0.533 | +0.331% | 0.549 | -0.058 |
| DISTRIBUTION | 40 | 0.600 | +0.725% | 0.549 | +0.009 |
| TRENDING_BEAR | 1 | 0.000 | +0.000% | 0.400 | -0.300 |
| TRENDING_BULL | 2 | 1.000 | +0.000% | 0.549 | +0.200 |
| VOLATILE_EXPANSION | 2 | 0.000 | +0.000% | 0.549 | -0.300 |