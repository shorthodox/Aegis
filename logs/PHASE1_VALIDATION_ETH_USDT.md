# AEGIS Phase-1 Validation — ETH/USDT

**Generated:** 2026-06-05 00:33:23

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.0936 | 0.0000 | -0.0936 ✓ improved |
| Brier  | 0.1398 | 0.1232 | -0.0166 |
| Precision | 0.3007 | 0.7143 | +0.4136 |
| Coverage | 0.1247 | 0.0012 | -0.1235 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.3100 | 0.0000 | -0.3100 |
| Coverage  | 0.4457  | 0.0000  | -0.4457 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 35 |
| WARNING | 15 |
| DEGRADED | 14 |
| CRITICAL | 9 |

**Top drifting features:**

- `vwap_decay_mean_24`: score=10.7938 [CRITICAL] PSI=20.758 KS=0.915
- `se_mid`: score=9.4778 [CRITICAL] PSI=18.159 KS=0.888
- `close`: score=9.4203 [CRITICAL] PSI=18.047 KS=0.886
- `open`: score=9.4178 [CRITICAL] PSI=18.042 KS=0.885
- `low`: score=8.1986 [CRITICAL] PSI=15.596 KS=0.888

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 310 | 0.355 | +0.238% | 0.299 | +0.064 |
| COMPRESSION | 195 | 0.190 | +0.037% | 0.274 | -0.101 |
| DISTRIBUTION | 817 | 0.286 | +0.130% | 0.274 | -0.004 |
| TRENDING_BEAR | 634 | 0.274 | +0.164% | 0.274 | -0.016 |
| VOLATILE_EXPANSION | 441 | 0.322 | +0.287% | 0.274 | +0.031 |