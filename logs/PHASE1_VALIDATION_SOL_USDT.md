# AEGIS Phase-1 Validation — SOL/USDT

**Generated:** 2026-06-05 00:42:58

---

## Calibration

| Metric | Before Calibration | After Calibration | Δ |
|--------|-------------------|------------------|---|
| ECE    | 0.1141 | 0.0000 | -0.1141 ✓ improved |
| Brier  | 0.1301 | 0.1106 | -0.0195 |
| Precision | 0.2796 | 0.0000 | -0.2796 |
| Coverage | 0.0989 | 0.0000 | -0.0989 |

**Calibrator selected:** ISOTONIC
✓ ECE target met (< 0.10)
✓ ECE preferred target met (< 0.05)

## Trading Performance

| Metric | DEV (OOF) | Holdout | Δ |
|--------|-----------|---------|---|
| Precision | 0.4211 | 0.0000 | -0.4211 |
| Coverage  | 0.0567  | 0.0000  | -0.0567 |

## Feature Drift Report

| State | Count |
|-------|-------|
| HEALTHY | 45 |
| WARNING | 24 |
| DEGRADED | 7 |
| CRITICAL | 8 |

**Top drifting features:**

- `close`: score=11.8125 [CRITICAL] PSI=22.772 KS=0.945
- `high`: score=11.7780 [CRITICAL] PSI=22.704 KS=0.945
- `vwap_decay_mean_24`: score=11.0958 [CRITICAL] PSI=21.349 KS=0.935
- `low`: score=10.6404 [CRITICAL] PSI=20.437 KS=0.945
- `vwap_decay_std_24`: score=2.4812 [CRITICAL] PSI=4.455 KS=0.583

## Regime Performance Report

| Regime | Trades | Precision | Expectancy | Threshold | Modifier |
|--------|--------|-----------|------------|-----------|----------|
| ACCUMULATION | 15 | 0.200 | -0.634% | 0.364 | -0.164 |
| CHOPPY | 103 | 0.311 | +0.190% | 0.364 | -0.053 |
| DISTRIBUTION | 19 | 0.263 | +0.019% | 0.364 | -0.101 |
| TRENDING_BEAR | 27 | 0.333 | +0.340% | 0.364 | -0.031 |
| TRENDING_BULL | 8 | 0.375 | +1.549% | 0.364 | +0.011 |
| VOLATILE_EXPANSION | 45 | 0.600 | +0.720% | 0.364 | +0.200 |