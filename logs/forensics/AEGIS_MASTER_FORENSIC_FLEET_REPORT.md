## Section X — Cross Token Forensics

### Fleet KPI Summary

| Symbol | Precision | PF | Expectancy | Sharpe | Drawdown | Coverage | BUY | SELL | Tradeable | Drift | Regime Prec | Mismatch |
|--------|-----------|----|------------|--------|----------|----------|-----|------|-----------|-------|-------------|----------|
| ADA/USDT | 54.9% | 5.25 | +0.20% | 25.38 | 1.13% | 16.0% | 100.0% | 99.0% | BUY/SELL | 10 | 42.4% | 0.0pp |
| AVAX/USDT | 50.3% | 2.53 | +0.24% | 27.38 | 8.54% | 50.5% | 86.0% | 74.5% | BUY/SELL | 0 | 0.0% | 0.0pp |
| BNB/USDT | 53.9% | 9.42 | +0.45% | 33.81 | 2.55% | 21.7% | 98.8% | 98.2% | BUY/SELL | 10 | 59.1% | 0.0pp |
| BTC/USDT | 54.1% | 4.47 | +0.19% | 33.34 | 2.50% | 29.9% | 99.2% | 95.0% | BUY/SELL | 10 | 42.4% | 0.0pp |
| DOGE/USDT | 56.3% | 4.50 | +0.29% | 30.14 | 6.18% | 24.6% | 100.0% | 88.8% | BUY/SELL | 10 | 0.0% | 0.0pp |
| DOT/USDT | 51.2% | 2.88 | +0.19% | 29.91 | 3.12% | 46.0% | 80.9% | 83.7% | BUY/SELL | 0 | 0.0% | 0.0pp |
| ETH/USDT | 40.9% | 2.19 | +0.10% | 6.57 | 2.06% | 4.0% | 83.3% | 100.0% | BUY/SELL | 10 | 29.1% | 0.0pp |
| LINK/USDT | 49.4% | 2.62 | +0.19% | 28.94 | 2.41% | 47.7% | 88.6% | 74.4% | BUY/SELL | 0 | 0.0% | 0.0pp |
| SOL/USDT | 44.7% | 3.94 | +0.57% | 34.55 | 18.88% | 41.6% | 87.8% | 72.5% | BUY/SELL | 10 | 36.4% | 0.0pp |
| TON/USDT | 40.6% | 0.93 | -0.02% | -0.70 | 13.18% | 7.7% | 74.2% | 100.0% | BUY/SELL | 0 | 0.0% | 0.0pp |
| TRX/USDT | 49.1% | 0.77 | -0.01% | -5.37 | 6.20% | 20.1% | 95.2% | 100.0% | BUY/SELL | 0 | 0.0% | 4.5pp |
| XRP/USDT | 48.7% | 2.25 | +0.24% | 26.45 | 12.08% | 58.8% | 89.9% | 66.7% | BUY/SELL | 10 | 46.9% | 0.0pp |

### Merged Fleet Comparison Summary

- This merged summary aggregates the fleet to expose retrain priorities and equalization gaps.
- Token count: **12**.
- Fleet average precision: **49.5%**, median: **49.8%**, std: **5.0%**.
- Fleet average profit factor: **3.48**; average expectancy: **+0.22%**.
- Average label/PnL mismatch: **0.4pp**.
- Tradeable ratio: **100.0%**; low coverage ratio: **0.0%**.
- Drift risk ratio: **58.3%**; regime risk ratio: **83.3%**; calibration risk ratio: **100.0%**.
- Recommendation: align retrain_model.py on regime-sensitive thresholding, buy/sell gate balance, drift normalization, and calibration consistency.

### Best / Worst Tokens

- Best precision: DOGE/USDT (56.3%), ADA/USDT (54.9%), BTC/USDT (54.1%), BNB/USDT (53.9%), DOT/USDT (51.2%)
- Best profit factor: BNB/USDT (9.42), ADA/USDT (5.25), DOGE/USDT (4.50), BTC/USDT (4.47), SOL/USDT (3.94)
- Best expectancy: SOL/USDT (+0.57%), BNB/USDT (+0.45%), DOGE/USDT (+0.29%), AVAX/USDT (+0.24%), XRP/USDT (+0.24%)
- Best sharpe: SOL/USDT (34.55), BNB/USDT (33.81), BTC/USDT (33.34), DOGE/USDT (30.14), DOT/USDT (29.91)
- Worst precision: TRX/USDT (49.1%), XRP/USDT (48.7%), SOL/USDT (44.7%), ETH/USDT (40.9%), TON/USDT (40.6%)

### Root Cause Difference Engine

- Failed tokens are characterized by: low precision, negative expectancy, weak regime precision, or disabled tradeability.
  - ETH/USDT: precision=40.9%, PF=2.19, expectancy=+0.10%, tradeable=True
  - LINK/USDT: precision=49.4%, PF=2.62, expectancy=+0.19%, tradeable=True
  - SOL/USDT: precision=44.7%, PF=3.94, expectancy=+0.57%, tradeable=True

### Label vs PnL Forensics

- Mismatch score measures absolute gap between meta label precision and panel win rate.
- TRX/USDT: mismatch=4.5pp, precision=49.1%, buy=95.2%, sell=100.0%.
- ADA/USDT: mismatch=0.0pp, precision=54.9%, buy=100.0%, sell=99.0%.
- AVAX/USDT: mismatch=0.0pp, precision=50.3%, buy=86.0%, sell=74.5%.
- BNB/USDT: mismatch=0.0pp, precision=53.9%, buy=98.8%, sell=98.2%.
- BTC/USDT: mismatch=0.0pp, precision=54.1%, buy=99.2%, sell=95.0%.

### Regime Dependence Forensics

- Fleet tokens with regime global precision < 45% are most vulnerable to regime shifts and threshold mismatch.
- Regime quality is a leading discriminator between BTC-like winners and ETH-like failures.
- Tokens with weak regime coverage: 10 / 12.

### Fleet Learning Audit

- Top tokens exhibit high meta precision and strong regime diversification. Weak tokens show the inverse.
- BTC vs ETH: BTC has precision=54.1%, PF=4.47, expectancy=+0.19%, tradeable=True.
- ETH has precision=40.9%, PF=2.19, expectancy=+0.10%, tradeable=True.
- Root cause: ETH's meta thresholds and regime hedge failed, while BTC's regime-sensitive thresholds and drift controls succeeded.

### Overfitting Forensics

- Overfitting is flagged when OOF precision exceeds holdout precision by >5pp and holdout sample size is low.
- High-risk tokens:
  - TON/USDT: precision=40.6%, PF=0.93, expectancy=-0.02%
  - TRX/USDT: precision=49.1%, PF=0.77, expectancy=-0.01%

### Executive Takeaway

- BTC-style success is driven by positive expectancy, strong regime precision, and tradeable threshold gating.
- ETH-style failure is driven by low holdout precision, disabled tradeability, and regime-dependent drift that violates the meta gate.
- For the fleet, prioritize tokens with both precision >60% and profit factor >1.5, while auditing any token with mismatch >10pp or drift_count ≥10.