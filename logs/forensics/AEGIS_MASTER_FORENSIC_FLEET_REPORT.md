## Section X — Cross Token Forensics

### Fleet KPI Summary

| Symbol | Precision | PF | Expectancy | Sharpe | Drawdown | Coverage | BUY | SELL | Tradeable | Drift | Regime Prec | Mismatch |
|--------|-----------|----|------------|--------|----------|----------|-----|------|-----------|-------|-------------|----------|
| BNB/USDT | 44.2% | 8.20 | +0.46% | 26.41 | 1600000000.00% | 20.0% | 100.0% | 97.5% | NONE | 10 | 59.1% | 0.0pp |
| BTC/USDT | 42.2% | 4.00 | +0.25% | 24.56 | 32.97% | 26.3% | 94.1% | 93.7% | NONE | 10 | 42.4% | 0.0pp |
| ETH/USDT | 27.4% | 1.48 | +0.07% | 7.01 | 48.16% | 27.0% | 63.9% | 86.8% | NONE | 10 | 29.1% | 0.0pp |

### Merged Fleet Comparison Summary

- This merged summary aggregates the fleet to expose retrain priorities and equalization gaps.
- Token count: **3**.
- Fleet average precision: **37.9%**, median: **42.2%**, std: **7.5%**.
- Fleet average profit factor: **4.56**; average expectancy: **+0.26%**.
- Average label/PnL mismatch: **0.0pp**.
- Tradeable ratio: **0.0%**; low coverage ratio: **0.0%**.
- Drift risk ratio: **100.0%**; regime risk ratio: **66.7%**; calibration risk ratio: **0.0%**.
- Recommendation: align retrain_model.py on regime-sensitive thresholding, buy/sell gate balance, drift normalization, and calibration consistency.

### Best / Worst Tokens

- Best precision: BNB/USDT (44.2%), BTC/USDT (42.2%), ETH/USDT (27.4%)
- Best profit factor: BNB/USDT (8.20), BTC/USDT (4.00), ETH/USDT (1.48)
- Best expectancy: BNB/USDT (+0.46%), BTC/USDT (+0.25%), ETH/USDT (+0.07%)
- Best sharpe: BNB/USDT (26.41), BTC/USDT (24.56), ETH/USDT (7.01)
- Worst precision: BNB/USDT (44.2%), BTC/USDT (42.2%), ETH/USDT (27.4%)

### Root Cause Difference Engine

- Failed tokens are characterized by: low precision, negative expectancy, weak regime precision, or disabled tradeability.
  - BNB/USDT: precision=44.2%, PF=8.20, expectancy=+0.46%, tradeable=False
  - BTC/USDT: precision=42.2%, PF=4.00, expectancy=+0.25%, tradeable=False
  - ETH/USDT: precision=27.4%, PF=1.48, expectancy=+0.07%, tradeable=False

### Label vs PnL Forensics

- Mismatch score measures absolute gap between meta label precision and panel win rate.
- BNB/USDT: mismatch=0.0pp, precision=44.2%, buy=100.0%, sell=97.5%.
- BTC/USDT: mismatch=0.0pp, precision=42.2%, buy=94.1%, sell=93.7%.
- ETH/USDT: mismatch=0.0pp, precision=27.4%, buy=63.9%, sell=86.8%.

### Regime Dependence Forensics

- Fleet tokens with regime global precision < 45% are most vulnerable to regime shifts and threshold mismatch.
- Regime quality is a leading discriminator between BTC-like winners and ETH-like failures.
- Tokens with weak regime coverage: 2 / 3.

### Fleet Learning Audit

- Top tokens exhibit high meta precision and strong regime diversification. Weak tokens show the inverse.
- BTC vs ETH: BTC has precision=42.2%, PF=4.00, expectancy=+0.25%, tradeable=False.
- ETH has precision=27.4%, PF=1.48, expectancy=+0.07%, tradeable=False.
- Root cause: ETH's meta thresholds and regime hedge failed, while BTC's regime-sensitive thresholds and drift controls succeeded.

### Overfitting Forensics

- Overfitting is flagged when OOF precision exceeds holdout precision by >5pp and holdout sample size is low.
- No obvious overfitting candidates detected in the fleet summary.

### Executive Takeaway

- BTC-style success is driven by positive expectancy, strong regime precision, and tradeable threshold gating.
- ETH-style failure is driven by low holdout precision, disabled tradeability, and regime-dependent drift that violates the meta gate.
- For the fleet, prioritize tokens with both precision >60% and profit factor >1.5, while auditing any token with mismatch >10pp or drift_count ≥10.