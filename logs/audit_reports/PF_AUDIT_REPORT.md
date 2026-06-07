# Profit Factor Audit Report

- **Total Trades Fired:** 50
- **Wins:** 33
- **Losses (including fee-only timeouts):** 17
- **Gross Profit (sum of win net_pnl):** 0.1790
- **Gross Loss (sum of loss net_pnl):** 0.0697
- **Calculated Profit Factor:** 2.566

### Root Cause of Old Massive PF (579,967,018)
In the previous run, Simulation A had 0 losses (gross_loss = 1e-9 fallback). This was caused by the model evaluated being the `deploy_primary` model, which was trained directly on the holdout set, enabling it to cheat and hit the target barrier with 100% precision. Because there were 0 prediction errors and timeouts were not counted as losses in the old calculations, the gross loss was exactly 0, leading to a division-by-zero anomaly.
To make the backtest statistically valid, the model must be trained *only* on the training pool, and evaluated *only* on the clean holdout set.