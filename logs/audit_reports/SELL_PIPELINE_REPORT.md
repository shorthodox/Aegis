# Sell Pipeline Audit Report

## Sell Signal Funnel Breakdown
- **Raw SELL predictions:** 1434
- **After Quality Filter:** 606 (prc_total <= 0.45, converted to Quality score >= 55)
- **After Edge Filter (threshold=68.91333434522844):** 58
- **After HMM Filter:** 58
- **After Confluence Filter:** 56
- **After Drift Filter:** 56

### Verdict
The SELL pipeline functions correctly now. Sells are successfully flowing through the entire pipeline and firing trades on holdout.