# AEGIS-1 BUY Precision Forensic Report

## SECTION 1 — CLASS DISTRIBUTION ANALYSIS
SELL (0): 0
HOLD (1): 6704
BUY (2): 0
Imbalance ratio BUY/HOLD: 0.000

## SECTION 2 — LABEL QUALITY AUDIT
Base ATR Multiplier used: 1.2
Total Barrier Hit Frequency: 0.00%
BUY vs SELL symmetry: 0.00 (1.0 is symmetric)

Regime Distribution of Labels:
target                     1
macro_confluence_score      
0.0                     6704

## SECTION 3 — LOOKAHEAD HORIZON ANALYSIS
Horizon 6h -> BUY: 0, HOLD: 6716, SELL: 0
Horizon 12h -> BUY: 0, HOLD: 6710, SELL: 0
Horizon 18h -> BUY: 0, HOLD: 6704, SELL: 0
Horizon 24h -> BUY: 0, HOLD: 6698, SELL: 0
Horizon 36h -> BUY: 0, HOLD: 6686, SELL: 0
Horizon 48h -> BUY: 0, HOLD: 6674, SELL: 0

## SECTION 5 — FEATURE DOMINANCE ANALYSIS
Missing features in re-run: ['prc_total', 'prc_candle', 'prc_volume', 'prc_trend', 'prc_smart_money', 'prc_bands', 'prc_momentum']
Confusion Matrix on entire dataset:
[[   0    0    0]
 [1272 4762  670]
 [   0    0    0]]
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       1.00      0.71      0.83      6704
           2       0.00      0.00      0.00         0

    accuracy                           0.71      6704
   macro avg       0.33      0.24      0.28      6704
weighted avg       1.00      0.71      0.83      6704


## SECTION 4 — REGIME ANALYSIS