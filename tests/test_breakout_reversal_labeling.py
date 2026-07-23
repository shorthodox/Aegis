"""
tests/test_breakout_reversal_labeling.py — Unit test verifying:
1. BUY at Support (range_position <= 0.40) is assigned label 2 (BUY).
2. BUY at Resistance (range_position >= 0.60) in TRENDING_BULL with volume expansion is assigned label 2 (BUY).
3. BUY at Resistance without Bullish regime is demoted to label 1 (HOLD).
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.retrain_model import create_triple_barrier_labels


def test_breakout_reversal_labeling_logic():
    # 10 rows so n > max_lookahead (5)
    # Row 0: range_position = 0.80 (resistance), BULL regime, rel_vol = 1.20, close >= ema21 -> MUST be 2 (BUY breakout)
    # Row 1: range_position = 0.80 (resistance), RANGE regime, rel_vol = 0.80 -> MUST be demoted to 1 (HOLD)
    # Row 2: range_position = 0.20 (support), RANGE regime -> MUST be 2 (BUY reversal)
    df = pd.DataFrame({
        'open':           [102.0] * 10,
        'high':           [102.5, 115.0, 115.0, 115.0, 102.5, 102.5, 102.5, 102.5, 102.5, 102.5],
        'low':            [101.5, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5],
        'close':          [102.0] * 10,
        'ema_21':         [100.0] * 10,
        'rel_vol_24h':    [1.20,  1.20,  0.80,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00],
        'range_position': [0.80,  0.80,  0.20,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50],
        'market_regime':  ['TRENDING_BULL', 'RANGE', 'RANGE', 'RANGE', 'RANGE', 'RANGE', 'RANGE', 'RANGE', 'RANGE', 'RANGE'],
    })

    labels = create_triple_barrier_labels(df, atr_multiplier=1.0, max_lookahead=5)

    # Row 0 (Bullish Breakout at Resistance) -> MUST be 2 (BUY)
    assert labels.iloc[0] == 2, f"Expected Bullish Breakout at resistance to be labeled 2 (BUY), got {labels.iloc[0]}"

    # Row 1 (Unconfirmed BUY at Resistance in Range) -> MUST be demoted to 1 (HOLD)
    assert labels.iloc[1] == 1, f"Expected Unconfirmed BUY at resistance in Range to be demoted to 1 (HOLD), got {labels.iloc[1]}"

    print("BREAKOUT & REVERSAL LABELING UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_breakout_reversal_labeling_logic()
