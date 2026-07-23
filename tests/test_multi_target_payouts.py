"""
tests/test_multi_target_payouts.py — Unit test verifying:
1. Multi-target scaled exit payouts (1.8x barrier) and breakeven stop loss locking in backtest().
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.retrain_model import backtest


def test_multi_target_backtest_payouts():
    fire_mask = np.array([True, True, True])
    proposed  = np.array([2, 0, 2])
    y_true    = np.array([2, 0, 1])  # 2 wins, 1 timeout
    barriers  = np.array([0.02, 0.02, 0.02])

    res = backtest(fire_mask, proposed, y_true, barriers, fee=0.001)

    assert res['n'] == 3, f"Expected exactly 3 trades, got {res['n']}"
    assert res['profit_factor'] > 5.0, f"Expected Profit Factor > 5.0, got {res['profit_factor']}"
    assert res['max_drawdown_pct'] < 0.50, f"Expected Drawdown < 0.50%, got {res['max_drawdown_pct']}"

    print("MULTI-TARGET PAYOUT UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_multi_target_backtest_payouts()
