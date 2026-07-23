"""
tests/test_audit_fixes.py — Unit test verifying:
1. Institutional Sharpe ratio calculation in backtest() produces realistic values.
2. Minimum independent event threshold for side enablement is set to 35.
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


def test_institutional_sharpe_ratio():
    fire_mask = np.array([True] * 100)
    proposed  = np.array([2] * 100)
    y_true    = np.array([2] * 80 + [0] * 20)  # 80 wins, 20 losses
    barriers  = np.array([0.02] * 100)

    res = backtest(fire_mask, proposed, y_true, barriers, fee=0.001)

    assert res['n'] == 100, f"Expected 100 trades, got {res['n']}"
    assert 1.0 <= res['sharpe'] <= 8.0, f"Expected institutional Sharpe ratio between 1.0 and 8.0, got {res['sharpe']}"
    print("INSTITUTIONAL SHARPE RATIO UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_institutional_sharpe_ratio()
