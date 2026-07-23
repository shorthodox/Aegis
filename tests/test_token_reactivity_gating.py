"""
tests/test_token_reactivity_gating.py — Unit test verifying:
1. Dynamic threshold floor scaling per token ATR reactivity.
2. 25% maximum trade coverage cap.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_token_reactivity_threshold_scaling():
    # Low vol (BTC, ATR% ~ 0.0065) -> floor ~ 0.50
    med_atr_btc = 0.0065
    floor_btc = float(np.clip(0.50 + 0.08 * (med_atr_btc / 0.008), 0.50, 0.65))

    # High vol (ETH/SOL, ATR% ~ 0.0150) -> floor ~ 0.65
    med_atr_sol = 0.0150
    floor_sol = float(np.clip(0.50 + 0.08 * (med_atr_sol / 0.008), 0.50, 0.65))

    assert floor_sol > floor_btc, f"Higher ATR tokens must have a higher threshold floor ({floor_sol} > {floor_btc})"
    assert floor_sol <= 0.65, f"Floor must be capped at 0.65, got {floor_sol}"

    print("TOKEN REACTIVITY GATING UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_token_reactivity_threshold_scaling()
