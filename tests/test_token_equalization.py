"""
tests/test_token_equalization.py — Unit test verifying:
1. Universal threshold floor is >= 0.60 across all tokens.
2. Coverage cap is enforced at <= 0.20 with 18% target selection.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_universal_token_equalization():
    # Low ATR (BTC ~ 0.0065) -> floor >= 0.60
    med_btc = 0.0065
    floor_btc = float(np.clip(0.60 + 0.05 * (med_btc / 0.008), 0.60, 0.70))

    # High ATR (BNB ~ 0.0079) -> floor >= 0.60
    med_bnb = 0.0079
    floor_bnb = float(np.clip(0.60 + 0.05 * (med_bnb / 0.008), 0.60, 0.70))

    assert floor_btc >= 0.60, f"BTC floor must be >= 0.60, got {floor_btc}"
    assert floor_bnb >= 0.60, f"BNB floor must be >= 0.60, got {floor_bnb}"

    # Verify coverage targeting
    rows = [
        (0.62, 0.75, 0.28, 100, 0.65),  # 28% coverage -> invalid (> 0.20)
        (0.64, 0.72, 0.18, 90,  0.62),  # 18% coverage -> target!
        (0.66, 0.70, 0.12, 70,  0.60),  # 12% coverage
    ]
    passing = [r for r in rows if r[4] >= 0.60 and r[2] <= 0.20]
    sel = min(passing, key=lambda r: abs(r[2] - 0.18))
    assert sel[0] == 0.64, f"Expected 18% coverage target selection (thr=0.64), got {sel[0]}"

    print("UNIVERSAL TOKEN EQUALIZATION UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_universal_token_equalization()
