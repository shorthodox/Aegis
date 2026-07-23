"""
tests/test_entry_proximity.py — Unit test verifying:
1. Thin entry proximity gap (STRUCT_LEVEL_PROXIMITY_ATR = 0.35, PENDING_NEAR_PCT = 0.35%).
2. Signals > 0.35 ATR away from support return WAIT (far from level).
3. Stop Loss for LONG is strictly below support level.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import LiveEngine, DynamicRiskEngine


def test_entry_proximity_and_sl_placement():
    engine = LiveEngine(token_configs=[])

    # 1. Test entry far from support (DOGE case: Entry 0.0724 vs Support 0.0719)
    result_far = {
        'support': 0.0719,
        'resistance': 0.0750,
        'range_position': 0.16,
        'atr': 0.0005,  # 0.35 ATR is 0.000175
        'rsi': 45,
    }

    async def mock_candles_reversal(symbol, tf, limit):
        # 5m candles turned bullish
        return [
            [0, 0.0723, 0.0725, 0.0722, 0.0724, 100],
            [0, 0.0724, 0.0726, 0.0723, 0.0725, 100],
            [0, 0.0725, 0.0727, 0.0724, 0.0726, 100],
            [0, 0.0726, 0.0728, 0.0725, 0.0727, 100],
            [0, 0.0727, 0.0729, 0.0726, 0.0728, 100],
        ]

    engine._fetch_candles = mock_candles_reversal

    # Entry at 0.0724 (dist to support = 0.0005 = 1.0 ATR > 0.35 ATR) -> MUST return WAIT
    verdict_far, detail_far = asyncio.run(engine._structure_gate('DOGE/USDT', 'BUY', 0.0724, result_far))
    assert verdict_far == 'WAIT', f"Expected WAIT for far entry (0.0724 vs support 0.0719), got {verdict_far}: {detail_far}"
    assert 'far' in detail_far or 'insufficient RR' in detail_far

    # Entry at 0.0720 (dist to support = 0.0001 = 0.2 ATR <= 0.35 ATR) -> MUST return PASS
    verdict_close, detail_close = asyncio.run(engine._structure_gate('DOGE/USDT', 'BUY', 0.0720, result_far))
    assert verdict_close == 'PASS', f"Expected PASS for close entry (0.0720 vs support 0.0719), got {verdict_close}: {detail_close}"

    # 2. Test DynamicRiskEngine SL placement for BUY
    risk_engine = DynamicRiskEngine()
    stops = risk_engine.calculate_stops(
        price=0.0720,
        side='BUY',
        atr=0.0005,
        support=0.0719,
        resistance=0.0750,
    )
    assert stops['sl'] < 0.0719, f"Stop Loss ({stops['sl']}) must be strictly below support (0.0719)"

    print("ALL ENTRY PROXIMITY UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_entry_proximity_and_sl_placement()
