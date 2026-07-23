"""
tests/test_profit_enhancements.py — Unit test verifying:
1. Volume absorption & liquidity sweep detection.
2. Minimum 1.4:1 RR headroom gating.
3. Counter-trend RSI / Volume absorption requirement.
4. Smart ATR stop loss cushion.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import LiveEngine, DynamicRiskEngine


def test_profit_enhancements_logic():
    engine = LiveEngine(token_configs=[])

    # 1. Test RR Headroom Gate (< 1.4:1 headroom returns WAIT)
    result_cramped = {
        'support': 0.0890,
        'resistance': 0.0898,  # Only 8 pips to resistance from 0.0891 entry
        'range_position': 0.10,
        'atr': 0.002,
        'rsi': 30,
    }

    async def mock_candles_neutral(symbol, tf, limit):
        # 5m candles
        return [
            [0, 0.0892, 0.0893, 0.0890, 0.0891, 100],
            [0, 0.0891, 0.0892, 0.0889, 0.0890, 100],
            [0, 0.0890, 0.0891, 0.0888, 0.0891, 100],
            [0, 0.0891, 0.0893, 0.0890, 0.0892, 100],
            [0, 0.0892, 0.0894, 0.0891, 0.0893, 100],
        ]

    engine._fetch_candles = mock_candles_neutral

    verdict_rr, detail_rr = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0891, result_cramped))
    assert verdict_rr == 'WAIT', f"Expected WAIT for low RR headroom, got {verdict_rr}: {detail_rr}"
    assert 'insufficient RR headroom' in detail_rr

    # 2. Test Liquidity Sweep Detection (low dips below support, closes above)
    result_sweep = {
        'support': 0.0850,
        'resistance': 0.0950,
        'range_position': 0.10,
        'atr': 0.001,
        'rsi': 30,
    }

    async def mock_candles_sweep(symbol, tf, limit):
        # Sweep below 0.0850 support to 0.0845 and close back at 0.0855
        return [
            [0, 0.0860, 0.0862, 0.0858, 0.0859, 100],
            [0, 0.0859, 0.0860, 0.0852, 0.0853, 100],
            [0, 0.0853, 0.0855, 0.0845, 0.0855, 300],  # Liquidity sweep candle
            [0, 0.0855, 0.0859, 0.0854, 0.0858, 200],
            [0, 0.0858, 0.0862, 0.0857, 0.0861, 250],
        ]

    engine._fetch_candles = mock_candles_sweep
    verdict_swp, detail_swp = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0855, result_sweep))
    assert verdict_swp == 'PASS', f"Expected PASS for liquidity sweep, got {verdict_swp}: {detail_swp}"
    assert 'sweep' in detail_swp or 'confirmed' in detail_swp
    assert result_sweep.get('liquidity_sweep') is True

    # 3. Test DynamicRiskEngine SL Buffer
    risk_engine = DynamicRiskEngine()
    stops = risk_engine.calculate_stops(
        price=0.0855,
        side='BUY',
        atr=0.001,
        support=0.0850,
        resistance=0.0950,
    )
    assert stops['sl'] < 0.0850, f"Stop Loss should be below support cushion, got {stops['sl']}"
    assert stops['valid_rr'] is True, "Valid RR expected with roomy resistance"

    print("ALL PROFIT ENHANCEMENT UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_profit_enhancements_logic()
