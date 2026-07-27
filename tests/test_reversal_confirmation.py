"""
tests/test_reversal_confirmation.py — Unit test verifying:
1. BUY near resistance (price <= resistance, range_pos >= 0.65) returns WAIT (approaching resistance) / BLOCKED (wrong zone).
2. BUY at support WITHOUT 5m 3-candle reversal confirmation returns WAIT.
3. BUY at support WITH 5m 3-candle reversal confirmation returns PASS.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import LiveEngine


def test_reversal_confirmation_logic():
    engine = LiveEngine(token_configs=[])

    # 1. Test _structure_gate: BUY near resistance without breakout
    result_near_res = {
        'support': 0.0817,
        'resistance': 0.0898,
        'range_position': 0.96,  # Near resistance
        'atr': 0.001,
    }
    # Mock candle fetch returning 5m candles
    async def mock_fetch_candles(symbol, tf, limit):
        # candles: [ts, o, h, l, c, v]
        # candles moving down (no bullish reversal)
        return [
            [0, 0.0896, 0.0897, 0.0894, 0.0895, 100],
            [0, 0.0895, 0.0896, 0.0893, 0.0894, 100],
            [0, 0.0894, 0.0895, 0.0892, 0.0893, 100],
            [0, 0.0893, 0.0894, 0.0891, 0.0892, 100],
            [0, 0.0892, 0.0893, 0.0890, 0.0891, 100],
        ]

    engine._fetch_candles = mock_fetch_candles

    verdict, detail = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0895, result_near_res))
    assert verdict == 'WAIT', f"Expected WAIT for BUY near resistance, got {verdict}: {detail}"
    assert 'approaching' in detail or 'breakout' in detail or 'insufficient RR headroom' in detail

    # 2. Test _structure_gate: BUY at support WITHOUT 5m 3-candle reversal confirmation
    result_at_sup = {
        'support': 0.0817,
        'resistance': 0.0898,
        'range_position': 0.10,  # At support
        'atr': 0.001,
    }

    async def mock_fetch_red_candles(symbol, tf, limit):
        # 5m candles all red / falling
        return [
            [0, 0.0825, 0.0826, 0.0821, 0.0822, 100],
            [0, 0.0822, 0.0823, 0.0818, 0.0819, 100],
            [0, 0.0819, 0.0820, 0.0817, 0.0817, 100],
            [0, 0.0817, 0.0818, 0.0815, 0.0816, 100],
            [0, 0.0816, 0.0817, 0.0814, 0.0815, 100],
        ]

    engine._fetch_candles = mock_fetch_red_candles
    verdict_unconf, detail_unconf = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0818, result_at_sup))
    assert verdict_unconf == 'WAIT', f"Expected WAIT for unconfirmed BUY at support, got {verdict_unconf}: {detail_unconf}"
    assert 'waiting for 5m 3-candle reversal confirmation' in detail_unconf

    # 3. Test _structure_gate: BUY at support WITH 5m 3-candle reversal confirmation
    async def mock_fetch_green_candles(symbol, tf, limit):
        # 5m candles turning green (bullish reversal)
        return [
            [0, 0.0815, 0.0818, 0.0814, 0.0817, 100],
            [0, 0.0817, 0.0820, 0.0816, 0.0819, 100],
            [0, 0.0819, 0.0822, 0.0818, 0.0821, 100],
            [0, 0.0821, 0.0825, 0.0820, 0.0824, 100],
            [0, 0.0824, 0.0828, 0.0823, 0.0827, 100],
        ]

    engine._fetch_candles = mock_fetch_green_candles
    verdict_conf, detail_conf = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0818, result_at_sup))
    assert verdict_conf == 'PASS', f"Expected PASS for confirmed BUY at support, got {verdict_conf}: {detail_conf}"
    assert 'reversal confirmed' in detail_conf

    print("ALL REVERSAL CONFIRMATION UNIT TESTS PASSED SUCCESSFULLY!")


def test_zone_entry_without_valid_target_waits():
    engine = LiveEngine(token_configs=[])
    engine._swing_sr = lambda symbol, price, atr: None

    result_no_target = {
        'support': 0.0817,
        'resistance': 0.0898,
        'range_position': 0.30,
        'atr': 0.001,
        'rsi': 40,
    }

    async def mock_fetch_candles(symbol, tf, limit):
        if tf == '5m':
            return [
                [0, 0.0818, 0.0820, 0.0816, 0.0819, 100],
                [0, 0.0819, 0.0821, 0.0817, 0.0820, 100],
                [0, 0.0820, 0.0823, 0.0819, 0.0822, 100],
                [0, 0.0822, 0.0824, 0.0821, 0.0823, 100],
                [0, 0.0823, 0.0825, 0.0822, 0.0824, 100],
            ]
        if tf == '15m':
            return [
                [0, 0.0818, 0.0821, 0.0816, 0.0819, 100],
                [0, 0.0819, 0.0822, 0.0817, 0.0820, 100],
                [0, 0.0820, 0.0824, 0.0818, 0.0821, 100],
            ]
        return []

    engine._fetch_candles = mock_fetch_candles
    verdict, detail = asyncio.run(engine._structure_gate('ARB/USDT', 'BUY', 0.0800, result_no_target))
    assert verdict == 'WAIT', f"Expected WAIT for zone BUY without a valid target, got {verdict}: {detail}"
    assert 'no tested support' in detail


if __name__ == '__main__':
    test_reversal_confirmation_logic()
    test_zone_entry_without_valid_target_waits()
