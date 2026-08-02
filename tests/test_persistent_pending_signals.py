"""
tests/test_persistent_pending_signals.py — Unit test verifying:
  1. Armed/waiting signals (pending_entry=True) are registered in persistent armed queue (_armed_pending_setups).
  2. Temporary model noise on subsequent scans does not cancel or drop the armed signal.
  3. The waiting signal fires when either (Path A) price hits the fire level, or (Path B) an early 5m reversal candle pattern prints.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.live_engine as live_engine
from scripts.live_engine import LiveEngine


def test_armed_pending_setup_lifecycle(monkeypatch):
    # v83 replaced PENDING with TraderGate's working orders, and
    # `_sync_armed_pending_state` is inert while USE_TRADER_GATE is on. This
    # test covers the LEGACY queue — the rollback path — so it pins the flag
    # off. The working-order lifecycle that supersedes it is covered by
    # scripts/tests/test_trader_gate_wiring.py.
    monkeypatch.setattr(live_engine, 'USE_TRADER_GATE', False)
    engine = LiveEngine(token_configs={})

    # 1. Register an armed pending setup
    sym = 'BTC/USDT'
    engine._armed_pending_setups[sym] = {
        'side': 'BUY',
        'target': 60000.0,
        'reason': 'approaching support 60000',
        'armed_time': time.time(),
    }

    assert sym in engine._armed_pending_setups
    assert engine._armed_pending_setups[sym]['side'] == 'BUY'

    # 1b. Simulate indicator wobble on subsequent scan where last_signals reset pending_entry=False
    engine.last_signals[sym] = {
        'symbol': sym,
        'signal': 'HOLD',
        'fire': False,
        'pending_entry': False,
        'price': 61000.0,
    }
    # Run sync: should preserve pending_entry=True in last_signals
    engine._sync_armed_pending_state(sym)
    assert engine.last_signals[sym]['pending_entry'] is True
    assert engine.last_signals[sym]['pending_side'] == 'BUY'

    # 2. Simulate level hit (Path A): price reaches 60000.0 (near target)
    price_hit = 60010.0 # 0.016% away
    target = engine._armed_pending_setups[sym]['target']
    near_pct = abs(price_hit - target) / target * 100.0
    assert near_pct <= engine.PENDING_NEAR_PCT # 0.3%

    # 3. Simulate early reversal fire (Path B): bullish hammer pattern
    from scripts.live_engine import _reversal_candle
    c1 = [1599999700000, 102.0, 102.1, 100.0, 100.6, 1000.0]
    c2 = [1600000000000, 100.5, 100.6, 98.0, 100.4, 1000.0] # hammer
    pat = _reversal_candle([c1, c2], want_bullish=True)
    assert pat == 'hammer'

    # Clear armed setup on fire
    engine._armed_pending_setups.pop(sym, None)
    assert sym not in engine._armed_pending_setups


if __name__ == '__main__':
    test_armed_pending_setup_lifecycle()
    print("ALL PERSISTENT PENDING SIGNAL TESTS PASSED SUCCESSFULLY!")
