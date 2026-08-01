"""Regression tests for the v82d fast exit monitor.

Until v82d the only call to _manage_exit lived inside _process_symbol, so a
stop was evaluated once per 300s scan and only after that symbol's turn through
a semaphore of MAX_CONCURRENT predict_realtime() calls. Prices streamed in
continuously from the websocket and nothing checked them.

Measured on the four trades closed 2026-08-01, every loss overshot its own
stop: 0.02, 0.14, 0.85 and 0.28 percentage points against stops of 1.1-2.3%.
"""
import asyncio
import time
from datetime import datetime, timezone

import pytest

from scripts.live_engine import (
    DynamicRiskEngine, LiveEngine, Position, VirtualWallet,
)

SYM = 'TEST/USDT'


class _Noop:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _position(**kw) -> Position:
    base = dict(
        symbol=SYM, direction='LONG', side='BUY',
        entry_price=100.0, position_value=1000.0, initial_value=1000.0,
        stop_loss=99.0, signal_id='sig-1',
        entry_time=datetime.now(timezone.utc).isoformat(),
        meta_confidence=0.0, atr_multiplier=1.8, atr=0.5,
        take_profit_1=101.0, take_profit_2=102.0, take_profit_3=105.0,
        take_profit_4=108.0, take_profit_5=112.0,
    )
    base.update(kw)
    return Position(**base)


@pytest.fixture
def engine(tmp_path):
    eng = LiveEngine.__new__(LiveEngine)
    eng.wallet = VirtualWallet(10_000.0, 1_000.0,
                               track_record_path=tmp_path / 'tr.json')
    eng.risk_engine = DynamicRiskEngine()
    eng.live_prices = {}
    eng.last_signals = {}
    eng._open_time = {}
    eng._peak_price = {}
    eng._tp1_hit, eng._tp2_hit, eng._tp3_hit, eng._tp4_hit = {}, {}, {}, {}
    eng._last_close_time, eng._last_close_side = {}, {}
    eng._last_close_reason, eng._last_loss_time = {}, {}
    eng.MAX_HOLD_SECONDS = 24 * 3600
    eng.MIN_HOLD_SECONDS = 3600
    eng._save_track_record = lambda: None
    for a in ('perf_tracker', 'drift_monitor', 'adaptive_orchestrator'):
        setattr(eng, a, _Noop())
    return eng


def _open(engine, pos):
    engine.wallet.open_positions[SYM] = pos
    engine._open_time[SYM] = time.time()
    engine._peak_price[SYM] = pos.entry_price


# ── the monitor exists and is wired in ───────────────────────────────────────

def test_exit_monitor_is_started_by_run():
    import inspect
    src = inspect.getsource(LiveEngine.run)
    assert '_exit_monitor_loop' in src, 'run() no longer starts the exit monitor'


def test_exit_check_interval_is_far_below_the_scan_interval():
    """The scan interval governs ENTRIES; this governs every stop and TP."""
    assert LiveEngine.EXIT_CHECK_SECONDS <= 10


# ── stops are now honoured near the level ────────────────────────────────────

def test_monitor_closes_a_long_that_breached_its_stop(engine):
    _open(engine, _position())
    engine.live_prices[SYM] = 98.95        # just through the 99.0 stop
    engine._manage_exit(SYM, engine.wallet.open_positions[SYM], {}, 98.95,
                        price_only=True)
    assert SYM not in engine.wallet.open_positions
    assert engine.wallet.trade_history[-1].exit_reason == 'STOP_HIT'


def test_monitor_closes_a_short_that_breached_its_stop(engine):
    pos = _position(direction='SHORT', side='SELL', stop_loss=101.0,
                    take_profit_1=99.0, take_profit_2=98.0, take_profit_3=95.0,
                    take_profit_4=92.0, take_profit_5=88.0)
    _open(engine, pos)
    engine.live_prices[SYM] = 101.05
    engine._manage_exit(SYM, pos, {}, 101.05, price_only=True)
    assert SYM not in engine.wallet.open_positions


def test_monitor_banks_tp1_between_scans(engine):
    _open(engine, _position())
    engine.live_prices[SYM] = 101.2
    engine._manage_exit(SYM, engine.wallet.open_positions[SYM], {}, 101.2,
                        price_only=True)
    assert engine._tp1_hit[SYM] is True
    assert engine.wallet.trade_history[-1].exit_reason == 'TP1_PARTIAL'


def test_monitor_leaves_an_untouched_position_alone(engine):
    _open(engine, _position())
    engine.live_prices[SYM] = 100.2
    engine._manage_exit(SYM, engine.wallet.open_positions[SYM], {}, 100.2,
                        price_only=True)
    assert SYM in engine.wallet.open_positions
    assert engine.wallet.trade_history == []


# ── price_only must not let stale signals drive model decisions ──────────────

def test_price_only_suppresses_the_model_reversal_exit(engine):
    """A stale opposing last_signals entry must not close the trade."""
    _open(engine, _position())
    engine._open_time[SYM] = time.time() - 7200        # past MIN_HOLD
    engine.live_prices[SYM] = 100.2
    stale = {'side': 'SELL', 'fire': True, 'edge_score': 99.0, 'atr': 0.5}
    engine._manage_exit(SYM, engine.wallet.open_positions[SYM], stale, 100.2,
                        price_only=True)
    assert SYM in engine.wallet.open_positions, 'reversal fired on a stale signal'


def test_scan_path_still_takes_the_model_reversal(engine):
    """The scan cycle keeps its behaviour — price_only defaults to False."""
    _open(engine, _position())
    engine._open_time[SYM] = time.time() - 7200
    engine.live_prices[SYM] = 100.2
    live = {'side': 'SELL', 'fire': True, 'edge_score': 99.0, 'atr': 0.5}
    engine._manage_exit(SYM, engine.wallet.open_positions[SYM], live, 100.2)
    assert SYM not in engine.wallet.open_positions
    assert engine.wallet.trade_history[-1].exit_reason == 'MODEL_REVERSAL_TP'


# ── the overshoot this was built to remove ───────────────────────────────────

def _overshoot(engine, path, check_every) -> float:
    """Walk a price path, sampling every `check_every` ticks. Returns the
    realised loss beyond the stop, in percent of entry."""
    pos = _position()
    _open(engine, pos)
    for i, px in enumerate(path):
        engine.live_prices[SYM] = px
        if i % check_every == 0:
            if SYM not in engine.wallet.open_positions:
                break
            engine._manage_exit(SYM, pos, {}, px, price_only=True)
    rec = engine.wallet.trade_history[-1] if engine.wallet.trade_history else None
    if rec is None:
        return float('nan')
    stop_pct = (pos.entry_price - pos.stop_loss) / pos.entry_price * 100
    return abs(rec.pnl_pct) - stop_pct - VirtualWallet.round_trip_cost_pct()


def test_frequent_checks_cut_the_stop_overshoot(engine, tmp_path):
    """A position sliding through its stop: sampling often fills near the
    level, sampling rarely fills wherever price ended up."""
    path = [100.0 - i * 0.05 for i in range(60)]      # 100.0 down to 97.05
    fast = _overshoot(engine, path, check_every=1)

    eng2 = LiveEngine.__new__(LiveEngine)
    for k, v in engine.__dict__.items():
        setattr(eng2, k, v)
    eng2.wallet = VirtualWallet(10_000.0, 1_000.0,
                                track_record_path=tmp_path / 'tr2.json')
    eng2._tp1_hit, eng2._tp2_hit = {}, {}
    eng2._tp3_hit, eng2._tp4_hit = {}, {}
    eng2._peak_price, eng2._open_time = {}, {}
    slow = _overshoot(eng2, path, check_every=40)     # ~one look per scan

    assert fast < slow, f'frequent checking did not help (fast={fast}, slow={slow})'
    assert fast <= 0.10, f'fast path still overshoots by {fast:.3f}pp'


# ── resilience ───────────────────────────────────────────────────────────────

def test_monitor_survives_a_bad_tick(engine):
    """An unsupervised book is worse than a logged error."""
    async def _drive():
        engine.EXIT_CHECK_SECONDS = 0.01
        _open(engine, _position())
        engine.live_prices[SYM] = 'not-a-number'      # type: ignore[assignment]
        task = asyncio.create_task(engine._exit_monitor_loop())
        await asyncio.sleep(0.08)
        alive = not task.done()
        task.cancel()
        return alive
    assert asyncio.run(_drive()), 'monitor died on a malformed price'


def test_monitor_closes_position_through_the_loop(engine):
    async def _drive():
        engine.EXIT_CHECK_SECONDS = 0.01
        _open(engine, _position())
        engine.live_prices[SYM] = 98.9
        task = asyncio.create_task(engine._exit_monitor_loop())
        await asyncio.sleep(0.08)
        task.cancel()
    asyncio.run(_drive())
    assert SYM not in engine.wallet.open_positions
