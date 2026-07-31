"""Regression tests for the v82 exit-ladder / cost / regime repairs.

These lock in the four behaviours whose absence made the live engine lose money
at a 60 % win rate:

  1. A TP1 tag followed by a pullback must NOT flatten the position
     (the deleted TP1_RECROSS capped every winner at 0.7R against a 1.0R stop).
  2. Break-even moves at TP2, not at TP1.
  3. Round-trip execution cost is charged on every close, full or partial.
  4. MarketRegimeDetector actually classifies (it was raising NameError on every
     call and silently returning the fail-safe RANGING regime).
"""
import time
from datetime import datetime, timezone

import pytest

from scripts.live_engine import (
    DynamicRiskEngine, LiveEngine, MarketRegimeDetector, Position, VirtualWallet,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _position(**kw) -> Position:
    base = dict(
        symbol='TEST/USDT', direction='LONG', side='BUY',
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
    """A LiveEngine with just enough wiring for _manage_exit, no network."""
    eng = LiveEngine.__new__(LiveEngine)          # bypass __init__ (loads models)
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
    # neutralise the side effects _manage_exit fires on close
    eng._save_track_record = lambda: None
    for attr in ('perf_tracker', 'drift_monitor', 'adaptive_orchestrator'):
        setattr(eng, attr, _Noop())
    return eng


class _Noop:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _drive(engine, pos, prices, side='BUY'):
    """Feed a price path through _manage_exit."""
    sym = pos.symbol
    engine.wallet.open_positions[sym] = pos
    engine._open_time[sym] = time.time()
    engine._peak_price[sym] = pos.entry_price
    result = {'atr': pos.atr, 'side': side, 'fire': False, 'edge_score': 0.0}
    for p in prices:
        engine.live_prices[sym] = p
        if sym not in engine.wallet.open_positions:
            break
        engine._manage_exit(sym, pos, result, p)
    return engine.wallet.open_positions.get(sym)


# ── 1. the deleted TP1_RECROSS ───────────────────────────────────────────────

def test_tp1_tag_then_pullback_keeps_position_open(engine):
    """The core v82 fix: tagging TP1 and falling back must not flatten."""
    pos = _position()
    still_open = _drive(engine, pos, [100.2, 101.3, 100.4])

    assert still_open is not None, (
        'position was flattened on a TP1 pullback — TP1_RECROSS is back'
    )
    assert engine._tp1_hit['TEST/USDT'] is True
    reasons = [t.exit_reason for t in engine.wallet.trade_history]
    assert reasons == ['TP1_PARTIAL'], reasons
    # only the TP1 slice was banked; the rest still rides
    assert still_open.position_value == pytest.approx(850.0, abs=1.0)


def test_stop_is_not_moved_to_breakeven_at_tp1(engine):
    """Break-even at 1R used to flatten runners on ordinary retracement."""
    pos = _position()
    _drive(engine, pos, [101.3])
    assert pos.stop_loss == 99.0, 'SL moved to break-even at TP1'


def test_breakeven_moves_at_tp2(engine):
    pos = _position()
    _drive(engine, pos, [101.3, 102.4])
    assert engine._tp2_hit['TEST/USDT'] is True
    assert pos.stop_loss == pos.entry_price


def test_short_side_tp1_pullback_also_holds(engine):
    pos = _position(direction='SHORT', side='SELL', stop_loss=101.0,
                    take_profit_1=99.0, take_profit_2=98.0, take_profit_3=95.0,
                    take_profit_4=92.0, take_profit_5=88.0)
    still_open = _drive(engine, pos, [99.8, 98.9, 99.4], side='SELL')
    assert still_open is not None
    assert [t.exit_reason for t in engine.wallet.trade_history] == ['TP1_PARTIAL']


# ── 2. the winner can now actually run ───────────────────────────────────────

def test_runner_reaches_tp3_instead_of_capping_at_tp1(engine):
    """Under the old ladder this path exited at TP1 for +1R on the whole size.

    Pullbacks here stay inside the ATR x TRAIL_MULTIPLIER band, so the runner
    survives to the structural target.
    """
    pos = _position()                      # atr 0.5 -> trail distance 0.5
    _drive(engine, pos, [101.2, 100.6, 102.3, 102.0, 105.4])
    reasons = [t.exit_reason for t in engine.wallet.trade_history]
    assert 'TP1_PARTIAL' in reasons and 'TP2_PARTIAL' in reasons
    assert 'TP3_PARTIAL' in reasons, f'runner never reached TP3: {reasons}'


def test_trailing_stop_exits_on_a_pullback_wider_than_the_trail(engine):
    """The trail is the profit protection that replaced the re-cross exits."""
    pos = _position()                      # trail distance = 1.0 x atr = 0.5
    still_open = _drive(engine, pos, [101.2, 102.3, 101.7])   # 1.2 atr pullback
    assert still_open is None
    assert engine.wallet.trade_history[-1].exit_reason == 'TRAILING_STOP'


def test_trailing_floor_never_gives_back_more_than_tp1(engine):
    """Floor is TP1, so a runner that reverses still banks >= 1R on the rest."""
    pos = _position()
    _drive(engine, pos, [101.2, 102.1, 99.0])
    last = engine.wallet.trade_history[-1]
    assert last.exit_reason == 'TRAILING_STOP'
    # ratcheted trail (peak 102.1 - 0.5) sits above the floor here
    assert last.exit_price >= pos.take_profit_1


def test_trailing_floor_binds_when_the_trail_would_sit_below_tp1(engine):
    """With a wide ATR the raw trail falls under TP1; the floor must hold it."""
    pos = _position(atr=1.5)               # trail distance 1.5 > TP2 - TP1
    _drive(engine, pos, [101.2, 102.1, 95.0])
    last = engine.wallet.trade_history[-1]
    assert last.exit_reason == 'TRAILING_STOP'
    assert last.exit_price == pytest.approx(pos.take_profit_1)


# ── 3. execution costs ───────────────────────────────────────────────────────

def test_round_trip_cost_is_charged_on_full_close(tmp_path):
    w = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'a.json')
    pos = _position()
    w.open_trade(pos)
    rec = w.close_trade('TEST/USDT', 101.0, 'TP_TEST')
    gross = 1.0                                  # 100 -> 101 on a LONG
    assert rec.pnl_pct == pytest.approx(gross - w.round_trip_cost_pct(), abs=1e-9)
    assert w.round_trip_cost_pct() == pytest.approx(0.10)


def test_round_trip_cost_is_charged_on_partial_close(tmp_path):
    w = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'b.json')
    pos = _position()
    w.open_trade(pos)
    rec = w.partial_close_trade('TEST/USDT', 101.0, 'TP1_PARTIAL', 0.15)
    assert rec.pnl_pct == pytest.approx(1.0 - w.round_trip_cost_pct(), abs=1e-9)


def test_a_flat_scratch_trade_is_a_loss_after_costs(tmp_path):
    """Exiting at entry is not free — this is what the old wallet pretended."""
    w = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'c.json')
    w.open_trade(_position())
    rec = w.close_trade('TEST/USDT', 100.0, 'BREAK_EVEN')
    assert rec.pnl_pct < 0
    assert rec.outcome == 'LOSS'


# ── 4. partial sizing off the original allocation ────────────────────────────

def test_partials_are_fractions_of_the_original_allocation(tmp_path):
    w = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'd.json')
    w.open_trade(_position())
    r1 = w.partial_close_trade('TEST/USDT', 101.0, 'TP1_PARTIAL', 0.15)
    r2 = w.partial_close_trade('TEST/USDT', 102.0, 'TP2_PARTIAL', 0.25)
    assert r1.position_value == pytest.approx(150.0)
    # old behaviour closed 0.25 * 850 = 212.50 here
    assert r2.position_value == pytest.approx(250.0)
    assert w.open_positions['TEST/USDT'].position_value == pytest.approx(600.0)


def test_partial_never_closes_more_than_remains(tmp_path):
    w = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'e.json')
    w.open_trade(_position())
    for pct in (0.15, 0.25, 0.25, 0.15, 0.20, 0.20):
        w.partial_close_trade('TEST/USDT', 101.0, 'TP_PARTIAL', pct)
    assert w.open_positions['TEST/USDT'].position_value >= 0


# ── 5. payoff geometry ───────────────────────────────────────────────────────

def test_tp1_is_one_R_not_zero_seven_R():
    """0.7R against a 1.0R stop needed a 58.8 % win rate just to break even."""
    re_ = DynamicRiskEngine()
    s = re_.calculate_stops(price=100.0, side='BUY', atr=1.0)
    risk = 100.0 - s['sl']
    assert (s['tp1'] - 100.0) / risk == pytest.approx(1.0, abs=1e-6)
    assert re_.TP1_MULTIPLIER == 1.0


# ── 6. the regime detector repair ────────────────────────────────────────────

def test_regime_detector_does_not_fall_back_on_a_screaming_trend():
    """It returned RANGING/0.4 for every input while _detect raised NameError."""
    d = MarketRegimeDetector()
    r = d.detect({'adx': 60.2, 'trend_regime': 'STRONG_UP', 'atr_pct': 0.85,
                  'market_bias': 'BULLISH', 'rsi': 65, 'macd_signal': 'BULLISH',
                  'volume_strength': 'HIGH', 'volume_zscore': 2.0})
    assert r.regime != 'RANGING'
    assert r.confidence != 0.4


def test_detect_raises_nothing_directly():
    """Guards the fail-safe in detect() from hiding a future paste accident."""
    d = MarketRegimeDetector()
    d._detect({'adx': 60.2, 'trend_regime': 'STRONG_UP', 'rsi': 65})


def test_regime_detector_discriminates_between_conditions():
    d = MarketRegimeDetector()
    trend = d.detect({'adx': 60, 'trend_regime': 'STRONG_UP', 'atr_pct': 0.9,
                      'market_bias': 'BULLISH', 'rsi': 65,
                      'macd_signal': 'BULLISH', 'volume_zscore': 2.0})
    trap = d.detect({'adx': 8, 'trend_regime': 'RANGING', 'atr_pct': 0.4,
                     'rsi': 50, 'volatility_regime': 'LOW',
                     'volume_strength': 'BELOW_AVERAGE', 'volume_zscore': -1.2})
    assert trend.regime != trap.regime
    assert trap.trade_allowed is False
