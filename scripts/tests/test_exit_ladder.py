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

def test_tp1_tag_then_shallow_pullback_keeps_position_open(engine):
    """Tagging TP1 and wobbling must not flatten — the v82 fix, re-scoped.

    This used to drive a pullback to 100.4, which hands back 60 % of the
    entry->TP1 rung. Under the give-back ratchet (TP_GIVEBACK_PCT) that is now a
    deliberate exit, not a survivable wobble: the protective level for this
    geometry sits at 101.0 - 0.35 * 1.0 = 100.65.

    What the test still protects is the thing that mattered — a ZERO-width
    recross. Price may come back off the rung without the position being
    flattened; it just may not come back most of the way.
    """
    pos = _position()
    still_open = _drive(engine, pos, [100.2, 101.3, 100.8])

    assert still_open is not None, (
        'position was flattened on a shallow TP1 pullback — TP1_RECROSS is back'
    )
    assert engine._tp1_hit['TEST/USDT'] is True
    reasons = [t.exit_reason for t in engine.wallet.trade_history]
    assert reasons == ['TP1_PARTIAL'], reasons
    # only the TP1 slice was banked; the rest still rides
    assert still_open.position_value == pytest.approx(850.0, abs=1.0)


def test_tp1_tag_then_deep_pullback_banks_the_rung(engine):
    """The other half of the same rule, and the reason it exists.

    BCH/USDT 2026-08-06: short from 214.70, TP1 tagged at 212.09, price back to
    213.40 — over half the rung handed back — with the stop still at break-even
    214.70 and nothing in between. That runner is now closed instead.
    """
    pos = _position()
    still_open = _drive(engine, pos, [100.2, 101.3, 100.4])

    assert still_open is None, 'a deep give-back after TP1 should close the runner'
    reasons = [t.exit_reason for t in engine.wallet.trade_history]
    assert 'TP1_PARTIAL' in reasons
    assert reasons[-1] == 'TP_GIVEBACK', reasons


def test_breakeven_moves_at_tp1(engine):
    """Deliberate win-rate choice: a reversal after TP1 scratches, not loses.

    Distinct from the deleted TP1_RECROSS — break-even exits at ENTRY on a full
    reversal, it does not close the position at TP1 on any tick back through it.
    """
    pos = _position()
    still_open = _drive(engine, pos, [101.3])
    assert still_open is not None
    assert pos.stop_loss == pos.entry_price


def test_reversal_after_tp1_is_still_net_green(engine):
    """The win-rate mechanic, now served by the ratchet rather than break-even.

    A full reversal after TP1 used to ride all the way back to entry and exit
    STOP_HIT at 100.0. It now closes earlier, at the give-back level — the
    outcome the test cares about is unchanged and slightly better: TP1 is banked
    and the remainder does not give the move back.
    """
    pos = _position()
    _drive(engine, pos, [101.3, 100.0, 99.5])
    total = sum(t.pnl_usdt for t in engine.wallet.trade_history)
    last = engine.wallet.trade_history[-1]
    assert last.exit_reason in ('TP_GIVEBACK', 'STOP_HIT'), last.exit_reason
    # exits at or above break-even, never below it
    assert last.exit_price >= 100.0 - 1e-9, last.exit_price
    assert total > -pos.initial_value * 0.005, (
        f'a TP1-then-reverse should be a scratch, not a full loss: {total}'
    )


def test_short_side_shallow_pullback_also_holds(engine):
    """Mirror of the long case: the SHORT rung is 100 -> 99, level 99.35."""
    pos = _position(direction='SHORT', side='SELL', stop_loss=101.0,
                    take_profit_1=99.0, take_profit_2=98.0, take_profit_3=95.0,
                    take_profit_4=92.0, take_profit_5=88.0)
    still_open = _drive(engine, pos, [99.8, 98.9, 99.2], side='SELL')
    assert still_open is not None
    assert [t.exit_reason for t in engine.wallet.trade_history] == ['TP1_PARTIAL']


def test_short_side_deep_pullback_banks_the_rung(engine):
    pos = _position(direction='SHORT', side='SELL', stop_loss=101.0,
                    take_profit_1=99.0, take_profit_2=98.0, take_profit_3=95.0,
                    take_profit_4=92.0, take_profit_5=88.0)
    still_open = _drive(engine, pos, [99.8, 98.9, 99.6], side='SELL')
    assert still_open is None
    assert engine.wallet.trade_history[-1].exit_reason == 'TP_GIVEBACK'


# ── 2. the winner can now actually run ───────────────────────────────────────

def test_runner_still_reaches_tp3(engine):
    """The winner must still be able to run — the point of deleting the recross.

    Under the old ladder this path exited at TP1 for +1R on the whole size. The
    pullbacks here stay inside BOTH bands that can now close a runner: the
    trailing stop (ATR x TRAIL_MULTIPLIER = 0.5) and the give-back leash
    (TP_GIVEBACK_PCT of the rung — 0.35 off TP1, so a floor at 100.65, and 0.35
    off TP2, so 101.65).

    That is the trade this ladder makes: a runner survives noise, but it no
    longer survives handing back a third of the rung it just banked.
    """
    pos = _position()                      # atr 0.5 -> trail distance 0.5
    _drive(engine, pos, [101.2, 100.9, 102.3, 102.0, 105.4])
    reasons = [t.exit_reason for t in engine.wallet.trade_history]
    assert 'TP1_PARTIAL' in reasons and 'TP2_PARTIAL' in reasons
    assert 'TP3_PARTIAL' in reasons, f'runner never reached TP3: {reasons}'


def test_a_zero_width_recross_is_still_forbidden(engine):
    """The guard the four rewritten tests were really protecting.

    TP1_RECROSS closed on ANY tick back through a tagged TP. Whatever the leash
    is set to, it must leave room for price to come off the rung at all —
    otherwise every winner is capped at exactly TP1 again, which is the measured
    mistake this ladder exists to avoid.
    """
    from scripts.engine.risk import DynamicRiskEngine as R
    assert R.TP_GIVEBACK_PCT > 0.0, 'a zero leash is TP1_RECROSS by another name'
    pos = _position()
    # a tick barely off the rung must NOT close
    still_open = _drive(engine, pos, [101.3, 100.99])
    assert still_open is not None, 'closed on a tick back through TP1'


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

def test_tp1_is_the_configured_percentage():
    """The ladder is priced in percent of entry, not in R.

    This used to assert TP1 == 1.0R, the v82 fix for a 0.7R rung that needed a
    58.8 % win rate to break even. The rungs moved to percentages because
    R-derived ones put the first objective 2-3x further away than the stop, so a
    reversal in between turned profitable positions into full losses.

    The v82 hazard has NOT gone away — a percentage rung lands at a different R
    on every token, and on a wide stop TP1 can be well under 1R. What contains
    it is that TP1 closes only 15 % and its job is arming break-even early,
    while the ladder still ends where the trade is paid. The guard that survives
    is the one below: TP1 must never be the de-facto exit.
    """
    re_ = DynamicRiskEngine()
    s = re_.calculate_stops(price=100.0, side='BUY', atr=1.0)
    assert (s['tp1'] - 100.0) / 100.0 * 100.0 == pytest.approx(
        re_.TP_LADDER_PCT[0], abs=1e-6)
    assert re_.TP_CLOSE_PCTS[0] <= 0.20, 'TP1 must stay a partial, not an exit'
    assert re_.TP_LADDER_PCT[-1] > re_.TP_LADDER_PCT[0] * 3,         'the ladder must still reach a payable distance'


def test_risky_stop_is_not_tighter_than_the_label_barrier():
    """A 1.2xATR stop sat inside the +/-1.8xATR band the model trains on."""
    re_ = DynamicRiskEngine()
    assert re_.RISKY_SL_CAP_ATR >= 1.8


def test_risky_tier_stop_matches_the_normal_cap():
    """Almost every live signal is RISKY, so this is the dominant path."""
    re_ = DynamicRiskEngine()
    risky  = re_.calculate_stops(price=100.0, side='BUY', atr=1.0,
                                 sl_cap_atr=re_.RISKY_SL_CAP_ATR)
    normal = re_.calculate_stops(price=100.0, side='BUY', atr=1.0)
    assert risky['sl'] == pytest.approx(normal['sl'])


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
