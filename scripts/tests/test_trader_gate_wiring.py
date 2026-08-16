"""
test_trader_gate_wiring.py — the desk is actually wired to the engine.

test_trader_gate.py proves the playbook decides correctly in isolation.  This
proves `_process_symbol` routes through it, that an approved plan becomes a real
position with the plan's OWN stop, and that a refusal leaves the book untouched
with an explanation attached.
"""

import asyncio
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import scripts.live_engine as le
from scripts.tests.characterise_process_symbol import (
    SYMBOL, _StubPredictor, _base_result, _build_engine,
)


def build(levels, ltf=('bull',), **over):
    eng = _build_engine(Path(tempfile.mkdtemp()))

    async def _levels(*a, **k):
        return list(levels)

    async def _ltf(*a, **k):
        return {'ltf_bull': 'bull' in ltf, 'ltf_bear': 'bear' in ltf}

    async def _tide(*a, **k):
        return over.get('tide', 'FLAT')

    eng._structural_levels = _levels     # type: ignore[assignment]
    eng._ltf_confirmation = _ltf         # type: ignore[assignment]
    eng._btc_tide = _tide                # type: ignore[assignment]
    eng._tide_strength = over.get('tide_strength', 0.0)
    return eng


async def drive(eng, result, min_quality=0.0):
    """Run one symbol through _process_symbol with the desk enabled.

    `min_quality` neutralises MIN_FIRE_QUALITY by default. These fixtures exist
    to pin the WIRING — plan becomes position, the plan's own stop is honoured,
    away-from-level rests instead of entering — and they do not set the fields
    SignalQualityFilter scores (adx, volume_zscore, confluence, funding, OI), so
    they score far below the live floor. Leaving the floor on would make every
    case here refuse for a reason none of them is about, and the wiring would go
    untested. The floor's own behaviour on this path is pinned separately, by
    test_the_quality_floor_blocks_the_desk_path below.
    """
    # scripts.engine.config is the single mutable source of truth for the flag;
    # scripts.live_engine only re-exports its value, so setting it there would
    # rebind a name the engine does not read.
    from scripts.engine import config as _cfg
    prev = _cfg.USE_TRADER_GATE
    prev_q = getattr(_cfg, 'MIN_FIRE_QUALITY', 0.0)
    _cfg.MIN_FIRE_QUALITY = min_quality
    _cfg.USE_TRADER_GATE = True
    loop = asyncio.get_running_loop()
    real = loop.run_in_executor

    def _inline(executor, fn, *args):
        fut = loop.create_future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:      # pragma: no cover - surfaced by the assert
            fut.set_exception(e)
        return fut

    loop.run_in_executor = _inline       # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            await eng._process_symbol(SYMBOL, _StubPredictor(result), asyncio.Semaphore(1))
    finally:
        loop.run_in_executor = real      # type: ignore[assignment]
        _cfg.USE_TRADER_GATE = prev
        _cfg.MIN_FIRE_QUALITY = prev_q
    return eng.last_signals.get(SYMBOL, {}), eng.wallet.open_positions.get(SYMBOL), buf.getvalue()


# NB: the engine resolves the live price itself (100.0 in this harness) — the
# 'price' key on the result dict is not what it trades off. To place a setup
# away from its level, move the LEVEL, not the price.
#
# A range-low long: price sitting on a well-touched support, real room overhead.
GOOD_LONG = dict(price=100.0, atr=1.0, atr_pct=1.0, support=99.8, resistance=112.0,
                 range_position=0.02, rsi=38.0, rsi_slope=0.5,
                 cdl_bull_reversal=True, trend_regime='RANGING',
                 p_buy=0.34, p_sell=0.33, p_hold=0.33)
GOOD_LEVELS = [(99.8, 4), (112.0, 4)]

# The same setup with its support 1.5 ATR below the live price — in reach, but
# not yet at the level, so it must rest rather than enter.
AWAY_LONG = dict(GOOD_LONG, support=98.5, range_position=0.11)
AWAY_LEVELS = [(98.5, 4), (112.0, 4)]


def test_an_approved_plan_opens_a_position_with_the_plans_own_stop():
    sig, pos, out = asyncio.run(drive(build(GOOD_LEVELS), _base_result(**GOOD_LONG)))
    assert pos is not None, f'the desk approved nothing:\n{out}'
    assert pos.side == 'BUY'
    plan = sig.get('trade_plan')
    assert plan, 'the plan was not published on the signal'
    assert pos.stop_loss == pytest.approx(plan['stop']), \
        'the engine placed a different stop from the one the payoff was approved on'
    assert sig['fire'] is True
    assert sig['setup_type']


def test_the_quality_floor_blocks_the_desk_path():
    """The setup the desk APPROVES is refused when quality is below the floor.

    This is the pairing that matters: GOOD_LONG opens a position in the test
    above, so anything that stops it here is the floor and nothing else.

    It has to be pinned on THIS path specifically. Suppressing result['fire'] and
    result['side'] upstream does not reach the book — result['tradeable'] is set
    True unconditionally, the call site checks only tradeable/price/existing,
    _classify never reads result['side'], and REQUIRE_MODEL_FIRE is False. A
    floor applied only at the fire flag would blank the published card and let
    the desk open the position regardless, which is exactly how the v83 rewrite
    left the hard vetoes.
    """
    sig, pos, out = asyncio.run(
        drive(build(GOOD_LEVELS), _base_result(**GOOD_LONG), min_quality=100.0))
    assert pos is None, f'a sub-floor signal opened a position:\n{out}'
    assert sig.get('fire') is False
    assert 'LOW_QUALITY_REFUSED' in out, 'the refusal was not attributed'
    assert 'quality' in (sig.get('structure_reason') or '').lower()


def test_the_floor_is_what_blocked_it_and_not_the_desk():
    """Guards the test above from rotting into a tautology: with the floor off,
    the identical fixture must still open a position."""
    _, pos, _ = asyncio.run(
        drive(build(GOOD_LEVELS), _base_result(**GOOD_LONG), min_quality=0.0))
    assert pos is not None, 'the fixture stopped opening a position for some other reason'


def test_a_refusal_opens_nothing_and_says_why():
    """Mid-range in a trend — no setup exists."""
    mid = dict(GOOD_LONG, price=105.0, range_position=0.5, support=99.8,
               resistance=112.0, trend_regime='TRENDING_BULL')
    sig, pos, out = asyncio.run(drive(build(GOOD_LEVELS), _base_result(**mid)))
    assert pos is None
    assert sig.get('fire') is False
    assert sig.get('structure_reason'), 'a refusal with no reason is the old system'
    assert 'NO TRADE' in out


def test_a_setup_away_from_its_level_becomes_a_working_order_not_a_position():
    sig, pos, out = asyncio.run(drive(build(AWAY_LEVELS), _base_result(**AWAY_LONG)))
    assert pos is None, f'a working order opened a position:\n{out}'
    assert sig.get('working_order') is True
    assert sig.get('pending_target') == pytest.approx(98.5)
    assert sig.get('expires_in_bars', 0) > 0, 'a working order with no clock is PENDING again'


def test_a_working_order_expires_instead_of_queueing_forever():
    """The single behaviour PENDING never had."""
    import time
    eng = build(AWAY_LEVELS)
    away = _base_result(**AWAY_LONG)

    sig, _, _ = asyncio.run(drive(eng, away))
    assert sig.get('working_order') is True
    key = next(iter(eng._working_orders))

    # Backdate the order past its 8-bar (8h) expiry and re-run the same scan.
    eng._working_orders[key] = time.time() - 9 * 3600
    sig2, pos2, out2 = asyncio.run(drive(eng, away))
    assert pos2 is None
    assert sig2.get('working_order') is not True
    assert 'EXPIRED' in out2
    assert not eng._working_orders, 'the expired order was not retired'


def test_the_model_cannot_fire_a_trade_the_desk_refuses():
    """A maximal-conviction model signal at a location with no setup."""
    hot = _base_result(**dict(GOOD_LONG, price=105.0, range_position=0.5,
                              fire=True, side='BUY', edge_score=100.0,
                              meta_confidence=1.0, p_buy=0.95, p_sell=0.02))
    sig, pos, _ = asyncio.run(drive(build(GOOD_LEVELS), hot))
    assert pos is None, 'model conviction bought its way past the desk'
    assert sig.get('fire') is False


def test_a_strong_counter_tide_short_never_reaches_the_book():
    """The 2026-07-20 basket, end to end through the engine."""
    short = _base_result(**dict(GOOD_LONG, price=109.5, support=99.8, resistance=110.0,
                                range_position=0.95, rsi=75.0, rsi_slope=-0.5,
                                cdl_bull_reversal=False, cdl_bear_reversal=True,
                                trend_regime='TRENDING_BULL',
                                p_sell=0.80, p_buy=0.05))
    eng = build([(110.0, 4), (99.8, 4), (95.0, 3)], ltf=('bear',),
                tide='UP', tide_strength=0.85)
    sig, pos, out = asyncio.run(drive(eng, short))
    assert pos is None, f'the basket trade reached the book:\n{out}'
    assert sig.get('fire') is False
