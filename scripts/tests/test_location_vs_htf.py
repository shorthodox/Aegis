"""Location says which side is on offer; the weekly and daily say if it is allowed.

The doctrine this pins:

    at resistance + higher timeframe bearish  -> only the SHORT is on offer
    at support    + higher timeframe bullish  -> only the LONG  is on offer
    overbought at the highs -> never a long
    oversold at the lows    -> never a short

Every setup except BREAK_RETEST already trades with its location — the fades
sell highs and buy lows, the pullbacks buy dips in a bull and sell rallies in a
bear. BREAK_RETEST deliberately does the opposite, buying the top of the range
on the argument that broken resistance is now support. That argument holds only
while the higher timeframe agrees.

Reference failure, ADA/USDT 2026-08-05 (published as "STRONG", LOW RISK):

    entry 0.197600, support 0.189900, resistance 0.199000  -> rp 0.846
    RSI 62, weekly BEAR, daily BULL, regime VOLATILE_COMPRESSION
    resistance_broken_recent = True

    _classify -> BREAK_RETEST BUY, because `bear` was False under a
    VOLATILE_COMPRESSION label, so nothing stopped a long into resistance
    beneath a bearish weekly. Gate 1.7 saw it and said "1 of 2 opposes this
    long (advisory)".
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import TraderGate


# the ADA signal, as published
ADA_RP = (0.197600 - 0.189900) / (0.199000 - 0.189900)   # 0.846


def _result(**over):
    r = {
        'range_position': ADA_RP,
        'rsi': 62.0,
        'macro_weekly': -1.0,      # weekly BEAR
        'macro_daily':  1.0,       # daily BULL
        'resistance_broken_recent': True,
        'support_broken_recent': False,
        'p_buy': 0.512, 'p_sell': 0.488,
    }
    r.update(over)
    return r


def test_the_ada_geometry_is_at_the_top_of_the_range():
    assert ADA_RP == pytest.approx(0.846, abs=0.002)
    assert ADA_RP >= TG.RANGE_EDGE_HIGH


def test_classify_still_offers_the_break_retest_long():
    """_classify is unchanged — the refusal is a separate, later stage."""
    setup, side, _ = TraderGate._classify(_result(), 'VOLATILE_COMPRESSION', 0.9)
    assert (setup, side) == (TG.SETUP_BREAK_RETEST, 'BUY')


def test_long_into_resistance_under_a_bearish_weekly_is_refused():
    reason = TraderGate._counter_location_refusal('BUY', ADA_RP, 62.0, _result())
    assert reason, 'the ADA long was permitted again'
    assert 'weekly' in reason and 'short' in reason


def test_short_into_support_under_a_bullish_weekly_is_refused():
    r = _result(range_position=0.12, macro_weekly=1.0, macro_daily=1.0,
                resistance_broken_recent=False, support_broken_recent=True)
    reason = TraderGate._counter_location_refusal('SELL', 0.12, 45.0, r)
    assert reason and 'weekly' in reason and 'long' in reason


@pytest.mark.parametrize('w,d,blocked', [
    (-1.0,  1.0, True),    # the ADA case: weekly alone is enough
    ( 1.0, -1.0, True),    # daily alone is enough
    (-1.0, -1.0, True),    # both
    ( 1.0,  1.0, False),   # both agree with the long -> retest is allowed
    ( 0.0,  0.0, False),   # no lean either way -> not this stage's call
])
def test_either_timeframe_can_veto_a_counter_location_long(w, d, blocked):
    r = _result(macro_weekly=w, macro_daily=d)
    reason = TraderGate._counter_location_refusal('BUY', ADA_RP, 62.0, r)
    assert bool(reason) is blocked, f'weekly {w}, daily {d}: {reason!r}'


def test_overbought_at_the_highs_is_refused_even_with_a_clean_htf():
    r = _result(macro_weekly=1.0, macro_daily=1.0)
    reason = TraderGate._counter_location_refusal('BUY', ADA_RP, 75.0, r)
    assert reason and 'overbought' in reason


def test_oversold_at_the_lows_is_refused_even_with_a_clean_htf():
    r = _result(macro_weekly=-1.0, macro_daily=-1.0, range_position=0.10)
    reason = TraderGate._counter_location_refusal('SELL', 0.10, 25.0, r)
    assert reason and 'oversold' in reason


# ── the rule must not touch setups that already agree with their location ────

@pytest.mark.parametrize('side,rp,rsi', [
    ('BUY',  0.10, 30.0),   # fade / exhaustion buy at the lows
    ('BUY',  0.40, 45.0),   # trend pullback buy in a bull
    ('SELL', 0.90, 70.0),   # fade / exhaustion sell at the highs
    ('SELL', 0.60, 55.0),   # trend pullback sell in a bear
    ('BUY',  0.50, 50.0),   # mid-range: not this stage's business
    ('SELL', 0.50, 50.0),
])
def test_side_that_agrees_with_its_location_is_untouched(side, rp, rsi):
    for w in (-1.0, 0.0, 1.0):
        for d in (-1.0, 0.0, 1.0):
            r = _result(range_position=rp, rsi=rsi, macro_weekly=w, macro_daily=d)
            assert TraderGate._counter_location_refusal(side, rp, rsi, r) is None, (
                f'{side} at rp {rp} was refused with weekly {w} / daily {d} — '
                f'this stage should only judge counter-location entries'
            )


def test_break_retest_long_survives_when_the_htf_agrees():
    """The setup is not banned — it is conditioned. This is the case it is for."""
    r = _result(macro_weekly=1.0, macro_daily=1.0, rsi=55.0)
    assert TraderGate._counter_location_refusal('BUY', ADA_RP, 55.0, r) is None


def test_refusal_is_wired_into_evaluate_with_its_own_stage():
    import inspect
    src = inspect.getsource(TraderGate.evaluate)
    assert '_counter_location_refusal' in src, 'the check is not wired into evaluate'
    assert "_reject('location'" in src, (
        'the refusal must carry its own stage name — an unattributed HOLD is '
        'what made the old guard chain impossible to debug'
    )
