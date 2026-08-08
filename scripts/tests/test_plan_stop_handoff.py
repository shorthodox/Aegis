"""
test_plan_stop_handoff.py — the plan's stop must be the stop that gets placed.

TraderGate approves a trade on a net R:R computed from a specific stop.  If the
engine then re-derives its own stop, the number the trade was approved on is not
the number the trade has, and the payoff stage is decoration.  `sl_override`
exists to close that gap; these tests hold it shut.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.live_engine import DynamicRiskEngine  # noqa: E402


@pytest.fixture
def eng():
    return DynamicRiskEngine()


def test_long_stop_override_is_used_verbatim(eng):
    out = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                              support=99.8, resistance=110.0,
                              sl_override=98.9)
    assert out['sl'] == pytest.approx(98.9)
    assert out['risk'] == pytest.approx(1.1)


def test_short_stop_override_is_used_verbatim(eng):
    out = eng.calculate_stops(price=100.0, side='SELL', atr=1.0,
                              support=90.0, resistance=100.2,
                              sl_override=101.1)
    assert out['sl'] == pytest.approx(101.1)
    assert out['risk'] == pytest.approx(1.1)


def test_the_tp_ladder_derives_from_the_overridden_risk(eng):
    """The REAL stop must drive risk and R:R, even though the rungs no longer
    derive from it.

    The ladder is priced in percent of entry now, so TP1 is no longer 1R off the
    stop. What this test still protects is the thing the override exists for:
    `risk` is measured from the stop the gate supplied, not from one the engine
    invented, so the reported R:R is the trade's real R:R.
    """
    out = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                              support=99.8, resistance=110.0,
                              sl_override=98.9)
    assert out['sl'] == pytest.approx(98.9), 'the plan stop was re-derived'
    assert out['risk'] == pytest.approx(100.0 - 98.9)
    assert out['tp1'] == pytest.approx(100.0 * (1 + eng.TP_LADDER_PCT[0] / 100.0))
    assert out['tp2'] > out['tp1'] and out['tp3'] > out['tp2']


def test_an_override_on_the_wrong_side_of_price_is_ignored(eng):
    """A malformed override must not invert the trade's geometry."""
    for side, bad in (('BUY', 101.0), ('SELL', 99.0)):
        out = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                                  support=99.0, resistance=101.0,
                                  sl_override=bad)
        if side == 'BUY':
            assert out['sl'] < 100.0
        else:
            assert out['sl'] > 100.0
        assert out['risk'] > 0


def test_no_override_leaves_the_legacy_geometry_untouched(eng):
    """The v80..v82 path must behave exactly as before when the flag is off."""
    before = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                                 support=99.0, resistance=110.0)
    after = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                                support=99.0, resistance=110.0,
                                sl_override=0.0)
    assert before == after
