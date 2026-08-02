"""
test_plan_target_handoff.py — the plan's TARGET must be the target that gets placed.

The mirror of test_plan_stop_handoff.py.  `sl_override` closed the gap on the
risk leg in v83, but the reward leg stayed open: TraderGate cleared the trade on
`MIN_NET_R` measured to `plan.target`, and the engine then built its TP ladder
from the rolling S/R — a different structure set.  `plan.target` only reached
`sig['suggested_tp']`, a display field, so the payoff floor was decoration on
the side it was actually floor-ing.  `tp_override` closes it; these tests hold
it shut.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.live_engine import DynamicRiskEngine  # noqa: E402


@pytest.fixture
def eng():
    return DynamicRiskEngine()


def test_long_target_override_is_used_verbatim(eng):
    out = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                              support=99.0, resistance=101.0,
                              sl_override=99.1, tp_override=104.0)
    assert out['tp3'] == pytest.approx(104.0)


def test_short_target_override_is_used_verbatim(eng):
    out = eng.calculate_stops(price=100.0, side='SELL', atr=1.0,
                              support=99.0, resistance=101.0,
                              sl_override=100.9, tp_override=96.0)
    assert out['tp3'] == pytest.approx(96.0)


def test_reported_rr_is_measured_to_the_plans_target(eng):
    """The R:R the engine reports must be the one the gate approved.

    Risk is read back off the result rather than assumed: the v84 rule that
    clears a stop past the level it defends can legitimately widen it, and this
    test is about the REWARD leg.
    """
    out = eng.calculate_stops(price=100.0, side='BUY', atr=1.0,
                              support=99.0, resistance=101.0,
                              sl_override=99.0, tp_override=104.0)
    assert out['reward'] == pytest.approx(4.0)          # measured to the plan target...
    assert out['reward'] != pytest.approx(1.0)          # ...not to `resistance`
    # risk_reward is rounded to 3dp on the way out.
    assert out['risk_reward'] == pytest.approx(out['reward'] / out['risk'], abs=1e-3)


@pytest.mark.parametrize('side,target', [('BUY', 101.6), ('SELL', 98.4)])
def test_a_near_target_is_not_pushed_out_by_the_monotonic_clamp(eng, side, target):
    """The regression that made the first fix a no-op.

    TraderGate approves setups from 1.6R gross, so the fixed 2.0R rung lands
    PAST the objective on a large share of real trades.  The old clamp
    (`tp3 = max(tp3, tp2 + 0.3R)`) then shoved tp3 beyond the level — re-inventing
    the unreachable target stage 3 exists to reject.  The banking rungs must
    compress inside the objective instead.
    """
    out = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                              support=99.0, resistance=101.0,
                              sl_override=(99.0 if side == 'BUY' else 101.0),
                              tp_override=target)
    assert out['tp3'] == pytest.approx(target)
    if side == 'BUY':
        assert 100.0 < out['tp1'] < out['tp2'] < out['tp3']
    else:
        assert 100.0 > out['tp1'] > out['tp2'] > out['tp3']


def test_a_target_override_does_not_move_the_stop(eng):
    """The two overrides are independent; the reward fix must not touch risk."""
    base = eng.calculate_stops(price=100.0, side='SELL', atr=1.0,
                               support=95.0, resistance=100.2, sl_override=101.1)
    with_t = eng.calculate_stops(price=100.0, side='SELL', atr=1.0,
                                 support=95.0, resistance=100.2, sl_override=101.1,
                                 tp_override=96.0)
    assert with_t['sl'] == pytest.approx(base['sl'])
    assert with_t['risk'] == pytest.approx(base['risk'])


@pytest.mark.parametrize('side,bad', [('BUY', 95.0), ('SELL', 105.0)])
def test_a_target_on_the_wrong_side_of_price_is_ignored(eng, side, bad):
    """A malformed override must fall back to the legacy structural target."""
    legacy = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                                 support=99.0, resistance=101.0)
    out = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                              support=99.0, resistance=101.0, tp_override=bad)
    assert out['tp3'] == pytest.approx(legacy['tp3'])


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_zero_override_is_a_no_op(eng, side):
    """Every pre-v85 caller passes nothing; their geometry must be untouched."""
    legacy = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                                 support=99.0, resistance=101.0)
    same = eng.calculate_stops(price=100.0, side=side, atr=1.0,
                               support=99.0, resistance=101.0, tp_override=0.0)
    assert same == legacy
