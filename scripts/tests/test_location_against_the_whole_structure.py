"""A long is not taken at the top of the chart, whatever the local band says.

Reported 2026-08-23 with ETH/USDT on screen: "engine gave this signal when the
market is at the top most resistance of the chart. We need long when market is
at the bottom, mid or 80% zone between the top most and bottom most support.
Vice versa for short."

The gate thought the trade was low in its range and the chart said the highs,
because they were measuring different ranges:

    gate   support 2372.17  resistance 2518.31  ->  rp 0.275   (a 6% band)
    chart  Sup 1876/1912    Res 2549           ->  rp 0.796   (a 36% range)

Both are honest. rp is computed against the NEAREST support and resistance, and
inside that 6% slice price really was near the bottom — but the location a trade
is taken at is its place in the STRUCTURE, which is what the subscriber sees.

Location is now judged against the full span of _structural_levels: top-most
level down to bottom-most. A long is refused in the top fifth of that, a short
in the bottom fifth.

NOT the nearest-bracketing-levels idea tried and reverted on 2026-08-21: that
measured the gap between two ADJACENT levels, a local position, and turned a
price at the top of its range into rp 0.009. A full span cannot do that.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import ACTION_REJECT, TraderGate

from scripts.tests.test_trader_gate import mk, run, TURNED_UP, TURNED_DOWN


# the reported ETH structure
ETH_LEVELS = [(1876.01, 3), (1912.44, 3), (2263.43, 4), (2372.17, 4),
              (2422.52, 4), (2444.30, 4), (2518.31, 4), (2549.40, 3)]


# -- the helper ---------------------------------------------------------------

def test_it_measures_the_whole_span_not_a_local_gap():
    rp = TraderGate._structural_rp(
        2412.29, ETH_LEVELS, {'support': 2372.17, 'resistance': 2518.31})
    assert rp == pytest.approx(0.796, abs=0.005), (
        'the reported ETH long measured somewhere other than 79.6% of its structure'
    )


def test_the_local_band_would_have_said_the_opposite():
    """The whole point: 0.275 by the local band, 0.796 by the structure."""
    local = (2412.29 - 2372.17) / (2518.31 - 2372.17)
    assert local == pytest.approx(0.275, abs=0.005)
    struct = TraderGate._structural_rp(
        2412.29, ETH_LEVELS, {'support': 2372.17, 'resistance': 2518.31})
    assert struct > local + 0.4, 'the two readings no longer diverge'


def test_the_extremes_are_what_they_should_be():
    assert TraderGate._structural_rp(1876.01, ETH_LEVELS, {}) == pytest.approx(0.0, abs=1e-6)
    assert TraderGate._structural_rp(2549.40, ETH_LEVELS, {}) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('levels', [[], [(100.0, 4)], None])
def test_too_little_structure_returns_none(levels):
    """Without a range there is nothing to be a fraction of; location falls back
    to the local checks rather than being guessed."""
    assert TraderGate._structural_rp(100.0, levels, {}) is None


def test_a_degenerate_span_returns_none():
    assert TraderGate._structural_rp(100.0, [(100.0, 4), (100.0, 4)], {}) is None


# -- the refusal --------------------------------------------------------------

def _long_at(frac):
    """A BUY that the LOCAL band likes, placed at `frac` of a wide structure."""
    lo, hi = 100.0, 200.0
    price = lo + frac * (hi - lo)
    d = mk(price=price, atr=2.0, support=price - 1.0, resistance=price + 9.0,
           rsi=45.0, **TURNED_UP)
    return d, [(lo, 3), (price - 1.0, 4), (price + 9.0, 4), (hi, 3)]


def test_a_long_in_the_top_fifth_is_refused():
    d, lv = _long_at(0.90)
    plan = run(d, regime='RANGING', levels=lv)
    assert plan.action == ACTION_REJECT and plan.stage == 'location'
    assert 'whole structure' in plan.reason


def test_a_long_in_the_middle_is_allowed_through_location():
    d, lv = _long_at(0.45)
    plan = run(d, regime='RANGING', levels=lv)
    assert not (plan.action == ACTION_REJECT and plan.stage == 'location')


def test_a_long_at_the_bottom_is_allowed_through_location():
    d, lv = _long_at(0.10)
    plan = run(d, regime='RANGING', levels=lv)
    assert not (plan.action == ACTION_REJECT and plan.stage == 'location')


def _short_at(frac):
    lo, hi = 100.0, 200.0
    price = lo + frac * (hi - lo)
    d = mk(price=price, atr=2.0, support=price - 9.0, resistance=price + 1.0,
           rsi=55.0, **TURNED_DOWN)
    return d, [(lo, 3), (price - 9.0, 4), (price + 1.0, 4), (hi, 3)]


def test_a_short_in_the_bottom_fifth_is_refused():
    """The mirror the request asked for explicitly."""
    d, lv = _short_at(0.10)
    plan = run(d, regime='RANGING', levels=lv)
    assert plan.action == ACTION_REJECT and plan.stage == 'location'
    assert 'whole structure' in plan.reason


def test_a_short_in_the_upper_zone_is_allowed_through_location():
    d, lv = _short_at(0.75)
    plan = run(d, regime='RANGING', levels=lv)
    assert not (plan.action == ACTION_REJECT and plan.stage == 'location')


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(TG, 'USE_STRUCTURAL_LOCATION', False)
    d, lv = _long_at(0.90)
    plan = run(d, regime='RANGING', levels=lv)
    assert not (plan.action == ACTION_REJECT and plan.stage == 'location')


def test_the_thresholds_are_the_ones_requested():
    assert TG.STRUCTURAL_RP_HIGH == 0.80
    assert TG.STRUCTURAL_RP_LOW == 0.20


def test_the_runtime_knobs_are_registered():
    import main
    for k in ('structural_rp_high', 'structural_rp_low', 'use_structural_location'):
        assert k in main._TUNABLES, k
