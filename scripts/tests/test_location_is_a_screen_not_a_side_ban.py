"""The structural location screen must judge a span that actually trades.

Reported 2026-08-23: "not even armed signals now".

The engine was healthy — ~58s cycle, 59 tokens, refusing every one. The funnel:

    NO TRADE (setup)     93   local rp says "uptrend at its highs" (0.81-0.96)
    NO TRADE (location)  36   ALL shorts, "short at 1-19% of the whole structure"

Both true at once: these tokens were bouncing off multi-week lows with real
momentum, so the local 24h window read "at the highs" while the deep structure
read "at the lows". Longs died at stage 1, shorts died at stage 1a, nothing
could arm.

The location half of that is a defect. _structural_levels pools daily pivots
from 1d/1500 — about FOUR YEARS — so "the whole structure" is anchored to a
cycle top that no longer trades. Measured across 20 tokens:

    span         median srp   <= 0.20 (short refused)   both sides open
    ~4 years        0.06            16/20                     20%
    ~1 year         0.20            11/20                     45%
    ~6 months       0.55             1/20                     65%
    ~4 months       0.70             1/20                     60%

At the full span the short side is refused fleet-wide for as long as price sits
under an old cycle high. That is not a location screen, it is a standing ban on
one direction, and it is not what the desk asked for: "we need long when market
is at the bottom, mid or 80% zone ... vice versa for short."

Stops and targets still use the full structure — the nearest significant level
to stand behind is exactly where deep history earns its keep. Only the location
screen is bounded.
"""
import pytest

import scripts.engine.engine as E
from src.trading import trader_gate as TG


def test_the_location_span_is_bounded():
    assert hasattr(E.LiveEngine, 'LOCATION_DAY_BARS')
    assert 0 < E.LiveEngine.LOCATION_DAY_BARS <= 400, (
        'the location span is back to reading a cycle top from years ago, '
        'which bans one side of the book fleet-wide'
    )


def test_the_deep_structure_is_still_deep_for_stops():
    """Bounding location must not shorten the levels used for stops/targets."""
    import inspect
    src = inspect.getsource(E.LiveEngine._scan_symbol) if hasattr(E.LiveEngine, '_scan_symbol') else ''
    body = inspect.getsource(E.LiveEngine)
    assert 'day_bars=self.LOCATION_DAY_BARS' in body
    # the unbounded call must still exist
    assert '_structural_levels(symbol, price, atr)' in body, (
        'the deep, unbounded level read is gone — stops and targets lost their '
        'history'
    )


def test_the_gate_takes_a_separate_location_span():
    import inspect
    sig = inspect.signature(TG.TraderGate.evaluate)
    assert 'location_levels' in sig.parameters


def test_location_falls_back_to_levels_when_not_given():
    """Existing callers and tests keep working unchanged."""
    import inspect
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'location_levels if location_levels is not None else levels' in src


# -- the screen must still enforce the rule it was built for ------------------

def _srp(price, lv):
    return TG.TraderGate._structural_rp(price, [(x, 3) for x in lv], {})


def test_a_short_at_the_lows_is_still_refused():
    assert _srp(10.5, [10.0, 50.0]) <= TG.STRUCTURAL_RP_LOW


def test_a_long_at_the_highs_is_still_refused():
    assert _srp(49.0, [10.0, 50.0]) >= TG.STRUCTURAL_RP_HIGH


def test_the_middle_is_open_both_ways():
    v = _srp(30.0, [10.0, 50.0])
    assert TG.STRUCTURAL_RP_LOW < v < TG.STRUCTURAL_RP_HIGH


def test_the_thresholds_are_the_ones_the_desk_asked_for():
    """'long when at the bottom, mid or 80% zone ... vice versa for short'."""
    assert TG.STRUCTURAL_RP_HIGH == 0.80
    assert TG.STRUCTURAL_RP_LOW == 0.20


def test_too_little_structure_leaves_location_alone():
    """None means 'do not guess', not 'reject'."""
    assert TG.TraderGate._structural_rp(10.0, [], {}) is None
    assert TG.TraderGate._structural_rp(10.0, [(10.0, 3)], {}) is None
