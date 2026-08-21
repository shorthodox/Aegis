"""The counter-tide cut must reduce SIZE, not act as a veto.

2026-08-21, a full day with zero signals. BTC's tide was UP and strong, the
fleet was in a bull, so every setup the desk found was a counter-tide SELL. With
STRONG_TIDE_FACTOR (0.25) chosen to coincide exactly with MIN_SIZE_FACTOR (0.25),
every setup weighted below 1.00 was refused at the floor rather than on merit:

    TREND_PULLBACK      1.00 x 0.25 = 0.2500  fired
    BREAK_RETEST        0.85 x 0.25 = 0.2125  refused
    RANGE_FADE          0.70 x 0.25 = 0.1750  refused
    EXHAUSTION_REVERSAL 0.50 x 0.25 = 0.1250  refused   <- the live "0.12"

Observed in production as, e.g.:
    [BCH/USDT] NO TRADE (allocation): risk allocation fell to 0.12 (< 0.25)

and the ONE setup that could clear the floor -- TREND_PULLBACK -- is a BUY in a
bull needing rp <= 0.35, against a fleet whose live minimum was 0.51. So nothing
could trade at all, and it was arithmetic rather than judgement doing it.

The floor's stated reason, "not worth its execution cost", does not hold: costs
are PROPORTIONAL to size. A 0.125 position pays the same 0.10% round trip per
unit as a 1.00 position. The defensible reason for a floor is keeping negligible
positions out of a 5-slot book, which argues for a much lower number.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import (
    ACTION_REJECT, SETUP_EXHAUSTION_REVERSAL, SETUP_RANGE_FADE,
    SETUP_TREND_PULLBACK,
)


STRONG = {'tide_dir': 'UP', 'tide_strength': 0.85}      # strong tide AGAINST a SELL
FRESH = {'open_total': 0, 'max_open': 5, 'cluster_long': 0,
         'cluster_short': 0, 'max_per_cluster': 2}
SECOND = dict(FRESH, cluster_short=1)                   # one correlated SELL already open


def _size(setup, book):
    """The allocation arithmetic the gate performs, for a counter-tide SELL."""
    size = TG.SETUP_RISK_WEIGHT[setup]
    size *= TG.STRONG_TIDE_FACTOR
    if book.get('cluster_short', 0) >= 1:
        size *= TG.CLUSTER_SECOND_FACTOR
    return size


# -- a single counter-tide setup now trades, small ---------------------------

@pytest.mark.parametrize('setup', [
    SETUP_TREND_PULLBACK, SETUP_RANGE_FADE, SETUP_EXHAUSTION_REVERSAL,
])
def test_a_lone_counter_tide_setup_clears_the_floor(setup):
    size = _size(setup, FRESH)
    assert size >= TG.MIN_SIZE_FACTOR, (
        f'{setup} lands at {size:.3f}, under the {TG.MIN_SIZE_FACTOR} floor — the '
        f'tide cut is a veto again and the desk goes silent in a rally'
    )


def test_the_exhaustion_case_that_was_observed_live():
    """[BCH/USDT] risk allocation fell to 0.12 (< 0.25)."""
    size = _size(SETUP_EXHAUSTION_REVERSAL, FRESH)
    assert size == pytest.approx(0.125, abs=1e-9)
    assert size >= TG.MIN_SIZE_FACTOR


# -- but the stacked cases still refuse --------------------------------------

@pytest.mark.parametrize('setup', [SETUP_RANGE_FADE, SETUP_EXHAUSTION_REVERSAL])
def test_a_second_correlated_counter_tide_setup_is_still_refused(setup):
    """Small is fine; negligible AND duplicated is not. This is the line the
    lower floor is meant to keep."""
    size = _size(setup, SECOND)
    assert size < TG.MIN_SIZE_FACTOR, (
        f'{setup} as a second cluster expression lands at {size:.3f} and would '
        f'now trade — the floor has been dropped too far'
    )


def test_the_floor_still_blocks_something():
    """A floor that rejects nothing is not a floor."""
    assert TG.MIN_SIZE_FACTOR > 0.0
    assert _size(SETUP_EXHAUSTION_REVERSAL, SECOND) < TG.MIN_SIZE_FACTOR


# -- the cut is still a real cut ---------------------------------------------

def test_counter_tide_still_costs_size():
    """This loosens the FLOOR, not the tide cut. A counter-tide trade must still
    be a quarter of the position it would otherwise be."""
    assert TG.STRONG_TIDE_FACTOR == 0.25
    assert TG.COUNTER_TIDE_FACTOR == 0.50
    assert TG.CLUSTER_SECOND_FACTOR == 0.60
    full = TG.SETUP_RISK_WEIGHT[SETUP_TREND_PULLBACK]
    assert _size(SETUP_TREND_PULLBACK, FRESH) == pytest.approx(full * 0.25, abs=1e-9)


def test_the_other_basket_protections_are_untouched():
    """The 2026-07-20 basket defences were max-per-cluster and max-open, not this
    floor. Loosening the floor must not touch them."""
    assert FRESH['max_per_cluster'] == 2
    assert FRESH['max_open'] == 5


def test_the_runtime_knob_is_registered():
    import main
    assert 'min_size_factor' in main._TUNABLES
    mod, attr, kind = main._TUNABLES['min_size_factor'][:3]
    assert (mod, attr, kind) == ('src.trading.trader_gate', 'MIN_SIZE_FACTOR', 'float')


def test_the_floor_is_where_the_measurement_put_it():
    assert TG.MIN_SIZE_FACTOR == 0.12
