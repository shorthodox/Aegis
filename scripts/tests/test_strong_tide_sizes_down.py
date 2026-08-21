"""A strong BTC tide sizes a counter-tide trade down; it no longer vetoes it.

The fleet fired ONE signal overnight. Attribution of the 44 live symbols on
2026-08-20:

    21 (48%)  no setup, mid-range          structure — correct, left alone
     9 (20%)  exhaustion fade refused      policy, measured NEGATIVE (2W/15L,
                                           p=1.74%) — deliberately left alone
     7 (16%)  SELL vs strong BTC UP tide   policy — THIS
     0 ( 0%)  quality floor                not the constraint

Widening the long-pullback threshold was measured first and discarded: every
bull symbol sat at rp 0.48-0.96, so even a 0.60 threshold produced 2 candidates
and 0 that cleared quality. It would have been a number that changed nothing.

The tide rule was refusing 12-of-17 bear symbols' TREND_PULLBACK SELL — the setup
with the BEST measured edge, +0.069R — because of what BTC was doing. A token in
its own downtrend was declined for an unrelated instrument's direction.

The veto existed for the 2026-07-20 basket (eight alt SHORTs in 55 minutes). Three
independent protections against that have been added since and all remain: max 2
per correlated cluster, max 5 open, and this size cut. The veto was the fourth and
bluntest, and the only one that could not tell one good trade from eight
correlated ones.

STRONG_TIDE_FACTOR = 0.25 was chosen to equal MIN_SIZE_FACTOR, so the EXISTING
floor decided what survived and no new veto was added.

UPDATE 2026-08-21 — that identity turned the size cut back into a veto. With
both at 0.25, only a setup weighted 1.00 could clear the floor, so three of the
four setups were refused by arithmetic rather than on merit. A full day with
zero signals: BTC's tide UP and strong, the fleet in a bull, every setup a
counter-tide SELL, and the one setup that could clear the floor was a BUY
needing rp <= 0.35 against a fleet whose live minimum was 0.51.

    [BCH/USDT] NO TRADE (allocation): risk allocation fell to 0.12 (< 0.25)

MIN_SIZE_FACTOR is now 0.12, so the cut is a SIZE cut again. The floor's old
reason -- "not worth its execution cost" -- never held: costs are proportional
to size, so a 0.125 position pays the same ratio as a 1.00 one.

The basket protection does NOT come from the floor and is unchanged. The ceiling
is max_open x the LARGEST allowed size, which the floor never gated:

    before (floor 0.25)   1/4 setups allowed   heaviest book 1.250 units
    after  (floor 0.12)   4/4 setups allowed   heaviest book 1.250 units

against the basket's own ~8.0 units. What changed is that WEAKER setups may now
participate at SMALLER sizes; the worst case is identical and the typical book
is lighter.
"""
import pytest

from src.trading import trader_gate as TG


def test_the_factor_no_longer_equals_the_size_floor():
    """The identity is deliberately BROKEN now. While the two were equal the
    size cut silently vetoed every setup below weight 1.00, which is what
    produced a day with no signals."""
    assert TG.STRONG_TIDE_FACTOR > TG.MIN_SIZE_FACTOR, (
        'the cut is a veto again — a counter-tide setup below weight 1.00 '
        'cannot clear the floor'
    )


@pytest.mark.parametrize('setup,expected', [
    ('TREND_PULLBACK', 0.2500),      # +0.069R — the best measured setup
    ('BREAK_RETEST', 0.2125),
    ('RANGE_FADE', 0.1750),
    ('EXHAUSTION_REVERSAL', 0.1250), # the live "0.12" that was being refused
])
def test_every_setup_survives_a_strong_counter_tide_at_reduced_size(setup, expected):
    """All four now trade, each cut to a quarter. Previously three of them were
    refused by the floor rather than on merit, which is what silenced the desk
    for a day in a rally."""
    size = TG.SETUP_RISK_WEIGHT[setup] * TG.STRONG_TIDE_FACTOR
    assert size == pytest.approx(expected, abs=1e-9)
    assert size >= TG.MIN_SIZE_FACTOR, (
        f'{setup} at {size:.4f} is under the {TG.MIN_SIZE_FACTOR} floor'
    )


def test_a_second_correlated_counter_tide_trade_now_fires_but_smaller():
    """A REAL loosening, recorded as such. The second expression of one cluster
    thesis used to be refused by the floor; it now trades at 0.15.

    What still bounds it: max 2 per correlated cluster, so there is no third,
    and max 5 open, so the book cannot fill with them."""
    size = (TG.SETUP_RISK_WEIGHT['TREND_PULLBACK']
            * TG.STRONG_TIDE_FACTOR * TG.CLUSTER_SECOND_FACTOR)
    assert size == pytest.approx(0.15, abs=1e-9)
    assert size >= TG.MIN_SIZE_FACTOR
    assert size < TG.SETUP_RISK_WEIGHT['TREND_PULLBACK'] * TG.STRONG_TIDE_FACTOR, (
        'the second correlated trade is no smaller than the first'
    )


def test_the_weakest_stacked_cases_are_still_refused():
    """The floor still has to reject something or it is not a floor."""
    for setup in ('RANGE_FADE', 'EXHAUSTION_REVERSAL'):
        size = (TG.SETUP_RISK_WEIGHT[setup]
                * TG.STRONG_TIDE_FACTOR * TG.CLUSTER_SECOND_FACTOR)
        assert size < TG.MIN_SIZE_FACTOR, (
            f'a second correlated {setup} sizes to {size:.4f} and would fire — '
            f'the floor has been dropped too far'
        )


def test_the_aggregate_counter_tide_ceiling_is_unchanged():
    """The basket protection is max_open x the LARGEST allowed size, and the
    floor never gated that. 2026-07-20 was ~8.0 units of correlated short."""
    heaviest = max(TG.SETUP_RISK_WEIGHT.values()) * TG.STRONG_TIDE_FACTOR
    assert heaviest * 5 == pytest.approx(1.25, abs=1e-9), (
        'the ceiling moved — lowering the floor was supposed to admit weaker '
        'setups at smaller sizes, not raise the maximum exposure'
    )


def test_a_moderate_tide_still_only_halves():
    """Unchanged below STRONG_TIDE — this change touches the strong case only."""
    assert TG.COUNTER_TIDE_FACTOR == 0.50
    assert TG.STRONG_TIDE == 0.65


def test_setting_the_factor_to_zero_restores_the_hard_veto():
    """The documented revert path must exist in the code, not just the comment."""
    import inspect
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'STRONG_TIDE_FACTOR <= 0' in src, (
        'the revert path is gone — STRONG_TIDE_FACTOR = 0.0 must restore the reject'
    )
    assert 'basket trade that bled' in src, 'the original refusal message was deleted'


def test_the_book_and_cluster_caps_are_untouched():
    """The other two basket protections must survive this change."""
    assert TG.CLUSTER_SECOND_FACTOR == 0.60
    import inspect
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'max_per_cluster' in src and 'max_open' in src
