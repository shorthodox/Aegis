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

STRONG_TIDE_FACTOR = 0.25 is exactly MIN_SIZE_FACTOR, so the EXISTING floor
decides what survives and no new veto was added.
"""
import pytest

from src.trading import trader_gate as TG


def test_the_factor_equals_the_size_floor():
    """The whole design rests on this identity: at 0.25 only a setup weighted
    1.00 clears MIN_SIZE_FACTOR, so the filtering is done by machinery that was
    already there."""
    assert TG.STRONG_TIDE_FACTOR == TG.MIN_SIZE_FACTOR


@pytest.mark.parametrize('setup,fires', [
    ('TREND_PULLBACK', True),        # +0.069R — the best measured setup
    ('BREAK_RETEST', False),
    ('RANGE_FADE', False),
    ('EXHAUSTION_REVERSAL', False),  # -0.064R — must stay out
])
def test_only_the_best_setup_survives_a_strong_counter_tide(setup, fires):
    size = TG.SETUP_RISK_WEIGHT[setup] * TG.STRONG_TIDE_FACTOR
    assert (size >= TG.MIN_SIZE_FACTOR) is fires, (
        f'{setup} at {size:.4f} vs floor {TG.MIN_SIZE_FACTOR}'
    )


def test_a_second_correlated_counter_tide_trade_is_refused():
    """The basket protection, arithmetically. Only ONE per cluster survives."""
    size = (TG.SETUP_RISK_WEIGHT['TREND_PULLBACK']
            * TG.STRONG_TIDE_FACTOR * TG.CLUSTER_SECOND_FACTOR)
    assert size < TG.MIN_SIZE_FACTOR, (
        f'a second correlated counter-tide trade sizes to {size:.4f} and would '
        f'fire — the 2026-07-20 basket becomes possible again'
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
