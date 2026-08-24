"""The oversold long — the one setup measured positive through the real ladder.

Asked for 2026-08-24, after the funnel was traced end to end and the engine was
found to be correctly silent with nothing to fade.

THE MEASUREMENT. 345,209 simulated entries, 59 tokens, ~250 days of 1H bars, run
through the desk's actual exits (TP_LADDER_PCT, TP_CLOSE_PCTS, break-even at
TP1, ATR trail from TP2, giveback ratchet), stop 1.386%, costs 0.10% round trip:

    setup              ALL              fit half        held out         n
    BUY unconditional  45.1% -0.092R    47.0% -0.051R   42.1% -0.154R    345,209
    BUY RSI<30         49.9% +0.020R    48.1% -0.020R   52.7% +0.080R     16,556
    BUY RSI<25         51.4% +0.044R    48.7% -0.019R   55.5% +0.139R      5,946

Read honestly: RSI<25 is only ABSOLUTELY positive in the held-out half. What
holds in BOTH halves is that it beats unconditional longs - +0.032R in fit,
+0.293R held out - and every setup improves in the second half, which is a
regime shift rather than an improvement, so only within-period comparisons mean
anything. Per-token at RSI<30, 37 of 59 are net positive.

WHY IT NEEDED ITS OWN BRANCH. Every other branch in _classify keys on
range_position FIRST and consults RSI only inside an rp bracket, so a deeply
oversold token sitting mid-range matched nothing and fell through to "mid-range,
no edge". Measured live 2026-08-24: 5 of 43 tokens under RSI 25, including
PENDLE at RSI 24.0 / rp 0.44 carrying the HIGHEST quality score on the fleet
(78) and being refused for having no recognised setup. The measured edge is on
RSI ALONE, unconditional on rp, which the existing shape could not express.

It is a FALLBACK on purpose. Verified over a 630-cell grid of regime x rp x rsi:
it fires on 112 cells, changes ZERO existing classifications, and every cell it
claims was previously SETUP_NONE.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import (
    SETUP_NONE, SETUP_OVERSOLD_LONG, SETUP_RISK_WEIGHT, TraderGate,
)


REGIMES = ('TRENDING_BULL', 'TRENDING_BEAR', 'RANGING')


def _classify(rp, rsi, regime='RANGING', conf=0.80):
    return TraderGate._classify({'range_position': rp, 'rsi': rsi}, regime, conf)


# -- it fires where the measurement says it should ----------------------------

def test_a_deeply_oversold_token_mid_range_is_now_a_setup():
    """PENDLE: RSI 24.0, rp 0.44, quality 78 - refused as 'no recognised setup'."""
    setup, side, why = _classify(0.44, 24.0)
    assert setup == SETUP_OVERSOLD_LONG
    assert side == 'BUY'
    assert 'RSI 24' in why


def test_the_threshold_is_the_measured_one():
    assert TG.OVERSOLD_LONG_RSI == 25.0


def test_it_fires_at_the_threshold_and_not_above():
    assert _classify(0.44, 25.0)[0] == SETUP_OVERSOLD_LONG
    assert _classify(0.44, 25.1)[0] == SETUP_NONE


@pytest.mark.parametrize('regime', REGIMES)
def test_it_works_in_every_regime(regime):
    assert _classify(0.50, 20.0, regime)[0] == SETUP_OVERSOLD_LONG


def test_it_does_not_need_the_range_edge():
    """The whole point - the measured edge is on RSI alone.

    Strictly INSIDE the range edges: at rp <= RANGE_EDGE_LOW the existing
    RANGE_FADE claims the bar first, which is correct - the fallback only takes
    what nothing else wanted.
    """
    for rp in (0.35, 0.44, 0.50, 0.60, 0.65):
        assert _classify(rp, 22.0)[0] == SETUP_OVERSOLD_LONG, rp


def test_it_never_buys_at_the_top_of_the_range():
    """The counter-location rule is absolute — it outranks this fallback.

    Caught by test_location_vs_htf before shipping: the first version ignored rp
    entirely and proposed a BUY at rp 0.75. Costs nothing to bound: of 402 bars
    with RSI < 25 across 29,250 hourly bars, ZERO sat at rp >= 0.70.
    """
    for rp in (0.70, 0.75, 0.85, 0.95, 1.0):
        setup, side, _ = _classify(rp, 20.0)
        assert setup != SETUP_OVERSOLD_LONG, f'oversold BUY at rp {rp}'


def test_an_existing_range_fade_still_claims_the_edge():
    assert _classify(0.30, 22.0)[0] == TG.SETUP_RANGE_FADE


# -- it never displaces a setup the desk already had --------------------------

def test_it_only_ever_claims_bars_that_had_no_setup():
    """630-cell grid: 112 claimed, 0 reclassified."""
    claimed = changed = 0
    cells = [(rg, i / 20, rsi)
             for rg in REGIMES
             for i in range(21)
             for rsi in (15, 20, 24, 25, 26, 32, 45, 55, 70, 80)]
    with_fb = {c: _classify(c[1], c[2], c[0])[:2] for c in cells}
    TG.ALLOW_OVERSOLD_LONG = False
    try:
        for c in cells:
            before = _classify(c[1], c[2], c[0])[:2]
            after = with_fb[c]
            if after[0] == SETUP_OVERSOLD_LONG:
                claimed += 1
                assert before[0] == SETUP_NONE, (
                    f'the oversold fallback displaced {before[0]} at {c}'
                )
            elif before != after:
                changed += 1
    finally:
        TG.ALLOW_OVERSOLD_LONG = True
    assert claimed > 0
    assert changed == 0, 'an existing classification changed'


def test_a_trend_pullback_still_wins_where_both_apply():
    """rp low AND oversold in a bull: the pullback is the better-measured setup
    (+0.069R) and must keep the bar."""
    setup, side, _ = _classify(0.20, 20.0, 'TRENDING_BULL')
    assert setup != SETUP_OVERSOLD_LONG
    assert side == 'BUY'


# -- sizing and the off switch ------------------------------------------------

def test_it_is_sized_below_the_best_measured_setup():
    """+0.044R against TREND_PULLBACK's +0.069R."""
    assert SETUP_RISK_WEIGHT[SETUP_OVERSOLD_LONG] == 0.85
    assert (SETUP_RISK_WEIGHT[SETUP_OVERSOLD_LONG]
            < SETUP_RISK_WEIGHT[TG.SETUP_TREND_PULLBACK])


def test_it_is_sized_above_the_worst_measured_setup():
    """EXHAUSTION_REVERSAL measured -0.064R fleet-wide."""
    assert (SETUP_RISK_WEIGHT[SETUP_OVERSOLD_LONG]
            > SETUP_RISK_WEIGHT[TG.SETUP_EXHAUSTION_REVERSAL])


def test_it_can_be_switched_off():
    TG.ALLOW_OVERSOLD_LONG = False
    try:
        assert _classify(0.44, 20.0)[0] == SETUP_NONE
    finally:
        TG.ALLOW_OVERSOLD_LONG = True


def test_switching_it_off_restores_the_old_refusal_text():
    TG.ALLOW_OVERSOLD_LONG = False
    try:
        _, _, why = _classify(0.44, 20.0)
        assert 'mid-range' in why
    finally:
        TG.ALLOW_OVERSOLD_LONG = True


# -- it is a long, and only a long --------------------------------------------

def test_there_is_no_overbought_short_counterpart():
    """Deliberate. The mirror - fading at resistance - is the WORST setup
    measured (-0.075R), so it does not get a fallback of its own."""
    for rp in (0.44, 0.55, 0.60):
        setup, side, _ = _classify(rp, 85.0)
        assert setup == SETUP_NONE, (
            f'an overbought mid-range bar produced {setup} — the mirror of this '
            f'fallback was never built, and must not appear by accident'
        )
        assert side == 'FLAT'
    assert _classify(0.50, 90.0)[0] == SETUP_NONE
