"""A long is taken at support, a short at resistance. No exceptions.

Location is absolute. Nothing buys an exception — not the higher timeframe, not
a broken level, not a regime that agrees with the direction.

That is stricter than it started. The first version of this rule conditioned a
counter-location entry on the weekly and daily agreeing, which sounded
principled and was the wrong control: in a TRENDING_BEAR the weekly agrees WITH
a short, so the condition passed and the trade was taken at the BOTTOM of the
range anyway. Measured 2026-08-07:

    OP/USDT  SHORT at rp -0.03 — below its own support —
                   4.25 ATR from the resistance it should have been selling
    SUI/USDT SHORT at rp  0.20, 2.63 ATR from that resistance

Both were BREAK_RETEST, which is counter-location by construction: it bought at
rp >= 0.70 and sold at rp <= 0.30. It has been retired from _classify, because a
setup whose every output is rejected at stage 1b is dead code that reads as
live.

Earlier reference, the other direction — ADA/USDT 2026-08-05, published STRONG:
a BREAK_RETEST LONG at rp 0.85 with a bearish weekly.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import TraderGate


ADA_RP = (0.197600 - 0.189900) / (0.199000 - 0.189900)      # 0.846, at resistance
OP_RP  = (0.085800 - 0.085900) / (0.089700 - 0.085900)      # -0.026, below support
SUI_RP = (0.667800 - 0.664300) / (0.681500 - 0.664300)      # 0.203, at support


def _result(**over):
    r = {
        'range_position': ADA_RP, 'rsi': 62.0,
        'macro_weekly': -1.0, 'macro_daily': 1.0,
        'resistance_broken_recent': True, 'support_broken_recent': False,
        'p_buy': 0.512, 'p_sell': 0.488,
    }
    r.update(over)
    return r


# ── the three reported trades ────────────────────────────────────────────────

def test_the_reported_geometries_are_what_we_think():
    assert ADA_RP == pytest.approx(0.846, abs=0.002)   # at resistance
    assert OP_RP < 0                                    # below its own support
    assert SUI_RP == pytest.approx(0.203, abs=0.002)
    assert ADA_RP >= TG.RANGE_EDGE_HIGH
    assert OP_RP <= TG.RANGE_EDGE_LOW and SUI_RP <= TG.RANGE_EDGE_LOW


def test_ada_long_at_resistance_is_refused():
    reason = TraderGate._counter_location_refusal('BUY', ADA_RP, 62.0, _result())
    assert reason and 'at support, not at resistance' in reason


@pytest.mark.parametrize('rp,rsi', [(OP_RP, 38.7), (SUI_RP, 39.5)])
def test_the_short_at_support_trades_are_refused(rp, rsi):
    """OP and SUI — shorts taken at the bottom of the range."""
    r = _result(range_position=rp, macro_weekly=-1.0, macro_daily=-1.0,
                resistance_broken_recent=False, support_broken_recent=True)
    reason = TraderGate._counter_location_refusal('SELL', rp, rsi, r)
    assert reason, 'a short at support was permitted again'
    assert 'at resistance, not at support' in reason


# ── the higher timeframe no longer buys an exception ─────────────────────────

@pytest.mark.parametrize('w,d', [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
                                 (-1.0, 1.0), (0.0, 0.0)])
def test_no_htf_combination_permits_a_long_at_resistance(w, d):
    r = _result(macro_weekly=w, macro_daily=d)
    assert TraderGate._counter_location_refusal('BUY', ADA_RP, 55.0, r), \
        f'weekly {w} / daily {d} let a long through at the top of the range'


@pytest.mark.parametrize('w,d', [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
                                 (-1.0, 1.0), (0.0, 0.0)])
def test_no_htf_combination_permits_a_short_at_support(w, d):
    """The OP/SUI case: a bearish weekly must not authorise a short at support."""
    r = _result(range_position=SUI_RP, macro_weekly=w, macro_daily=d)
    assert TraderGate._counter_location_refusal('SELL', SUI_RP, 45.0, r), \
        f'weekly {w} / daily {d} let a short through at the bottom of the range'


def test_the_htf_still_appears_in_the_reason_when_it_agrees_with_the_refusal():
    """Useful detail, not a condition — the refusal stands either way."""
    r = _result(macro_weekly=-1.0, macro_daily=-1.0)
    assert 'lean bearish' in TraderGate._counter_location_refusal('BUY', ADA_RP, 55.0, r)


# ── sides that agree with their location are untouched ───────────────────────

@pytest.mark.parametrize('side,rp,rsi', [
    ('BUY',  0.10, 30.0),   # fade / exhaustion buy at the lows
    ('BUY',  0.25, 45.0),   # trend pullback buy
    ('SELL', 0.90, 70.0),   # fade / exhaustion sell at the highs
    ('SELL', 0.75, 55.0),   # trend pullback sell
    ('BUY',  0.50, 50.0),   # mid-range: not this stage's business
    ('SELL', 0.50, 50.0),
])
def test_a_side_that_agrees_with_its_location_is_untouched(side, rp, rsi):
    for w in (-1.0, 0.0, 1.0):
        for d in (-1.0, 0.0, 1.0):
            r = _result(range_position=rp, rsi=rsi, macro_weekly=w, macro_daily=d)
            assert TraderGate._counter_location_refusal(side, rp, rsi, r) is None, (
                f'{side} at rp {rp} refused with weekly {w} / daily {d} — this '
                f'stage judges counter-location entries only'
            )


# ── BREAK_RETEST is gone ─────────────────────────────────────────────────────

def test_classify_no_longer_produces_break_retest():
    """It only ever fired counter-location, so it could only ever be rejected."""
    for regime, conf in (('TRENDING_BULL', 0.9), ('TRENDING_BEAR', 0.9),
                         ('RANGING', 0.6), ('VOLATILE_COMPRESSION', 0.7)):
        for rp in (0.05, 0.2, 0.5, 0.8, 0.95):
            for rb, sb in ((True, False), (False, True), (True, True)):
                setup, _side, _why = TraderGate._classify(
                    _result(range_position=rp, resistance_broken_recent=rb,
                            support_broken_recent=sb), regime, conf)
                assert setup != TG.SETUP_BREAK_RETEST, (
                    f'BREAK_RETEST returned at rp {rp} in {regime} — it is '
                    f'counter-location by construction and would be rejected'
                )


def test_the_constant_survives_for_historical_records():
    """Old plans and the risk-weight table still have to resolve."""
    assert TG.SETUP_BREAK_RETEST in TG.SETUP_RISK_WEIGHT


# ── no setup can propose a counter-location side ─────────────────────────────

def test_no_setup_anywhere_proposes_a_counter_location_side():
    """The property the whole rule exists to guarantee."""
    for regime, conf in (('TRENDING_BULL', 0.9), ('TRENDING_BEAR', 0.9),
                         ('RANGING', 0.6), ('VOLATILE_COMPRESSION', 0.7),
                         ('ACCUMULATION', 0.7), ('DISTRIBUTION', 0.7)):
        for rp in (0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0):
            for rsi in (20, 35, 50, 65, 80):
                setup, side, _ = TraderGate._classify(
                    _result(range_position=rp, rsi=rsi), regime, conf)
                if side == 'BUY':
                    assert rp < TG.RANGE_EDGE_HIGH, f'{setup} BUY at rp {rp}'
                elif side == 'SELL':
                    assert rp > TG.RANGE_EDGE_LOW, f'{setup} SELL at rp {rp}'


def test_refusal_is_wired_into_evaluate_with_its_own_stage():
    import inspect
    src = inspect.getsource(TraderGate.evaluate)
    assert '_counter_location_refusal' in src
    assert "_reject('location'" in src
