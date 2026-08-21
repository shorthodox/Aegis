"""The fade refusal pooled two setups with opposite signs.

ALLOW_EXHAUSTION_REVERSAL refuses "buy oversold / sell overbought" as one thing,
on a -0.064R/trade that was measured across BOTH directions at once.

Re-measured 2026-08-21 on 345,209 entries run through the desk's real 5-rung
ladder (TP_LADDER_PCT, TP_CLOSE_PCTS, break-even at TP1, ATR trail from TP2,
giveback ratchet), split 60/40 by time:

    BUY  oversold  RSI<25          51.4% win  +0.044R   held out +0.139R
    BUY  oversold  RSI<30          49.9% win  +0.020R   held out +0.080R
    SELL overbought at resistance  46.1% win  -0.075R   fit      -0.143R

The BUY half is the best setup measured on this desk and 37 of 59 tokens are net
positive on it. The SELL half is the worst. A single flag threw the good half
away to be rid of the bad one.

CAVEAT kept deliberately visible: the live corroboration behind the original
refusal (17 closed, 2W/15L, every loss a STOP_HIT) is not broken out by
direction, and the fleet was overwhelmingly bear-labelled over that window --
which is when the BUY half fires. So live may be arguing against the backtest
rather than with it. Hence the runtime kill switch, and hence this file asserting
the SELL half stays refused.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import (
    ACTION_REJECT, SETUP_EXHAUSTION_REVERSAL, TraderGate,
)

from scripts.tests.test_trader_gate import mk, run, TURNED_UP, TURNED_DOWN


def _oversold(**kw):
    """A bear at the bottom of its range, washed out. rp 0.10, RSI 28."""
    d = mk(price=100.0, atr=1.2, support=99.0, resistance=109.0,
           rsi=28.0, **TURNED_UP)
    d.update(kw)
    return d


def _overbought(**kw):
    """The mirror: a bull at the top of its range, stretched. rp 0.90, RSI 72."""
    d = mk(price=100.0, atr=1.2, support=91.0, resistance=101.0,
           rsi=72.0, **TURNED_DOWN)
    d.update(kw)
    return d


OVERSOLD_LEVELS = [(99.0, 4), (109.0, 4), (95.0, 3)]
OVERBOUGHT_LEVELS = [(101.0, 4), (91.0, 4), (105.0, 3)]


# -- the BUY half is allowed --------------------------------------------------

def test_buying_oversold_is_no_longer_refused():
    plan = run(_oversold(), regime='TRENDING_BEAR', levels=OVERSOLD_LEVELS)
    assert not (plan.action == ACTION_REJECT
                and 'exhaustion reversal refused' in (plan.reason or '')), (
        'the BUY half is still refused — the best setup measured on this desk '
        'is being thrown away with the worst one'
    )


def test_the_buy_half_classifies_as_an_exhaustion_reversal():
    plan = run(_oversold(), regime='TRENDING_BEAR', levels=OVERSOLD_LEVELS)
    if plan.action == ACTION_REJECT and plan.stage != 'setup':
        pytest.skip(f'refused later for an unrelated reason: {plan.reason}')
    assert plan.setup == SETUP_EXHAUSTION_REVERSAL and plan.side == 'BUY'


# -- the SELL half stays refused ---------------------------------------------

def test_selling_overbought_is_still_refused():
    """-0.075R through the ladder, -0.143R in the fit half. It must not come
    back as a side effect of freeing the BUY half."""
    plan = run(_overbought(), regime='TRENDING_BULL', levels=OVERBOUGHT_LEVELS)
    assert plan.action == ACTION_REJECT and plan.stage == 'setup'
    assert 'exhaustion reversal refused' in (plan.reason or '')


def test_the_sell_half_is_not_freed_by_the_buy_flag(monkeypatch):
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL_BUY', True)
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL', False)
    plan = run(_overbought(), regime='TRENDING_BULL', levels=OVERBOUGHT_LEVELS)
    assert plan.action == ACTION_REJECT and plan.stage == 'setup'


# -- the switches ------------------------------------------------------------

def test_the_buy_half_can_be_killed_at_runtime(monkeypatch):
    """It ships killable from /control because the live 2W/15L may be arguing
    against the backtest, and that has to be answerable without a deploy."""
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL_BUY', False)
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL', False)
    plan = run(_oversold(), regime='TRENDING_BEAR', levels=OVERSOLD_LEVELS)
    assert plan.action == ACTION_REJECT and plan.stage == 'setup'
    assert 'exhaustion reversal refused' in (plan.reason or '')


def test_the_old_flag_still_enables_both(monkeypatch):
    """Backwards compatible: the original switch keeps its original meaning."""
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL', True)
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_REVERSAL_BUY', False)
    plan = run(_overbought(), regime='TRENDING_BULL', levels=OVERBOUGHT_LEVELS)
    assert not (plan.action == ACTION_REJECT
                and 'exhaustion reversal refused' in (plan.reason or ''))


def test_the_runtime_knob_is_registered():
    """It is only killable if /control can actually reach it."""
    import main
    assert 'allow_exhaustion_reversal_buy' in main._TUNABLES
    mod, attr, kind = main._TUNABLES['allow_exhaustion_reversal_buy'][:3]
    assert mod == 'src.trading.trader_gate'
    assert attr == 'ALLOW_EXHAUSTION_REVERSAL_BUY'
    assert kind == 'bool'


# -- nothing else moved ------------------------------------------------------

def test_no_threshold_moved():
    assert TG.ALLOW_EXHAUSTION_REVERSAL is False, (
        'the SELL half must stay refused — it measured -0.075R'
    )
    assert TG.EXHAUSTION_RSI_LO == 32.0
    assert TG.EXHAUSTION_RSI_HI == 68.0
    assert TG.EXTREME_RP_LOW == 0.20
    assert TG.EXTREME_RP_HIGH == 0.80
    assert TG.PULLBACK_RP_LONG == 0.35, (
        'the pullback long measured -0.002R, the best on the board — it does '
        'not need loosening to make signals appear'
    )
    assert TG.SETUP_RISK_WEIGHT[SETUP_EXHAUSTION_REVERSAL] == 0.50, (
        'the freed BUY half keeps the conservative allocation while the live '
        'sample that argued against it is still unresolved'
    )
