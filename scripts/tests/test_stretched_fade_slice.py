"""Only a genuinely stretched fade is allowed back; the rest stays refused.

A full day with zero signals. Production over 55 minutes: 492 decisions, 0
fires, 96% dying at setup, and ONE reason 57% of everything -- the exhaustion
refusal, every instance a SELL (the bull branch, since the fleet is in a rally).

Re-measured with the conditions this branch ACTUALLY uses (uptrend, rp >= 0.80
on the rolling 24h range, RSI >= 68), not the looser proxy behind the -0.075R
that justified the blanket refusal (72h range, RSI > 70, no trend filter -- a
different population). Through the real 5-rung ladder, 9,846 bars:

    the whole refused branch    48.9% win  -0.008R   held out -0.002R

Breakeven. The refusal was spending 57% of the desk's decisions to avoid a setup
that costs about nothing. One slice is positive in BOTH halves:

    RSI >= 75 AND rp >= 0.95    1,242 bars  49.2% win
                                +0.023R all / +0.009R fit / +0.045R held out

Eleven variants were tested and this is the one that survived both halves, so
some of it may be selection, and the effect is small. Hence: unchanged 0.50
allocation, killable from /control, and everything less stretched stays refused.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import (
    ACTION_REJECT, SETUP_EXHAUSTION_REVERSAL,
)

from scripts.tests.test_trader_gate import mk, run, TURNED_DOWN


def _top(rsi, rp, **kw):
    """A bull at the top of its range. rp is set via the support/resistance span."""
    support, resistance = 90.0, 100.0
    price = support + rp * (resistance - support)
    d = mk(price=price, atr=1.0, support=support, resistance=resistance,
           rsi=rsi, **TURNED_DOWN)
    d.update(kw)
    return d


def _levels(rp):
    return [(100.0, 4), (90.0, 4), (104.0, 3)]


def _plan(rsi, rp):
    return run(_top(rsi, rp), regime='TRENDING_BULL', levels=_levels(rp))


def _refused(plan):
    return (plan.action == ACTION_REJECT
            and 'exhaustion reversal refused' in (plan.reason or ''))


# -- the slice that measured positive is allowed -----------------------------

def test_a_genuinely_stretched_fade_is_allowed():
    plan = _plan(rsi=78.0, rp=0.97)
    assert not _refused(plan), (
        'the stretched slice is still refused — this is the 57% of decisions '
        'that produced a day with no signals'
    )


def test_it_classifies_as_an_exhaustion_reversal_sell():
    plan = _plan(rsi=78.0, rp=0.97)
    if plan.action == ACTION_REJECT and plan.stage != 'setup':
        pytest.skip(f'refused later for an unrelated reason: {plan.reason}')
    assert plan.setup == SETUP_EXHAUSTION_REVERSAL and plan.side == 'SELL'


# -- everything less stretched stays refused ---------------------------------

@pytest.mark.parametrize('rsi,rp,why', [
    (70.0, 0.97, 'overbought but not stretched'),
    (78.0, 0.85, 'stretched but not at the top of the range'),
    (70.0, 0.85, 'neither'),
    (74.9, 0.97, 'just under the RSI bar'),
    (78.0, 0.94, 'just under the range bar'),
])
def test_the_rest_of_the_branch_stays_refused(rsi, rp, why):
    plan = _plan(rsi=rsi, rp=rp)
    assert _refused(plan), (
        f'{why} (RSI {rsi}, rp {rp}) was allowed — the branch as a whole '
        f'measured -0.008R and only the stretched slice earned its way back'
    )


def test_the_slice_can_be_killed_at_runtime(monkeypatch):
    monkeypatch.setattr(TG, 'ALLOW_EXHAUSTION_SELL_STRETCHED', False)
    assert _refused(_plan(rsi=78.0, rp=0.97))


def test_the_buy_half_is_untouched():
    """Freed earlier on its own measurement; this change must not disturb it."""
    assert TG.ALLOW_EXHAUSTION_REVERSAL_BUY is True


def test_the_runtime_knob_is_registered():
    import main
    assert 'allow_exhaustion_sell_stretched' in main._TUNABLES
    mod, attr, kind = main._TUNABLES['allow_exhaustion_sell_stretched'][:3]
    assert (mod, attr, kind) == ('src.trading.trader_gate',
                                 'ALLOW_EXHAUSTION_SELL_STRETCHED', 'bool')


def test_nothing_else_moved():
    assert TG.ALLOW_EXHAUSTION_REVERSAL is False, (
        'the blanket switch must stay off — only the measured slice is back'
    )
    assert TG.EXHAUSTION_SELL_RSI == 75.0
    assert TG.EXHAUSTION_SELL_RP == 0.95
    assert TG.EXHAUSTION_RSI_HI == 68.0
    assert TG.EXTREME_RP_HIGH == 0.80
    assert TG.SETUP_RISK_WEIGHT[SETUP_EXHAUSTION_REVERSAL] == 0.50, (
        'the slice comes back at the SAME small allocation — a +0.023R edge '
        'does not earn a bigger bet'
    )
