"""The location veto must test the side that will actually be traded.

config.py has said since v82e that FAR_FROM_SR is a HARD veto — "the majority of
fires were entries taken nowhere near their own structure ... the engine
reporting that the setup's premise was absent, then taking the trade regardless".
That was true of the veto and false of the fleet.

gate_scorer computed `_far` against its OWN `winner`:

    if winner == 'BUY':   _far = ...
    elif winner == 'SELL': _far = ...
    else:                  _far = False        # <-- winner == 'HOLD'

But the engine is MODEL-FIRST: `result['side']` is what gets traded and `winner`
only picks the dashboard card. score_hold carries a structural floor — engine.py
says at COMPOSITE_HOLD_MARGIN that it "edges out the model's side on almost every
signal" — so HOLD is the COMMON case and the veto silently short-circuited on
exactly the signals that had a tradeable side.

Measured live, 2026-08-16, 44 scored symbols: 11 sat more than AT_LEVEL_ATR from
the level their own side leans on, and 8 of those carried NO veto because UWGS
said HOLD:

    ATOM  SELL 1.69   STX BUY 3.01   XRP BUY 2.70   GMX BUY 2.57
    ETC   BUY  2.50   INJ BUY 1.47   STORJ BUY 1.47  DOGE SELL 1.05

ATOM and DOGE were OPEN POSITIONS at the time.
"""
import pytest

import src.trading.gate_scorer as gs

ATR_PCT = 1.0          # so 1 ATR == 1.0 price unit at price 100


def _far(traded_side, level_price, winner_side, price=100.0):
    """Reproduce the veto's decision for one side/level pair.

    Mirrors the branch under test rather than importing it, because the branch
    lives mid-way through a 100-line scoring pass.
    """
    atr = (ATR_PCT / 100.0) * price
    result = {'side': traded_side, 'price': price}
    if traded_side == 'BUY' or winner_side == 'BUY':
        result['support'] = level_price
    else:
        result['resistance'] = level_price

    t = str(result.get('side') or '').upper()
    if t not in ('BUY', 'SELL'):
        t = winner_side
    sup, res = result.get('support', 0.0), result.get('resistance', 0.0)
    if t == 'BUY':
        return (price - sup) / atr > gs.AT_LEVEL_ATR
    if t == 'SELL':
        return (res - price) / atr > gs.AT_LEVEL_ATR
    return False


# ── the hole ─────────────────────────────────────────────────────────────────

def test_a_far_buy_is_vetoed_even_when_uwgs_says_hold():
    """The regression that mattered. 2.5 ATR from support, UWGS winner HOLD."""
    assert _far('BUY', 97.5, winner_side='HOLD'), (
        'a BUY 2.5 ATR above its support escaped FAR_FROM_SR because the '
        'composite happened to land on HOLD'
    )


def test_a_far_sell_is_vetoed_even_when_uwgs_says_hold():
    assert _far('SELL', 101.7, winner_side='HOLD')


def test_the_eight_live_escapes_are_all_caught():
    """Every symbol that slipped through on 2026-08-16, at its measured gap."""
    live = [('ATOM', 'SELL', 1.69), ('STX', 'BUY', 3.01), ('XRP', 'BUY', 2.70),
            ('GMX', 'BUY', 2.57), ('ETC', 'BUY', 2.50), ('INJ', 'BUY', 1.47),
            ('STORJ', 'BUY', 1.47), ('DOGE', 'SELL', 1.05)]
    for sym, side, gap in live:
        lvl = 100.0 - gap if side == 'BUY' else 100.0 + gap
        assert _far(side, lvl, winner_side='HOLD'), (
            f'{sym} {side} at {gap} ATR from its level still escapes the veto'
        )


# ── and it must not become indiscriminate ────────────────────────────────────

def test_a_trade_at_its_level_still_fires():
    assert not _far('BUY', 99.5, winner_side='HOLD')
    assert not _far('SELL', 100.5, winner_side='HOLD')


def test_the_boundary_is_at_level_atr():
    just_in  = 100.0 - (gs.AT_LEVEL_ATR * 1.0) + 0.01
    just_out = 100.0 - (gs.AT_LEVEL_ATR * 1.0) - 0.01
    assert not _far('BUY', just_in, winner_side='HOLD')
    assert _far('BUY', just_out, winner_side='HOLD')


def test_the_uwgs_winner_is_still_used_when_there_is_no_model_side():
    """The fallback must stay — gate_scorer is also called where result['side']
    is absent, and losing the check entirely would be the same bug inverted."""
    assert _far('', 97.5, winner_side='BUY')
    assert not _far('', 99.5, winner_side='BUY')


def test_a_flat_model_and_a_hold_winner_veto_nothing():
    assert not _far('FLAT', 97.5, winner_side='HOLD')


def test_the_veto_is_wired_as_hard():
    """The veto being correct is worthless if the engine only tags it."""
    from scripts.engine.config import HARD_VETOES
    assert 'FAR_FROM_SR' in HARD_VETOES
