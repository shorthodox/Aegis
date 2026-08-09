"""Every virtual trade pays a fee and slippage, and the site has to say so.

The wallet has charged a round trip since v82 — the published figures were
already net. Six places on the site said the opposite ("gross · no fees"),
which understated the product: a subscriber discounting the numbers for costs
was discounting them twice.

These tests pin both halves, because the failure mode is drift between them.
"""
import inspect
import re
from pathlib import Path

import pytest

from scripts.engine.portfolio import VirtualWallet
from src.trading import trader_gate as TG


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web' / 'src'


# ── the cost model ───────────────────────────────────────────────────────────

def test_a_fee_and_slippage_are_both_charged_per_side():
    assert VirtualWallet.TAKER_FEE_PCT > 0, 'the taker fee is off'
    assert VirtualWallet.SLIPPAGE_PCT > 0, 'slippage is off'
    assert VirtualWallet.round_trip_cost_pct() == pytest.approx(
        2 * (VirtualWallet.TAKER_FEE_PCT + VirtualWallet.SLIPPAGE_PCT))


@pytest.mark.parametrize('fn', ['close_trade', 'partial_close_trade'])
def test_both_close_paths_charge_the_round_trip(fn):
    """A partial that escaped the charge would make banking look free."""
    src = inspect.getsource(getattr(VirtualWallet, fn))
    assert 'round_trip_cost_pct()' in src, f'{fn} books a slice without costs'
    # ...and it must be charged BEFORE the record is written, or the stored
    # pnl_pct is the gross one whatever the balance says
    assert src.index('round_trip_cost_pct()') < src.index('pnl_pct         ='), (
        f'{fn} records pnl_pct before deducting the round trip')


def test_the_gate_prices_costs_the_same_way_the_book_does():
    """A gate that approves trades on a cheaper cost model than the book
    charges is approving trades the book cannot pay for."""
    assert TG.ROUND_TRIP_COST_PCT == pytest.approx(VirtualWallet.round_trip_cost_pct())


def test_a_flat_trade_loses_exactly_the_round_trip():
    """The clearest statement of what the cost model does."""
    w = VirtualWallet(10_000.0, 1_000.0)
    gross = 0.0
    net = gross - w.round_trip_cost_pct()
    assert net == pytest.approx(-0.10, abs=1e-9), (
        'a round trip is no longer 0.10% — update the site copy, which quotes '
        'the number')


# ── the site must not contradict it ──────────────────────────────────────────

_PAGES = ['pages/track-record.html', 'pages/trader-record.html',
          'pages/bot-record.html', 'pages/chart.html', 'pages/dashboard.html',
          'scripts/track-record.js']

_STALE = re.compile(r'gross[^<>\n]{0,24}(no fees|P\s*/\s*L)|no fees or slippage',
                    re.IGNORECASE)


@pytest.mark.parametrize('rel', _PAGES)
def test_no_page_still_claims_the_figures_are_gross(rel):
    p = WEB / rel
    if not p.exists():
        pytest.skip(f'{rel} not present')
    text = p.read_text(encoding='utf-8')
    hits = [m.group(0) for m in _STALE.finditer(text)]
    assert not hits, (
        f'{rel} still tells the subscriber the numbers exclude costs, but the '
        f'wallet has already deducted them: {hits[:3]}')


def test_the_quoted_rate_matches_the_wallet():
    """The disclosure quotes real numbers, so they have to be the real ones."""
    js = (WEB / 'scripts' / 'track-record.js').read_text(encoding='utf-8')
    assert f'{VirtualWallet.TAKER_FEE_PCT}% taker fee' in js, (
        'the published taker fee does not match VirtualWallet.TAKER_FEE_PCT')
    assert f'{VirtualWallet.SLIPPAGE_PCT}% slippage' in js
    assert f'{VirtualWallet.round_trip_cost_pct():.2f}% round trip' in js
