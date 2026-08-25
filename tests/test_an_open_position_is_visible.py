"""An engine position must be visible WHILE it is open, not only after it closes.

Reported 2026-08-24, with the Telegram log as evidence:

    AEGIS BUY - ENA/USDT   entry 0.1609   12:37 UTC
    WIN CLOSED  +1.40%  held 11m  TP1 PARTIAL
    WIN CLOSED  +0.95%  held 26m  TP GIVEBACK   13:04 UTC

    "in between i never seen this ena open in track record and in the
     dashboard as well"

Correct on both counts, and for two different reasons.

TRACK RECORD - by design. track_record.json carried summary / signals /
performance / drift / portfolio / edge, where `signals` is the CLOSED record
only. The page says as much ("Only closed signals shown publicly", "Open signals
are visible on the dashboard (subscribers only)"). The wallet's open_positions
were simply never written into the payload, so even a subscriber view had no
open data to render.

DASHBOARD - a genuine gap. gatekeeper.js deliberately keeps data.open_trades out
of renderTrades, because that panel is for positions the USER executed. That
separation is right. But nothing else rendered them either, so the engine's own
position appeared nowhere at all for its entire life and then materialised,
already finished, with a PnL attached.

The fix keeps the separation and adds a section of its own.
"""
import inspect
import io

from scripts.engine import positions as P


def _js(path):
    return io.open(path, encoding='utf-8').read()


# -- the data exists ----------------------------------------------------------

def test_the_track_record_payload_carries_open_trades():
    src = inspect.getsource(P)
    assert "'open_trades':" in src, (
        'the track record payload still has no open positions, so nothing '
        'downstream can render one'
    )
    assert 'self.wallet.open_positions.values()' in src


def test_the_open_trades_serialiser_tolerates_any_shape():
    """A dataclass, a dict, or a plain object must all survive."""
    from dataclasses import dataclass

    @dataclass
    class _D:
        symbol: str = 'X/USDT'
        entry_price: float = 1.0

    class _O:
        def __init__(self):
            self.symbol = 'Y/USDT'
            self._hidden = 'skip me'

    assert P._asdict_safe(_D())['symbol'] == 'X/USDT'
    assert P._asdict_safe({'symbol': 'Z/USDT'})['symbol'] == 'Z/USDT'
    out = P._asdict_safe(_O())
    assert out['symbol'] == 'Y/USDT'
    assert '_hidden' not in out


def test_a_serialiser_failure_returns_a_dict_not_an_exception():
    assert P._asdict_safe(object()) == {}


# -- the dashboard renders them ----------------------------------------------

def test_the_dashboard_has_a_place_to_put_them():
    html = _js('web/src/pages/dashboard.html')
    assert 'id="enginePositions"' in html
    assert 'id="enginePositionsWrap"' in html
    assert 'Live Engine Positions' in html


def test_open_trades_are_actually_rendered():
    js = _js('web/src/scripts/gatekeeper.js')
    assert 'renderEnginePositions(data.open_trades);' in js, (
        'open_trades is still received and dropped on the floor'
    )
    assert 'function renderEnginePositions(' in js


def test_they_stay_out_of_the_user_trades_panel():
    """The existing separation is deliberate and must survive: renderTrades is
    for positions the USER executed."""
    js = _js('web/src/scripts/gatekeeper.js')
    assert 'renderTrades(data.open_trades' not in js


def test_the_panel_hides_itself_when_there_is_nothing_open():
    js = _js('web/src/scripts/gatekeeper.js')
    i = js.index('function renderEnginePositions(')
    body = js[i:i + 2500]
    assert "wrap.style.display = 'none'" in body


def test_unrealised_pnl_is_computed_not_trusted():
    """pnl_pct on a position is stamped on close; an open row needs it derived
    from the live tick or it reads 0.00% for the whole life of the trade."""
    js = _js('web/src/scripts/gatekeeper.js')
    i = js.index('function renderEnginePositions(')
    body = js[i:i + 2500]
    assert 'currentTickers' in body
    assert 'isLong ? live - entry : entry - live' in body


def test_a_short_is_not_shown_as_a_long():
    js = _js('web/src/scripts/gatekeeper.js')
    i = js.index('function renderEnginePositions(')
    body = js[i:i + 2500]
    assert "side === 'BUY' || side === 'LONG'" in body
    assert 'SHORT' in body


def test_it_survives_a_missing_or_malformed_payload():
    js = _js('web/src/scripts/gatekeeper.js')
    i = js.index('function renderEnginePositions(')
    body = js[i:i + 2500]
    assert 'Array.isArray(trades) ? trades : []' in body


# ── the Discord community invite ─────────────────────────────────────────────
# Added 2026-08-25. There was no "join Discord" button anywhere on the site —
# the only Discord element in Settings was the WEBHOOK URL input, which is a
# different feature (where this user's own alerts get posted). Repurposing that
# field would have broken notification config, so the invite got its own panel.

def test_the_settings_tab_links_to_the_discord_server():
    html = _js('web/src/pages/dashboard.html')
    assert 'https://discord.gg/BjK4nhQZN' in html


def test_the_invite_opens_safely_in_a_new_tab():
    html = _js('web/src/pages/dashboard.html')
    i = html.index('https://discord.gg/BjK4nhQZN')
    anchor = html[max(0, i - 200): i + 200]
    assert 'target="_blank"' in anchor
    assert 'rel="noopener noreferrer"' in anchor, (
        'a target=_blank link without noopener hands the opened page a handle '
        'on this one'
    )


def test_the_webhook_field_is_not_repurposed():
    """The webhook input configures where a user's OWN alerts go. It is not the
    community invite and must keep working."""
    html = _js('web/src/pages/dashboard.html')
    assert 'id="notif-discord-url"' in html
    assert 'placeholder="https://discord.com/api/webhooks/..."' in html


def test_the_invite_is_not_a_webhook():
    """An invite is public; a webhook is a credential. Confusing the two is how
    a credential ends up rendered into a page."""
    html = _js('web/src/pages/dashboard.html')
    i = html.index('https://discord.gg/BjK4nhQZN')
    anchor = html[max(0, i - 300): i + 300]
    assert 'api/webhooks' not in anchor
