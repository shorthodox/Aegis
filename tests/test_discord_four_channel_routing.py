"""Four Discord channels, one router, and a refusal feed that stays readable.

Asked 2026-08-25: post accepted / rejected / closed / open events to four
separate Discord channels, with the reasoning attached to both the rejected and
the executed ones.

ROUTING
    accepted / live   -> signals    (already existed)
    closed  (final)   -> records    (already existed)
    rejected          -> refusals   (new)
    open / partial    -> positions  (new)

The partial-close split is the subtle one. A five-rung ladder banks profit up to
five times per trade, and every one of those went through send_exit. Routed
naively, ONE trade would appear in the track record five times and the published
win count would be inflated by its own exit mechanics. A partial leaves the
position open, so it is a position update; only the final close is a record.

THE REFUSAL FEED IS DEDUPLICATED, AND HAS TO BE.

The desk refuses roughly 200 setups per scan and scans about every 60 seconds:
~290,000 messages a day against a webhook limit near 30 a minute. Posting each
occurrence would be rate-limited into uselessness within the first minute, and
unreadable even if it were not. What carries information is a CHANGE of verdict
— a symbol refused all day for "mid-range, no edge" is one fact, not a thousand
— so a refusal posts when the reason changes, and again only after a quiet
period. Every distinct refusal still reaches the channel with its full reasoning.

FAILURE HANDLING lives in discord_notifier.send_discord: retry with full jitter
on 429 and 5xx, honouring Discord's own retry_after, and never raising — a
notification must never be able to interrupt an exit.
"""
import inspect

import pytest

from scripts.notifications import discord_notifier as N
from scripts.notifications import formatter as F
from scripts.notifications.dispatcher import NotificationDispatcher as D


# -- four channels, four credentials ------------------------------------------

def test_all_four_channels_exist():
    assert sorted(D._CHANNELS) == ['positions', 'records', 'refusals', 'signals']


def test_each_channel_has_its_own_environment_variable():
    envs = set(D._ENV_KEYS.values())
    for name in ('AEGIS_DISCORD_WEBHOOK_SIGNALS', 'AEGIS_DISCORD_WEBHOOK_RECORDS',
                 'AEGIS_DISCORD_WEBHOOK_REFUSALS', 'AEGIS_DISCORD_WEBHOOK_POSITIONS'):
        assert name in envs, name


def test_every_channel_maps_to_a_real_settings_key():
    from scripts.notifications.dispatcher import _DEFAULT_SETTINGS
    for ch, key in D._CHANNELS.items():
        assert key in _DEFAULT_SETTINGS, f'{ch} points at a key that does not exist'


def test_an_unconfigured_channel_falls_back_rather_than_dropping():
    src = inspect.getsource(D._do_send)
    assert 'or cfg.get("discord_webhook_url", "")' in src


# -- a partial close is not a track-record entry ------------------------------

def test_a_partial_close_goes_to_positions_not_records():
    src = inspect.getsource(D.send_exit)
    assert '"position still open" in (exit_reason or "")' in src
    assert '"positions" if _partial else "records"' in src, (
        'a partial close still lands in the track record, so one laddered trade '
        'is counted up to five times'
    )


# -- refusals are deduplicated ------------------------------------------------

def _disp():
    d = D.__new__(D)
    d._refusal_seen = {}
    return d


def test_the_same_refusal_twice_posts_once(monkeypatch):
    sent = []
    d = _disp()
    monkeypatch.setattr(d, '_load_settings', lambda: {'enabled': True}, raising=False)
    monkeypatch.setattr(d, '_pool', type('P', (), {
        'submit': lambda self, *a, **k: sent.append(a)})(), raising=False)

    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    assert len(sent) == 1, 'a repeated refusal posted twice'


def test_a_changed_reason_posts_again(monkeypatch):
    sent = []
    d = _disp()
    monkeypatch.setattr(d, '_load_settings', lambda: {'enabled': True}, raising=False)
    monkeypatch.setattr(d, '_pool', type('P', (), {
        'submit': lambda self, *a, **k: sent.append(a)})(), raising=False)

    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    d.send_refusal('ENA/USDT', 'BUY', 'quality 40/100 is below the 60 floor')
    assert len(sent) == 2, 'a new verdict was suppressed as a duplicate'


def test_different_symbols_do_not_suppress_each_other(monkeypatch):
    sent = []
    d = _disp()
    monkeypatch.setattr(d, '_load_settings', lambda: {'enabled': True}, raising=False)
    monkeypatch.setattr(d, '_pool', type('P', (), {
        'submit': lambda self, *a, **k: sent.append(a)})(), raising=False)

    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    d.send_refusal('SOL/USDT', 'BUY', 'mid-range, no edge')
    assert len(sent) == 2


def test_refusals_go_to_the_refusals_channel(monkeypatch):
    sent = []
    d = _disp()
    monkeypatch.setattr(d, '_load_settings', lambda: {'enabled': True}, raising=False)
    monkeypatch.setattr(d, '_pool', type('P', (), {
        'submit': lambda self, *a, **k: sent.append(a)})(), raising=False)
    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    assert sent[0][-1] == 'refusals'


def test_refusals_do_not_go_to_telegram(monkeypatch):
    """A refusal every few minutes on a subscriber's phone is unbearable."""
    sent = []
    d = _disp()
    monkeypatch.setattr(d, '_load_settings', lambda: {'enabled': True}, raising=False)
    monkeypatch.setattr(d, '_pool', type('P', (), {
        'submit': lambda self, *a, **k: sent.append(a)})(), raising=False)
    d.send_refusal('ENA/USDT', 'BUY', 'mid-range, no edge')
    # submit(fn, cfg, discord_payload, tg_text, wa_text, channel)
    _fn, _cfg, _dp, tg_text, wa_text, _ch = sent[0]
    assert tg_text is None and wa_text is None


def test_the_repeat_window_is_sane():
    assert 60.0 <= D.REFUSAL_REPEAT_S <= 3600.0


# -- embeds carry the reasoning -----------------------------------------------

def test_a_refusal_embed_leads_with_the_reason():
    e = F.format_refusal_discord(
        'ENA/USDT', 'SELL', 'short at 6% of the whole structure',
        stage='location', setup='RANGE_FADE', price=0.16, quality=62)['embeds'][0]
    assert e['description'] == 'short at 6% of the whole structure'
    names = [f['name'] for f in e['fields']]
    assert 'Refused at' in names and 'Setup' in names


def test_a_refusal_is_amber_not_red():
    """A refusal is the system working, not a loss."""
    e = F.format_refusal_discord('X/USDT', 'BUY', 'r')['embeds'][0]
    assert e['color'] == 0xE8A06A
    loss = F.format_exit_discord('X/USDT', 'BUY', 'LOSS', -2.0, 60, 'STOP HIT')['embeds'][0]
    assert e['color'] != loss['color']


def test_an_open_position_is_blue_and_carries_its_why():
    e = F.format_position_open_discord(
        'ENA/USDT', 'BUY', 0.1609, 0.1540, [0.1633, 0.1657],
        1700, 'three consecutive 5m candles turned', 'TREND_PULLBACK', 2.31)['embeds'][0]
    assert e['color'] == 0x3B82F6
    assert 'three consecutive 5m candles turned' in e['description']


def test_the_open_embed_states_risk_as_a_percentage():
    e = F.format_position_open_discord('X/USDT', 'BUY', 100.0, 99.0)['embeds'][0]
    stop_field = [f for f in e['fields'] if f['name'] == 'Stop'][0]
    assert '1.00%' in stop_field['value']


def test_a_missing_reason_does_not_render_an_empty_embed():
    assert F.format_refusal_discord('X/USDT', 'BUY', '')['embeds'][0]['description']


# -- failures never reach the trading logic -----------------------------------

def test_the_notifier_retries_rate_limits_and_server_errors():
    assert 429 in N.RETRY_STATUS
    assert {500, 502, 503, 504} <= N.RETRY_STATUS


def test_it_honours_discords_own_retry_after():
    src = inspect.getsource(N.send_discord)
    assert 'retry_after' in src


def test_a_permanent_error_is_not_retried():
    """404 (deleted webhook) or 400 (bad embed) needs a human, not four tries."""
    assert 404 not in N.RETRY_STATUS
    assert 400 not in N.RETRY_STATUS


def test_backoff_uses_jitter():
    src = inspect.getsource(N._sleep)
    assert 'random.uniform' in src, 'lockstep retries turn a rate limit into a herd'


def test_send_never_raises(monkeypatch):
    class _Boom:
        def post(self, *a, **k): raise RuntimeError('network gone')
    monkeypatch.setitem(__import__('sys').modules, 'requests', _Boom())
    monkeypatch.setattr(N, 'MAX_ATTEMPTS', 1)
    monkeypatch.setattr(N, '_sleep', lambda *a, **k: None)
    assert N.send_discord('https://example.invalid/x', {'embeds': []}) is False


def test_an_empty_url_is_a_no_op():
    assert N.send_discord('', {'embeds': []}) is False


def test_the_dispatch_methods_swallow_their_own_errors():
    for m in (D.send_refusal, D.send_position_open):
        assert 'except Exception' in inspect.getsource(m)


# -- wired into the pipeline --------------------------------------------------

def test_refusals_are_wired_where_the_desk_refuses():
    import scripts.engine.engine as E
    src = inspect.getsource(E.LiveEngine._publish_no_trade)
    assert 'send_refusal(' in src


def test_position_open_is_wired_where_a_position_opens():
    import scripts.engine.positions as P
    src = inspect.getsource(P)
    assert 'send_position_open(' in src


def test_the_manual_check_script_exists():
    from pathlib import Path
    p = Path('scripts/check_discord_channels.py')
    assert p.exists()
    body = p.read_text(encoding='utf-8')
    assert '--send' in body, 'the script must default to a dry run'
    for ch in ('signals', 'records', 'refusals', 'positions'):
        assert ch in body


# ── a swallowed notification must not be a silent one ────────────────────────
# The `time` import bug: `except Exception: pass` around send_refusal hid a bare
# NameError — time was never imported in dispatcher.py — so the method raised on
# its first line and posted nothing, for every refusal, with no log line
# anywhere. It was indistinguishable from a channel nobody had configured.

def test_a_programming_error_is_logged_as_a_bug(caplog):
    from scripts.notifications.dispatcher import _swallow
    with caplog.at_level('ERROR'):
        _swallow('send_refusal', NameError("name 'time' is not defined"))
    assert 'is broken, not merely failing' in caplog.text
    assert 'send_refusal' in caplog.text


def test_an_outside_world_failure_is_only_a_warning(caplog):
    from scripts.notifications.dispatcher import _swallow
    with caplog.at_level('WARNING'):
        _swallow('send_exit', ConnectionError('reset by peer'))
    assert 'failed' in caplog.text
    assert 'is broken' not in caplog.text


def test_the_exact_bug_that_hid_would_now_be_reported(caplog):
    """NameError is the one that actually bit."""
    from scripts.notifications.dispatcher import _swallow
    for exc in (NameError('x'), AttributeError('x'), TypeError('x'),
                KeyError('x'), ImportError('x')):
        caplog.clear()
        with caplog.at_level('ERROR'):
            _swallow('send_refusal', exc)
        assert 'is broken' in caplog.text, type(exc).__name__


def test_time_is_actually_imported():
    """The bug itself, asserted directly."""
    import scripts.notifications.dispatcher as m
    assert hasattr(m, 'time')


def test_the_notify_boundaries_still_swallow():
    """They must not start raising — a notification cannot be allowed to
    interrupt a fill or an exit."""
    import inspect
    from scripts.notifications.dispatcher import NotificationDispatcher as D
    for m in (D.send_refusal, D.send_position_open):
        src = inspect.getsource(m)
        assert 'except Exception as exc:' in src
        assert '_swallow(' in src


def test_swallow_is_module_level_not_a_class_attribute():
    """Defining it inside the class body silently ended the class and turned
    every method after it into a module-level function."""
    from scripts.notifications.dispatcher import NotificationDispatcher as D
    import scripts.notifications.dispatcher as m
    assert callable(m._swallow)
    assert not hasattr(D, '_swallow')
    for name in ('send_refusal', 'send_position_open', 'send_entry',
                 'send_exit', '_do_send', '_load_settings'):
        assert hasattr(D, name), f'{name} fell out of the class'
