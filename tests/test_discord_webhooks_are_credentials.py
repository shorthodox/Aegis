"""Discord webhook URLs are credentials, and closes are not signals.

Set up 2026-08-24, when the two webhooks were created:

    #live-signals  -> AEGIS
    #track-record  -> AEGIS Records

A Discord webhook URL is a bearer credential: anyone holding one can post as the
bot into that channel, with no token and no auth step. THIS REPOSITORY IS PUBLIC
and has leaked committed secrets before, so the URLs are read from the
environment and never live in a tracked file.

Three things were wrong with the plumbing before the split.

1 - ONE WEBHOOK. Fires and closes went to the same channel, so #live-signals
    would have been a mixed feed and #track-record silent.

2 - THE SETTINGS FILE WAS ON THE CONTAINER FILESYSTEM. _SETTINGS_PATH resolved
    to _ROOT/"data", not the volume, so anything saved through the settings API
    - a webhook URL, a chat id - was wiped by the next deploy. Same class of bug
    as shadow_exits.json.

3 - THE FILE WAS THE ONLY SOURCE. A URL saved there is a secret sitting in the
    working tree next to tracked files, one `git add -A` away from being public.
    The environment now wins over the file, so a rotation on the host takes
    effect without editing anything on disk.
"""
import inspect
import io
import os
import re

from scripts.notifications import dispatcher as D


SRC = inspect.getsource(D)


# -- the URLs are environment credentials -------------------------------------

def test_both_webhooks_read_from_the_environment():
    assert D.NotificationDispatcher._ENV_KEYS['discord_webhook_url'] == \
        'AEGIS_DISCORD_WEBHOOK_SIGNALS'
    assert D.NotificationDispatcher._ENV_KEYS['discord_webhook_records_url'] == \
        'AEGIS_DISCORD_WEBHOOK_RECORDS'


def test_the_environment_beats_the_settings_file(monkeypatch, tmp_path):
    """A value on the host is current intent; a file value may predate a
    rotation."""
    f = tmp_path / 'notification_settings.json'
    f.write_text('{"discord_webhook_url": "https://example.invalid/STALE"}',
                 encoding='utf-8')
    monkeypatch.setattr(D, '_SETTINGS_PATH', f)
    monkeypatch.setenv('AEGIS_DISCORD_WEBHOOK_SIGNALS', 'https://example.invalid/FRESH')
    cfg = D.NotificationDispatcher._load_settings(D.NotificationDispatcher)
    assert cfg['discord_webhook_url'] == 'https://example.invalid/FRESH'


def test_an_unset_variable_does_not_blank_the_file_value(monkeypatch, tmp_path):
    """Only a NON-EMPTY env value overrides — otherwise an unset host would
    silently disable a configured webhook."""
    f = tmp_path / 'notification_settings.json'
    f.write_text('{"discord_webhook_url": "https://example.invalid/FROMFILE"}',
                 encoding='utf-8')
    monkeypatch.setattr(D, '_SETTINGS_PATH', f)
    monkeypatch.delenv('AEGIS_DISCORD_WEBHOOK_SIGNALS', raising=False)
    cfg = D.NotificationDispatcher._load_settings(D.NotificationDispatcher)
    assert cfg['discord_webhook_url'] == 'https://example.invalid/FROMFILE'


def test_no_webhook_url_is_committed_anywhere():
    """The property that actually matters. A real Discord webhook is
    https://discord.com/api/webhooks/<snowflake>/<token>."""
    import subprocess
    out = subprocess.run(['git', 'grep', '-lE',
                          r'discord\.com/api/webhooks/[0-9]{5,}'],
                         capture_output=True, text=True)
    hits = [l for l in out.stdout.splitlines() if l.strip()]
    assert not hits, f'a Discord webhook URL is committed in: {hits}'


def test_the_defaults_ship_empty():
    assert D._DEFAULT_SETTINGS['discord_webhook_url'] == ''
    assert D._DEFAULT_SETTINGS['discord_webhook_records_url'] == ''


# -- settings survive a deploy ------------------------------------------------

def test_the_settings_file_is_on_the_volume():
    assert 'AEGIS_STATE_DIR' in SRC, (
        'the settings file is back on the container filesystem, so anything '
        'saved through the API is wiped by the next deploy'
    )


# -- closes go to the records channel ----------------------------------------

def test_a_close_is_routed_to_the_records_webhook():
    src = inspect.getsource(D.NotificationDispatcher.send_exit)
    assert 'cfg, dp, tg, wa, True' in src, (
        'a close still posts to #live-signals, so #track-record stays empty'
    )


def test_a_fire_is_not_routed_to_records():
    src = inspect.getsource(D.NotificationDispatcher.send_entry)
    assert 'cfg, dp, tg, wa, True' not in src


def test_the_router_falls_back_when_only_one_is_configured():
    """A half-set-up server must still post rather than drop closes silently."""
    src = inspect.getsource(D.NotificationDispatcher._do_send)
    assert 'if not _wh:' in src
    assert 'to_records' in src


def test_records_routing_is_off_by_default():
    sig = inspect.signature(D.NotificationDispatcher._do_send)
    assert sig.parameters['to_records'].default is False
