"""Telegram connections must live on the volume, and must say so when they don't.

The user connected Telegram, it worked, and then signals stopped arriving. Nothing
was broken about the bot, the token, or the notification pipeline — which is wired
correctly (positions.py:331 send_entry, exits.py:188/211 send_exit).

The connections file was on the container overlay:

    main.py               Path("data/telegram_connections.json")   # relative to CWD
    dispatcher.py         _ROOT / "data" / "telegram_connections.json"

Both resolve to /app/data on Railway, which is wiped on every deploy. Four deploys
on 2026-08-17 erased it four times. Each time the dispatcher found no chat_ids and
`_tg_send_all` returned having sent nothing and logged nothing, so the engine
looked healthy while every subscriber was silently unsubscribed.

This is the SAME defect as instance 2 in docs/SYSTEMS_REVIEW.md — "state persists
on a mounted volume / no volume existed, /app/data wiped every deploy". When the
volume was attached and STATE_DIR introduced, the track record moved and this file
was missed. That is why the track record survives redeploys and Telegram did not.

Two properties are pinned here:
  1. the path resolves under STATE_DIR, not the repo or the CWD
  2. the writer and the reader derive it from the SAME place

Property 2 is the one that matters. Two independent expressions for one file is
how this broke, and how TRACK_RECORD_PATH broke twice before it (see
docs/SYSTEMS_REVIEW.md instances 5 and 7).
"""
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
FILENAME = 'telegram_connections.json'


def _src(rel: str) -> str:
    return (REPO / rel).read_text(encoding='utf-8', errors='replace')


def test_the_reader_derives_the_path_from_state_dir():
    src = _src('scripts/notifications/dispatcher.py')
    line = next((l for l in src.splitlines() if FILENAME in l and 'conn_path' in l), '')
    assert line, 'the dispatcher no longer builds a connections path'
    assert 'STATE_DIR' in src.split(FILENAME)[0][-400:], (
        f'dispatcher reads connections from a non-STATE_DIR path: {line.strip()}'
    )


def test_the_writer_derives_the_path_from_state_dir():
    src = _src('main.py')
    m = re.search(r'^_TG_CONNECTIONS_PATH\s*=\s*(.+)$', src, re.M)
    assert m, '_TG_CONNECTIONS_PATH is gone from main.py'
    assert 'STATE_DIR' in m.group(1), (
        f'main.py writes connections outside the volume: {m.group(1).strip()}'
    )


def test_neither_side_uses_a_bare_relative_path():
    """`Path("data/...")` is resolved against the process CWD, so a process
    started from anywhere else reads and writes a different file.

    The one-time migration in main.py is exempt: reading the OLD ephemeral path is
    the whole point of it. It is matched by variable name rather than by skipping
    every relative path, so a genuine regression still fails here.
    """
    for rel in ('main.py', 'scripts/notifications/dispatcher.py'):
        for line in _src(rel).splitlines():
            stripped = line.strip()
            if FILENAME not in line or stripped.startswith('#'):
                continue
            if '_tg_legacy' in line:
                continue                      # deliberate: migration source
            assert not re.search(r'Path\(\s*[\'"]data/', line), (
                f'{rel} still uses a CWD-relative connections path: {stripped}'
            )


def test_the_migration_only_reads_the_legacy_path_never_writes_it():
    """Migration must be one-way. Writing back to the ephemeral copy would
    reintroduce the split it exists to close."""
    src = _src('main.py')
    for line in src.splitlines():
        if '_tg_legacy' not in line:
            continue
        assert not re.search(r'_tg_legacy\s*\.\s*write_text', line), (
            f'migration writes to the ephemeral path: {line.strip()}'
        )


def test_the_two_sides_resolve_to_the_same_file(tmp_path, monkeypatch):
    """The property that actually broke. Independent expressions for one file is
    how the writer and reader drifted apart."""
    monkeypatch.setenv('AEGIS_STATE_DIR', str(tmp_path))
    import importlib
    from scripts.engine import config as cfg
    importlib.reload(cfg)
    assert (cfg.STATE_DIR / FILENAME) == (tmp_path / FILENAME)
    assert 'app' not in str(cfg.STATE_DIR / FILENAME).replace(str(tmp_path), '')


def test_the_silent_paths_now_report():
    """A delivery failure that logs nothing is indistinguishable from success.

    Both quiet failures are covered: an unset token, and zero connected chat_ids
    after a deploy wiped the file. The per-chat send result is also no longer
    discarded — send_telegram returns a bool and it is now counted.
    """
    src = _src('scripts/notifications/dispatcher.py')
    # The whole method, not a fixed-size window: a slice too short to reach the
    # code under test passes or fails for the wrong reason.
    start = src.index('def _tg_send_all')
    nxt = src.find('\n    def ', start + 1)
    body = src[start: nxt if nxt != -1 else len(src)]
    assert 'TELEGRAM_BOT_TOKEN is not set' in body, 'a missing token is silent again'
    assert 'no connected chat_ids' in body, 'an empty registry is silent again'
    assert re.search(r'if\s+chat_id\s+and\s+send_telegram\(', body), (
        'the send result is discarded again — failures cannot be counted'
    )
