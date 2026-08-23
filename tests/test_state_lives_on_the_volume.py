"""Move what we can off Firestore and onto the Railway volume.

Asked 2026-08-23: "let's shift telegram notification to railway volume, we
should shift almost everything to railway volume that we can".

Context: aegis-d78e1 is on the Firestore FREE tier (20,000 writes/day) and was
exhausting it by mid-morning every day, taking the dashboard and Telegram silent
while the engine kept working. Rationing the writes helped; removing the need
for them is the durable fix.

THE SIGNALS COLLECTION was the dominant consumer (~8,640 writes/day) and was
pure duplication. The dashboard already receives every signal twice over:

  1. /ws/dashboard - full payloads every ~0.5s, straight from LIVE_STATE.
     gatekeeper.js _buildSignalObj writes them into window.latestSignals - the
     SAME store the Firestore listener wrote to.
  2. /web/src/data/live_signals.json - rewritten every producer tick, polled
     every 30s as the state backstop.

It was also the LEAST reliable of the three: the v80 note on _pollSnapshotState
records production carrying armed signals in the snapshot file while the
Firestore docs held only stale fires, because one NaN fails a whole batch
silently.

TELEGRAM delivery was already volume-backed (telegram_connections.json,
telegram_pending.json). What still reached for Firestore was refreshing
access_until. Paid grants are written to the volume by _ent_grant BEFORE the
Firestore write, so the volume can answer for exactly the users who receive
signals - and must, because Firestore being over quota is precisely when a
paying subscriber must not silently stop receiving them.
"""
import inspect
import io
import os
import re

import main


SRC = inspect.getsource(main)


# -- signals are no longer published to Firestore ------------------------------

def test_signal_publishing_is_off_by_default():
    assert '_PUBLISH_SIGNALS_TO_FIRESTORE' in SRC
    m = re.search(r"_PUBLISH_SIGNALS_TO_FIRESTORE = os\.getenv\(\s*'([^']+)',\s*'([^']+)'", SRC)
    assert m, 'the switch is not env-driven'
    assert m.group(2) == '0', 'Firestore signal publishing defaults back ON'


def test_the_env_var_can_restore_it():
    assert 'AEGIS_FIRESTORE_PUBLISH_SIGNALS' in SRC


def test_the_push_is_gated_on_the_switch():
    assert '_PUBLISH_SIGNALS_TO_FIRESTORE\n' in SRC or '_PUBLISH_SIGNALS_TO_FIRESTORE' in SRC
    assert 'should_push = (_PUBLISH_SIGNALS_TO_FIRESTORE' in SRC, (
        'the push still runs regardless of the switch'
    )


def test_the_restart_stale_sweep_is_gated_too():
    """A full-collection write on every restart, for docs nothing reads."""
    assert 'if not _stale_sweep_done and _PUBLISH_SIGNALS_TO_FIRESTORE:' in SRC


def test_the_producer_loop_still_sleeps():
    """A bare `continue` in this block would skip the loop's own
    `await asyncio.sleep(1)` and spin the producer flat out."""
    i = SRC.index('_PUBLISH_SIGNALS_TO_FIRESTORE and not _signals_off_announced')
    j = SRC.index('Warmup in progress', i)
    guard = SRC[i:j]
    assert 'continue' not in guard, (
        'the guard uses `continue`, which skips the producer loop sleep and '
        'busy-spins the event loop'
    )


# -- the client no longer subscribes ------------------------------------------

def _gatekeeper():
    return io.open('web/src/scripts/gatekeeper.js', encoding='utf-8').read()


def test_the_client_does_not_subscribe_to_firestore_signals():
    js = _gatekeeper()
    assert '_useFirestoreSignals' in js
    assert 'signalsUnsubscribe = !_useFirestoreSignals ? null : onSnapshot(' in js, (
        'the dashboard still opens a Firestore listener for signals'
    )


def test_the_websocket_still_fills_the_same_store():
    """The whole argument for removing the listener."""
    js = _gatekeeper()
    assert 'window.latestSignals[key] = signalObj;' in js


def test_the_snapshot_poller_survives():
    js = _gatekeeper()
    assert '_pollSnapshotState' in js
    assert 'live_signals.json' in js


def test_the_store_is_not_wiped_on_resubscribe():
    """It used to be reset to {}, which would now discard WebSocket state."""
    js = _gatekeeper()
    assert 'window.latestSignals = window.latestSignals || {}; // populated by the WS' in js


# -- telegram reads the volume first ------------------------------------------

def test_telegram_access_checks_the_volume_before_firestore():
    src = inspect.getsource(main._tg_access_until)
    vol = src.index('_ent_load()')
    fs = src.index('get_user_doc(')
    assert vol < fs, 'Firestore is still consulted before the volume'


def test_a_cancelled_grant_still_honours_its_end_date():
    """Cancellation means 'will not renew', not 'ends now'."""
    src = inspect.getsource(main._tg_access_until)
    assert 'canceled' in src
    assert 'if _end:' in src


def test_a_volume_read_failure_falls_through_rather_than_denying():
    src = inspect.getsource(main._tg_access_until)
    assert 'volume entitlement read failed' in src


def test_telegram_stores_are_on_the_volume():
    assert '_STATE_DIR / "telegram_connections.json"' in SRC
    assert '_STATE_DIR / "telegram_pending.json"' in SRC
    assert '_STATE_DIR / "entitlements.json"' in SRC
