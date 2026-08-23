"""Signals live on the volume, not only on a filesystem that gets wiped.

Asked 2026-08-23: "so signals will be stored into volume now?" — and then
"shift all the signals into volume then, every kind of".

Honest answer at the time was NO. Taking signals off Firestore removed the write
cap problem, but it left them in exactly two places:

  * LIVE_STATE, in memory - gone on restart by definition
  * WEB_ROOT/src/data/live_signals.json - the CONTAINER filesystem, wiped on
    every deploy and only rebuilt on the next producer tick

So every deploy opened a window where the dashboard fetched the snapshot and got
nothing until the engine had warmed up and scanned again. To a subscriber that
is indistinguishable from the engine being down.

The served copy has to stay under WEB_ROOT because it is fetched as a static
asset. So the volume holds the durable copy, the served copy is seeded from it
at boot, and the producer writes both.

The volume write is gated on a content change: the producer loop ticks every
second while signals only change once a scan, and the volume is persistent disk
rather than throwaway container FS.
"""
import inspect
import os

import main
from scripts.engine import config as ECFG


SRC = inspect.getsource(main)


def test_the_snapshot_has_a_volume_path():
    assert hasattr(main, '_SIGNALS_SNAPSHOT_PATH')
    assert str(main._SIGNALS_SNAPSHOT_PATH).endswith('live_signals.json')
    assert main._SIGNALS_SNAPSHOT_PATH.parent == main._STATE_DIR, (
        'the durable snapshot is not on the state volume'
    )


def test_the_volume_copy_is_written_by_the_producer():
    assert '_SIGNALS_SNAPSHOT_PATH.parent.mkdir' in SRC
    assert 'os.replace(_vol_tmp, _SIGNALS_SNAPSHOT_PATH)' in SRC, (
        'the volume copy is not written atomically'
    )


def test_the_volume_write_is_gated_on_a_real_change():
    """The loop ticks every second; the signals change once a scan."""
    assert '_last_snapshot_hash' in SRC
    assert 'if _h != _last_snapshot_hash:' in SRC


def test_the_served_copy_is_still_written_every_tick():
    """It is cheap container FS and it is what the browser fetches."""
    assert "signals_dir = WEB_ROOT_PATH / 'src' / 'data'" in SRC
    assert 'os.replace(temp_file, signals_file)' in SRC


def test_the_served_copy_is_seeded_from_the_volume_at_boot():
    assert hasattr(main, '_seed_signals_snapshot_from_volume')
    src = inspect.getsource(main._seed_signals_snapshot_from_volume)
    assert '_SIGNALS_SNAPSHOT_PATH' in src
    assert 'WEB_ROOT_PATH' in src


def test_the_seeder_runs_in_lifespan():
    assert '_seed_signals_snapshot_from_volume()' in SRC
    i = SRC.index('async def lifespan')
    j = SRC.index('engine_task', i)
    assert '_seed_signals_snapshot_from_volume()' in SRC[i:j], (
        'the seeder does not run before the engine starts'
    )


def test_the_seeder_never_breaks_boot():
    """A missing or unreadable volume copy is the old behaviour, not a crash."""
    src = inspect.getsource(main._seed_signals_snapshot_from_volume)
    assert 'except Exception' in src
    assert 'if not _SIGNALS_SNAPSHOT_PATH.exists():' in src
    assert 'return' in src


def test_an_empty_volume_file_is_not_served():
    src = inspect.getsource(main._seed_signals_snapshot_from_volume)
    assert 'if not blob.strip():' in src


# -- the rest of the engine's durable state --------------------------------

def test_every_engine_store_is_on_the_state_dir():
    for name in ('TRACK_RECORD_PATH', 'ALPHA_TRACK_RECORD_PATH', 'PERF_STATE_PATH',
                 'DRIFT_STATE_PATH', 'WORKING_ORDER_LOG_PATH', 'SCALP_RECORD_PATH',
                 'SHADOW_EXITS_PATH'):
        path = getattr(ECFG, name)
        assert path.parent == ECFG.STATE_DIR, f'{name} is not on the volume'


def test_the_shadow_book_no_longer_writes_to_the_container_fs():
    """It used to be <repo>/data/shadow_exits.json — the container filesystem —
    so every deploy threw away the observation history the book exists to
    accumulate.

    Asserted at SOURCE level rather than by comparing the bound value: _STORE is
    resolved once at import, and test_state_dir_mount_guard reloads config with a
    tmp STATE_DIR, so the two module attributes legitimately disagree mid-suite.
    In production the env is set before anything imports, which is the case that
    matters.
    """
    import inspect
    from scripts.engine import shadow_exits
    src = inspect.getsource(shadow_exits)
    assert 'from scripts.engine.config import SHADOW_EXITS_PATH as _STORE' in src, (
        'the shadow book no longer takes its path from config, so it is free to '
        'drift back onto the container filesystem'
    )
    assert shadow_exits._STORE.name == 'shadow_exits.json'


def test_the_shadow_book_path_follows_the_state_dir():
    """Reload both together: they must agree on where the volume is."""
    import importlib
    from scripts.engine import config as _c, shadow_exits as _se
    _c = importlib.reload(_c)
    _se = importlib.reload(_se)
    assert _se._STORE == _c.SHADOW_EXITS_PATH
    assert _se._STORE.parent == _c.STATE_DIR


def test_the_telegram_and_entitlement_stores_are_on_the_volume():
    for attr in ('_TG_CONNECTIONS_PATH', '_TG_PENDING_PATH', '_ENTITLEMENTS_PATH',
                 '_RUNTIME_PATH'):
        assert getattr(main, attr).parent == main._STATE_DIR, f'{attr} is not on the volume'
