"""A STATE_GENERATION bump must actually clear the track record.

Three admin resets were run and the record kept coming back. Two separate reasons,
both now closed:

  1. POST /api/admin/reset-track-record wrote web/track_record.json twice and
     never touched STATE_DIR/track_record.json — fixed earlier (commit 40caf00d).

  2. Nothing could clear it WITHOUT the admin key, and the on-disk file is
     self-perpetuating: _save_track_record rebuilds the file from the wallet and
     then re-adds every on-disk record the wallet no longer holds, under a comment
     that says outright "records are never lost due to restarts or wallet resets".
     With an empty wallet and a populated file, all 40 records were re-orphaned
     and written straight back on the very next save.

The generation guard was supposed to be the escape hatch — its own comment says
"a bump wipes stale history exactly once" — but it only ever compared Firestore's
generation, and it returned early before loading anything whenever a populated
LOCAL file existed. So it could never see the file that was the problem.

Now the local record carries `generation`, the guard checks it, and a bump is a
deploy-triggered wipe that needs no credentials.
"""
import json

import pytest

from scripts.engine import config as _cfg
from scripts.engine import state as _state


def test_the_generation_is_an_integer_that_can_be_bumped():
    assert isinstance(_cfg.STATE_GENERATION, int)
    assert _cfg.STATE_GENERATION >= 3, (
        'generation regressed below 3 — the 2026-08-19 wipe bump was undone'
    )


def test_the_saved_payload_stamps_the_generation():
    """Without the stamp the guard has nothing to compare and every local file
    reads as generation 1."""
    import inspect
    from scripts.engine.positions import PositionsMixin
    src = inspect.getsource(PositionsMixin._save_track_record)
    assert "'generation'" in src, 'the saved payload no longer stamps a generation'


def test_a_stale_local_record_is_wiped(tmp_path, monkeypatch):
    """The behaviour that was missing: a populated file from an OLD generation
    must be emptied at boot, not preserved."""
    rec = tmp_path / 'track_record.json'
    rec.write_text(json.dumps({
        'generation': _cfg.STATE_GENERATION - 1,
        'signals': [{'signal_id': 'x', 'symbol': 'BTC/USDT', 'outcome': 'WIN'}],
    }), encoding='utf-8')
    monkeypatch.setattr(_state, 'TRACK_RECORD_PATH', rec)
    monkeypatch.setattr(_state, '_fs_clear_track_record', lambda: None)
    monkeypatch.setattr(_state, '_fs_load_track_record', lambda: None)

    _state._hydrate_track_record_from_firestore()

    after = json.loads(rec.read_text(encoding='utf-8'))
    assert after['signals'] == [], 'a stale-generation record survived the wipe'
    assert after['generation'] == _cfg.STATE_GENERATION, (
        'the wiped file was not re-stamped — the next boot would wipe again, '
        'forever'
    )


def test_a_current_generation_record_is_left_alone(tmp_path, monkeypatch):
    """The wipe is ONE-TIME. A current-generation record must survive untouched,
    or every restart would destroy live history."""
    rec = tmp_path / 'track_record.json'
    payload = {
        'generation': _cfg.STATE_GENERATION,
        'signals': [{'signal_id': 'keep', 'symbol': 'ETH/USDT', 'outcome': 'WIN'}],
    }
    rec.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(_state, 'TRACK_RECORD_PATH', rec)
    monkeypatch.setattr(_state, '_fs_clear_track_record',
                        lambda: pytest.fail('Firestore cleared on a current record'))
    monkeypatch.setattr(_state, '_fs_load_track_record',
                        lambda: pytest.fail('hydrate ran on a current record'))

    _state._hydrate_track_record_from_firestore()

    assert json.loads(rec.read_text(encoding='utf-8'))['signals'] == payload['signals']


def test_a_failed_firestore_clear_does_not_undo_the_local_wipe(tmp_path, monkeypatch):
    """Firestore's write quota is exhausted, so _fs_clear_track_record raising is
    the EXPECTED path right now. The local wipe must still stand."""
    rec = tmp_path / 'track_record.json'
    rec.write_text(json.dumps({'generation': 1, 'signals': [{'signal_id': 'old'}]}),
                   encoding='utf-8')
    monkeypatch.setattr(_state, 'TRACK_RECORD_PATH', rec)
    def _boom():
        raise RuntimeError('429 quota exhausted')
    monkeypatch.setattr(_state, '_fs_clear_track_record', _boom)
    monkeypatch.setattr(_state, '_fs_load_track_record', lambda: None)

    _state._hydrate_track_record_from_firestore()

    assert json.loads(rec.read_text(encoding='utf-8'))['signals'] == []
