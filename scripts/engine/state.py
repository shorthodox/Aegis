"""Durable track-record state: local file + best-effort Firestore mirror.

Railway's filesystem is ephemeral, so the local track record cannot be the only
copy — see config.py for the full note. Everything here is best-effort by
design: if Firestore is unreachable the engine falls back to local-file
behaviour and signal generation continues. Nothing in this module may raise into
the scan loop.

`_FS_DOWN` is a module-level circuit breaker. The backend Firebase project may
have no Firestore database at all, in which case every call fails and retries
would stall the scan loop; after the first failure we stop trying for the
lifetime of the process.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from scripts.engine.config import (
    FS_STATE_COLLECTION, FS_STATE_DOC, ROOT, STATE_GENERATION, TRACK_RECORD_PATH,
)

__all__ = [
    "_fs_state_client", "_fs_save_track_record", "_fs_clear_track_record",
    "_fs_load_track_record", "_hydrate_track_record_from_firestore",
    "is_firestore_down",
]

_FS_DOWN = False


def is_firestore_down() -> bool:
    """Whether the circuit breaker has tripped (for status reporting)."""
    return _FS_DOWN


def _fs_state_client():
    """Best-effort Firestore client for durable state (None if unavailable or the
    circuit breaker has tripped)."""
    if _FS_DOWN:
        return None
    try:
        import firebase_admin
        from firebase_admin import credentials as _creds, firestore as _fs
        cred_path = ROOT / 'config' / 'serviceAccountKey.json'
        if not cred_path.exists():
            return None
        if not firebase_admin._apps:
            firebase_admin.initialize_app(_creds.Certificate(str(cred_path)))
        return _fs.client()
    except Exception:
        return None


def _fs_save_track_record(payload: dict) -> None:
    """Mirror the track record to Firestore (best-effort, never raises)."""
    try:
        db = _fs_state_client()
        if db is None:
            return
        # Firestore's per-doc limit is ~1 MB; store only the restore-critical
        # subset (the capped signals list + summary), which the wallet reads back.
        slim = {
            'signals':      payload.get('signals', []),
            'summary':      payload.get('summary', {}),
            'gate_version': payload.get('engine_version', ''),
            'generated_at': payload.get('generated_at', ''),
            'generation':   STATE_GENERATION,
        }
        db.collection(FS_STATE_COLLECTION).document(FS_STATE_DOC).set(slim)
    except Exception:
        global _FS_DOWN
        _FS_DOWN = True   # trip breaker — stop retrying a broken datastore


def _fs_clear_track_record() -> None:
    """Delete the durable track record from Firestore.  Used by the admin reset so
    a deliberate wipe is NOT resurrected by the hydrate on the next redeploy."""
    try:
        db = _fs_state_client()
        if db is None:
            return
        db.collection(FS_STATE_COLLECTION).document(FS_STATE_DOC).delete()
    except Exception:
        pass


def _fs_load_track_record() -> Optional[dict]:
    """Fetch the durable track record from Firestore (None if absent/unreachable)."""
    global _FS_DOWN
    if _FS_DOWN:            # breaker already tripped — don't retry a broken datastore
        return None
    try:
        db = _fs_state_client()
        if db is None:
            return None
        # snap typed Any: _fs_state_client() returns the SYNC firestore client, so
        # .get() yields a DocumentSnapshot — but the stubs resolve to the AsyncClient
        # (Awaitable[DocumentSnapshot]), falsely flagging .exists/.to_dict(). This is
        # a sync call; do NOT await it.
        snap: Any = db.collection(FS_STATE_COLLECTION).document(FS_STATE_DOC).get()
        if not snap.exists:
            return None
        return snap.to_dict()
    except Exception:
        _FS_DOWN = True   # trip breaker — stop retrying a broken datastore
        return None


def _write_empty_track_record() -> None:
    """Empty the local record, stamped with the CURRENT generation.

    Stamping matters: an empty file written without a generation would read as
    generation 1 on the next boot and trip the wipe again every restart.
    """
    from datetime import datetime, timezone
    TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'generation':   STATE_GENERATION,
        'summary':      {},
        'signals':      [],
    }
    tmp = TRACK_RECORD_PATH.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    tmp.replace(TRACK_RECORD_PATH)


def _hydrate_track_record_from_firestore() -> None:
    """On boot, if the local track record is missing/empty (ephemeral FS after a
    redeploy), restore it from Firestore so history is never lost.  Writes the
    same file VirtualWallet._load_history reads, so restore is transparent."""
    try:
        if TRACK_RECORD_PATH.exists() and TRACK_RECORD_PATH.stat().st_size > 2:
            # A STATE_GENERATION bump is the deploy-triggered wipe, and it has to
            # be able to reach THIS file. Previously the generation was only ever
            # compared against the Firestore copy — which this early return meant
            # we never even loaded whenever a populated local record existed. So a
            # bump could not clear the one file that was keeping stale history
            # alive, and _save_track_record's orphan-preservation (positions.py:
            # "records are never lost due to restarts or wallet resets") wrote
            # every one of them straight back on the next save.
            try:
                with open(TRACK_RECORD_PATH, 'r', encoding='utf-8') as _f:
                    _local = json.load(_f)
            except Exception:
                _local = {}
            if int(_local.get('generation', 1) or 1) != STATE_GENERATION:
                _n = len(_local.get('signals') or [])
                print(f'[state] local track record is generation '
                      f'{_local.get("generation", 1)} != {STATE_GENERATION} — '
                      f'wiping {_n} stale entries (one-time)')
                _write_empty_track_record()
                try:
                    _fs_clear_track_record()
                except Exception as _e:
                    print(f'[state] Firestore copy not cleared ({_e}) — '
                          f'the local wipe still stands')
                return
            return  # current generation, local state present — nothing to restore
        data = _fs_load_track_record()
        if not data or not data.get('signals'):
            return
        # Generation guard: ignore any record from an older state generation so a
        # bump wipes stale history exactly once, no matter what an older engine
        # wrote to Firestore before this deploy took over.
        if int(data.get('generation', 1)) != STATE_GENERATION:
            print(f'[state] Firestore record is generation '
                  f'{data.get("generation", 1)} != {STATE_GENERATION} — starting fresh (one-time wipe)')
            _fs_clear_track_record()
            return
        TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACK_RECORD_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f'[state] restored {len(data.get("signals", []))} track-record entries from Firestore')
    except Exception as e:
        print(f'[state] Firestore hydrate skipped: {e}')
