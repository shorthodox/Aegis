"""Deleting a track-record row must remove it from EVERY store.

Reported 2026-08-20: "delete 3 track records that are in loss SAND/USDT,
BCH/USDT and ETC/USDT". All three came back from the public endpoint tagged
source='live_engine' — they lived in live_engine's own track_record.json, which
sits on the RAILWAY VOLUME.

DELETE /api/admin/track-record/{id} only filtered main.py's in-memory
_track_store and rewrote WEB_ROOT/track_record.json. For a record the engine
owned it therefore:

  * found nothing in memory,
  * raised 404 "signal_id not found",
  * and left the row on the public page.

Which is the same symptom as three earlier wipes ("records are still in the
track record page ... you should clear the railway volume for it") and has the
same cause: the public endpoint MERGES three stores and the writers only ever
touched one of them.
"""
import asyncio
import json

import pytest

import main


LOSS_IDS = {
    "ee759ce5-b111-4c47-bf1e-1b5d340a0726",   # SAND/USDT
    "e35c5a3c-7a0d-4b71-a1da-438224c62aea",   # BCH/USDT
    "c8f45500-73a3-4e36-ba02-e613151b4fdf",   # ETC/USDT
}


def _rec(sid, symbol, outcome, pnl):
    return {"signal_id": sid, "symbol": symbol, "outcome": outcome,
            "pnl_pct": pnl, "entry_time": "2026-08-20T08:00:00Z"}


def _payload():
    """The live record as the public endpoint reported it: 2 wins, 3 losses."""
    sigs = [
        _rec("win-trx", "TRX/USDT", "WIN", 1.0699),
        _rec("win-storj", "STORJ/USDT", "WIN", 4.288),
        _rec("c8f45500-73a3-4e36-ba02-e613151b4fdf", "ETC/USDT", "LOSS", -1.4454),
        _rec("e35c5a3c-7a0d-4b71-a1da-438224c62aea", "BCH/USDT", "LOSS", -1.438),
        _rec("ee759ce5-b111-4c47-bf1e-1b5d340a0726", "SAND/USDT", "LOSS", -1.6097),
    ]
    return {"generated_at": "2026-08-20T13:00:00Z",
            "summary": {"total_signals": 5, "wins": 2, "losses": 3,
                        "win_rate_pct": 40.0, "total_pnl_pct": 0.865,
                        "balance": 10086.5},
            "signals": sigs}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Three separate files, exactly as production has them."""
    engine = tmp_path / "volume" / "track_record.json"
    web = tmp_path / "web" / "track_record.json"
    mem = tmp_path / "webroot" / "track_record.json"
    for p in (engine, web, mem):
        p.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "_engine_record_path", lambda: engine)
    monkeypatch.setattr(main, "_web_record_path", lambda: web)
    monkeypatch.setattr(main, "TRACK_RECORD_PATH", mem)
    return engine, web, mem


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# -- the regression ----------------------------------------------------------

def test_a_record_only_on_the_volume_is_actually_deleted(stores):
    """The exact reported shape: the row exists ONLY in live_engine's file."""
    engine, web, mem = stores
    _write(engine, _payload())
    removed = main._purge_ids_from_disk(LOSS_IDS)
    assert sum(removed.values()) == 3, (
        "the three losses were not removed from the volume - this is the bug "
        "that made the delete endpoint 404 while the rows stayed on the page"
    )
    left = {r["signal_id"] for r in _read(engine)["signals"]}
    assert left == {"win-trx", "win-storj"}
    assert not (left & LOSS_IDS)


def test_the_stored_summary_stops_counting_deleted_trades(stores):
    """The file's summary is what the wallet figures are read from. Leaving it
    at 2W/3L would publish a win rate for trades no longer in the record."""
    engine, _web, _mem = stores
    _write(engine, _payload())
    main._purge_ids_from_disk(LOSS_IDS)
    s = _read(engine)["summary"]
    assert s["wins"] == 2 and s["losses"] == 0
    assert s["win_rate_pct"] == 100.0
    assert s["total_signals"] == 2
    assert s["total_pnl_pct"] == pytest.approx(5.358, abs=1e-3)


def test_every_store_is_purged_not_just_the_first(stores):
    """The same signal_id can sit in more than one file; all copies must go."""
    engine, web, mem = stores
    for p in (engine, web, mem):
        _write(p, _payload())
    removed = main._purge_ids_from_disk(LOSS_IDS)
    assert len(removed) == 3, f"only {len(removed)} store(s) were touched: {removed}"
    for p in (engine, web, mem):
        assert not ({r["signal_id"] for r in _read(p)["signals"]} & LOSS_IDS), p


def test_wins_and_unrelated_rows_are_never_touched(stores):
    engine, _web, _mem = stores
    _write(engine, _payload())
    main._purge_ids_from_disk({"ee759ce5-b111-4c47-bf1e-1b5d340a0726"})
    left = [r["signal_id"] for r in _read(engine)["signals"]]
    assert left == ["win-trx", "win-storj",
                    "c8f45500-73a3-4e36-ba02-e613151b4fdf",
                    "e35c5a3c-7a0d-4b71-a1da-438224c62aea"]


# -- it must stay safe -------------------------------------------------------

def test_an_unknown_id_removes_nothing_and_rewrites_nothing(stores):
    engine, _web, _mem = stores
    _write(engine, _payload())
    before = engine.read_text(encoding="utf-8")
    assert main._purge_ids_from_disk({"not-a-real-id"}) == {}
    assert engine.read_text(encoding="utf-8") == before


def test_missing_files_are_not_an_error(stores):
    assert main._purge_ids_from_disk(LOSS_IDS) == {}


def test_a_corrupt_store_does_not_stop_the_others(stores):
    """One unreadable file must not leave the row live in the file that matters."""
    engine, web, _mem = stores
    web.write_text("{not json", encoding="utf-8")
    _write(engine, _payload())
    removed = main._purge_ids_from_disk(LOSS_IDS)
    assert sum(removed.values()) == 3
    assert not ({r["signal_id"] for r in _read(engine)["signals"]} & LOSS_IDS)


def test_the_write_is_atomic(stores):
    """os.replace via a .tmp sibling - a torn write here corrupts the record."""
    engine, _web, _mem = stores
    _write(engine, _payload())
    main._purge_ids_from_disk(LOSS_IDS)
    assert not engine.with_suffix(".tmp").exists(), "temp file left behind"
    _read(engine)          # raises if the file is not valid JSON


# -- the endpoint itself -----------------------------------------------------

def test_the_endpoint_no_longer_404s_on_a_volume_only_record(stores, monkeypatch):
    engine, _web, _mem = stores
    _write(engine, _payload())
    monkeypatch.setattr(main, "_track_store", [], raising=False)
    monkeypatch.setattr(main, "_tr_seen_ids", set(), raising=False)
    monkeypatch.setattr(main, "_save_track_record", lambda: None)

    sid = "ee759ce5-b111-4c47-bf1e-1b5d340a0726"
    out = asyncio.run(main.delete_track_record_entry(sid, None))
    assert out["success"] is True
    assert out["removed"] == 1
    assert sid not in {r["signal_id"] for r in _read(engine)["signals"]}


def test_the_endpoint_still_404s_when_the_id_exists_nowhere(stores, monkeypatch):
    from fastapi import HTTPException
    engine, _web, _mem = stores
    _write(engine, _payload())
    monkeypatch.setattr(main, "_track_store", [], raising=False)
    monkeypatch.setattr(main, "_tr_seen_ids", set(), raising=False)
    monkeypatch.setattr(main, "_save_track_record", lambda: None)
    with pytest.raises(HTTPException) as e:
        asyncio.run(main.delete_track_record_entry("ghost", None))
    assert e.value.status_code == 404


def test_the_response_says_which_store_it_came_from(stores, monkeypatch):
    """Without this the operator cannot tell a real delete from a no-op."""
    engine, _web, _mem = stores
    _write(engine, _payload())
    monkeypatch.setattr(main, "_track_store", [], raising=False)
    monkeypatch.setattr(main, "_tr_seen_ids", set(), raising=False)
    monkeypatch.setattr(main, "_save_track_record", lambda: None)
    out = asyncio.run(main.delete_track_record_entry(
        "e35c5a3c-7a0d-4b71-a1da-438224c62aea", None))
    assert str(engine) in out["stores"]
    assert out["stores"][str(engine)] == 1
