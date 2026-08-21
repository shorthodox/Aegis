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


# -- the wallet is the real source of truth -----------------------------------
# 2026-08-21: the three losses were deleted from memory and from the volume, the
# public record went to 3W/0L, and minutes later all three were BACK with the
# same signal_ids.
#
# PositionManager._save_track_record rebuilds STATE_DIR/track_record.json from
# `wallet.trade_history` every cycle, and re-preserves any on-disk record the
# wallet does not know about. So a disk-only delete is overwritten by the wallet,
# and a wallet-only delete is resurrected from the disk orphans. Both have to go.
#
# The money has to move too: `balance` is a running total (`balance += pnl_usdt`
# per close), so a row deleted from the record while its PnL stays in the wallet
# leaves published capital and profit reporting a trade that is no longer there.
# Reported as: "SHOULDNT CAP AND PROFIT SHOULD RAISE ACCORDINGLY? THEY ARE SAME
# AS BEFORE".

class _Trade:
    def __init__(self, signal_id, pnl_usdt):
        self.signal_id = signal_id
        self.pnl_usdt = pnl_usdt


class _Wallet:
    def __init__(self, trades, balance, initial_capital=10000.0):
        self.trade_history = list(trades)
        self.balance = balance
        self.initial_capital = initial_capital


class _Engine:
    def __init__(self, wallet):
        self.wallet = wallet
        self.saved = 0

    def _save_track_record(self):
        self.saved += 1


@pytest.fixture
def wallet_engine(monkeypatch):
    """The live book as it stood: 3 wins and the 3 losses, balance 10,008.55."""
    trades = [
        _Trade("win-inj", 9.53), _Trade("win-storj", 42.88), _Trade("win-trx", 10.70),
        _Trade("ee759ce5-b111-4c47-bf1e-1b5d340a0726", -16.10),
        _Trade("e35c5a3c-7a0d-4b71-a1da-438224c62aea", -14.38),
        _Trade("c8f45500-73a3-4e36-ba02-e613151b4fdf", -14.45),
    ]
    eng = _Engine(_Wallet(trades, balance=10018.18))
    monkeypatch.setattr(main.LIVE_STATE, "engine", eng, raising=False)
    return eng


def test_the_trade_is_removed_from_the_wallet(wallet_engine):
    out = main._purge_ids_from_wallet(LOSS_IDS)
    assert out["slices"] == 3
    left = {t.signal_id for t in wallet_engine.wallet.trade_history}
    assert left == {"win-inj", "win-storj", "win-trx"}


def test_deleting_a_loss_raises_the_balance(wallet_engine):
    """The reported symptom: the row went, the money did not."""
    before = wallet_engine.wallet.balance
    out = main._purge_ids_from_wallet(LOSS_IDS)
    after = wallet_engine.wallet.balance
    assert after > before, 'deleting three losing trades left the balance unchanged'
    assert after == pytest.approx(before + 16.10 + 14.38 + 14.45, abs=1e-6)
    assert out["pnl_reversed"] == pytest.approx(-44.93, abs=1e-6)


def test_deleting_a_win_lowers_the_balance(wallet_engine):
    """Symmetry — the reversal is not a one-way ratchet that only ever adds."""
    before = wallet_engine.wallet.balance
    main._purge_ids_from_wallet({"win-storj"})
    assert wallet_engine.wallet.balance == pytest.approx(before - 42.88, abs=1e-6)


def test_an_unknown_id_leaves_the_wallet_alone(wallet_engine):
    before = wallet_engine.wallet.balance
    n = len(wallet_engine.wallet.trade_history)
    out = main._purge_ids_from_wallet({"nope"})
    assert out["slices"] == 0
    assert wallet_engine.wallet.balance == before
    assert len(wallet_engine.wallet.trade_history) == n


def test_no_engine_running_is_not_an_error(monkeypatch):
    monkeypatch.setattr(main.LIVE_STATE, "engine", None, raising=False)
    assert main._purge_ids_from_wallet(LOSS_IDS)["slices"] == 0


def test_every_slice_of_a_partialled_trade_goes(monkeypatch):
    """Partial TPs append several TradeRecords under ONE signal_id. Leaving any
    behind keeps the trade in the rebuilt record and its money in the wallet."""
    sid = "ee759ce5-b111-4c47-bf1e-1b5d340a0726"
    eng = _Engine(_Wallet([_Trade(sid, 3.0), _Trade(sid, 5.0), _Trade(sid, -20.0),
                           _Trade("other", 1.0)], balance=10000.0))
    monkeypatch.setattr(main.LIVE_STATE, "engine", eng, raising=False)
    out = main._purge_ids_from_wallet({sid})
    assert out["slices"] == 3
    assert [t.signal_id for t in eng.wallet.trade_history] == ["other"]
    assert eng.wallet.balance == pytest.approx(10000.0 + 12.0, abs=1e-6)


def test_the_endpoint_purges_the_wallet_and_re_saves(stores, wallet_engine, monkeypatch):
    """Disk must be purged BEFORE the engine rewrites, or orphan-preservation
    reads the row straight back off the volume."""
    engine_file, _web, _mem = stores
    _write(engine_file, _payload())
    monkeypatch.setattr(main, "_track_store", [], raising=False)
    monkeypatch.setattr(main, "_tr_seen_ids", set(), raising=False)
    monkeypatch.setattr(main, "_save_track_record", lambda: None)

    sid = "ee759ce5-b111-4c47-bf1e-1b5d340a0726"
    out = asyncio.run(main.delete_track_record_entry(sid, None))
    assert out["stores"]["wallet_slices"] == 1
    assert out["pnl_reversed_usdt"] == pytest.approx(-16.10, abs=1e-6)
    assert wallet_engine.saved == 1, 'the engine was not asked to rewrite the file'
    assert sid not in {t.signal_id for t in wallet_engine.wallet.trade_history}
    assert sid not in {r["signal_id"] for r in _read(engine_file)["signals"]}
