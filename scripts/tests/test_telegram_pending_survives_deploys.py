"""A connect code must survive a deploy, and a user must never be ignored.

Reported repeatedly, most recently 2026-08-22: "TELEGRAM STILL IS NOT POPULATING
THE SIGNAL NOTIFICATIONS". Production said it plainly once a signal finally
fired:

    [Telegram] no connected chat_ids — nothing sent. Checked TELEGRAM_CHAT_ID
    and /data/telegram_connections.json (exists=True)

The dispatcher was fine and the connections file was on the volume — that half
was fixed on 2026-08-17. The defect was one step EARLIER and had the same shape,
which is why the earlier fix did not catch it:

    /connect  ->  {code: email} written to _tg_pending, a dict IN RAM
    deploy    ->  _tg_pending wiped
    /start C  ->  code not in _tg_pending  ->  SILENTLY IGNORED
    signal    ->  "no connected chat_ids"

No reply to the user, no log line. The person did everything right and the only
trace was an empty connections file hours later. With ~10 deploys on 2026-08-22
the window that loses a connection was most of the day.

Two rules here: the code lives on the VOLUME with a TTL, and an unmatched /start
always gets an answer.
"""
import json
import time

import pytest

import main


@pytest.fixture
def stores(tmp_path, monkeypatch):
    conn = tmp_path / "telegram_connections.json"
    pend = tmp_path / "telegram_pending.json"
    monkeypatch.setattr(main, "_TG_CONNECTIONS_PATH", conn)
    monkeypatch.setattr(main, "_TG_PENDING_PATH", pend)
    monkeypatch.setattr(main, "_tg_connections", {}, raising=False)
    monkeypatch.setattr(main, "_tg_pending", {}, raising=False)
    monkeypatch.setattr(main, "_tg_load_failed", False, raising=False)
    return conn, pend


# -- the code must outlive the process ---------------------------------------

def test_a_pending_code_is_written_to_the_volume(stores):
    _conn, pend = stores
    main._tg_pending["A1B2"] = {"email": "u@example.test", "created": time.time()}
    main._tg_save_pending()
    assert pend.exists(), "the connect code never reached the volume"
    assert json.loads(pend.read_text())["A1B2"]["email"] == "u@example.test"


def test_a_pending_code_survives_a_restart(stores):
    """The exact failure: issued before a deploy, redeemed after it."""
    _conn, pend = stores
    main._tg_pending["A1B2"] = {"email": "u@example.test", "created": time.time()}
    main._tg_save_pending()

    main._tg_pending.clear()          # the deploy
    main._tg_load_pending()

    assert "A1B2" in main._tg_pending, (
        "the code did not survive the restart — this is the bug that silently "
        "dropped every connection made near a deploy"
    )
    assert main._tg_pending["A1B2"]["email"] == "u@example.test"


def test_an_expired_code_is_dropped_on_load(stores):
    _conn, pend = stores
    old = time.time() - main._TG_PENDING_TTL_SECONDS - 60
    pend.write_text(json.dumps({"OLD1": {"email": "u@example.test", "created": old}}))
    main._tg_load_pending()
    assert "OLD1" not in main._tg_pending


def test_saving_prunes_expired_codes(stores):
    _conn, pend = stores
    old = time.time() - main._TG_PENDING_TTL_SECONDS - 60
    main._tg_pending.update({
        "OLD1": {"email": "a@example.test", "created": old},
        "NEW1": {"email": "b@example.test", "created": time.time()},
    })
    main._tg_save_pending()
    on_disk = json.loads(pend.read_text())
    assert "NEW1" in on_disk and "OLD1" not in on_disk


def test_unreadable_pending_starts_empty_without_raising(stores):
    _conn, pend = stores
    pend.write_text("{not json")
    main._tg_load_pending()
    assert main._tg_pending == {}


# -- a corrupt connections file must not become an empty one -----------------

def test_a_failed_load_refuses_to_save_over_the_file(stores):
    """The quiet data-loss path: `except Exception: pass` left the dict empty
    and the next save wrote {} over real subscribers."""
    conn, _pend = stores
    conn.write_text("{not json")
    main._tg_load_connections()
    assert main._tg_load_failed is True

    main._tg_connections["someone@example.test"] = {"chat_id": "1"}
    main._tg_save_connections()
    assert conn.read_text() == "{not json", (
        "the unreadable connections file was overwritten — a parse error just "
        "became permanent data loss"
    )


def test_a_good_load_saves_normally(stores):
    conn, _pend = stores
    conn.write_text(json.dumps({"a@example.test": {"chat_id": "42"}}))
    main._tg_load_connections()
    assert main._tg_load_failed is False
    assert main._tg_connections["a@example.test"]["chat_id"] == "42"

    main._tg_connections["b@example.test"] = {"chat_id": "43"}
    main._tg_save_connections()
    assert json.loads(conn.read_text())["b@example.test"]["chat_id"] == "43"


def test_a_missing_connections_file_is_not_a_failure(stores):
    main._tg_load_connections()
    assert main._tg_load_failed is False


def test_the_write_is_atomic(stores):
    conn, _pend = stores
    main._tg_connections["a@example.test"] = {"chat_id": "42"}
    main._tg_save_connections()
    assert not conn.with_suffix(".tmp").exists(), "temp file left behind"
    json.loads(conn.read_text())


# -- both stores live on the volume ------------------------------------------

def test_both_stores_are_on_the_volume():
    """A deploy wipes the container filesystem; neither of these may live there.

    Asserted against main's own STATE_DIR alias, bound at import. Reading
    scripts.engine.config.STATE_DIR here instead makes this test order-dependent
    — test_telegram_connections_persist.py repoints that module attribute and
    does not put it back.
    """
    assert main._TG_CONNECTIONS_PATH.parent == main._STATE_DIR
    assert main._TG_PENDING_PATH.parent == main._STATE_DIR
    assert main._TG_CONNECTIONS_PATH.name == 'telegram_connections.json'
    assert main._TG_PENDING_PATH.name == 'telegram_pending.json'


def test_the_dispatcher_reads_the_same_connections_file():
    """The original bug was two halves deriving the path separately and drifting."""
    src = (main.Path(main.__file__).resolve().parent / 'scripts' / 'notifications'
           / 'dispatcher.py').read_text(encoding='utf-8', errors='replace')
    assert 'STATE_DIR' in src, 'the dispatcher no longer resolves off STATE_DIR'
    assert 'telegram_connections.json' in src
    assert main._TG_CONNECTIONS_PATH == main._STATE_DIR / 'telegram_connections.json'
