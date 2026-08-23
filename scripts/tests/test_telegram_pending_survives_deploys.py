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


# -- an unreadable datastore must not disconnect anyone -----------------------
# Reported 2026-08-23: "site is getting disconnected from telegram
# automatically". The hourly sweep is the ONLY thing that deletes a connection,
# and it decided with is_trial_expired(), which opens:
#
#     user_doc = get_user_doc(email)
#     if not user_doc:
#         return True          # <- a FAILED READ read as "expired"
#
# get_user_doc() collapses "no such user" and "Firestore did not answer" into
# the same None. That is harmless for GRANTING access — both mean no
# entitlement — and destructive for REVOKING it. This project is on the
# Firestore free tier and has exhausted its daily quota before, so a datastore
# hiccup silently unsubscribed people whose plan had not changed.
#
# Delivery never depended on the sweep: dispatcher._tg_send_all checks
# access_until on every send. A wrong disconnect needs the user to notice and
# reconnect; a wrong keep costs one skipped send. So this fails OPEN.

class _Boom:
    def collection(self, *_a, **_k):
        raise RuntimeError('quota exceeded')


def test_a_firestore_failure_does_not_disconnect(monkeypatch):
    monkeypatch.setattr(main, 'db', _Boom())
    monkeypatch.setattr(main, '_ent_overlay', lambda e, base: base)
    drop, why = main._tg_should_disconnect('u@example.test')
    assert drop is False, (
        'a failed datastore read disconnected a subscriber — this is the '
        'automatic disconnection that was reported'
    )
    assert 'unreadable' in why


def test_a_missing_document_does_not_disconnect(monkeypatch):
    """No document is not evidence a plan ended — it is no evidence at all."""
    monkeypatch.setattr(main, '_user_doc_read', lambda e: (None, True))
    drop, why = main._tg_should_disconnect('u@example.test')
    assert drop is False
    assert 'no user document' in why


def test_a_live_paid_plan_is_kept(monkeypatch):
    monkeypatch.setattr(main, '_user_doc_read',
                        lambda e: ({'plan': 'pro'}, True))
    assert main._tg_should_disconnect('u@example.test')[0] is False


def test_a_genuinely_expired_trial_is_still_disconnected(monkeypatch):
    """Failing open must not mean never disconnecting."""
    doc = {'plan': 'trial', 'trial_end': '2020-01-01T00:00:00Z'}
    monkeypatch.setattr(main, '_user_doc_read', lambda e: (doc, True))
    monkeypatch.setattr(main, 'get_user_doc', lambda e: doc)
    drop, why = main._tg_should_disconnect('u@example.test')
    assert drop is True
    assert 'trial' in why


def test_a_genuinely_lapsed_paid_plan_is_still_disconnected(monkeypatch):
    doc = {'plan': 'pro',
           'subscription': {'status': 'active', 'current_period_end': '2020-01-01T00:00:00Z'}}
    monkeypatch.setattr(main, '_user_doc_read', lambda e: (doc, True))
    monkeypatch.setattr(main, 'get_user_doc', lambda e: doc)
    drop, why = main._tg_should_disconnect('u@example.test')
    assert drop is True
    assert 'paid plan ended' in why


def test_the_sweep_leaves_the_stamp_alone_when_the_read_failed(stores, monkeypatch):
    """Rewriting access_until from a failed lookup replaces a good stamp with a
    guess — the same damage one layer down."""
    monkeypatch.setattr(main, '_tg_should_disconnect',
                        lambda e: (False, 'datastore unreadable — keeping the connection'))
    monkeypatch.setattr(main, '_tg_access_until', lambda e: '')
    main._tg_connections['u@example.test'] = {'chat_id': '42',
                                              'access_until': '2099-01-01T00:00:00Z'}
    main._telegram_cleanup_pass()
    assert main._tg_connections['u@example.test']['access_until'] == '2099-01-01T00:00:00Z'
    assert main._tg_connections['u@example.test']['chat_id'] == '42'


# -- lapsed access PAUSES the link, it does not delete it ---------------------
# Reported 2026-08-23: "the bot is disconnecting from the site by its own" and
# "when i am closing telegram, bot disconnecting". Read off the volume: the user
# connected at 06:27 and by 06:41 the file was {} again, with the reason in the
# log —
#
#     [TG cleanup] ...: paid plan ended — disconnecting Telegram
#     [TG cleanup] Disconnected 1 expired user(s)
#
# — correct about the entitlement (a 5-day dev code that ended 2026-07-06) and
# wrong about what to do with it. Delivery is ALREADY refused at send time by
# access_until. Deleting the chat_id is a second, destructive copy of the same
# rule and the only one that costs the user anything: a full /connect + /start
# again on renewal, and until then a site that just looks broken.

@pytest.fixture
def lapsed(stores, monkeypatch):
    monkeypatch.setattr(main, '_tg_should_disconnect',
                        lambda e: (True, 'paid plan ended'))
    monkeypatch.setattr(main, '_tg_access_until', lambda e: '2026-07-01T10:58:49Z')
    sent = []
    monkeypatch.setattr(main, '_tg_notify', lambda cid, text: sent.append((cid, text)))
    main._tg_connections['u@example.test'] = {'chat_id': '6376199309',
                                              'access_until': '2026-07-01T10:58:49Z'}
    return sent


def test_a_lapsed_plan_keeps_the_link(lapsed):
    main._telegram_cleanup_pass()
    assert 'u@example.test' in main._tg_connections, (
        'the connection was deleted again — this is the "disconnecting by itself" '
        'the user reported, and it forces a full reconnect on renewal'
    )
    assert main._tg_connections['u@example.test']['chat_id'] == '6376199309'


def test_the_pause_is_recorded_with_its_reason(lapsed):
    main._telegram_cleanup_pass()
    e = main._tg_connections['u@example.test']
    assert e['paused_reason'] == 'paid plan ended'
    assert e['paused_at']


def test_delivery_is_still_refused_while_paused(lapsed):
    """Pausing must not become a way to keep receiving signals for free."""
    main._telegram_cleanup_pass()
    from datetime import datetime, timezone
    until = main._tg_connections['u@example.test']['access_until']
    end = datetime.fromisoformat(until.replace('Z', '+00:00'))
    assert end < datetime.now(timezone.utc), (
        'a paused link carries a live access_until — the send-time gate would '
        'let it through'
    )


def test_an_empty_stamp_never_survives_a_pause(stores, monkeypatch):
    """'' means open-ended to the dispatcher. A paused link must not carry one."""
    monkeypatch.setattr(main, '_tg_should_disconnect', lambda e: (True, 'trial access ended'))
    monkeypatch.setattr(main, '_tg_access_until', lambda e: '')
    monkeypatch.setattr(main, '_tg_notify', lambda cid, text: None)
    main._tg_connections['u@example.test'] = {'chat_id': '42', 'access_until': ''}
    main._telegram_cleanup_pass()
    assert main._tg_connections['u@example.test']['access_until'] not in ('', None)


def test_the_user_is_told_once_not_every_hour(lapsed):
    """The sweep runs hourly. Telling them each time is spam."""
    main._telegram_cleanup_pass()
    assert len(lapsed) == 1, 'no message sent on the first pause'
    assert 'paused' in lapsed[0][1].lower()
    main._telegram_cleanup_pass()
    main._telegram_cleanup_pass()
    assert len(lapsed) == 1, 'the pause notice repeated on a later sweep'


def test_status_reports_paused_rather_than_disconnected(stores):
    """Saying "not connected" for a linked-but-lapsed account is what made an
    expired plan look like a broken integration."""
    main._tg_connections['u@example.test'] = {
        'chat_id': '42', 'access_until': '2026-07-01T00:00:00Z',
        'paused_reason': 'paid plan ended', 'paused_at': '2026-08-23T06:41:00Z'}
    entry = main._tg_connections['u@example.test']
    assert main._tg_chat_id(entry) == '42'
    assert entry.get('paused_reason') == 'paid plan ended'


def test_a_live_plan_is_not_paused(stores, monkeypatch):
    monkeypatch.setattr(main, '_tg_should_disconnect', lambda e: (False, 'paid access is live'))
    monkeypatch.setattr(main, '_tg_access_until', lambda e: '2099-01-01T00:00:00Z')
    main._tg_connections['u@example.test'] = {'chat_id': '42', 'access_until': ''}
    main._telegram_cleanup_pass()
    assert 'paused_reason' not in main._tg_connections['u@example.test']
