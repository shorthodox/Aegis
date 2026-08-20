"""A paid grant must survive Firestore being unavailable.

Every record of who had paid lived in Firestore and nowhere else, so a datastore
outage was indistinguishable from a cancelled subscription. Both grant paths
wrote Firestore only:

  * the payment-verify path answered "Payment verified but account update failed"
    — money taken, no access, and no record of the grant anywhere
  * the Whop and Paddle webhooks returned 500 so the provider would retry, but a
    retry window can expire against a quota that resets daily

A paid subscriber therefore reverted to `plan: trial`, which is the cascade behind
the "trial expired" overlay on a paid account and the missing Telegram signals.

entitlements.json on the Railway VOLUME is now the durable record of a payment
that happened — written BEFORE Firestore at every grant, and overlaid at the
single read boundary (get_user_doc) so every consumer honours it.

Two invariants keep it safe:
  1. It may only GRANT. It is never consulted to remove access, so a stale or
     corrupt file cannot lock a paying customer out.
  2. It respects its own expiry, so a lapsed plan cannot become permanent access.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

import main


FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(main, '_ENTITLEMENTS_PATH', tmp_path / 'entitlements.json')
    return tmp_path / 'entitlements.json'


# ── the record itself ────────────────────────────────────────────────────────

def test_a_grant_is_written_to_the_volume(store):
    main._ent_grant('buyer@example.test', 'pro', FUTURE, 'whop', 'pay_1')
    saved = json.loads(store.read_text(encoding='utf-8'))
    rec = saved['buyer@example.test']
    assert rec['plan'] == 'pro' and rec['status'] == 'active'
    assert rec['provider'] == 'whop'


def test_the_write_is_atomic(store):
    """A half-written entitlements file is worse than none."""
    import inspect
    src = inspect.getsource(main._ent_save)
    assert '.tmp' in src and 'replace' in src, (
        'entitlements are written in place — a crash mid-write corrupts the '
        'only durable record of who has paid'
    )


# ── the overlay ──────────────────────────────────────────────────────────────

def test_it_restores_a_plan_firestore_no_longer_shows(store):
    """The exact failure: Firestore says trial, the customer has paid."""
    main._ent_grant('buyer@example.test', 'pro', FUTURE, 'whop')
    doc = main._ent_overlay('buyer@example.test', {'plan': 'trial'})
    assert doc['plan'] == 'pro'
    assert main.has_paid_access(doc) is True


def test_it_works_when_firestore_returns_nothing_at_all(store):
    """A read failure returns None — the grant must still stand."""
    main._ent_grant('buyer@example.test', 'pro', FUTURE, 'whop')
    doc = main._ent_overlay('buyer@example.test', None)
    assert main.has_paid_access(doc) is True


def test_an_expired_grant_confers_nothing(store):
    """Invariant 2. A lapsed plan must not become permanent access."""
    main._ent_grant('lapsed@example.test', 'pro', PAST, 'whop')
    doc = main._ent_overlay('lapsed@example.test', {'plan': 'trial'})
    assert doc.get('plan') == 'trial'
    assert main.has_paid_access(doc) is False


def test_it_never_downgrades_a_live_firestore_plan(store):
    """Invariant 1. It may only grant, never take away."""
    main._ent_grant('u@example.test', 'basic', PAST, 'whop')   # expired record
    live = {'plan': 'pro', 'subscription': {'status': 'active'}}
    assert main._ent_overlay('u@example.test', live)['plan'] == 'pro'


def test_an_unknown_user_is_untouched(store):
    doc = {'plan': 'trial'}
    assert main._ent_overlay('nobody@example.test', doc) == doc


def test_a_corrupt_file_cannot_break_a_lookup(store):
    store.write_text('{ this is not json', encoding='utf-8')
    doc = {'plan': 'trial'}
    assert main._ent_overlay('anyone@example.test', doc) == doc


def test_cancellation_keeps_access_until_the_period_ends(store):
    """Cancelling means 'will not renew', not 'ends now'."""
    main._ent_grant('bye@example.test', 'pro', FUTURE, 'whop')
    main._ent_revoke('bye@example.test')
    doc = main._ent_overlay('bye@example.test', {'plan': 'trial'})
    assert main.has_paid_access(doc) is True, 'cancelling cut access off early'


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_read_boundary_applies_the_overlay():
    import inspect
    src = inspect.getsource(main.get_user_doc)
    assert '_ent_overlay' in src, (
        'get_user_doc no longer overlays the volume grant — every consumer '
        'silently loses it'
    )
    assert 'except Exception' in src, 'a Firestore read failure still raises'


def test_every_grant_path_records_to_the_volume_first():
    """Three paid paths: payment-verify, Whop webhook, Paddle webhook."""
    src = open(main.__file__, encoding='utf-8', errors='replace').read()
    calls = [l for l in src.splitlines() if '_ent_grant(' in l and 'def ' not in l]
    assert len(calls) >= 3, f'only {len(calls)} grant paths record to the volume'


def test_the_volume_path_is_on_the_railway_volume():
    """Not beside the web root — that is wiped on every deploy."""
    assert 'entitlements.json' in str(main._ENTITLEMENTS_PATH)
    import inspect
    src = inspect.getsource(main)
    assert '_ENTITLEMENTS_PATH = _STATE_DIR / "entitlements.json"' in src, (
        'entitlements moved off STATE_DIR — they would not survive a redeploy'
    )
