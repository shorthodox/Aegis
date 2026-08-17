"""A paid plan must not be read as an expired trial.

Reported: Telegram connects, the bot replies "AEGIS Signal Bot connected!", and no
signal ever arrives. The bot, the token and the chat were all fine — that
confirmation is posted straight to the Telegram API and passes neither gate below,
which is exactly why the failure was invisible.

Two gates in main.py stopped a paying subscriber, both for the same reason:

  * _tg_access_until() required `plan in PAID and subscription.status == "active"`.
    Anything else fell through to `return trial_end` — an already-elapsed
    timestamp — which was written beside the chat_id at connect time.
    dispatcher._tg_send_all then silently `continue`d past that user on every send.

  * the hourly sweep called is_trial_expired(), which had the identical condition
    and so returned True for the same user, DELETING their connection outright.

The condition appeared five times in main.py, each slightly different; one copy
even listed "active" as a plan name.

The fix inverts the test. Requiring one known-live status made every unrecognised
or absent status mean "cancelled". Enumerating the DEAD statuses instead lets
missing bookkeeping fail OPEN for someone who has paid, while a real cancellation
still fails closed.
"""
import pytest

import main


PAID = ['pro', 'premium', 'intermediate', 'basic', 'pro-dev']


def _doc(plan, status=None, **kw):
    d = {'plan': plan}
    if status is not None:
        d['subscription'] = {'status': status}
    d.update(kw)
    return d


# ── the regression ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('plan', PAID)
def test_a_paid_plan_with_no_subscription_status_has_access(plan):
    """The exact shape that broke it. No status field at all — which is the
    normal state for a plan set by hand or by an older payment flow."""
    assert main.has_paid_access(_doc(plan)) is True


@pytest.mark.parametrize('plan', PAID)
def test_a_paid_plan_with_an_empty_status_has_access(plan):
    assert main.has_paid_access(_doc(plan, status='')) is True


@pytest.mark.parametrize('plan', PAID)
def test_a_paid_plan_with_status_active_has_access(plan):
    assert main.has_paid_access(_doc(plan, status='active')) is True


def test_an_unrecognised_status_fails_open_for_a_paid_plan():
    """A status nobody anticipated must not silently revoke a paid plan."""
    assert main.has_paid_access(_doc('pro', status='trialing')) is True
    assert main.has_paid_access(_doc('pro', status='whatever')) is True


# ── and it must still fail closed ────────────────────────────────────────────

@pytest.mark.parametrize('status', ['cancelled', 'canceled', 'expired',
                                    'past_due', 'unpaid', 'halted', 'paused'])
def test_a_cancelled_subscription_has_no_access(status):
    assert main.has_paid_access(_doc('pro', status=status)) is False


def test_status_matching_is_case_insensitive():
    assert main.has_paid_access(_doc('pro', status='CANCELLED')) is False
    assert main.has_paid_access(_doc('PRO', status='active')) is True


@pytest.mark.parametrize('plan', ['trial', 'none', 'expired', '', 'free', 'foo'])
def test_an_unpaid_plan_never_has_paid_access(plan):
    assert main.has_paid_access(_doc(plan)) is False
    assert main.has_paid_access(_doc(plan, status='active')) is False, (
        'a status of active must not promote an unpaid plan'
    )


def test_missing_and_malformed_docs_have_no_access():
    assert main.has_paid_access(None) is False
    assert main.has_paid_access({}) is False
    # subscription present but not a dict — must not raise
    assert main.has_paid_access({'plan': 'pro', 'subscription': 'active'}) is True
    assert main.has_paid_access({'plan': 'trial', 'subscription': None}) is False


# ── the Telegram timestamp ───────────────────────────────────────────────────

def test_a_paid_plan_never_gets_a_trial_timestamp(monkeypatch):
    """The precise defect: an elapsed trial_end written beside a paying user's
    chat_id, which the sender then reads as lapsed access."""
    monkeypatch.setattr(main, 'get_user_doc', lambda e: {
        'plan': 'pro',
        'trial_end': '2020-01-01T00:00:00Z',      # long gone
    })
    until = main._tg_access_until('someone@example.test')
    assert until != '2020-01-01T00:00:00Z', (
        'a paid user was gated on their expired trial date — this is what silently '
        'dropped every Telegram signal'
    )
    assert until == '', 'paid with no subscription end date means no timestamp gate'


def test_a_paid_plan_uses_its_own_end_date_when_present(monkeypatch):
    monkeypatch.setattr(main, 'get_user_doc', lambda e: {
        'plan': 'pro',
        'subscription': {'status': 'active', 'current_period_end': '2026-12-01T00:00:00Z'},
        'trial_end': '2020-01-01T00:00:00Z',
    })
    assert main._tg_access_until('x@example.test') == '2026-12-01T00:00:00Z'


def test_a_trial_user_is_still_gated_on_trial_end(monkeypatch):
    monkeypatch.setattr(main, 'get_user_doc', lambda e: {
        'plan': 'trial', 'trial_end': '2026-09-01T00:00:00Z',
    })
    assert main._tg_access_until('t@example.test') == '2026-09-01T00:00:00Z'


def test_the_sweep_no_longer_deletes_a_paid_connection(monkeypatch):
    """is_trial_expired() gates the hourly sweep, which REMOVES connections."""
    monkeypatch.setattr(main, 'get_user_doc', lambda e: {
        'plan': 'pro', 'trial_end': '2020-01-01T00:00:00Z',
    })
    assert main.is_trial_expired('paid@example.test') is False, (
        'the sweep would disconnect a paying subscriber whose trial has elapsed'
    )


def test_the_skip_is_no_longer_silent():
    """A dropped send that logs nothing is indistinguishable from no signal."""
    from pathlib import Path
    src = (Path(main.__file__).resolve().parent / 'scripts' / 'notifications'
           / 'dispatcher.py').read_text(encoding='utf-8', errors='replace')
    body = src.split('_tg_send_all')[1]
    body = body[:body.find('\n    def ')] if '\n    def ' in body else body
    assert 'SKIPPED' in body, 'the entitlement skip is a bare `continue` again'
