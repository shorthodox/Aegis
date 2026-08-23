"""The site must not say "active" while delivery says expired.

Reported 2026-08-23: "telegram bot is disconnecting from the site by its own
even though my plan is active". Read off the volume and the live user document:

    plan            : pro
    subscription    : status "active", expires_at 2026-07-06   (7 weeks past)
    has_paid_access : False        <- delivery correctly stopped
    _tg_access_until: 2026-07-01   <- the trial end, written beside the chat_id

The disconnection was CORRECT. The bug was that nothing told the user: three
places computed access and only one was right.

    has_paid_access()          respects the end date          CORRECT
    /auth/me subscription_active   `status == "active"` only  WRONG
    gatekeeper.isPaidPlan()        plan NAME only             WRONG

So the dashboard showed a live Pro plan while Telegram silently delivered
nothing, and the only way to find out was to read the volume by hand.

/auth/me now returns has_access from the same function that gates delivery, and
both frontend paths prefer it over re-deriving.
"""
import pytest

import main


PAID_LIVE = {'plan': 'pro',
             'subscription': {'status': 'active',
                              'current_period_end': '2099-01-01T00:00:00Z'}}
PAID_ELAPSED = {'plan': 'pro',
                'subscription': {'status': 'active',
                                 'expires_at': '2026-07-06T05:34:59Z'}}


def test_the_reported_document_has_no_access():
    """status 'active', term seven weeks past. The exact shape observed."""
    assert main.has_paid_access(PAID_ELAPSED) is False


def test_a_live_paid_plan_still_has_access():
    assert main.has_paid_access(PAID_LIVE) is True


def test_subscription_active_no_longer_means_the_status_string(monkeypatch):
    """The line that told the site a lapsed plan was live.

    It read `subscription.status == "active"`, which is a bookkeeping field
    nothing updates when a term runs out.
    """
    import inspect
    src = inspect.getsource(main.get_me)
    assert '"subscription_active": has_paid_access(user_doc)' in src, (
        'subscription_active is derived from something other than the function '
        'that gates delivery again'
    )
    assert 'has_access' in src, '/auth/me does not expose the authoritative flag'


def test_subscription_end_reads_a_key_that_exists():
    """It used to read a top-level "subscription_end" that has never existed, so
    the frontend was handed None and could not have checked expiry."""
    import inspect
    src = inspect.getsource(main.get_me)
    assert 'user_doc.get("subscription_end")' not in src
    for k in ('current_period_end', 'expires_at', 'end_date'):
        assert k in src, k


@pytest.mark.parametrize('doc,expected', [
    (PAID_LIVE, True),
    (PAID_ELAPSED, False),
])
def test_delivery_and_the_api_now_agree(doc, expected):
    """The whole point: one rule, one answer."""
    assert main.has_paid_access(doc) is expected


# -- the frontend must prefer the server's answer ----------------------------

def _js(path):
    return (main.Path(main.__file__).resolve().parent / path).read_text(
        encoding='utf-8', errors='replace')


def test_gatekeeper_prefers_the_server_flag():
    src = _js('web/src/scripts/gatekeeper.js')
    assert 'serverHasAccess' in src
    assert 'if (serverHasAccess !== null) return serverHasAccess;' in src, (
        'hasActiveAccess() re-derives from the plan name again'
    )
    assert "typeof userData.has_access === 'boolean'" in src, (
        'nothing feeds the server flag in from /auth/me'
    )


def test_authmanager_prefers_the_server_flag():
    src = _js('web/src/auth/authManager.js')
    assert "typeof user.has_access === 'boolean'" in src, (
        'isTrialValid() still decides from the plan name alone'
    )
