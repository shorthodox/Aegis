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
    """/auth/me must report a real end date.

    Updated 2026-08-23. The original premise here — that a top-level
    "subscription_end" has never existed — turned out to be wrong: the Whop and
    Razorpay handlers write exactly that, and write NONE of the in-subscription
    keys every reader looked for. So the writers and readers were on different
    keys and the term never ended for anyone.

    Both shapes are now read through _user_sub_end, so this asserts the property
    rather than the spelling.
    """
    import inspect
    src = inspect.getsource(main.get_me)
    assert '_user_sub_end(user_doc)' in src, (
        '/auth/me re-derives the end date instead of using the one reader that '
        'knows where providers actually write it'
    )


def test_the_end_date_is_found_in_either_shape():
    """A document written by the provider webhooks, and one written the
    canonical way, must both expire."""
    from datetime import datetime, timedelta, timezone
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()

    # what the Whop handler actually wrote: top level only
    top_past = {'plan': 'pro', 'subscription': {'status': 'active'},
                'subscription_end': past}
    top_live = {'plan': 'pro', 'subscription': {'status': 'active'},
                'subscription_end': future}
    assert main._user_sub_end(top_past) is not None, (
        'the top-level end date is invisible, so a lapsed plan never expires'
    )
    assert main.has_paid_access(top_past) is False
    assert main.has_paid_access(top_live) is True

    # the canonical shape
    inner_past = {'plan': 'pro',
                  'subscription': {'status': 'active', 'current_period_end': past}}
    assert main.has_paid_access(inner_past) is False


def test_a_plan_with_no_end_date_anywhere_still_fails_open():
    """Absent bookkeeping is not a cancellation — the documented asymmetry."""
    assert main.has_paid_access({'plan': 'pro', 'subscription': {'status': 'active'}}) is True


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


# -- the lock screen must not flicker -----------------------------------------
# Reported 2026-08-23: "this is coming when I am getting into dashboard, it
# disappears when I refresh it".
#
# hasActiveAccess() falls back to the plan NAME plus a cached trial flag until
# /auth/me lands. Both can disagree with the server for the first moment of a
# load, and a user object cached BEFORE has_access existed has no such field at
# all — so that path answers from the name forever. Two code paths reaching
# opposite conclusions in the same second is what made the lock screen flash on
# and off.
#
# The overlay is a hard, disruptive statement. It is now drawn only on a KNOWN
# negative; an unknown answer draws nothing and waits.

def test_the_overlay_waits_for_a_known_answer():
    src = _js('web/src/scripts/gatekeeper.js')
    assert 'function accessKnown()' in src
    assert src.count('accessKnown() && !hasActiveAccess()') >= 3, (
        'an overlay call site can still fire before the server has answered — '
        'that is the flicker'
    )


def test_no_overlay_site_decides_on_the_fallback_alone():
    """Any bare `!hasActiveAccess()` guarding the overlay is the bug returning."""
    src = _js('web/src/scripts/gatekeeper.js')
    for line in src.splitlines():
        if 'showSubscriptionExpiredOverlay' not in line and '!hasActiveAccess()' in line:
            assert 'accessKnown()' in line, f'unguarded overlay decision: {line.strip()}'


def test_feature_gating_may_still_be_optimistic():
    """Only the OVERLAY needs certainty. hasActiveAccess() keeps its permissive
    fallback so the UI is not blank for a second on every load."""
    src = _js('web/src/scripts/gatekeeper.js')
    assert 'return isPaidPlan() || trialActive === true;' in src
