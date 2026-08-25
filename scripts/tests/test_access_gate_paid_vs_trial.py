"""A paid subscriber must never be told their trial expired.

Reported: log in with a valid subscription, land on the dashboard, get an
"expired" overlay, refresh, and it is gone.

Not a timing glitch — a missing condition. gatekeeper.js held the paid-plan list
in THREE places. checkAuthAndLoad and applyUserData both tested

    !PAID.includes(userPlan) && !trialActive

but the 'trial-status-updated' listener tested only

    trialActive = AuthManager.isTrialValid();
    if (!trialActive) showSubscriptionExpiredOverlay();

isTrialValid() knows nothing about a paid plan, and a paid subscriber has no live
trial BY DEFINITION — so it returned false and the overlay fired over a valid
subscription. It vanished on refresh because the reload takes the cached-token
fast path, which never emits 'trial-status-updated'. Hence "on login, gone on F5".

Also fixed here: the copy. The overlay markup is hardcoded "Subscription Expired /
Your trial period has ended", which is wrong for a subscriber whose 30-day plan
lapsed — they never had a trial, and being told they did reads as the product
having lost their payment. Paid users now get "Plan Expired" and a Renew CTA.

These tests read the JS as text because there is no JS test runner in this repo.
That is a real limitation: they pin the CONDITIONS and the strings, not runtime
behaviour. They would still have caught this bug, which is what matters.
"""
import re
from pathlib import Path

import pytest

GK = (Path(__file__).resolve().parent.parent.parent
      / 'web' / 'src' / 'scripts' / 'gatekeeper.js')


@pytest.fixture(scope='module')
def src():
    return GK.read_text(encoding='utf-8', errors='replace')


# ── one definition ───────────────────────────────────────────────────────────

def test_the_paid_plan_list_is_defined_exactly_once(src):
    """Three copies is how one of them ended up missing the check."""
    lits = re.findall(r"\[\s*'pro',\s*'premium',\s*'intermediate',\s*'basic',\s*'pro-dev'\s*\]", src)
    assert len(lits) == 1, f'paid-plan list appears {len(lits)} times — it must be one constant'


def test_there_is_a_shared_access_helper(src):
    assert re.search(r'function hasActiveAccess\s*\(', src), 'hasActiveAccess() is gone'
    assert re.search(r'function isPaidPlan\s*\(', src), 'isPaidPlan() is gone'


def test_access_is_a_paid_plan_OR_a_live_trial(src):
    """Either alone is sufficient. Requiring both would lock out paid users
    (the reported bug); requiring neither would let expired users in."""
    body = src.split('function hasActiveAccess')[1].split('}')[0]
    assert 'isPaidPlan()' in body and 'trialActive' in body
    assert '||' in body, 'access must be paid OR trial, not AND'


# ── the bug itself ───────────────────────────────────────────────────────────

def test_the_trial_status_listener_consults_the_plan(src):
    """The exact regression. This listener must not gate on the trial alone."""
    m = re.search(r"addEventListener\('trial-status-updated'.*?\n\}\);", src, re.S)
    assert m, "the 'trial-status-updated' listener is gone"
    body = m.group(0)
    assert 'hasActiveAccess()' in body, (
        'the listener gates on the trial alone again — this is what showed '
        '"trial expired" to paying subscribers'
    )
    assert not re.search(r'if\s*\(\s*!\s*trialActive\s*&&', body), (
        'reverted to `if (!trialActive && ...)`, which ignores a paid plan'
    )


def test_every_overlay_call_site_uses_the_helper(src):
    """No call site may re-derive the condition inline."""
    for m in re.finditer(r'\n[^\n]*showSubscriptionExpiredOverlay\(\)', src):
        line = src[m.start():src.index(chr(10), m.start() + 1)]
        # The DEFINITION is not a call site. Recognise it from the matched
        # line itself rather than a fixed-size window of surrounding text --
        # the window version broke the moment a guard was added at the top of
        # the function body, and began auditing the definition as a caller.
        if 'function showSubscriptionExpiredOverlay' in line:
            continue
        window = src[max(0, m.start() - 320): m.start()]
        assert ('hasActiveAccess()' in window
                or 'window.setExpiredView' in window
                or 'trialExpired' in window), (
            f'an overlay call site re-derives its own condition: '
            f'...{window[-120:].strip()}'
        )


# ── the copy ─────────────────────────────────────────────────────────────────

def test_paid_and_trial_get_different_words(src):
    assert '_applyExpiredCopy' in src, 'the copy is hardcoded again'
    body = src.split('function _applyExpiredCopy')[1][:1400]
    assert 'Plan Expired' in body and 'Trial Expired' in body, (
        'both headings must exist — a lapsed plan is not an ended trial'
    )
    assert '30-day plan has ended' in body, 'paid users need plan wording, not trial wording'
    assert 'Renew' in body, 'a lapsed subscriber renews; they do not subscribe afresh'


def test_the_plan_badge_also_distinguishes_them(src):
    assert 'PLAN EXPIRED' in src, 'the badge still says TRIAL EXPIRED for paid users'
    m = re.search(r'PLAN EXPIRED', src)
    window = src[max(0, m.start() - 260): m.start()]
    assert 'isPaidPlan(' in window, 'the badge does not branch on plan type'


# ── the sign-in error that vanished ──────────────────────────────────────────

def test_redirect_to_login_does_not_reload_the_login_page(src):
    """The other reported bug, same file.

    handleEmailLogin signs the user back out when Firebase authenticates them but
    their Firestore profile is missing. That signOut fires onAuthStateChanged(null)
    -> redirectToLogin() -> href='/' — reloading the landing page out from under
    the modal that had just rendered "No account found". Hence the message
    flashing for about a second.
    """
    m = re.search(r'function redirectToLogin\s*\(\)\s*\{(.*?)\n\}', src, re.S)
    assert m, 'redirectToLogin() is gone'
    body = m.group(1)
    assert re.search(r"pathname|location\.pathname", body), (
        'redirectToLogin no longer checks where it already is — it will reload '
        'the login page and discard any sign-in error on screen'
    )
    assert re.search(r"===\s*'/'", body), "the '/' case is not special-cased"
    assert 'return' in body, 'it must bail out rather than navigate'
