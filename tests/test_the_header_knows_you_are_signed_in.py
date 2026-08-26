"""The header must not report "signed out" while it is merely still asking.

Reported 2026-08-25: the nav shows "Dashboard / Sign Out" on /, /track-record
and /reviews but "Login / Create Account" on /pricing, /refund-policy and
/risk-disclosure — in the same authenticated session.

It is NOT the page code. Those pages load the same scripts, carry the same
element ids, ship byte-identical initial markup for #authButtonContainer, and
pass byte-identical callbacks to subscribeToAuthState. The difference was in
subscribeToAuthState itself, and it is two faults stacked.

1 - THE WRONG ANSWER WAS FAST, THE RIGHT ANSWER WAS SLOW

    if (user) {
      const userData = await getDoc(doc(db, 'users', user.uid));   // network
      callback({ authenticated: true, ... });
    } else {
      callback({ authenticated: false, ... });                     // instant
    }

The signed-IN path waited on a Firestore round trip before it would admit the
session existed; the signed-OUT path answered immediately with no await. So the
header rendered "Login / Create Account" for as long as that read took, on a
page where a valid session existed. Which pages showed it came down to how much
else was contending for Firestore.

2 - A THROWN AWAIT DROPPED THE CALLBACK ENTIRELY

An async onAuthStateChanged handler that raises is not caught by Firebase, so if
that getDoc threw the subscriber was NEVER CALLED. On a day when Firestore is
over its daily quota — which happened on this project — that is not a flicker,
it is a header permanently stuck on "Login" for a paying subscriber.

THE FIX: announce identity first, enrich second. `user` is already the verified
Firebase user by the time the handler runs — the profile document is extra
detail and must never gate the fact of being signed in. The read is wrapped, and
a second callback carries userData if and when it arrives.

THIS IS THE THIRD TIME this project has collapsed a THREE-state truth
(yes / no / not known yet) into two and rendered the guess:
  * the expired-subscription overlay drew itself before /auth/me answered
  * checkUserSubscriptionStatus returned 'expired' when it simply could not tell
  * and this
The shape to watch for is a UI that renders an authoritative claim about
entitlement or identity from a value that has not arrived yet.
"""
import io
import re


AUTH = 'web/src/scripts/auth.js'


def _src():
    return io.open(AUTH, encoding='utf-8').read()


def _subscribe_body():
    s = _src()
    i = s.index('export function subscribeToAuthState')
    j = s.index('\nexport function', i + 10)
    return s[i:j]


def test_identity_is_announced_before_the_profile_read():
    body = _subscribe_body()
    cb = body.index('callback({ authenticated: true')
    read = body.index('await getDoc(')
    assert cb < read, (
        'the signed-in callback still waits on a Firestore round trip, so the '
        'header claims "signed out" until the network answers'
    )


def test_the_profile_read_cannot_drop_the_callback():
    body = _subscribe_body()
    i = body.index('await getDoc(')
    assert 'try {' in body[:i], 'the profile read is not guarded'
    assert 'catch (err)' in body, (
        'a throw here rejects the onAuthStateChanged handler and the subscriber '
        'is never called at all'
    )


def test_a_failed_profile_read_still_leaves_the_session_standing():
    body = _subscribe_body()
    assert 'session stands regardless' in body


def test_userdata_still_arrives_when_it_can():
    body = _subscribe_body()
    assert 'if (userData) callback({ authenticated: true, user, userData });' in body


def test_the_signed_out_branch_is_unchanged():
    body = _subscribe_body()
    assert 'authenticated: false' in body
    assert body.count('authStateChange') == 2


def test_both_branches_dispatch_the_event():
    body = _subscribe_body()
    assert "detail: { authenticated: true }" in body
    assert "detail: { authenticated: false }" in body


# -- the pages themselves were never the problem ------------------------------

PAGES = ['index', 'track-record', 'reviews', 'pricing', 'refund-policy',
         'risk_disclosure']


def test_every_page_ships_the_same_nav_container():
    for p in PAGES:
        html = io.open(f'web/src/pages/{p}.html', encoding='utf-8').read()
        assert 'id="authButtonContainer"' in html, p
        assert 'id="signOutBtn"' in html, p


def test_the_working_and_failing_pages_had_identical_callbacks():
    """Guards the diagnosis: if these ever diverge, the cause is elsewhere and
    this test should be the thing that says so."""
    def cb(page):
        html = io.open(f'web/src/pages/{page}.html', encoding='utf-8').read()
        i = html.find('subscribeToAuthState')
        i = html.find('subscribeToAuthState', i + 1)
        return ' '.join(html[i:i + 260].split())
    assert cb('track-record') == cb('refund-policy')
