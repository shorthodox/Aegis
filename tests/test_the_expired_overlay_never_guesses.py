"""The SUBSCRIPTION EXPIRED overlay must never appear on a guess.

Reported 2026-08-24, by a user whose account was unlocked with a dev token:

    "when login back ... i recieve this as i login, it disappear when i refresh
     ... i want it not to appear at first time, also check if this is happening
     for the paid user's and trial user too"

It was, and for all three kinds of user, because it is a RACE rather than an
entitlement problem.

dashboard.js re-derived access itself, from Firestore, independently of the
server. That is the third implementation of access in this codebase — memory of
the earlier fix records has_access as the single authority and gatekeeper.js was
moved onto it, but this file never was.

On first load AuthManager has not been populated yet, so:

  * the fast path needs u.subscription_active === true, which only exists after
    /auth/me answers -> misses, for PAID users as readily as anyone
  * the Firestore walk then looks for subscription.status == 'active', which a
    dev-token or volume-entitled user simply does not have
  * it falls through and returns 'expired' -> setExpiredView() -> overlay

A refresh finds AuthManager populated and the overlay does not appear. That is
the whole of "it disappear when i refresh".

THE FIX has two halves:

1. the server's has_access is consulted FIRST and outranks every local guess
2. setExpiredView refuses to draw while access is merely UNKNOWN

The guard lives at the chokepoint because there are ELEVEN call sites for
setExpiredView, and guarding them one at a time is how this kept coming back.

SAFETY: the overlay is UX, not enforcement. /api/signals gates on the plan
server-side and returns 403 regardless of what the browser draws, so failing
open here cannot leak signals to a lapsed subscriber — while failing closed
shows a paying customer a lie they have to refresh away.
"""
import io
import re


def _js(path):
    return io.open(path, encoding='utf-8').read()


DASH = 'web/src/scripts/dashboard.js'
GATE = 'web/src/scripts/gatekeeper.js'


# -- the overlay refuses to guess ---------------------------------------------

def test_the_overlay_is_gated_on_access_being_known():
    js = _js(DASH)
    i = js.index('function setExpiredView() {')
    body = js[i:i + 1200]
    assert '__aegisAccessKnown' in body, (
        'setExpiredView can still draw before the server has answered, which is '
        'the flicker'
    )
    assert 'return;' in body


def test_the_guard_is_the_first_thing_in_the_function():
    """It must outrank the isPremium and debounce checks, which are themselves
    derived from the same not-yet-known state."""
    js = _js(DASH)
    i = js.index('function setExpiredView() {')
    body = js[i:i + 1200]
    known = body.index('__aegisAccessKnown')
    premium = body.index('_subState.isPremium')
    assert known < premium


def test_the_guard_sits_at_the_chokepoint_not_the_call_sites():
    """There are eleven callers; guarding each is how this regressed before."""
    js = _js(DASH)
    assert js.count('setExpiredView()') >= 8


# -- the server's answer outranks the local derivation ------------------------

def test_the_server_answer_is_consulted_first():
    js = _js(DASH)
    i = js.index('async function checkUserSubscriptionStatus(uid)')
    body = js[i:i + 1400]
    srv = body.index('_serverAccess()')
    fs = body.index('AuthManager.getUser()')
    assert srv < fs, 'the local Firestore derivation still runs before the server'


def test_a_definite_server_answer_marks_access_known():
    js = _js(DASH)
    i = js.index('async function checkUserSubscriptionStatus(uid)')
    body = js[i:i + 1400]
    assert body.count('__aegisAccessKnown = true') >= 2, (
        'a definite yes AND a definite no must both mark access as known'
    )


def test_server_access_returns_null_when_it_cannot_tell():
    """null, not false — 'I do not know' is not 'no'."""
    js = _js(DASH)
    i = js.index('function _serverAccess()')
    body = js[i:i + 700]
    assert 'return null;' in body
    assert "typeof u.has_access === 'boolean'" in body


def test_a_thrown_error_is_not_read_as_no_access():
    js = _js(DASH)
    i = js.index('function _serverAccess()')
    body = js[i:i + 700]
    assert 'catch (_) { return null; }' in body


# -- the flag genuinely flips, or the overlay could never show ----------------

def test_the_flag_flips_when_the_server_answers():
    """Without this the overlay is unreachable and a lapsed subscriber keeps
    full access — a worse bug than the flicker."""
    js = _js(GATE)
    i = js.index('window.setServerAccess = function')
    body = js[i:i + 700]
    assert '__aegisAccessKnown = true' in body
    assert 'serverHasAccess !== null' in body


def test_it_only_flips_on_a_real_boolean():
    js = _js(GATE)
    i = js.index('window.setServerAccess = function')
    body = js[i:i + 700]
    assert "(typeof v === 'boolean') ? v : null" in body


def test_the_server_answer_is_persisted_for_the_dashboard_to_read():
    """dashboard.js reads has_access off AuthManager, so applyUserData must
    store the whole /auth/me payload."""
    js = _js(GATE)
    i = js.index('function applyUserData(')
    body = js[i:i + 900]
    assert 'AuthManager.setUser(userData)' in body
    assert "typeof userData.has_access === 'boolean'" in body


def test_the_gatekeeper_overlay_is_gated_at_its_chokepoint_too():
    """It is reachable through a standalone branch when dashboard.js has not
    loaded, so the call-site guards are not sufficient on their own."""
    js = _js(GATE)
    i = js.index('function showSubscriptionExpiredOverlay() {')
    body = js[i:i + 700]
    assert 'accessKnown()' in body
    assert '__aegisAccessKnown' in body


# -- the server still owns enforcement ----------------------------------------

def test_the_api_gates_signals_server_side():
    """The overlay is UX. Failing open in the browser cannot leak signals."""
    main = io.open('main.py', encoding='utf-8').read()
    i = main.index('def api_signals(')
    body = main[i:i + 900]
    assert "plan != 'pro'" in body
    assert '403' in body


# ── Whop Pixel ───────────────────────────────────────────────────────────────
# Installed 2026-08-27 on all 22 pages, verbatim. It loads a third-party script
# from https://t.whop.tw, so it interacts directly with the CSP shipped two days
# earlier: without that host in script-src the pixel is blocked the moment the
# policy goes from Report-Only to enforcing, and the failure is silent — the
# dashboard simply shows no events and nothing says why.

PIXEL_ID = 'biz_Pa8k1dIsPYugdt'


def test_the_pixel_is_on_every_page():
    import glob
    pages = glob.glob('web/src/pages/*.html')
    missing = [p for p in pages if PIXEL_ID not in _js(p)]
    assert not missing, f'pages without the pixel: {missing}'


def test_it_sits_inside_the_head():
    js = _js('web/src/pages/index.html')
    assert js.index(PIXEL_ID) < js.index('</head>')


def test_the_snippet_is_unaltered():
    """Whop's instruction was explicit: do not change a character."""
    js = _js('web/src/pages/index.html')
    for fragment in (
        '!function(w,d,s,u,n,a,b){if(w[n])return;',
        'a.q.push([+new Date].concat([].slice.call(arguments)))',
        '"script","https://t.whop.tw","whop");',
        f'whop.setScope("{PIXEL_ID}");whop.track("page");',
    ):
        assert fragment in js, fragment[:50]


def test_it_appears_exactly_once_per_page():
    """A duplicated pixel double-counts every page view."""
    import glob
    for p in glob.glob('web/src/pages/*.html'):
        assert _js(p).count('whop.track("page")') == 1, p


def test_the_csp_allows_the_pixel_to_load():
    """Anchored on the DIRECTIVE, not the word. The comment above the policy
    also contains "script-src", and matching that measured the wrong span."""
    main_src = io.open('main.py', encoding='utf-8').read()
    i = main_src.index(""""script-src 'self'""")
    j = main_src.index('style-src', i)
    assert 't.whop.tw' in main_src[i:j], (
        'the pixel script is not in script-src, so it is blocked the moment the '
        'CSP is enforced — silently'
    )


def test_the_csp_allows_the_pixel_to_report():
    """Loading is not enough; it beacons events back."""
    main_src = io.open('main.py', encoding='utf-8').read()
    i = main_src.index(""""connect-src 'self'""")
    j = main_src.index('frame-src', i)
    assert 't.whop.tw' in main_src[i:j]
