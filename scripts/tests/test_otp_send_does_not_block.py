"""Sending a signup OTP must not hang, and must not stall the whole server.

Reported: "Sending verification code…" spins forever on the signup form.

Nothing was wrong with SMS. _send_sms_otp already bounds MSG91 at 12s. The
endpoint made FOUR blocking Firestore round trips before reaching it —
get_user_doc, is_cooldown_active (_otp_get), and _otp_set — each a synchronous
google-cloud-firestore call issued from inside an `async def`.

Two consequences, and the second is worse than the reported symptom:

  1. With the daily WRITE quota exhausted, _otp_set retries with backoff for up to
     a minute. The browser had no timeout of its own, so the spinner never
     resolved.
  2. A synchronous call inside `async def` blocks the EVENT LOOP. So that same
     stalled write froze every other request too — one signup attempt made the
     whole site sticky.

These tests pin the structure, not the timing: that the blocking calls are dropped
off the loop with a deadline, and that the browser bounds its own fetch. Timing
cannot be asserted here without a live Firestore.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN = (ROOT / 'main.py').read_text(encoding='utf-8', errors='replace')
AUTH = (ROOT / 'web' / 'src' / 'scripts' / 'auth.js').read_text(encoding='utf-8', errors='replace')


def _endpoint_body() -> str:
    i = MAIN.index('async def send_otp_for_registration')
    j = MAIN.index('\n@app.', i)
    return MAIN[i:j]


# ── server: off the loop, with a deadline ────────────────────────────────────

def test_the_off_loop_helper_exists():
    assert 'async def _fs_await' in MAIN, 'the off-loop Firestore helper is gone'
    body = MAIN[MAIN.index('async def _fs_await'):]
    body = body[:body.index('\ndef ')]
    assert 'asyncio.to_thread' in body, 'the call is no longer moved off the event loop'
    assert 'asyncio.wait_for' in body, 'the call is no longer bounded by a deadline'
    assert 'TimeoutError' in body, 'a timeout is not handled'


@pytest.mark.parametrize('fn', ['get_user_doc', 'is_cooldown_active'])
def test_every_blocking_read_goes_through_the_helper(fn):
    """The regression that matters. A direct call here re-blocks the loop.

    _otp_set is deliberately NOT in this list any more. It was superseded by a
    better fix than a deadline: the write was removed from the request path
    entirely — see test_the_otp_write_no_longer_touches_firestore below.
    """
    body = _endpoint_body()
    assert f'_fs_await({fn}' in body, f'{fn} is not routed through _fs_await'


def test_the_otp_write_no_longer_touches_firestore():
    """Replaces an earlier assertion that required a tight Firestore deadline.

    A deadline was the right FIRST fix — it turned a minute-long hang into a
    6-second error — but the error still read "Our datastore is not responding"
    and still meant nobody could register.

    The store is now memory-authoritative with a daemon-thread Firestore mirror,
    so the write cannot hang, cannot fail on quota, and needs no deadline. A
    bounded call here would mean the dependency came back.
    """
    body = _endpoint_body()
    assert '_fs_await(_otp_set' not in body, (
        'the OTP write is a Firestore round trip again — signup is coupled to '
        'the datastore once more'
    )
    assert '_otp_set(email, {' in body, (
        '_otp_set is not called directly — is the store still memory-first?'
    )


def test_the_sms_call_was_already_bounded():
    """Guards against 'fixing' the wrong layer next time: MSG91 was never the
    problem, and its 12s ceiling should stay."""
    i = MAIN.index('async def _send_sms_otp')
    assert 'timeout=12.0' in MAIN[i:i + 2000]


# ── browser: the spinner must always resolve ─────────────────────────────────

def test_the_client_fetch_is_bounded():
    i = AUTH.index('export async function sendOTPForSignup')
    body = AUTH[i:AUTH.index('\nexport ', i + 10)]
    assert 'AbortController' in body, 'fetch is unbounded again — fetch has no default timeout'
    assert 'signal: ctl.signal' in body, 'the abort signal is not passed to fetch'
    assert 'AbortError' in body, 'an aborted request has no user-facing message'
    assert 'clearTimeout' in body, 'the timer leaks on the success path'


def test_the_client_deadline_exceeds_every_server_deadline():
    """So a real server error wins the race and the user sees the real reason
    rather than a generic timeout."""
    i = AUTH.index('export async function sendOTPForSignup')
    body = AUTH[i:AUTH.index('\nexport ', i + 10)]
    m = re.search(r'ctl\.abort\(\),\s*(\d+)\)', body)
    assert m, 'no client abort deadline found'
    client_ms = int(m.group(1))
    server_max_ms = 12_000          # the SMS ceiling, the longest server-side wait
    assert client_ms > server_max_ms, (
        f'client aborts at {client_ms}ms, before the server can finish a '
        f'{server_max_ms}ms SMS attempt — the user would see a timeout instead of '
        f'the real error'
    )
