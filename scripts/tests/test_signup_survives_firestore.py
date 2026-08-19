"""Signup must not depend on Firestore being healthy.

Reported: the signup form showed "Our datastore is not responding right now."
That message is correct — it is the 6s guard added when the OTP write was hanging
— but it means no new user can register whenever Firestore is unhappy, which on a
free tier with an exhausted daily write quota is most of the time.

The real defect is architectural. A six-digit, single-use token that lives five
minutes was being stored in a quota-limited datastore, making a transient
datastore problem a hard signup outage.

The store is now memory-authoritative with a best-effort Firestore mirror written
on a daemon thread. This is safe because the process is SINGLE-WORKER (start.sh:
`uvicorn main:app`, no --workers) so send and verify land in the same process.
Adding --workers later without revisiting this would break verification.

The two remaining Firestore reads on the path fail OPEN:
  * duplicate-email pre-check — a courtesy message. Firebase Auth is the actual
    uniqueness authority and rejects duplicates at account creation regardless.
  * OTP cooldown — abuse is already bounded by two in-memory _rate_limit calls.
"""
import re
from pathlib import Path

import pytest

MAIN = (Path(__file__).resolve().parent.parent.parent / 'main.py').read_text(
    encoding='utf-8', errors='replace')


def _endpoint() -> str:
    i = MAIN.index('async def send_otp_for_registration')
    return MAIN[i:MAIN.index('\n@app.', i)]


def test_the_otp_store_is_memory_authoritative():
    assert '_OTP_MEM' in MAIN, 'the in-memory OTP store is gone'
    body = MAIN[MAIN.index('def _otp_set('):]
    body = body[:body.index('\ndef ')]
    assert '_OTP_MEM[email]' in body, '_otp_set no longer writes memory'
    assert '_otp_mirror' in body, 'the Firestore mirror is gone'


def test_the_mirror_cannot_block_or_raise():
    body = MAIN[MAIN.index('def _otp_mirror('):]
    body = body[:body.index('\ndef ')]
    assert 'daemon=True' in body, 'the mirror is not on a daemon thread'
    assert 'except Exception' in body, 'the mirror can raise into the request'


def test_the_otp_write_is_no_longer_awaited_against_firestore():
    """The call that hung. It must now be a plain in-process assignment."""
    body = _endpoint()
    assert re.search(r'\n\s*_otp_set\(email, \{', body), '_otp_set is not called directly'
    assert '_fs_await(_otp_set' not in body, 'the OTP write goes through Firestore again'


@pytest.mark.parametrize('what', ['user lookup', 'OTP cooldown check'])
def test_the_remaining_firestore_reads_fail_open(what):
    """A courtesy lookup timing out must not block registration."""
    body = _endpoint()
    i = body.index(what)
    window = body[max(0, i - 400): i + 400]
    assert 'except HTTPException' in window, (
        f'the "{what}" read still 503s the whole signup when Firestore is slow'
    )


def test_lookup_helpers_search_memory_first():
    """Memory-authoritative storage is useless if the verify step only queries
    Firestore for the record this process just wrote."""
    for fn in ('_otp_find_by_signup_token', '_otp_find_by_phone'):
        body = MAIN[MAIN.index(f'def {fn}('):]
        body = body[:body.index('\ndef ')]
        assert '_OTP_MEM' in body, f'{fn} does not consult the in-memory store'


def test_single_worker_assumption_is_documented():
    """If --workers is ever added, send and verify can land in different
    processes and the memory store silently breaks."""
    assert 'SINGLE-WORKER' in MAIN or 'single-worker' in MAIN.lower()
