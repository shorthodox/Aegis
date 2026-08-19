"""A failed profile write must not strand an account it already created.

Reported: the OTP arrived, the code was entered, and "Verify & Create Account"
answered with the literal string `firestore-write-timeout`.

Two defects in one screen.

1. THE FAILURE MODE WAS WORSE THAN THE FAILURE. handleEmailSignup calls
   createUserWithEmailAndPassword FIRST, so the Firebase Auth account exists
   before the Firestore profile write is attempted. Throwing on that write does
   not cancel the signup — it leaves an Auth account with no profile, which is
   exactly the state that makes the next sign-in report "No account found for
   this email" (auth.js needsSignup path). The abort created the broken account
   it was trying to avoid.

   It was also unnecessary. users/{email} written by /api/users/provision is the
   authoritative record, and gatekeeper.js provisions lazily from the Firebase
   user on the next dashboard load if it is missing. Two independent paths
   already rebuild what this write does.

   Note the inversion: the BACKEND provision was explicitly non-fatal
   ("console.warn ... non-fatal") while this CLIENT write was fatal. The less
   important of the two was the one that could fail a signup.

2. AN INTERNAL IDENTIFIER REACHED THE USER. The catch fell through to
   `error.message` for anything without a Firebase error code, so the sentinel
   thrown by the 8s write guard was rendered as UI copy.
"""
import re
from pathlib import Path

import pytest

AUTH = (Path(__file__).resolve().parent.parent.parent
        / 'web' / 'src' / 'scripts' / 'auth.js').read_text(encoding='utf-8', errors='replace')


def _signup_body() -> str:
    i = AUTH.index('export async function handleEmailSignup')
    return AUTH[i:AUTH.index('\nexport ', i + 10)]


def test_the_profile_write_cannot_fail_the_signup():
    body = _signup_body()
    assert 'catch (docErr)' in body, (
        'ensureUserDocumentV2 is fatal again — a timeout here strands an Auth '
        'account with no profile, which reads as "No account found" on next login'
    )


def test_the_auth_account_is_created_before_the_profile_write():
    """The ordering is what makes an abort harmful rather than clean."""
    body = _signup_body()
    assert body.index('createUserWithEmailAndPassword') < body.index('ensureUserDocumentV2'), (
        'if the profile write ever precedes account creation, aborting on it '
        'would be safe and this protection could be reconsidered'
    )


def test_no_internal_identifier_can_reach_the_user():
    """`firestore-write-timeout` is a sentinel, not a sentence."""
    body = _signup_body()
    line = next((l for l in body.splitlines() if 'else if (error.message' in l), None)
    assert line is not None, 'the error.message fallback is gone entirely'
    # It must be GUARDED. A bare `else if (error.message)` republishes any Error
    # thrown anywhere inside signup as user-facing copy.
    assert line.strip() != '} else if (error.message) {', (
        'error.message is surfaced unfiltered again — an internal sentinel like '
        'firestore-write-timeout would reach the form verbatim'
    )
    assert 'test(error.message)' in line, 'the human-readability guard is gone'


def test_the_session_survives_a_missing_profile():
    """userData is null when the write failed; the session still needs an email."""
    body = _signup_body()
    assert 'userData || {' in body, (
        'AuthManager.setUser(null) — the dashboard would read the session as a '
        'broken account'
    )


def test_the_backend_provision_is_still_non_fatal():
    """Guards the fix from being 'balanced' by making the other path fatal."""
    body = _signup_body()
    assert 'non-fatal' in body
    assert body.count('console.warn') >= 2
