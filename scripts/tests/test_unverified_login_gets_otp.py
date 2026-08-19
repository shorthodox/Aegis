"""Signing in to an unverified account must offer the code, not a blank form.

Requested: "if id is created and it is not verified... an otp verification popup
should arrive when trying to login through that id."

Before this, an unfinished account produced one of two dead ends:

  * Firestore profile missing  -> needsSignup   -> "No account found for this
    email. Please create an account first." — on an address that demonstrably HAS
    an account, since the password had just authenticated against Firebase Auth.
  * /auth/me not OK            -> needsVerification -> "Complete Sign Up ->", a
    link that reopened a BLANK registration form.

Either way the user retyped everything to create an account that already existed,
and nothing ever asked for the one thing actually missing: the code.

Now sign-in dispatches `verifyExistingAccount` with the email and the
just-authenticated password. signup-ui opens the modal directly at the OTP step,
and on verify the normal completion path runs: createUserWithEmailAndPassword
returns auth/email-already-in-use and the recovery branch adopts the account,
rebuilds its profile and backend record, and signs them in.

The password matters and is not incidental — it is the proof the recovery branch
requires before adopting an existing account.
"""
from pathlib import Path

import pytest

W = Path(__file__).resolve().parent.parent.parent / 'web' / 'src' / 'scripts'
AUTH = (W / 'auth.js').read_text(encoding='utf-8', errors='replace')
SIGNIN = (W / 'signin-ui.js').read_text(encoding='utf-8', errors='replace')
SIGNUP = (W / 'signup-ui.js').read_text(encoding='utf-8', errors='replace')


def test_a_missing_profile_is_verification_not_a_missing_account():
    """The Auth account exists and the password worked — that is unfinished
    signup, not 'no account'."""
    i = AUTH.index('export async function handleEmailLogin')
    body = AUTH[i:AUTH.index('\nexport ', i + 10)]
    j = body.index('if (!docSnap.exists())')
    branch = body[j:j + 700]
    assert 'needsVerification: true' in branch, (
        'a missing profile still reports needsSignup — the user is sent to a '
        'blank form to re-create an account they already own'
    )
    assert 'needsSignup: true' not in branch


def test_signin_dispatches_the_verification_popup():
    i = SIGNIN.index('} else if (result.needsVerification)')
    # Stop at the NEXT branch: needsSignup legitimately still opens the blank
    # signup form, because there really is no account in that case.
    branch = SIGNIN[i:SIGNIN.index('} else if (result.needsSignup)', i)]
    assert "verifyExistingAccount" in branch, (
        'sign-in no longer opens the OTP step for an unverified account'
    )
    assert 'openSignup' not in branch, (
        'still reopening a blank signup form instead of the code step'
    )


def test_the_known_good_password_is_carried_over():
    """Without it the recovery branch cannot prove ownership and adoption fails."""
    i = SIGNIN.index('} else if (result.needsVerification)')
    branch = SIGNIN[i:i + 900]
    assert 'password' in branch, 'the authenticated password is not passed along'


def test_signup_exposes_the_verification_entry_point():
    assert 'openVerifyExistingAccount' in SIGNUP
    assert "addEventListener('verifyExistingAccount'" in SIGNUP, (
        'nothing listens for the event sign-in dispatches'
    )


def test_the_entry_point_sends_a_code_and_shows_the_otp_step():
    i = SIGNUP.index('export async function openVerifyExistingAccount')
    body = SIGNUP[i:SIGNUP.index('\nfunction listenForSignupEvent', i)]
    assert 'sendOTPForSignup' in body, 'no code is sent'
    assert 'showStep2' in body, 'the OTP step is never shown'
    assert '_pending' in body, 'the completion path has no email/password to use'


def test_a_failed_send_is_reported_not_silent():
    i = SIGNUP.index('export async function openVerifyExistingAccount')
    body = SIGNUP[i:SIGNUP.index('\nfunction listenForSignupEvent', i)]
    assert 'if (!result.success)' in body and 'showError' in body, (
        'a failed OTP send would leave an empty modal with no explanation'
    )
