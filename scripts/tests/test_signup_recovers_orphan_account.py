"""A half-created account must not lock the user out of their own email.

Reported: "when we re try creating the account when otp is not verified it says
account already exist, it should not."

What actually happened — the account is NOT created before the OTP. Step 1 only
calls checkPhoneUnique and sendOTPForSignup, neither of which touches Firebase
Auth. The confusion comes from the UI: handleStep2Verify renders a STEP 2 failure
back on step 1 (`if (signupResult.needsSignin) { showStep1(); showError(...) }`),
so a failure that happened after OTP entry appears on the form that precedes it.

The real defect is that createUserWithEmailAndPassword creates a PERMANENT Auth
account, and anything failing after it burns the email:

    1. OTP entered -> Auth account created -> profile write times out -> error
    2. retry       -> auth/email-already-in-use -> "sign in instead"
    3. sign in     -> no Firestore profile -> "No account found for this email"

Three screens, each technically accurate, that together lock someone out of an
address they own because of a datastore timeout.

Recovery is safe at this point and nowhere else: OTP entry already proved they own
the address, and signInWithEmailAndPassword proves they know the password. Both
proofs, then adopt. Wrong password falls through to the original message, so this
cannot be used to take over somebody else's account.
"""
import re
from pathlib import Path

import pytest

AUTH = (Path(__file__).resolve().parent.parent.parent
        / 'web' / 'src' / 'scripts' / 'auth.js').read_text(encoding='utf-8', errors='replace')


def _signup_body() -> str:
    i = AUTH.index('export async function handleEmailSignup')
    return AUTH[i:AUTH.index('\nexport ', i + 10)]


def _dup_branch() -> str:
    b = _signup_body()
    i = b.index("error.code === 'auth/email-already-in-use'")
    return b[i:b.index("else if (error.code === 'auth/invalid-email')", i)]


def test_a_duplicate_email_attempts_recovery():
    assert 'signInWithEmailAndPassword' in _dup_branch(), (
        'signup dead-ends on auth/email-already-in-use again — a half-created '
        'account permanently burns the email'
    )


def test_recovery_requires_the_password():
    """The security property. Adoption happens only if they can authenticate."""
    branch = _dup_branch()
    assert 'signInWithEmailAndPassword(auth, email, password)' in branch, (
        'the account is adopted without proving the password — that would be an '
        'account-takeover path'
    )


def test_a_wrong_password_still_refuses():
    branch = _dup_branch()
    assert 'catch (recoverErr)' in branch
    tail = branch[branch.index('catch (recoverErr)'):]
    assert 'needsSignin: true' in tail, (
        'a genuine duplicate must still be refused with the sign-in prompt'
    )


def test_recovery_reprovisions_the_backend_record():
    """Adopting an orphan is pointless if it stays an orphan."""
    branch = _dup_branch()
    assert 'ensureUserDocumentV2' in branch, 'the profile is not rebuilt on adoption'
    assert '/api/users/provision' in branch, 'the backend record is not rebuilt'


def test_recovery_establishes_a_session():
    branch = _dup_branch()
    assert 'AuthManager.setToken' in branch and 'AuthManager.setUser' in branch, (
        'the adopted account is not signed in — the user would land on a '
        'dashboard with no session'
    )


def test_the_profile_write_during_recovery_cannot_re_fail_the_signup():
    """The same trap as the original bug: recovery must not be undone by the very
    write whose failure created the orphan."""
    branch = _dup_branch()
    m = re.search(r'try \{ await ensureUserDocumentV2\([^)]*\); \} catch', branch)
    assert m, 'ensureUserDocumentV2 is unguarded inside the recovery path'
