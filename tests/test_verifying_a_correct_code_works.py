"""A correct verification code must verify the account, conveniently.

Asked 2026-08-24: "make sure, filling the correct verification code, verify the
account conveniently."

Two things stood between a correct code and a verified account.

1 - THE COMPARISON WAS ON KEYSTROKES, NOT DIGITS

    if record["otp"] != otp:

`otp` arrived raw off the request. A code copied out of an email carries a
trailing newline, a leading space, a non-breaking space, or gets typed as
"123 456". Every one of those was rejected as "Invalid OTP" while being exactly
right - and the user then burned a verify attempt and a 60-second resend
cooldown proving it was not their fault.

2 - A RESTART MADE A VALID CODE UNCHECKABLE

_otp_get falls back to the Firestore mirror whenever this process restarted
mid-signup, which on Railway is often. `record["expires_at"]` then held whatever
the mirror returned rather than the datetime memory had, and

    datetime.now(timezone.utc) > record["expires_at"]

raised TypeError against a string - a 500, not a verdict. Direct [] indexing on
"otp" and "expires_at" could KeyError the same way. So a user holding a perfectly
valid code got a server error and no way forward.

Both are fixed at the comparison rather than by asking the user to be tidier.
The client already stripped non-digits on paste; the server now agrees with it.

The client also submits as soon as the sixth digit lands - typed, pasted, or
autofilled - because making someone enter six digits and then hunt for a button
is the kind of friction that gets blamed on the code being wrong.
"""
import inspect
import io
import re

import main


SRC = inspect.getsource(main.verify_otp_for_registration)
UI = io.open('web/src/scripts/signup-ui.js', encoding='utf-8').read()


# -- the comparison is on digits ---------------------------------------------

def test_the_submitted_code_is_stripped_to_digits():
    assert "_given = re.sub(r'\\D', '', str(otp or \"\"))" in SRC, (
        'a pasted code with a space or newline is still rejected as invalid'
    )


def test_the_stored_code_is_stripped_the_same_way():
    """Both sides, or the normalisation just moves the mismatch."""
    assert "_sent = re.sub(r'\\D', '', str(record.get(\"otp\") or \"\"))" in SRC


def test_the_raw_inequality_is_gone():
    assert 'record["otp"] != otp' not in SRC


def test_an_empty_code_never_passes():
    """Stripping must not turn '' == '' into a successful verification."""
    assert 'if not _sent or not _given' in SRC


def test_the_compare_is_timing_safe():
    assert 'secrets.compare_digest' in SRC


def test_the_normalisation_actually_accepts_the_real_cases():
    """The formats a code genuinely arrives in."""
    sent = '123456'
    for given in ('123456', ' 123456', '123456\n', '123 456', '1 2 3 4 5 6',
                  ' 123456 ', '123-456'):
        assert re.sub(r'\D', '', given) == sent, given


def test_it_does_not_accept_a_different_code():
    assert re.sub(r'\D', '', '123457') != '123456'
    assert re.sub(r'\D', '', '1234567') != '123456'


# -- a restart does not make a valid code uncheckable ------------------------

def test_the_expiry_is_coerced_before_comparison():
    assert 'isinstance(_exp, datetime)' in SRC
    assert '_parse_ts(_exp)' in SRC


def test_a_naive_expiry_is_given_a_timezone():
    """Comparing naive against aware raises, which was a 500 not a verdict."""
    assert '_exp.tzinfo is None' in SRC


def test_an_uncheckable_record_gets_a_clean_message():
    assert 'could not be checked' in SRC
    assert 'status_code=400' in SRC


def test_the_fields_are_read_defensively():
    """record["expires_at"] KeyErrors into a 500 on a partial mirror row."""
    assert 'record["expires_at"]' not in SRC
    assert 'record.get("expires_at")' in SRC


# -- the client submits without a second action ------------------------------

def test_the_sixth_digit_submits():
    assert 'function _maybeAutoVerify()' in UI
    assert UI.count('_maybeAutoVerify();') >= 2, (
        'auto-verify must fire after typing AND after a paste'
    )


def test_auto_verify_requires_all_six():
    i = UI.index('function _maybeAutoVerify()')
    body = UI[i:i + 700]
    assert '/^\\d{6}$/.test(otp)' in body


def test_auto_verify_cannot_double_fire():
    i = UI.index('function _maybeAutoVerify()')
    body = UI[i:i + 700]
    assert '_autoVerifying' in body
    assert 'finally' in body, 'a failed verify must release the latch'


def test_the_paste_handler_still_strips_non_digits():
    """The client half of the same agreement."""
    assert "getData('text') || '').replace(/\\D/g, '')" in UI


def test_step2_listeners_bind_once():
    """They are document-level; binding twice fires auto-advance twice per
    keystroke and skips a box."""
    assert '_step2Bound' in UI
    i = UI.index('function attachStep2Listeners()')
    body = UI[i:i + 300]
    assert 'if (_step2Bound) return;' in body


def test_the_boxes_accept_platform_autofill():
    assert 'autocomplete="one-time-code"' in UI
