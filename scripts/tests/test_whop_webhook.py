"""Security tests for Whop webhook signature verification.

This endpoint grants paid plans. If verification is wrong in the permissive
direction it becomes an unauthenticated "upgrade this user" API, so these tests
lean on the rejection cases.

Scheme (Standard Webhooks, docs.whop.com/developer/guides/webhooks):
  headers : webhook-id, webhook-timestamp, webhook-signature
  payload : "<id>.<timestamp>." + RAW body, unmodified
  hmac    : SHA-256 keyed with the BASE64-DECODED secret
  encoding: signature is base64, sent as "v1,<sig>"; several space-separated
            versions may appear during a secret rotation

Every one of those differs from its neighbours in main.py — Paddle signs
"<ts>:<body>" and sends hex in a single header; DODO compares a plain hex HMAC
of the body alone. The decode-the-secret step is the easy one to miss, and
missing it fails closed (nothing verifies), which is the safe direction.
"""
import base64
import hashlib
import hmac
import json
import time

import pytest

import main


# a realistic base64 secret, as Whop issues them
SECRET = base64.b64encode(b'whop-test-secret-value-0123456789').decode()


def _sign(body: bytes, secret: str = SECRET, wid: str = 'msg_01',
          ts: int | None = None) -> tuple[str, str, str]:
    ts = int(time.time()) if ts is None else ts
    raw = secret.split('_', 1)[1] if secret.startswith('whsec_') else secret
    mac = hmac.new(base64.b64decode(raw),
                   f'{wid}.{ts}.'.encode() + body, hashlib.sha256).digest()
    return wid, str(ts), f'v1,{base64.b64encode(mac).decode()}'


@pytest.fixture
def body() -> bytes:
    return json.dumps({
        'type': 'payment.succeeded',
        'data': {
            'id': 'pay_01',
            'metadata': {'user_id': 'u1', 'plan': 'pro'},
            'membership': {'id': 'mem_01', 'plan': {'id': 'plan_pro'}},
        },
    }).encode()


# ── accepts a genuine event ──────────────────────────────────────────────────

def test_valid_signature_passes(body):
    wid, ts, sig = _sign(body)
    assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is True


def test_whsec_prefixed_secret_is_accepted(body):
    prefixed = 'whsec_' + SECRET
    wid, ts, sig = _sign(body, prefixed)
    assert main.whop_verify_signature(body, wid, ts, sig, prefixed) is True


def test_multiple_versioned_signatures_any_match_passes(body):
    """During a rotation Whop may send several space-separated signatures."""
    wid, ts, good = _sign(body)
    decoy = 'v1,' + base64.b64encode(b'x' * 32).decode()
    assert main.whop_verify_signature(body, wid, ts, f'{decoy} {good}', SECRET) is True


# ── rejects everything else ──────────────────────────────────────────────────

def test_tampered_body_fails(body):
    wid, ts, sig = _sign(body)
    assert main.whop_verify_signature(body + b' ', wid, ts, sig, SECRET) is False


def test_wrong_secret_fails(body):
    wid, ts, sig = _sign(body)
    other = base64.b64encode(b'a-completely-different-secret-val').decode()
    assert main.whop_verify_signature(body, wid, ts, sig, other) is False


def test_wrong_webhook_id_fails(body):
    """The id is part of the signed payload — swapping it must not verify."""
    wid, ts, sig = _sign(body)
    assert main.whop_verify_signature(body, 'msg_other', ts, sig, SECRET) is False


def test_stale_timestamp_fails(body):
    wid, ts, sig = _sign(body, ts=int(time.time()) - 10_000)
    assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is False


def test_future_timestamp_fails(body):
    """Clock skew must be rejected in both directions, not just the past."""
    wid, ts, sig = _sign(body, ts=int(time.time()) + 10_000)
    assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is False


@pytest.mark.parametrize('wid,ts,sig', [
    ('', '123', 'v1,abc'),          # no id
    ('msg_01', '', 'v1,abc'),       # no timestamp
    ('msg_01', '123', ''),          # no signature
    ('msg_01', 'not-a-number', 'v1,abc'),
])
def test_malformed_headers_fail(body, wid, ts, sig):
    assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is False


def test_missing_secret_fails_closed(body):
    wid, ts, sig = _sign(body)
    assert main.whop_verify_signature(body, wid, ts, sig, '') is False
    assert main.whop_verify_signature(body, wid, ts, sig, None) is False \
        or main.WHOP_WEBHOOK_SECRET  # None falls back to the env secret


def test_undecodable_secret_fails_closed(body):
    wid, ts, sig = _sign(body)
    assert main.whop_verify_signature(body, wid, ts, sig, '!!!not-base64!!!') is False


def test_signature_without_version_prefix_fails(body):
    """A bare base64 blob is not the documented format."""
    wid, ts, sig = _sign(body)
    bare = sig.split(',', 1)[1]
    assert main.whop_verify_signature(body, wid, ts, bare, SECRET) is False


# ── the secret format Whop actually issues ───────────────────────────────────
# The docs describe Standard Webhooks: `whsec_` + base64, base64-decoded before
# use. The dashboard issues `ws_` + 64 hex characters — a 256-bit key in hex.
# base64-decoding that raises, so the first version of the verifier rejected
# every genuine webhook. These pin both encodings.

WS_SECRET = 'ws_' + '0c' * 32          # ws_ + 64 hex chars, same shape as a real one


def _sign_with_key(body: bytes, key: bytes, wid='msg_01', ts=None, encoding='b64'):
    ts = int(time.time()) if ts is None else ts
    digest = hmac.new(key, f'{wid}.{ts}.'.encode() + body, hashlib.sha256).digest()
    sig = base64.b64encode(digest).decode() if encoding == 'b64' else digest.hex()
    return wid, str(ts), f'v1,{sig}'


def test_ws_prefixed_hex_secret_verifies(body):
    """The format the dashboard actually hands you."""
    key = bytes.fromhex(WS_SECRET.split('_', 1)[1])
    wid, ts, sig = _sign_with_key(body, key)
    assert main.whop_verify_signature(body, wid, ts, sig, WS_SECRET) is True


def test_ws_secret_also_verifies_a_hex_signature(body):
    """Signature encoding is checked both ways for the same reason."""
    key = bytes.fromhex(WS_SECRET.split('_', 1)[1])
    wid, ts, sig = _sign_with_key(body, key, encoding='hex')
    assert main.whop_verify_signature(body, wid, ts, sig, WS_SECRET) is True


def test_ws_secret_used_as_raw_ascii_also_verifies(body):
    """A naive implementation would key on the ASCII string; accept that too."""
    key = WS_SECRET.split('_', 1)[1].encode()
    wid, ts, sig = _sign_with_key(body, key)
    assert main.whop_verify_signature(body, wid, ts, sig, WS_SECRET) is True


def test_ws_secret_still_rejects_a_wrong_key(body):
    """Trying several encodings must not become 'accepts anything'."""
    wid, ts, sig = _sign_with_key(body, b'a-totally-different-32-byte-key!')
    assert main.whop_verify_signature(body, wid, ts, sig, WS_SECRET) is False


def test_ws_secret_still_rejects_a_tampered_body(body):
    key = bytes.fromhex(WS_SECRET.split('_', 1)[1])
    wid, ts, sig = _sign_with_key(body, key)
    assert main.whop_verify_signature(body + b'!', wid, ts, sig, WS_SECRET) is False


def test_ws_secret_still_rejects_a_stale_timestamp(body):
    key = bytes.fromhex(WS_SECRET.split('_', 1)[1])
    wid, ts, sig = _sign_with_key(body, key, ts=int(time.time()) - 10_000)
    assert main.whop_verify_signature(body, wid, ts, sig, WS_SECRET) is False


def test_key_derivation_covers_both_documented_and_actual_formats():
    hex_body = '0c' * 32
    keys = main._whop_secret_keys('ws_' + hex_body)
    assert bytes.fromhex(hex_body) in keys, 'hex secret not decoded'
    assert hex_body.encode() in keys, 'raw ASCII fallback missing'
    b64 = base64.b64encode(b'x' * 32).decode()
    assert b'x' * 32 in main._whop_secret_keys('whsec_' + b64), 'base64 secret not decoded'
    assert main._whop_secret_keys('') == []


# ── the scheme is not a copy of its neighbours ───────────────────────────────

def test_whop_does_not_accept_a_paddle_style_signature(body):
    """Guards against someone 'unifying' the two verifiers."""
    ts = str(int(time.time()))
    paddle_style = hmac.new(SECRET.encode(), f'{ts}:'.encode() + body,
                            hashlib.sha256).hexdigest()
    assert main.whop_verify_signature(body, 'msg_01', ts,
                                      f'v1,{paddle_style}', SECRET) is False


def test_documented_base64_decoding_is_supported(body):
    """The documented derivation must keep working.

    An earlier version of this test asserted the OPPOSITE — that signing with
    the undecoded secret must fail — to catch a forgotten b64decode. That
    assumption did not survive contact with a real secret: Whop issues
    `ws_` + hex, which base64-decoding cannot even parse, so the verifier now
    derives several candidate keys from the secret's own format. Accepting the
    raw-ASCII derivation is a deliberate consequence of that, not an oversight.

    What still must hold is below: a wrong secret never verifies.
    """
    wid, ts = 'msg_01', str(int(time.time()))
    key = base64.b64decode(SECRET)
    digest = hmac.new(key, f'{wid}.{ts}.'.encode() + body, hashlib.sha256).digest()
    sig = 'v1,' + base64.b64encode(digest).decode()
    assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is True


def test_trying_several_encodings_never_accepts_a_wrong_secret(body):
    """The property that actually protects the endpoint.

    The verifier tries hex, base64 and raw-ASCII derivations of ONE secret.
    That must not degrade into accepting a signature made with a DIFFERENT
    secret under any of them.
    """
    wid, ts = 'msg_01', str(int(time.time()))
    wrong = base64.b64encode(b'the-wrong-secret-abcdefghijklmno').decode()
    for key in main._whop_secret_keys(wrong):
        digest = hmac.new(key, f'{wid}.{ts}.'.encode() + body, hashlib.sha256).digest()
        for sig in ('v1,' + base64.b64encode(digest).decode(), 'v1,' + digest.hex()):
            assert main.whop_verify_signature(body, wid, ts, sig, SECRET) is False


# ── wiring ───────────────────────────────────────────────────────────────────

def test_webhook_route_rejects_unverified_whop_events():
    """The route must verify-or-reject, like Paddle — never warn-and-continue."""
    import inspect
    src = inspect.getsource(main.payments_webhook)
    assert 'whop_verify_signature' in src
    assert 'status_code=401' in src, 'an unverified Whop event must be refused'
    assert 'status_code=503' in src, 'a missing secret must refuse, not accept'


def test_grant_and_revoke_events_are_declared():
    assert 'payment.succeeded' in main._WHOP_GRANT_EVENTS
    assert 'membership.activated' in main._WHOP_GRANT_EVENTS
    assert main._WHOP_GRANT_EVENTS.isdisjoint(main._WHOP_REVOKE_EVENTS)
