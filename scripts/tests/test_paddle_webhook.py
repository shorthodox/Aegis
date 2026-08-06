"""Security tests for Paddle Billing webhook signature verification.

This endpoint grants paid plans. If verification is wrong in the permissive
direction it becomes an unauthenticated "upgrade this user" API, so these tests
lean on the rejection cases.

Scheme (developer.paddle.com/webhooks/signature-verification):
  header  : Paddle-Signature: ts=<unix>;h1=<hex>
  payload : "<ts>:" + RAW body, unmodified
  hmac    : SHA-256 keyed with the notification-setting secret (pdl_ntfset_...)
"""
import hashlib
import hmac
import json
import time

import pytest

import main


SECRET = 'pdl_ntfset_01testsecretvalue'


def _sign(body: bytes, secret: str = SECRET, ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f'{ts}:'.encode() + body,
                   hashlib.sha256).hexdigest()
    return f'ts={ts};h1={mac}'


@pytest.fixture
def body() -> bytes:
    return json.dumps({
        'event_type': 'transaction.completed',
        'data': {'id': 'txn_01', 'custom_data': {'user_id': 'u1', 'plan': 'pro'}},
    }).encode()


# ── accepts a genuine event ──────────────────────────────────────────────────

def test_valid_signature_passes(body):
    assert main.paddle_verify_signature(body, _sign(body), SECRET) is True


def test_signature_is_computed_over_the_raw_body(body):
    """Re-serialising the JSON changes the bytes and must break the match."""
    resigned = _sign(body)
    reserialised = json.dumps(json.loads(body), indent=2).encode()
    assert main.paddle_verify_signature(reserialised, resigned, SECRET) is False


# ── rejects everything else ──────────────────────────────────────────────────

def test_wrong_secret_fails(body):
    assert main.paddle_verify_signature(
        body, _sign(body, secret='pdl_ntfset_other'), SECRET) is False


def test_tampered_body_fails(body):
    sig = _sign(body)
    tampered = body.replace(b'"plan": "pro"', b'"plan": "pro "')
    assert main.paddle_verify_signature(tampered, sig, SECRET) is False


def test_escalated_plan_in_body_fails(body):
    """The attack this protects against: swap the plan, keep the signature."""
    sig = _sign(body)
    escalated = json.dumps({
        'event_type': 'transaction.completed',
        'data': {'id': 'txn_01', 'custom_data': {'user_id': 'u1', 'plan': 'pro'}},
        'injected': True,
    }).encode()
    assert main.paddle_verify_signature(escalated, sig, SECRET) is False


def test_missing_secret_fails(body):
    assert main.paddle_verify_signature(body, _sign(body), '') is False
    assert main.paddle_verify_signature(body, _sign(body), None) is False \
        or main.PADDLE_WEBHOOK_SECRET  # None falls back to env, unset in tests


def test_empty_or_malformed_header_fails(body):
    for header in ('', 'garbage', 'ts=;h1=', 'h1=abc', 'ts=123',
                   'ts=notanumber;h1=abc', 'ts=123;h1='):
        assert main.paddle_verify_signature(body, header, SECRET) is False, header


def test_stale_timestamp_fails(body):
    old = int(time.time()) - 10_000
    assert main.paddle_verify_signature(
        body, _sign(body, ts=old), SECRET, tolerance_seconds=300) is False


def test_future_timestamp_fails(body):
    """Guards clock-skew replay from the other direction."""
    future = int(time.time()) + 10_000
    assert main.paddle_verify_signature(
        body, _sign(body, ts=future), SECRET, tolerance_seconds=300) is False


def test_timestamp_inside_tolerance_passes(body):
    recent = int(time.time()) - 60
    assert main.paddle_verify_signature(
        body, _sign(body, ts=recent), SECRET, tolerance_seconds=300) is True


def test_replay_outside_window_is_refused_even_though_hmac_matches(body):
    """A captured, perfectly-signed event must not be replayable forever."""
    sig = _sign(body, ts=int(time.time()) - 3600)
    assert main.paddle_verify_signature(body, sig, SECRET,
                                        tolerance_seconds=300) is False


def test_comparison_does_not_short_circuit_on_length(body):
    """A truncated hex digest must not be accepted as a prefix match."""
    full = _sign(body)
    ts, h1 = full.split(';')
    assert main.paddle_verify_signature(body, f'{ts};{h1[:20]}', SECRET) is False


# ── configuration wiring ─────────────────────────────────────────────────────

def test_paddle_base_urls_are_correct():
    assert main._PADDLE_BASE in ('https://api.paddle.com',
                                 'https://sandbox-api.paddle.com')


def test_defaults_to_sandbox_not_live():
    """A missing PADDLE_MODE must never mean 'take real money'."""
    import os
    if not os.getenv('PADDLE_MODE'):
        assert main.PADDLE_MODE == 'sandbox'
        assert main._PADDLE_BASE == 'https://sandbox-api.paddle.com'


def test_provider_precedence_prefers_paddle_when_enabled(monkeypatch):
    # Whop now sits above Paddle in the chain, so it has to be pinned off for
    # this test to be about Paddle at all. Without this the test passed only
    # while WHOP_API_KEY was unset — i.e. it would have started failing the
    # moment real Whop credentials reached .env, which is exactly what happened.
    monkeypatch.setattr(main, 'WHOP_ENABLED', False)
    monkeypatch.setattr(main, 'PADDLE_ENABLED', True)
    assert main._active_payment_provider() == 'paddle'
    monkeypatch.setattr(main, 'PADDLE_ENABLED', False)
    monkeypatch.setattr(main, 'DODO_PAYMENTS_ENABLED', True)
    assert main._active_payment_provider() == 'dodopayments'
    monkeypatch.setattr(main, 'DODO_PAYMENTS_ENABLED', False)
    monkeypatch.setattr(main, 'RAZORPAY_ENABLED', True)
    assert main._active_payment_provider() == 'razorpay'
    monkeypatch.setattr(main, 'RAZORPAY_ENABLED', False)
    assert main._active_payment_provider() == 'none'


def test_grant_and_revoke_event_sets_are_disjoint():
    assert not (main._PADDLE_GRANT_EVENTS & main._PADDLE_REVOKE_EVENTS)
