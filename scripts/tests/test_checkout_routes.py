"""Guards the checkout endpoints against route shadowing.

FastAPI serves the FIRST route registered for a path+method. Two Razorpay-only
endpoints were declared ~2,800 lines above their provider-aware namesakes and
silently shadowed them:

  * /api/create-order   took {amount} in paise. Every real caller
    (gatekeeper.js, simple-auth-client.js) sends {plan, currency}, so live
    checkout answered 422 "Field required: amount" — nobody could subscribe,
    on any gateway.
  * /api/verify-payment checked a Razorpay signature and returned
    {"status": "ok"} WITHOUT upgrading the plan. The upgrade lived in the
    shadowed twin.

Nothing failed loudly: no import error, no startup warning, just a dead route.
These tests make a recurrence fail in CI.
"""
import collections

import pytest
from fastapi.testclient import TestClient

import main


PAYMENT_PATHS = ('/api/create-order', '/api/verify-payment')


def _routes_for(path: str):
    return [r for r in main.app.routes if getattr(r, 'path', None) == path]


@pytest.fixture(scope='module')
def client():
    main.app.dependency_overrides[main.get_current_user] = lambda: 'test-user'
    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()


# ── no shadowing on the payment paths ────────────────────────────────────────

@pytest.mark.parametrize('path', PAYMENT_PATHS)
def test_payment_path_has_exactly_one_route(path):
    names = [r.endpoint.__name__ for r in _routes_for(path)]
    assert len(names) == 1, (
        f'{path} has {len(names)} routes {names}; FastAPI serves only the '
        f'first, so the rest are dead code'
    )


def test_create_order_is_the_provider_aware_one():
    assert _routes_for('/api/create-order')[0].endpoint.__name__ == 'create_order'


def test_verify_payment_is_the_one_that_upgrades_the_plan():
    assert _routes_for('/api/verify-payment')[0].endpoint.__name__ == 'verify_payment'


def test_checkout_requires_authentication():
    """user_id becomes Paddle custom_data — it must come from the token."""
    import inspect
    sig = inspect.signature(_routes_for('/api/create-order')[0].endpoint)
    assert 'user_id' in sig.parameters


# ── the request shape the real frontend actually sends ───────────────────────

def test_checkout_accepts_the_shape_the_frontend_sends(client):
    """Regression for the 422. gatekeeper.js/simple-auth-client.js post this."""
    r = client.post('/api/create-order', json={'plan': 'pro', 'currency': 'USD'})
    assert r.status_code != 422, (
        f'checkout is rejecting the frontend payload again: {r.text[:200]}'
    )


def test_checkout_rejects_an_unknown_plan(client):
    r = client.post('/api/create-order', json={'plan': 'nope', 'currency': 'USD'})
    assert r.status_code == 400


def test_paddle_takes_precedence_and_reports_its_missing_price_id(client, monkeypatch):
    """With Paddle on, checkout must route there — and say what is missing."""
    monkeypatch.setattr(main, 'WHOP_ENABLED', False)
    monkeypatch.setattr(main, 'PADDLE_ENABLED', True)
    monkeypatch.setattr(main, 'PADDLE_PRICE_IDS',
                        {'basic': None, 'intermediate': None, 'pro': None})
    r = client.post('/api/create-order', json={'plan': 'pro', 'currency': 'USD'})
    assert r.status_code == 500
    assert 'PADDLE_PRICE_ID_PRO' in r.json().get('detail', '')


def test_whop_takes_precedence_and_reports_its_missing_plan_id(client, monkeypatch):
    """Whop outranks Paddle, and says which env var to set."""
    monkeypatch.setattr(main, 'WHOP_ENABLED', True)
    monkeypatch.setattr(main, 'PADDLE_ENABLED', True)      # still on underneath
    monkeypatch.setattr(main, 'WHOP_PLAN_IDS',
                        {'basic': None, 'intermediate': None, 'pro': None})
    r = client.post('/api/create-order', json={'plan': 'pro', 'currency': 'USD'})
    assert r.status_code == 500
    detail = r.json().get('detail', '')
    assert 'WHOP_PLAN_ID_PRO' in detail, detail
    assert 'PADDLE' not in detail, 'Paddle handled the request despite Whop being on'


@pytest.mark.parametrize('flags,expected', [
    ({'WHOP_ENABLED': True,  'PADDLE_ENABLED': True},  'whop'),
    ({'WHOP_ENABLED': False, 'PADDLE_ENABLED': True},  'paddle'),
    ({'WHOP_ENABLED': False, 'PADDLE_ENABLED': False,
      'DODO_PAYMENTS_ENABLED': False, 'RAZORPAY_ENABLED': False}, 'none'),
])
def test_provider_precedence(monkeypatch, flags, expected):
    """Whop -> Paddle -> DODO -> Razorpay, decided only by which creds exist.

    The rollback story depends on this: unsetting WHOP_API_KEY must fall
    straight back to Paddle with no code change.
    """
    for name, value in flags.items():
        monkeypatch.setattr(main, name, value)
    assert main._active_payment_provider() == expected


def test_payment_config_advertises_whop(client, monkeypatch):
    """gatekeeper.js keys its redirect flow off this payload."""
    monkeypatch.setattr(main, 'WHOP_ENABLED', True)
    cfg = client.get('/payment/config').json()
    assert cfg['provider'] == 'whop'
    assert cfg['whop']['enabled'] is True
    assert cfg['whop']['checkout'] == 'redirect'
    # the API key must never reach the browser
    assert 'api_key' not in str(cfg).lower()
    assert not (main.WHOP_API_KEY and main.WHOP_API_KEY in str(cfg))


# ── the duplicate request models are gone ────────────────────────────────────

def test_create_order_model_is_plan_based_not_amount_based():
    fields = set(main.CreateOrderRequest.model_fields)
    assert 'plan' in fields
    assert 'amount' not in fields, 'the paise-based CreateOrderRequest is back'


# ── visibility on the two shadowed routes NOT fixed here ─────────────────────

def test_no_route_is_shadowed():
    """No path may be registered twice.

    This used to permit one known pair — a second /api/signals and
    /api/public/signals — on the grounds that they were unreachable. They were,
    but only because FastAPI matches in registration order and the gated
    versions happened to come first. The shadowed /api/signals took
    HTTPBearer(auto_error=False) and never checked the plan, so it would have
    served the entire paid signal feed to anyone, authenticated or not, the
    moment anything reordered this file. Both are deleted; the allowance is
    gone with them.
    """
    seen = collections.defaultdict(list)
    for r in main.app.routes:
        p, m = getattr(r, 'path', None), getattr(r, 'methods', None)
        if p and m:
            for meth in m:
                if meth in ('GET', 'POST', 'PUT', 'DELETE'):
                    seen[(meth, p)].append(r.endpoint.__name__)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dups, (
        f'route shadowing reintroduced: {dups}. Whichever registers FIRST wins '
        f'— never rely on that for access control.'
    )


def test_the_surviving_signals_route_is_the_gated_one():
    """The paid feed must require a plan, not merely a token."""
    import inspect
    handlers = [r.endpoint for r in main.app.routes if r.path == '/api/signals']
    assert len(handlers) == 1
    src = inspect.getsource(handlers[0])
    assert "plan != 'pro'" in src or 'plan not in' in src, (
        '/api/signals no longer gates on the plan'
    )
