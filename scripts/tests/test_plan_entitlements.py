"""Buying a plan must grant that plan's features — and only that plan's.

The chain a purchase travels:

    /api/create-order   attaches metadata {user_id, plan} to the checkout
    Whop webhook        reads metadata.plan back, writes users/{id}.plan
    the gates below     read that plan string and decide what is returned

Every link keys off the same three strings — 'basic', 'intermediate', 'pro' —
so a typo or a renamed tier anywhere in that chain silently downgrades a paying
customer. These tests pin the vocabulary and the grading.

The reference for what each tier is SOLD is pricing.html:
    Basic     — all 63 tokens, market bias/regime/volatility, S/R + pivot +
                price targets, indicators, volume/funding/OI, session,
                fear & greed, trader views
    Sentinel  — everything in Basic + AI conviction label + all 8 timeframes
                + raw probabilities in token analysis
    AEGIS Pro — everything in Sentinel + the full fire signal (entry, TP, SL)
                + Alpha Mode
"""
import pytest

import main


TIERS = ('basic', 'intermediate', 'pro')


# ── the vocabulary is shared end to end ──────────────────────────────────────

def test_the_three_tier_strings_are_the_same_everywhere():
    """Checkout, plan IDs, prices and display names must agree on the keys."""
    assert set(main.USD_PLAN_PRICES) == set(TIERS)
    assert set(main.PLAN_DISPLAY_NAMES) == set(TIERS)
    assert set(main.WHOP_PLAN_IDS) == set(TIERS)
    assert set(main.PADDLE_PRICE_IDS) == set(TIERS)


def test_display_names_match_what_the_site_sells():
    """/payment/config serves these to the browser; they are customer-facing."""
    from pathlib import Path
    pricing = (Path(main.__file__).parent / 'web' / 'src' / 'pages' /
               'pricing.html')
    if not pricing.exists():
        pytest.skip('pricing.html not present')
    html = pricing.read_text(encoding='utf-8', errors='ignore')
    for tier, label in main.PLAN_DISPLAY_NAMES.items():
        assert label in html, (
            f'{tier} is shown as {label!r}, which appears nowhere on the '
            f'pricing page — a subscriber would be told they are on a tier '
            f'that is not for sale'
        )


def test_prices_are_ordered_by_tier():
    p = main.USD_PLAN_PRICES
    assert p['basic'] < p['intermediate'] < p['pro'], p


# ── the webhook writes a plan the gates understand ───────────────────────────

@pytest.mark.parametrize('tier', TIERS)
def test_webhook_accepts_only_the_known_tiers(tier):
    """_handle_whop_webhook validates metadata.plan against this exact set."""
    import inspect
    src = inspect.getsource(main._handle_whop_webhook)
    assert '("basic", "intermediate", "pro")' in src or \
           "('basic', 'intermediate', 'pro')" in src, \
           'the webhook no longer validates the plan against the known tiers'
    assert tier in src


def test_webhook_falls_back_to_the_plan_id_map():
    """If metadata is missing a plan, the plan ID must resolve the tier —
    never a silent default, which would grant a tier nobody paid for."""
    import inspect
    src = inspect.getsource(main._handle_whop_webhook)
    assert 'WHOP_PLAN_IDS' in src
    assert 'unresolved plan' in src, 'an unresolvable plan must be ignored'


# ── the gates grade correctly ────────────────────────────────────────────────

def _payload(plan: str) -> dict:
    sig = {
        'symbol': 'BTC/USDT', 'price': 1.0, 'support': 0.9, 'resistance': 1.1,
        'pivot': 1.0, 'bull_tp1': 1.05, 'rsi': 55.0, 'adx': 20.0,
        'confluence': {'total': 5.0}, 'fear_greed': 50, 'session': 'LONDON',
        'meta_confidence': 0.8, 'threshold': 0.6,
        'fire': True, 'signal': 'BUY', 'direction': 'LONG',
        'suggested_tp': 1.2, 'suggested_sl': 0.8, 'signal_id': 'x',
        'p_buy': 0.7, 'p_sell': 0.2, 'p_hold': 0.1,
    }
    return main._build_insight_payload(sig, plan)


@pytest.mark.parametrize('tier', TIERS)
def test_every_paid_tier_gets_the_basic_market_picture(tier):
    """Basic is sold S/R, pivot, price targets, indicators, session, sentiment."""
    out = _payload(tier)
    for field in ('symbol', 'price', 'support', 'resistance', 'pivot',
                  'bull_tp1', 'rsi', 'adx', 'confluence', 'fear_greed', 'session'):
        assert field in out, f'{tier} lost {field}, which Basic is sold'


def test_basic_does_not_get_the_conviction_label():
    assert 'ai_conviction' not in _payload('basic')


@pytest.mark.parametrize('tier', ('intermediate', 'pro'))
def test_sentinel_and_above_get_the_conviction_label(tier):
    assert _payload(tier).get('ai_conviction') in ('HIGH', 'MEDIUM', 'LOW', 'NO_DATA')


@pytest.mark.parametrize('tier', ('basic', 'intermediate'))
def test_only_pro_gets_the_fire_signal(tier):
    """'Full AI fire signal — exact Entry, TP & Stop-Loss' is the Pro promise."""
    out = _payload(tier)
    for field in ('fire', 'signal', 'direction', 'suggested_tp', 'suggested_sl',
                  'meta_confidence', 'p_buy'):
        assert field not in out, f'{tier} received {field}, which is Pro-only'


def test_pro_gets_the_fire_signal():
    out = _payload('pro')
    for field in ('fire', 'signal', 'direction', 'suggested_tp', 'suggested_sl'):
        assert field in out, f'Pro is missing {field}, which it is sold'


def test_all_plans_get_the_full_token_fleet():
    """'All 63 live tokens' is a BASIC feature, not an upsell."""
    import inspect
    src = inspect.getsource(main.get_allowed_tokens)
    assert 'PRO_TOKENS' in src


def test_timeframes_are_a_sentinel_feature():
    """'All 8 chart timeframes' is sold from Sentinel up."""
    import inspect
    src = inspect.getsource(main.get_allowed_timeframes)
    assert 'intermediate' in src and 'pro' in src
    assert '"1m"' in src or "'1m'" in src


def test_alpha_mode_is_pro_only():
    import inspect
    src = inspect.getsource(main.get_user_limits)
    assert 'alpha_mode_enabled' in src
    assert '"pro"' in src or "'pro'" in src
    # intermediate must NOT appear in the alpha_mode line
    line = [l for l in src.splitlines() if 'alpha_mode_enabled' in l][0]
    assert 'intermediate' not in line, 'Alpha Mode leaked to Sentinel'


def test_raw_probabilities_are_hidden_from_basic():
    import inspect
    src = inspect.getsource(main.token_analysis)
    assert "'trial', 'basic'" in src or '"trial", "basic"' in src
    assert 'probabilities' in src
