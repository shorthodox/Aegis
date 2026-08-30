"""Refunds: issued through Whop, bounded by our own window, access ends at once.

Asked 2026-08-27: "make sure we do have an automated system to make the refund
to the users that is only available for 3 days".

There was none. What existed was an <a href> to customer.dodopayments.com — a
processor no longer in use — beside a localStorage-driven label that greyed the
link out. No refund endpoint existed anywhere in main.py. A user inside the
window clicked through to a dead portal and nobody was refunded by anything.

WHAT THE WINDOW IS AND WHO ENFORCES IT

Whop documents NO maximum age of its own: it will refund a payment of any date.
So the 3-day window is entirely ours to enforce, and if we do not check it,
nothing does. REFUND_WINDOW_DAYS is the single place it is expressed.

A REFUND IS NOT A CANCELLATION

The existing revoke path deliberately keeps access until subscription_end,
because a cancellation means "will not renew" and the customer keeps the period
they paid for. A refund is the opposite: the money went back, so access ends
NOW. Reusing the cancellation branch would hand a refunded customer the rest of
the month for free.

DRIVEN BY refund.created, NOT membership.deactivated

Whop's docs do not confirm that a refund is among membership.deactivated's
triggers, so relying on it alone would leave refunded customers with full
access. refund.created is the primary signal; membership.deactivated stays
wired as an independent second one.

ORDER OF OPERATIONS: check our window, call Whop, and revoke only once Whop
confirms. Revoking first would cut a customer off for a refund that may never
have been issued.
"""
import inspect

import pytest

import main


SRC = inspect.getsource(main)


# -- the window is ours, and it is three days ---------------------------------

def test_the_window_is_three_days():
    assert main.REFUND_WINDOW_DAYS == 3.0


def test_the_window_is_overridable_by_configuration():
    assert 'AEGIS_REFUND_WINDOW_DAYS' in SRC


def test_a_fresh_purchase_is_inside_the_window():
    from datetime import datetime, timedelta, timezone
    ok, why = main._refund_window_open(
        (datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
    assert ok is True
    assert 'left in the window' in why


def test_an_old_purchase_is_outside_it():
    from datetime import datetime, timedelta, timezone
    ok, why = main._refund_window_open(
        (datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
    assert ok is False
    assert 'refund window is 3 days' in why


def test_the_boundary_is_three_days_not_seven():
    from datetime import datetime, timedelta, timezone
    inside, _ = main._refund_window_open(
        (datetime.now(timezone.utc) - timedelta(days=2, hours=23)).isoformat())
    outside, _ = main._refund_window_open(
        (datetime.now(timezone.utc) - timedelta(days=3, hours=1)).isoformat())
    assert inside is True
    assert outside is False


def test_an_unknown_purchase_date_is_not_auto_approved():
    """Unknown is a reason to involve a human, not to say yes."""
    ok, why = main._refund_window_open(None)
    assert ok is False
    assert 'could not be established' in why


# -- the API client -----------------------------------------------------------

def test_it_calls_the_documented_endpoint():
    src = inspect.getsource(main._whop_refund_payment)
    assert '/payments/{payment_id}/refund' in src
    assert 'Bearer {WHOP_API_KEY}' in src


def test_it_rejects_a_non_payment_identifier():
    """pay_xxx, not a membership or receipt id — those 404 misleadingly."""
    src = inspect.getsource(main._whop_refund_payment)
    assert "startswith(\"pay_\")" in src


def test_a_bad_identifier_never_reaches_whop():
    """Driven with asyncio.run rather than pytest.mark.asyncio — pytest-asyncio
    is not installed here and a refund test is not worth a new dependency."""
    import asyncio
    out = asyncio.run(main._whop_refund_payment('mem_123'))
    assert out['ok'] is False
    assert out['status'] == 'bad_identifier'


def test_an_unset_api_key_says_what_to_do_instead(monkeypatch):
    import asyncio
    monkeypatch.setattr(main, 'WHOP_API_KEY', '')
    out = asyncio.run(main._whop_refund_payment('pay_abc123'))
    assert out['ok'] is False
    assert out['status'] == 'unconfigured'
    assert 'dashboard' in out['detail'].lower()


def test_partial_refunds_are_supported():
    src = inspect.getsource(main._whop_refund_payment)
    assert 'partial_amount' in src
    sig = inspect.signature(main._whop_refund_payment)
    assert 'partial_amount' in sig.parameters
    assert sig.parameters['partial_amount'].default is None


def test_every_documented_error_maps_to_an_action():
    src = inspect.getsource(main._whop_refund_payment)
    for code in (400, 401, 403, 404, 422, 429):
        assert f'{code}:' in src, code
    assert 'payment:manage' in src, 'a 403 must name the scope that is missing'


def test_a_disputed_payment_is_called_final():
    """Whop will not create a refund record on a lost dispute. Retrying or
    telling an operator to do it manually is wrong advice."""
    src = inspect.getsource(main._whop_refund_payment)
    assert 'disputed' in src
    assert 'already left' in src


def test_failures_degrade_to_a_human_not_an_exception():
    """No raise STATEMENT in the body — the docstring says "never raises", and a
    naive substring scan matches its own explanation."""
    import ast
    tree = ast.parse(inspect.getsource(main._whop_refund_payment).lstrip())
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert not raises, 'the refund client can raise into a request handler'
    assert 'Whop dashboard' in inspect.getsource(main._whop_refund_payment)


# -- a refund ends access immediately -----------------------------------------

def test_refund_events_are_handled_separately_from_cancellations():
    assert main._WHOP_REFUND_EVENTS == {'refund.created', 'refund.updated'}
    assert not (main._WHOP_REFUND_EVENTS & main._WHOP_REVOKE_EVENTS)


def test_a_refund_ends_access_now_not_at_period_end():
    i = SRC.index('elif event in _WHOP_REFUND_EVENTS:')
    branch = SRC[i:SRC.index('elif event in _WHOP_REVOKE_EVENTS:', i)]
    assert '"subscription_end": _now' in branch, (
        'a refunded customer keeps access to period end, which is the '
        'cancellation rule, not the refund rule'
    )
    assert '"plan": "trial"' in branch


def test_a_cancellation_still_runs_to_period_end():
    """The distinction must survive: cancelling is not being refunded."""
    i = SRC.index('elif event in _WHOP_REVOKE_EVENTS:')
    branch = SRC[i:i + 1200]
    assert 'Do NOT downgrade `plan` here' in branch


def test_a_pending_refund_does_not_revoke():
    i = SRC.index('elif event in _WHOP_REFUND_EVENTS:')
    branch = SRC[i:SRC.index('elif event in _WHOP_REVOKE_EVENTS:', i)]
    assert "_status not in (\"succeeded\", \"\")" in branch


def test_the_volume_is_updated_before_firestore():
    i = SRC.index('elif event in _WHOP_REFUND_EVENTS:')
    branch = SRC[i:SRC.index('elif event in _WHOP_REVOKE_EVENTS:', i)]
    assert branch.index('_ent_save') < branch.index('user_ref.update')


# -- the admin path -----------------------------------------------------------

def test_the_window_is_checked_before_whop_is_called():
    src = inspect.getsource(main.admin_refund)
    assert src.index('_refund_window_open') < src.index('_whop_refund_payment')


def test_access_is_revoked_only_after_whop_confirms():
    src = inspect.getsource(main.admin_refund)
    assert src.index('_whop_refund_payment') < src.index('_ent_revoke')


def test_an_out_of_window_refund_can_be_overridden_deliberately():
    src = inspect.getsource(main.admin_refund)
    assert 'override_window' in src


def test_a_failed_bookkeeping_write_is_not_reported_as_a_failed_refund():
    """The money is already back. An operator who retries would double-refund."""
    src = inspect.getsource(main.admin_refund)
    assert 'AFTER a successful' in src


def test_both_endpoints_are_admin_only():
    for fn in (main.admin_refund, main.admin_refund_eligibility):
        assert '_require_admin' in inspect.getsource(fn)


def test_the_eligibility_endpoint_does_not_block_the_event_loop():
    """It calls Firestore synchronously and awaits nothing."""
    assert not inspect.iscoroutinefunction(main.admin_refund_eligibility)


# ── Whop must charge what the site sells ─────────────────────────────────────
# Found 2026-08-30, days before a planned ad spend. Whop was configured as:
#
#   Basic     plan_type one_time, initial 0.00, renewal 0.00   (site: $6/mo)
#   Sentinel  plan_type renewal,  initial 0.00, renewal 12.00  (site: $12/mo)
#   Pro       plan_type renewal,  initial 0.00, renewal 18.00  (site: $18/mo)
#
# Basic would have given the product away permanently, with no renewal to bill
# against. The other two had a free FIRST PERIOD of 30 days against an
# advertised 3-day trial. None of that is visible from this repository — it
# lives in a dashboard — so the only way to catch it is to ask Whop and compare.

def test_the_trial_length_has_one_definition():
    """It was a bare timedelta(days=3) that no other check could see."""
    assert main.TRIAL_DAYS == 3
    assert 'timedelta(days=TRIAL_DAYS)' in SRC
    assert 'timedelta(days=3)' not in SRC


def test_the_audit_catches_a_one_time_plan():
    src = inspect.getsource(main._whop_plan_audit)
    assert "ptype != \"renewal\"" in src
    assert 'never' in src and 'bill again' in src


def test_the_audit_catches_a_price_mismatch():
    src = inspect.getsource(main._whop_plan_audit)
    assert 'renewal - float(advertised)' in src
    assert 'the site sells' in src


def test_the_audit_catches_a_free_first_period():
    """initial_price 0 is a trial by another name, and a 30-day one."""
    src = inspect.getsource(main._whop_plan_audit)
    assert 'initial <= 0.0' in src
    assert 'TRIAL_DAYS' in src


def test_it_compares_against_the_advertised_prices():
    assert main.USD_PLAN_PRICES == {'basic': 6.00, 'intermediate': 12.00, 'pro': 18.00}
    src = inspect.getsource(main._whop_plan_audit)
    assert 'USD_PLAN_PRICES.items()' in src


def test_a_missing_plan_id_is_a_finding_not_a_crash():
    src = inspect.getsource(main._whop_plan_audit)
    assert 'no Whop plan id configured' in src


def test_the_audit_never_raises():
    import ast
    tree = ast.parse(inspect.getsource(main._whop_plan_audit).lstrip())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]


def test_an_unset_key_reports_rather_than_pretending_to_pass():
    src = inspect.getsource(main._whop_plan_audit)
    i = src.index('if not WHOP_API_KEY')
    assert 'out["ok"] = False' in src[i:i + 300], (
        'a missing key must not read as a clean audit'
    )


def test_it_runs_on_every_boot():
    assert '_plan_audit_on_boot()' in SRC
    src = inspect.getsource(main._plan_audit_on_boot)
    assert 'await asyncio.sleep' in src, 'the audit must not delay boot'
    assert 'DOES NOT MATCH' in src


def test_there_is_an_endpoint_to_re_check_after_fixing_whop():
    assert hasattr(main, 'admin_plan_audit')
    assert '_require_admin' in inspect.getsource(main.admin_plan_audit)
