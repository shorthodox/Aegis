"""The signal publisher must fit inside the Firestore free-tier write cap.

Reported 2026-08-23: "no signals since last night", repeatedly, across days.

The engine was never the problem. It scanned every 61s, armed orders and
refused trades for good reasons the whole time. What died was the PUBLISH:

    [PRODUCER] batch push failed on QUOTA (RetryError) - 32 doc(s) dropped
    [PRODUCER] Firestore: 0/32 written, 16 unchanged skipped (0 fired)

aegis-d78e1 is on the Firestore FREE tier: 20,000 document writes per day.

Two things combined to blow through that.

1 - The 290s interval was OR'd with "did anything change", so it was a CEILING
    that forced an extra push, never a floor that held one back.

2 - The change test is a GLOBAL fingerprint over all ~60 tokens keyed on
    (signal, fire, pending_entry). One token relabelling, or one order arming,
    marked the whole fleet as changed. With 60 tokens rescanned every cycle at
    least one of those flips essentially always.

    1,416 cycles/day x ~32 changed docs = ~45,000 writes/day vs a 20,000 cap.

It ran out roughly ten hours into each day and everything downstream went
silent. Note the interaction with order churn: pending_entry was flipping
constantly because resting orders were being superseded every scan, so that bug
was also feeding this one.

The floor is now a floor, with fires exempt - a genuine fire is the one event
where latency is the product.
"""
import inspect
import re

import main


SRC = inspect.getsource(main)


def _producer_src():
    i = SRC.index('_FIRESTORE_MIN_INTERVAL')
    j = SRC.index('unchanged skipped', i)
    return SRC[i:j]


def test_the_interval_is_a_floor_not_a_ceiling():
    body = _producer_src()
    # The switch in front of it came later (signals moved off Firestore
    # entirely); what this guards is the OR'ing of the interval with "did
    # anything change", which made it a ceiling rather than a floor.
    assert '_fires_changed or _interval_elapsed' in body, (
        'routine churn still pushes the moment anything changes, so the '
        'interval only ever forces EXTRA writes'
    )
    assert 'should_push = _signals_changed or _interval_elapsed' not in body


def test_a_fire_still_publishes_immediately():
    """Latency on a real signal is the product. Fires bypass the floor."""
    body = _producer_src()
    assert '_fires_changed' in body
    assert "v.get('fire', False)" in body


def test_the_fire_set_is_tracked_across_pushes():
    assert '_last_fire_set' in SRC, 'nothing remembers which symbols were firing'
    assert 'nonlocal' in SRC and '_last_fire_set' in SRC


def test_the_daily_budget_actually_fits():
    """The arithmetic that matters, stated so a future change has to face it."""
    FREE_TIER_WRITES_PER_DAY = 20_000
    interval = main.__dict__.get('_FIRESTORE_MIN_INTERVAL')
    if interval is None:
        m = re.search(r'_FIRESTORE_MIN_INTERVAL\s*=\s*([\d.]+)', SRC)
        interval = float(m.group(1))
    pushes_per_day = 86_400 / interval
    docs_per_push = 60           # worst case: the whole fleet changed
    routine = pushes_per_day * docs_per_push
    hourly_refresh = 24 * 60     # _FIRESTORE_FULL_REFRESH_S heals drift
    # Real headroom, not a pass by 3%. The cap has to absorb logins,
    # entitlement writes and track-record writes on top of this.
    assert routine + hourly_refresh < 0.70 * FREE_TIER_WRITES_PER_DAY, (
        f'{routine + hourly_refresh:,.0f} writes/day is {(routine + hourly_refresh) / FREE_TIER_WRITES_PER_DAY:.0%} '
        f'of the {FREE_TIER_WRITES_PER_DAY:,} cap - too little room for user '
        f'traffic; the publish will die partway through the day and the site '
        f'goes silent while the engine keeps working'
    )


def test_the_old_every_cycle_rate_would_have_blown_the_cap():
    """Guards the diagnosis itself: the pre-fix rate really was over budget."""
    old = (86_400 / 61) * 32
    assert old > 20_000, 'the stated root cause does not reproduce'


def test_quota_failures_skip_the_per_doc_fallback():
    """Retrying doc-by-doc against an exhausted quota costs 60s per document
    and cannot succeed."""
    assert 'skipping per-doc fallback' in SRC
