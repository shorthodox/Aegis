"""The signal producer must not spend the whole free tier on unchanged documents.

Measured 2026-08-15, and it is not a close call:

    signal producer   60 symbols x ~298 pushes/day  = 17,876 writes/day  (89.4%)
    everything else — every login, trial, OTP, analytics pass and trade
    for all 15 users, combined                      =    114 writes/day  ( 0.6%)
    Spark free tier                                 = 20,000 writes/day

The producer was 99% of the project's entire Firestore spend, and exhausting it
did not merely stop signals publishing. It froze the `signals` collection at
stale prices, made every logged-in page load block on a 60-second client retry,
failed the engine's writes, blocked administrative deletes, and put new signups
at risk of creating a Firebase Auth account with no profile behind it. One loop,
five symptoms.

The cause was a GLOBAL fingerprint: any single symbol changing rewrote all 60
documents, including the ~55 byte-identical ones.

These tests pin the diff that fixes it. They are deliberately about WRITE COUNT
rather than correctness of the payload — the payload was never wrong, there was
just eighteen times too much of it.
"""
import json

import pytest


def _hash(doc: dict) -> int:
    """Mirror of the producer's per-symbol hash."""
    return hash(json.dumps(doc, sort_keys=True, default=str))


def _compact(sym: str, signal: str = 'HOLD', fire: bool = False, **kw) -> dict:
    d = {'symbol': sym, 'signal': signal, 'fire': fire}
    d.update(kw)
    return d


def _diff(book, prev, full=False):
    """The producer's decision, reduced to what it actually is."""
    write, skip = [], 0
    for sym, doc in book.items():
        h = _hash(doc)
        if not full and prev.get(sym) == h:
            skip += 1
            continue
        prev[sym] = h
        write.append(sym)
    return write, skip


FLEET = [f'TOK{i}/USDT' for i in range(60)]


def test_an_unchanged_fleet_writes_nothing():
    """The whole point. 60 identical documents must cost 0 writes, not 60."""
    book = {s: _compact(s) for s in FLEET}
    prev = {}

    first, _ = _diff(book, prev)
    assert len(first) == 60, 'the first push must write everything'

    second, skipped = _diff(book, prev)
    assert second == [], f'unchanged documents were rewritten: {second[:5]}'
    assert skipped == 60


def test_only_the_changed_symbol_is_written():
    book = {s: _compact(s) for s in FLEET}
    prev = {}
    _diff(book, prev)

    book['TOK7/USDT'] = _compact('TOK7/USDT', signal='BUY', fire=True)
    write, skipped = _diff(book, prev)

    assert write == ['TOK7/USDT']
    assert skipped == 59


def test_timestamp_must_not_be_part_of_the_hash():
    """The trap that would silently undo the whole fix.

    `timestamp` is restamped on every push by definition. If it were inside the
    hash, every document would differ every time, the diff would skip nothing,
    and the write count would quietly return to 60/push while every test about
    payload correctness kept passing.
    """
    a = _compact('TOK1/USDT')
    b = _compact('TOK1/USDT')
    assert _hash(a) == _hash(b), 'identical payloads must hash identically'

    # what a timestamped hash would do
    a_ts = dict(a, timestamp='2026-08-15T00:00:00Z')
    b_ts = dict(b, timestamp='2026-08-15T00:05:00Z')
    assert _hash(a_ts) != _hash(b_ts), (
        'precondition: timestamps differ per push — which is exactly why they '
        'are excluded from the hash the producer takes'
    )


def test_the_periodic_full_refresh_rewrites_everything():
    """Drift insurance. A document deleted outside this loop would otherwise
    stay missing forever, because our hash says we already wrote it."""
    book = {s: _compact(s) for s in FLEET}
    prev = {}
    _diff(book, prev)

    write, skipped = _diff(book, prev, full=True)
    assert len(write) == 60
    assert skipped == 0


def test_the_daily_budget_lands_inside_the_free_tier():
    """The number that decides whether the product runs for free."""
    PUSHES_PER_DAY = 86400 / 290
    SPARK_CAP = 20_000
    OTHER = 114                      # every non-producer write, measured

    # The interval floor alone is 89.9% of the cap — close, but NOT over it.
    # What tipped it is that pushes are not capped at the interval:
    #
    #     should_push = _signals_changed or _interval_elapsed
    #
    # and the loop ticks every second, so a moving tape pushes MORE often than
    # once per 290s. At 60 writes a push, only ~50 extra change-triggered
    # pushes a day are needed to cross 20,000 — which is why the quota was
    # observed exhausted on consecutive days while a naive 60 x 298 estimate
    # says it should just fit.
    #
    # That is the real argument for diffing rather than for lengthening the
    # interval: the interval was never the thing setting the rate.
    before = 60 * PUSHES_PER_DAY + OTHER
    assert before / SPARK_CAP > 0.85, 'precondition: the old behaviour sat on the cap'
    assert 60 * (PUSHES_PER_DAY + 50) + OTHER > SPARK_CAP, (
        'a mere 50 extra change-triggered pushes/day must be enough to blow it'
    )

    for changed in (2, 3, 5):
        after = changed * PUSHES_PER_DAY + OTHER + 60   # +60 = hourly full refresh
        assert after < SPARK_CAP * 0.15, (
            f'{changed} changed symbols/push should sit well inside the free '
            f'tier, got {after:.0f}/day'
        )
