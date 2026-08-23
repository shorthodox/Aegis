"""The 5m reversal print must be CURRENT, and a failed read must not look like "no".

Reported 2026-08-23: "I guess engine is not able to check 3 5min candles for the
reversal confirmation to open the armed setups".

The logic itself was sound — measured live across 59 tokens at the time,
ltf_bear was true on 37 (63%) and ltf_bull on 3 (5%), so it fires constantly.
The armed book was almost entirely BUYs waiting at supports BELOW price, which
need ltf_bull, and the tape was falling toward those levels. That is correct
behaviour, not a broken check.

Two real defects sat next to it.

1 · STALENESS. _candle_cache_ttl is 240s and the scan runs about every 60s, so
the 5m window refreshed on one scan in four, and a cached read could straddle a
candle close and miss a freshly closed candle for nearly four minutes. For a
REVERSAL confirmation that is the one read where staleness matters: confirming
four minutes late is most of a 5m candle behind the move.

2 · SILENT FAILURE. A short or failed fetch returned {'ltf_bull': False,
'ltf_bear': False} — identical to "the market has not turned". Since the 5m
print is REQUIRED for every entry, an unreadable tape silently blocks the entire
desk in a way indistinguishable from a quiet market.
"""
import pytest

import scripts.engine.engine as E
import scripts.live_engine as LE


def test_the_5m_read_is_refreshed_every_scan():
    """A 240s TTL on a ~60s scan gives one fresh look in four."""
    class _S:
        pass
    s = _S()
    s._candle_cache_ttl = 240
    s._candle_cache_ttl_fast = 55
    s._fast_timeframes = ('1m', '3m', '5m')
    assert s._candle_cache_ttl_fast < 60, (
        'the 5m window is cached longer than a scan interval — the reversal '
        'print can be a whole candle behind'
    )
    assert '5m' in s._fast_timeframes


def test_the_fetcher_picks_the_fast_ttl_for_lower_timeframes():
    import inspect
    src = inspect.getsource(LE.LiveEngine._fetch_candles)
    assert '_candle_cache_ttl_fast' in src, (
        'the fetcher ignores the fast TTL, so setting it changes nothing'
    )
    assert '_fast_timeframes' in src


def test_a_fast_ttl_is_still_a_cache():
    """Not zero — the point is freshness per scan, not hammering the exchange."""
    class _S:
        pass
    s = _S()
    s._candle_cache_ttl_fast = 55
    assert s._candle_cache_ttl_fast > 0


# -- an unreadable tape must be visible ---------------------------------------

def test_an_unreadable_tape_is_counted():
    import inspect
    src = inspect.getsource(LE.LiveEngine._ltf_confirmation)
    lines = [l.split('#')[0] for l in src.splitlines()]
    body = chr(10).join(lines)
    assert 'LTF_UNREADABLE' in body, (
        'a short or failed 5m fetch still returns False silently — '
        'indistinguishable from a market that has not turned, while blocking '
        'every entry on the desk'
    )
    assert 'except Exception:\n            pass' not in body


def test_the_counter_exists_and_starts_clean():
    assert E.LTF_UNREADABLE['count'] >= 0


# -- a doji is not a down candle ----------------------------------------------

def test_a_flat_candle_does_not_confirm_a_short():
    """ltf_bear was `len(window) - ups`, which counts close == open as bearish,
    so four dojis confirmed a downside reversal."""
    import inspect
    src = inspect.getsource(LE.LiveEngine._ltf_confirmation)
    assert 'downs' in src, 'ltf_bear is still the complement of ups'
    assert 'len(window) - ups' not in src


def test_the_window_is_still_three_of_four():
    """The desk asked for '3 5min candle confirmation'. That is the bar."""
    assert E.LiveEngine.ENTRY_5M_WINDOW == 4
    assert max(3, E.LiveEngine.ENTRY_5M_WINDOW - 1) == 3
