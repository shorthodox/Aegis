"""A stop must be hit on the market we actually trade, not a correlated one.

Reported 2026-08-26: "when signal in the loss zone... not touching SL, signal is
still closing and saying SL hit even though it didn't."

WHAT IT WAS NOT
    * Not a mislabel. All five STOP_HIT closes on the volume exited within 0.1%
      of their recorded stop (-0.016%, -0.011%, -0.029%, -0.103%, +0.057%), so
      the tag was accurate and the stop really was reached on the engine's feed.
    * Not a moved stop being reported as the original. GMX's recorded stop sat
      2.83% from entry — the structural stop, not a break-even.

WHAT IT IS: TWO DIFFERENT MARKETS

    candles, levels, and therefore the STOP   ->  ccxt.binanceusdm   (PERP)
    the live tick that triggers the exit      ->  stream.binance.com (SPOT)

The engine derives a stop from perp structure, shows it on a chart captioned
"Binance USD-M Futures - the exact market AEGIS trades", and then tests it
against a spot price.

A stop is a THRESHOLD, and at a threshold the basis decides the outcome.
Measured across 20 tokens: median |perp-spot| 0.064%, p90 0.150%, max 0.196%.
Against a ~1.30% stop that is 5-15% of the entire distance. GMX closed 0.103%
through its stop on a day its basis was 0.094% - the two numbers are the same
size, which is the whole bug.

WHY NOT JUST SWITCH THE FEED

Because it does not work. Measured from the Railway container: the futures
websocket fstream.binance.com delivered ZERO messages in 18 seconds while the
spot stream delivered 753 symbols covering the whole fleet. The spot ticker is
not an oversight, it is the only stream reachable from that host. Switching it
would have silently ended all live pricing and with it every intra-scan exit.

Futures REST, however, is fine from there - 749 rows in 0.09s. So the stop is
confirmed against the perp at the moment it would fire. Stops are rare, so the
cost is a handful of calls a day.

FAIL DIRECTION: an unreachable perp returns 0.0, meaning UNKNOWN, and the close
proceeds exactly as before. A data failure must never be able to hold a position
open past its stop.
"""
import inspect

from scripts.engine import exits as E
from scripts.engine import market_data as M


SRC = inspect.getsource(E)


def _stop_block():
    i = SRC.index('if pos.stop_loss > 0:')
    return SRC[i:SRC.index("_close('STOP_HIT')", i) + 40]


# -- the confirmation exists and gates the close ------------------------------

def test_the_stop_is_confirmed_against_the_perp():
    b = _stop_block()
    assert 'fetch_perp_price' in b, (
        'the stop still fires on the spot tick alone, on a market the subscriber '
        'is not shown'
    )


def test_a_perp_that_disagrees_holds_the_position():
    b = _stop_block()
    assert 'if not _perp_hit:' in b
    assert 'return' in b.split('if not _perp_hit:')[1][:400]


def test_the_disagreement_is_logged_with_both_prices():
    b = _stop_block()
    assert 'STOP NOT CONFIRMED' in b
    assert 'basis' in b


def test_the_confirmation_respects_direction():
    b = _stop_block()
    assert "pos.direction == 'LONG'  and _perp <= pos.stop_loss" in b
    assert "pos.direction == 'SHORT' and _perp >= pos.stop_loss" in b


# -- an unknown perp must never strand a position -----------------------------

def test_an_unreachable_perp_still_closes():
    """0.0 means UNKNOWN, not 'the stop was not hit'."""
    b = _stop_block()
    assert 'if _perp > 0:' in b, (
        'a failed perp lookup could hold a position open past its stop'
    )


def test_the_helper_returns_zero_rather_than_raising(monkeypatch):
    monkeypatch.setattr(M, '_PERP_PX', {})
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('blocked')))
    assert M.fetch_perp_price('GMX/USDT') == 0.0


def test_the_lookup_is_wrapped_at_the_call_site_too():
    b = _stop_block()
    assert 'except Exception:' in b
    assert '_perp = 0.0' in b


# -- the helper itself --------------------------------------------------------

def test_it_targets_the_futures_endpoint():
    src = inspect.getsource(M.fetch_perp_price)
    assert 'fapi.binance.com' in src, 'the confirmation reads spot again'
    assert 'api.binance.com/api' not in src


def test_symbols_are_normalised():
    src = inspect.getsource(M.fetch_perp_price)
    assert "replace('/', '')" in src


def test_results_are_cached_briefly():
    """A stop cascade across symbols must not become a burst of REST calls."""
    assert 0 < M._PERP_PX_TTL <= 15.0
    src = inspect.getsource(M.fetch_perp_price)
    assert '_PERP_PX' in src


def test_a_cached_price_is_returned_without_a_call(monkeypatch):
    import time
    monkeypatch.setattr(M, '_PERP_PX', {'GMXUSDT': (7.404, time.time())})
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('should not have been called')))
    assert M.fetch_perp_price('GMX/USDT') == 7.404


# -- the spot ticker stays, deliberately --------------------------------------

def test_the_live_ticker_is_still_spot():
    """fstream is unreachable from the deploy host; switching it would end all
    live pricing. This test exists so that is a decision, not a regression."""
    import scripts.engine.engine as ENG
    src = inspect.getsource(ENG)
    assert 'stream.binance.com' in src
    assert 'wss://fstream.binance.com' not in src
