"""Would a resting limit AT the level have filled? Observation only.

The fill-rate cost of entry-at-level cannot be recovered from history: no
archived signal kept the level it was waiting at, so "what fraction of
ACTION_WORK cards would a resting limit have filled within 8 bars, and are the
fills adversely selected?" is unanswerable from the archive. Only new cards
answer it.

This records them. The entire value depends on it changing NOTHING, so that is
what these tests pin hardest: a broken observer must not raise, must not print
into the decision stream, and must not touch a decision. The characterisation
baseline is the other half of that proof — it is byte-identical with this wired
in.

Nothing here is read back into a decision. It exists so the working-order
question can be answered with numbers BEFORE fills are actually built —
_working_orders is a Dict[str, float] of timestamps today, so real fill tracking
is genuine work and should follow the evidence, not precede it.
"""
import json
import time

import pytest

import scripts.live_engine as LE
from scripts.engine.engine import WO_OBSERVE_ERRORS


class _Plan:
    def __init__(self, side='SELL', level=100.0, stop=101.0, target=94.0,
                 setup='TREND_PULLBACK', r_net=2.0, expiry_bars=8):
        self.side, self.level, self.stop, self.target = side, level, stop, target
        self.setup, self.r_net, self.expiry_bars = setup, r_net, expiry_bars


def _eng(atr=1.0):
    """A bare object carrying only what the observer touches."""
    class _E:
        pass
    e = _E()
    e._wo_observed = {}
    e.last_signals = {'X/USDT': {'atr': atr}}
    e._wo_observe = LE.LiveEngine._wo_observe.__get__(e)
    e._wo_close = LE.LiveEngine._wo_close.__get__(e)
    return e


@pytest.fixture
def logpath(tmp_path, monkeypatch):
    import scripts.engine.config as cfg
    p = tmp_path / 'wo.jsonl'
    monkeypatch.setattr(cfg, 'WORKING_ORDER_LOG_PATH', p)
    return p


def _lines(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


# -- it records what the archive could not --------------------------------

def test_a_resting_order_is_recorded_with_its_level(logpath):
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(level=100.0), price=98.0, now=now)
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', now + 8 * 3600)
    rows = _lines(logpath)
    assert len(rows) == 1
    r = rows[0]
    assert r['level'] == 100.0 and r['side'] == 'SELL'
    assert r['stop'] == 101.0 and r['target'] == 94.0
    assert r['outcome'] == 'EXPIRED'


def test_price_reaching_the_level_marks_it_fillable(logpath):
    """A SELL limit at 100 fills when price trades up THROUGH 100."""
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(level=100.0), price=98.0, now=now)
    e._wo_observe('X/USDT', _Plan(level=100.0), price=100.4, now=now + 3600)
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', now + 8 * 3600)
    r = _lines(logpath)[0]
    assert r['would_have_filled'] is True
    assert r['bars_to_touch'] == pytest.approx(1.0, abs=0.01)


def test_a_level_never_reached_is_recorded_as_unfilled(logpath):
    """The number the decision actually turns on."""
    e = _eng()
    now = time.time()
    for i in range(4):
        e._wo_observe('X/USDT', _Plan(level=100.0), price=97.0 + i * 0.2, now=now + i * 3600)
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', now + 8 * 3600)
    r = _lines(logpath)[0]
    assert r['would_have_filled'] is False
    assert r['closest_atr'] == pytest.approx(2.4, abs=0.01)


def test_a_buy_limit_fills_from_above(logpath):
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(side='BUY', level=100.0, stop=99.0, target=106.0),
                  price=102.0, now=now)
    e._wo_observe('X/USDT', _Plan(side='BUY', level=100.0, stop=99.0, target=106.0),
                  price=99.6, now=now + 3600)
    e._wo_close('X/USDT', 'BUY', 'EXPIRED', now + 2 * 3600)
    assert _lines(logpath)[0]['would_have_filled'] is True


@pytest.mark.parametrize('outcome', ['FILLED_AT_LEVEL', 'EXPIRED', 'SUPERSEDED'])
def test_every_retirement_path_closes_the_record(logpath, outcome):
    """Four call sites retire a working order; a leak means a silent undercount."""
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(), price=98.0, now=now)
    e._wo_close('X/USDT', 'SELL', outcome, now + 3600)
    assert _lines(logpath)[0]['outcome'] == outcome
    assert not e._wo_observed, 'the observation leaked'


def test_a_relevelled_order_starts_a_new_observation(logpath):
    """If the plan moves to a different level it is a different question."""
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(level=100.0), price=98.0, now=now)
    e._wo_observe('X/USDT', _Plan(level=105.0), price=98.0, now=now + 3600)
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', now + 2 * 3600)
    assert _lines(logpath)[0]['level'] == 105.0


# -- and it must never affect anything ------------------------------------

def test_a_partially_built_engine_does_not_raise(logpath):
    """The characterisation harness builds an engine without the store. An
    observer that raises there is an observer that changed behaviour."""
    class _E:
        pass
    e = _E()
    e._wo_observe = LE.LiveEngine._wo_observe.__get__(e)
    e._wo_close = LE.LiveEngine._wo_close.__get__(e)
    e._wo_observe('X/USDT', _Plan(), price=98.0, now=time.time())
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', time.time())
    assert _lines(logpath) == []


def test_failures_are_counted_not_printed(logpath, capsys):
    """A line printed into the decision stream IS a behaviour change — it lands
    in the published card's reasoning."""
    e = _eng()
    before = WO_OBSERVE_ERRORS['count']
    e._wo_observe('X/USDT', object(), price=98.0, now=time.time())   # no .side
    assert capsys.readouterr().out == ''
    assert WO_OBSERVE_ERRORS['count'] > before


def test_an_unwritable_log_does_not_raise(tmp_path, monkeypatch):
    import scripts.engine.config as cfg
    monkeypatch.setattr(cfg, 'WORKING_ORDER_LOG_PATH', tmp_path / 'nope' / '\x00bad')
    e = _eng()
    now = time.time()
    e._wo_observe('X/USDT', _Plan(), price=98.0, now=now)
    e._wo_close('X/USDT', 'SELL', 'EXPIRED', now + 3600)      # must not raise


def test_the_log_lives_on_the_volume():
    """Days of accumulation; a deploy must not reset the answer."""
    import scripts.engine.config as cfg
    assert cfg.WORKING_ORDER_LOG_PATH.parent == cfg.STATE_DIR
