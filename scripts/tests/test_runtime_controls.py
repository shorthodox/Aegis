"""Operating dials must be changeable without a deploy — and stay changed.

Six times in one day these constants were changed by editing code, running the
suite and redeploying: MIN_FIRE_QUALITY twice (60 -> 45 -> 60), the tide policy,
STRONG_TIDE_FACTOR, and a pause that did not exist. Every round trip needed a
computer, and by the time one landed the tape had moved.

A quality floor and a size factor are decisions a desk makes against a live
market. They are OPERATING dials, not code.

The property that makes this possible: every knob is read at CALL time by its
owner — engine.py reads MIN_FIRE_QUALITY through the config MODULE, trader_gate
reads its own globals inside evaluate()/_classify(). A knob captured at import
would accept a change and silently ignore it, which is worse than no control at
all, so that is asserted directly below.
"""
import importlib
import inspect
import json

import pytest

import main
from scripts.engine import config as cfg
from src.trading import trader_gate as TG


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(main, '_RUNTIME_PATH', tmp_path / 'runtime_overrides.json')
    saved = {k: getattr(importlib.import_module(m), a)
             for k, (m, a, *_r) in main._TUNABLES.items()}
    yield tmp_path / 'runtime_overrides.json'
    for k, (m, a, *_r) in main._TUNABLES.items():      # never leak into other tests
        setattr(importlib.import_module(m), a, saved[k])


# ── the property everything rests on ─────────────────────────────────────────

def test_every_knob_is_read_at_call_time():
    """A knob captured at import would accept a change and silently ignore it."""
    gate = inspect.getsource(main.LiveEngine._run_trader_gate) \
        if hasattr(main, 'LiveEngine') else ''
    from scripts.engine.engine import LiveEngine
    gate = inspect.getsource(LiveEngine._run_trader_gate)
    assert "getattr(_cfg, 'MIN_FIRE_QUALITY'" in gate
    assert "getattr(_cfg, 'TRADING_PAUSED'" in gate
    assert 'STRONG_TIDE_FACTOR' in inspect.getsource(TG.TraderGate.evaluate)
    assert 'ALLOW_EXHAUSTION_REVERSAL' in inspect.getsource(TG.TraderGate._classify)


# ── applying ─────────────────────────────────────────────────────────────────

def test_a_change_takes_effect_on_the_live_module(store):
    main._rt_apply({'min_fire_quality': 52})
    assert cfg.MIN_FIRE_QUALITY == 52.0


def test_it_reaches_a_different_module_too(store):
    main._rt_apply({'strong_tide_factor': 0.5})
    assert TG.STRONG_TIDE_FACTOR == 0.5


def test_the_pause_switch_is_honoured_where_positions_open():
    """One early return in the single path that opens a position."""
    from scripts.engine.engine import LiveEngine
    src = inspect.getsource(LiveEngine._run_trader_gate)
    i = src.index('TRADING_PAUSED')
    assert 'return True' in src[i:i + 400], 'pause does not stop the gate'
    assert src.index('TRADING_PAUSED') < src.index('LOSS_COOLDOWN_SECONDS'), (
        'the pause check runs after other logic — it should be first'
    )


# ── persistence ──────────────────────────────────────────────────────────────

def test_a_change_survives_a_restart(store):
    main._rt_apply({'min_fire_quality': 52, 'trading_paused': True})
    cfg.MIN_FIRE_QUALITY, cfg.TRADING_PAUSED = 60.0, False      # simulate restart
    main._rt_load_at_startup()
    assert cfg.MIN_FIRE_QUALITY == 52.0 and cfg.TRADING_PAUSED is True


def test_it_persists_to_the_railway_volume():
    assert '_RUNTIME_PATH = _STATE_DIR / "runtime_overrides.json"' in \
        open(main.__file__, encoding='utf-8', errors='replace').read(), (
            'overrides moved off STATE_DIR — they would not survive a redeploy'
        )


def test_the_write_is_atomic(store):
    src = inspect.getsource(main._rt_apply)
    assert '.tmp' in src and 'replace' in src


# ── bounds ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bad', [
    {'min_fire_quality': 900},      # would silently stop all trading
    {'min_fire_quality': -5},
    {'strong_tide_factor': -1},
    {'strong_tide_factor': 2},
])
def test_out_of_range_values_are_refused(store, bad):
    with pytest.raises(ValueError):
        main._rt_apply(bad)


def test_an_unknown_control_is_refused(store):
    with pytest.raises(ValueError):
        main._rt_apply({'delete_everything': 1})


def test_a_refused_value_changes_nothing(store):
    before = cfg.MIN_FIRE_QUALITY
    with pytest.raises(ValueError):
        main._rt_apply({'min_fire_quality': 900})
    assert cfg.MIN_FIRE_QUALITY == before


# ── reset ────────────────────────────────────────────────────────────────────

def test_reset_restores_the_values_committed_in_code(store):
    """The escape hatch: it must not depend on remembering the previous value."""
    main._rt_capture_defaults()
    committed = main._RUNTIME_DEFAULTS['min_fire_quality']
    main._rt_apply({'min_fire_quality': 12})
    main._rt_apply(dict(main._RUNTIME_DEFAULTS))
    assert cfg.MIN_FIRE_QUALITY == committed


def test_defaults_are_captured_once_not_overwritten(store):
    main._rt_capture_defaults()
    committed = main._RUNTIME_DEFAULTS['min_fire_quality']
    main._rt_apply({'min_fire_quality': 33})
    main._rt_capture_defaults()          # must be a no-op
    assert main._RUNTIME_DEFAULTS['min_fire_quality'] == committed, (
        'defaults were re-captured after an override — reset would restore the '
        'override instead of the committed value'
    )


# ── surface ──────────────────────────────────────────────────────────────────

def test_the_endpoints_require_admin():
    src = open(main.__file__, encoding='utf-8', errors='replace').read()
    i = src.index('@app.get("/api/admin/runtime")')
    seg = src[i:i + 1400]
    assert seg.count('_require_admin') >= 2, 'a runtime endpoint is unauthenticated'


def test_read_reports_bounds_and_defaults(store):
    c = main._rt_read()
    for name, meta in c.items():
        assert {'value', 'default', 'kind', 'min', 'max', 'description'} <= set(meta), name
