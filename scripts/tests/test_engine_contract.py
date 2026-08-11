"""What scripts.live_engine promises its 28 dependents, and how the engine is put together.

This replaces test_engine_extraction_parity.py. That file ran the old
single-file implementations against the newly extracted ones over randomised
inputs and required identical output — it is what caught, among other things, a
`from __future__ import annotations` that would have changed every dataclass
field's runtime type. It has now done its job: live_engine.py is a pure
re-export shim, so `live_engine.X is engine.<mod>.X` and every one of those
comparisons had become a tautology. Deleting a test that can no longer fail is
better than keeping one that looks like cover.

What is worth asserting from here on is different, and is what lives below:

  1. The import surface. main.py and ~27 other modules import by name from
     scripts.live_engine. Those names are a contract; breaking one breaks the
     app at import time, in production, not here.
  2. The composition. LiveEngine is assembled from four mixins, and a method
     silently lost during a future move would show up as an AttributeError deep
     in a scan rather than at boot.
  3. The flag gotcha. The shim re-exports USE_TRADER_GATE as a VALUE, so
     patching it there is a no-op — a trap worth one test.
  4. Geometry invariants that hold regardless of implementation.
"""
import inspect

import pytest

import scripts.live_engine as LE
from contextlib import contextmanager

from src.trading import trader_gate as TG

# v87 percent budget band. Read the switch instead of assuming it, so these
# contracts keep their meaning whichever way it is set.
_BAND_ON = TG.MIN_STOP_PCT > 0 and TG.MAX_STOP_PCT > 0


@contextmanager
def _band_off():
    """Assert a structural invariant on the placement the band clamps."""
    lo, hi = TG.MIN_STOP_PCT, TG.MAX_STOP_PCT
    TG.MIN_STOP_PCT = TG.MAX_STOP_PCT = 0.0
    try:
        yield
    finally:
        TG.MIN_STOP_PCT, TG.MAX_STOP_PCT = lo, hi


# ── 1. the import surface ────────────────────────────────────────────────────

# Exactly what the dependents pull from scripts.live_engine today. Sourced by
# grepping the repo for `from scripts.live_engine import ...`; if you remove a
# name here, remove its importer in the same commit.
REQUIRED_SURFACE = [
    # main.py
    'LiveEngine', 'automated_setup', 'TRACK_RECORD_PATH',
    'ALPHA_TRACK_RECORD_PATH', '_fs_clear_track_record',
    # scripts/tests/*
    'DynamicRiskEngine', 'Position', 'VirtualWallet', 'MarketRegimeDetector',
    'DriftMonitor', 'PerformanceTracker', 'PortfolioGuard', 'SignalQualityFilter',
    '_HARD_VETOES',
    # tests/*
    'RegimeState', 'TokenConfig', '_detect_bos_choch', '_reversal_candle',
    '_REGIME_TRENDING_BULL', '_REGIME_TRENDING_BEAR', '_REGIME_VOLATILE_COMPRESS',
    '_REGIME_RANGING',
    # engine internals other modules reach for
    'ScalpBot', 'MODEL_STORE', 'USE_TRADER_GATE', 'USE_WEIGHTED_SCORER',
]


@pytest.mark.parametrize('name', REQUIRED_SURFACE)
def test_surface_name_is_exported(name):
    assert hasattr(LE, name), (
        f'scripts.live_engine no longer exports {name!r} — something that '
        f'imports it will fail at boot, not here'
    )


def test_all_is_accurate():
    """__all__ must not advertise something the module does not have."""
    missing = [n for n in LE.__all__ if not hasattr(LE, n)]
    assert not missing, f'__all__ lists names that do not exist: {missing}'


def test_surface_is_covered_by_all():
    uncovered = [n for n in REQUIRED_SURFACE if n not in LE.__all__]
    assert not uncovered, (
        f'these are imported by other modules but missing from __all__: {uncovered}'
    )


# ── 2. composition ───────────────────────────────────────────────────────────

MIXINS = ['LevelsMixin', 'GatesMixin', 'ExitsMixin', 'PositionsMixin']

# One representative method per mixin, plus the core loop. If a mixin stops
# being composed these vanish silently until a scan touches them.
COMPOSED_METHODS = [
    '_sr_levels', '_swing_sr', '_htf_sr', '_btc_tide', '_daily_bias',   # levels
    '_structure_gate', '_confirmation_gate', '_retest_held',            # gates
    '_manage_exit',                                                     # exits
    '_open_position', '_build_signal_entry', '_save_track_record',      # positions
    '_process_symbol', '_run_trader_gate', 'run', '_scan_all',          # core
]


@pytest.mark.parametrize('mixin', MIXINS)
def test_mixin_is_in_the_mro(mixin):
    names = [c.__name__ for c in LE.LiveEngine.__mro__]
    assert mixin in names, f'{mixin} is no longer composed into LiveEngine'


@pytest.mark.parametrize('method', COMPOSED_METHODS)
def test_method_survives_composition(method):
    assert hasattr(LE.LiveEngine, method), f'LiveEngine lost {method}'


def test_no_method_is_defined_twice_across_mixins():
    """Two mixins defining the same name would let MRO order decide silently."""
    from scripts.engine import exits, gates, levels, positions

    seen: dict = {}
    clashes = []
    for mod, cls_name in ((levels, 'LevelsMixin'), (gates, 'GatesMixin'),
                          (exits, 'ExitsMixin'), (positions, 'PositionsMixin')):
        cls = getattr(mod, cls_name)
        for name, val in vars(cls).items():
            if name.startswith('__') or not callable(getattr(val, '__func__', val)):
                continue
            if name in seen:
                clashes.append(f'{name}: {seen[name]} and {cls_name}')
            seen[name] = cls_name
    assert not clashes, f'method defined in more than one mixin: {clashes}'


# ── 3. the flag gotcha ───────────────────────────────────────────────────────

def test_config_is_the_authority_for_runtime_flags():
    """Patching the shim must not be how you flip a flag.

    engine.py reads the flags through the config MODULE precisely so a patch is
    honoured. If someone reverts that to `from config import USE_TRADER_GATE`,
    every test and rollback that flips the flag becomes a silent no-op.
    """
    # USE_TRADER_GATE survives in exactly one place now that the legacy chain is
    # gone: the legacy PENDING-state sync, which is inert while the desk is on.
    src = inspect.getsource(LE.LiveEngine._sync_armed_pending_state)
    assert '_cfg.USE_TRADER_GATE' in src, (
        'the engine is reading USE_TRADER_GATE as a module-level value again — '
        'flipping it at runtime will silently do nothing'
    )
    # UWGS is consulted on the live decision path.
    src = inspect.getsource(LE.LiveEngine._process_symbol)
    assert '_cfg.USE_WEIGHTED_SCORER' in src, (
        'USE_WEIGHTED_SCORER is a module-level value copy again'
    )


# ── 4. geometry invariants ───────────────────────────────────────────────────

def test_tp_ladder_is_monotonic_and_ordered():
    r = LE.DynamicRiskEngine()
    for side in ('BUY', 'SELL'):
        out = r.calculate_stops(100.0, side, 1.0, support=97.0, resistance=106.0)
        tps = [out[f'tp{i}'] for i in range(1, 6)]
        ordered = tps == sorted(tps) if side == 'BUY' else tps == sorted(tps, reverse=True)
        assert ordered, f'{side}: TP ladder is not monotonic: {tps}'
        if side == 'BUY':
            assert out['sl'] < 100.0 < tps[0]
        else:
            assert tps[0] < 100.0 < out['sl']
        assert out['risk'] > 0


def test_stop_clears_the_level_it_defends():
    """v84: a stop resting ON the level is the one place it reliably gets taken.

    v87 qualifies this and the qualification is the important part. With the
    percent budget band on, a stop is capped at MAX_STOP_PCT of entry, and the
    levels this strategy leans on are routinely FURTHER than that — so the stop
    lands between price and the level rather than beyond it. It is then taken out
    by the very move that tests the level, before the thesis has been tested at
    all. The structural placement is still what the band clamps, so this asserts
    the invariant on that placement, and pins the consequence separately.
    """
    r = LE.DynamicRiskEngine()
    with _band_off():
        long = r.calculate_stops(100.0, 'BUY', 1.0, support=99.0, resistance=110.0)
        assert long['sl'] < 99.0, 'LONG stop is not below the support it leans on'
        short = r.calculate_stops(100.0, 'SELL', 1.0, support=90.0, resistance=101.0)
        assert short['sl'] > 101.0, 'SHORT stop is not above the resistance it leans on'


def test_the_budget_band_is_what_stops_the_stop_clearing_the_level():
    """Names the cost of the band so it cannot be paid by accident.

    A support 1.0% away cannot be cleared by a stop capped at MAX_STOP_PCT when
    that cap is under 1.0%. This test does not judge the trade-off — it fails
    loudly if someone tightens the band without knowing they are buying it.
    """
    if not _BAND_ON:
        pytest.skip('budget band disabled')
    r = LE.DynamicRiskEngine()
    long = r.calculate_stops(100.0, 'BUY', 1.0, support=99.0, resistance=110.0)
    risk_pct = (100.0 - long['sl'])
    assert risk_pct <= TG.MAX_STOP_PCT + 1e-9, 'the band did not bind'
    if TG.MAX_STOP_PCT < 1.0:
        assert long['sl'] > 99.0, (
            'a sub-1% cap is expected to leave the stop IN FRONT of a support '
            '1% away — if this now passes, the geometry changed and the '
            'level-clearing guarantee above may be recoverable')


def test_plan_target_is_honoured_verbatim():
    """v85: no rung may sit beyond the objective the payoff stage cleared.

    The ladder is priced in percent of entry now, so the objective CAPS it
    rather than placing tp3: nearer than the top rung, the ladder scales onto
    it; further, the percentages stand and the trade banks on the way. What must
    never happen is a rung past the level the floor vouched for — that is how
    the published R:R stopped matching the R:R the gate approved.
    """
    r = LE.DynamicRiskEngine()
    out = r.calculate_stops(100.0, 'BUY', 1.0, support=97.0, resistance=130.0,
                            tp_override=104.0)
    assert max(out[f'tp{i}'] for i in range(1, 6)) <= 104.0 + 1e-9, (
        'a rung was placed beyond the plan target')
    assert out['tp1'] < out['tp2'] <= out['tp3']

    short = r.calculate_stops(100.0, 'SELL', 1.0, support=70.0, resistance=103.0,
                              tp_override=96.0)
    assert min(short[f'tp{i}'] for i in range(1, 6)) >= 96.0 - 1e-9, (
        'a rung was placed beyond the plan target')
    assert short['tp1'] > short['tp2'] >= short['tp3']


def test_plan_stop_is_honoured_but_never_left_on_the_level():
    """v83 says take the gate's stop verbatim; v84 says clear the level anyway.

    Both are true, and the order matters. The plan's stop is used as-is unless a
    level sits at or inside it, in which case the wick buffer wins — a stop
    resting on the support it defends is collected both when price undercuts and
    turns, and when it breaks and retests. My first draft of this test asserted
    the override survives a support sitting right above it; it does not, and that
    is deliberate.
    """
    r = LE.DynamicRiskEngine()
    buf = LE.DynamicRiskEngine.STRUCT_SL_BUFFER_ATR   # in ATR

    # No level in the way -> verbatim.
    clean = r.calculate_stops(100.0, 'BUY', 1.0, support=0.0, resistance=130.0,
                              sl_override=98.5)
    assert clean['sl'] == pytest.approx(98.5), 'plan stop was re-derived'

    # A support sitting above the plan's stop -> pushed clear of it by the buffer.
    guarded = r.calculate_stops(100.0, 'BUY', 1.0, support=97.0, resistance=130.0,
                                sl_override=98.5)
    assert guarded['sl'] == pytest.approx(97.0 - buf * 1.0)
    assert guarded['sl'] < 97.0, 'LONG stop was left at/above its support'

    # Mirror for a SHORT.
    short = r.calculate_stops(100.0, 'SELL', 1.0, support=70.0, resistance=101.0,
                              sl_override=100.5)
    assert short['sl'] == pytest.approx(101.0 + buf * 1.0)
    assert short['sl'] > 101.0, 'SHORT stop was left at/below its resistance'


def test_degenerate_inputs_do_not_produce_a_trade():
    r = LE.DynamicRiskEngine()
    for price, atr in ((0.0, 1.0), (100.0, 0.0), (-1.0, 1.0)):
        out = r.calculate_stops(price, 'BUY', atr)
        assert out['valid_rr'] is False
        assert out['risk'] == 0.0


def test_reversal_candle_needs_a_real_pattern():
    """Three green candles is not a reversal — the whole point of the function."""
    flat = [[0, 100, 100.5, 99.5, 100.2, 1000] for _ in range(5)]
    assert LE._reversal_candle(flat, True) is None
    assert LE._reversal_candle(flat, False) is None
    assert LE._reversal_candle([], True) is None


def test_range_pos_preserves_a_genuine_zero():
    """0.0 means 'at the absolute bottom' — the best fade setup there is."""
    assert LE._range_pos({'range_position': 0.0}) == 0.0
    assert LE._range_pos({}) == 0.5
    assert LE._range_pos({'range_position': None}) == 0.5
    assert LE._range_pos({'range_position': 'junk'}) == 0.5


def test_wallet_charges_the_same_round_trip_the_labels_assume():
    """The wallet and the training pipeline must price costs identically."""
    assert LE.VirtualWallet.round_trip_cost_pct() == pytest.approx(0.10)
