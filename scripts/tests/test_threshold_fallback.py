"""A model that fires nothing is a failed run, not a cautious one.

BTC/USDT, 2026-08-11: trained to 64.7% directional precision, then fired 0 of
1676 holdout bars. The log carried no [THR-SWEEP] line and no [SMALL VAL] line,
which pins the path exactly — the validation sweep RAN and every row was
rejected, so the threshold kept its initialiser:

    _best_thr_po = 0.85  # conservative default (used when sweep finds nothing)

The sweep scans from a floor of 0.60-0.70 and needs 50 fires per row. That
floor is on the RAW confidence scale, but the value being compared is a
calibrated P(direction correct) whose base rate the same run reported as 0.386.
Nothing reached the lowest rung, so no row qualified and the model shipped with
a threshold it could never clear.

Two fixes, and these tests cover both: the empty sweep now falls through to the
same training-pool percentile the small-val branch already used, and the engine
refuses to enable a model whose own holdout fired nothing.
"""
import inspect
import re

import pytest


# ── the threshold fallback ───────────────────────────────────────────────────

def _src() -> str:
    import scripts.retrain_model as R
    return inspect.getsource(R)


def test_the_empty_sweep_no_longer_falls_through_to_the_initialiser():
    src = _src()
    i = src.index('_rows_po: list = []')
    j = src.index('_primary_conf_thr = _best_thr_po')
    block = src[i:j]
    assert 'EMPTY-SWEEP' in block, (
        'the sweep can still come up empty and leave _best_thr_po at 0.85, '
        'which is the threshold that fired 0 of 1676 on BTC')


def test_both_fallback_paths_share_one_implementation():
    """They were separate, and only one of them was reachable from the sweep."""
    src = _src()
    assert src.count('_threshold_from_training_pool(') >= 3, (
        'expected one definition and two call sites (EMPTY-SWEEP, SMALL-VAL)')


def test_the_fallback_is_a_percentile_not_a_constant():
    """A fixed number is what broke: it assumed a scale. A percentile of the
    actual confidences works whether they are raw or calibrated."""
    src = _src()
    i = src.index('def _threshold_from_training_pool')
    block = src[i:i + 2200]
    assert 'np.percentile' in block
    assert '60)' in block


def test_the_fallback_reports_the_range_it_measured():
    """A threshold of 0.85 against a pool that tops out at 0.42 is the whole
    bug, and it was invisible. Printing the range makes it obvious."""
    src = _src()
    i = src.index('def _threshold_from_training_pool')
    block = src[i:i + 2200]
    assert 'pool range' in block


def test_an_empty_pool_still_returns_something_usable():
    src = _src()
    i = src.index('def _threshold_from_training_pool')
    block = src[i:i + 2200]
    assert 'len(_conf) == 0' in block and 'return 0.60' in block


# ── the loud failure ─────────────────────────────────────────────────────────

def test_zero_fires_is_announced_as_a_failure():
    src = _src()
    assert 'TRAINING FAILED' in src, (
        'a 0-fire run still reads as a routine "No signals fired" line, which '
        'is easy to scroll past in a 100-token sweep')


def test_the_message_distinguishes_a_threshold_fault_from_a_model_fault():
    """The actionable half: a model with real precision that fires nothing has
    a gate problem, and the fix is a different one."""
    src = _src()
    i = src.index('TRAINING FAILED')
    block = src[i:i + 900]
    assert 'CONFIDENCE THRESHOLD' in block
    assert '_primary_dir_prec' in block, 'the check reads a variable that exists'


# ── the engine must not enable it anyway ─────────────────────────────────────

def _engine_load_src() -> str:
    from scripts.engine.engine import LiveEngine
    src = inspect.getsource(LiveEngine)
    i = src.index('Binary dual-model pair')
    return src[max(0, i - 2000):i + 900]


def test_a_zero_fire_model_is_benched_before_the_dual_model_override():
    """The override marks a token tradeable whenever both side files exist. It
    would have enabled BTC, which then never fires — a bug that presents as a
    quiet market."""
    block = _engine_load_src()
    assert 'holdout_trading' in block
    assert block.index('holdout_trading') < block.index('Binary dual-model pair'), (
        'the zero-fire check runs after the override, so it can never stop it')


def test_the_bench_requires_both_zero_coverage_and_zero_fires():
    """One field could be missing or stale on an old sidecar; both being zero
    is unambiguous."""
    block = _engine_load_src()
    assert "get('coverage')" in block and "get('fired')" in block


def test_a_sidecar_without_holdout_stats_is_left_alone():
    """Legacy sidecars predate the field. Treating a missing verdict as a
    failing one would bench most of the fleet on no evidence."""
    block = _engine_load_src()
    assert 'if _ht and' in block, (
        'the check fires on an empty holdout_trading block, so any sidecar '
        'lacking the field would be benched')


def test_the_narrow_scope_is_stated():
    """This deliberately does NOT make the override respect `tradeable` in
    general — that flag is stale on many sidecars."""
    block = _engine_load_src()
    assert 'Deliberately narrow' in block
