"""Guards against direction-blind copy on the chart page.

These are customer-facing explanations of a live trade, and they were describing
indicators relative to THEMSELVES rather than to the trade. Observed on
INJ/USDT 2026-08-01, an open SHORT:

  * "MACD signal is BULLISH — momentum crossed above signal line, supporting
    continuation."  On a short. Bullish momentum opposes it.
  * Gate 1.7 reported "Weekly BULL · Daily BEAR" as ALIGNED, because the check
    only asked whether BOTH higher timeframes opposed.
  * "ADX 60 confirms a strong trending move — trend-following entries have
    higher edge here", while live_engine hard-blocks trend-following
    ("fade-only strategy").

chart.html has no test harness, so these assert on the source. They are coarse
by design: they fail if the old shape returns, not if the wording is edited.
"""
import re
from pathlib import Path

import pytest

_CHART = Path(__file__).resolve().parents[2] / 'web' / 'src' / 'pages' / 'chart.html'


@pytest.fixture(scope='module')
def src() -> str:
    return _CHART.read_text(encoding='utf-8')


# ── Gate 1.7 must be three-state ─────────────────────────────────────────────

def test_htf_verdict_is_not_a_two_state_collapse(src):
    """The old form reported '1 of 2 opposing' as ALIGNED."""
    assert 'const htfOppose = isShort ? (_mw > 0 && _md > 0)' not in src, (
        'Gate 1.7 is back to binary — a mixed HTF read will render as ALIGNED'
    )


def test_htf_counts_how_many_timeframes_oppose(src):
    assert '_htfAgainst' in src
    assert 'htfMixed' in src
    assert "'MIXED'" in src, 'the MIXED verdict is gone'


def test_htf_conflict_still_exists(src):
    """CONFLICT must survive — it is what Guard F hard-blocks on."""
    assert "'CONFLICT'" in src


# ── the trade story must describe indicators relative to the trade ───────────

def test_macd_narrative_is_not_direction_blind(src):
    assert 'momentum crossed above signal line, supporting continuation.' not in src, (
        'the MACD story line is direction-blind again — it will claim bullish '
        'momentum supports a short'
    )


def test_macd_narrative_branches_on_trade_direction(src):
    macd = src[src.index('// MACD'):src.index('// RSI')]
    assert 'dir ===' in macd, 'MACD copy no longer considers the trade direction'
    assert 'against' in macd.lower(), 'no branch tells the user momentum opposes'


def test_adx_does_not_recommend_trend_following(src):
    """live_engine blocks trend-following outright; recommending it is wrong."""
    assert 'trend-following entries have higher edge here' not in src, (
        'ADX copy recommends trend-following, which the engine refuses to trade'
    )


def test_adx_frames_a_strong_trend_as_a_headwind(src):
    adx = src[src.index('// ADX'):src.index('// Funding')]
    assert 'headwind' in adx.lower()


# ── the engine really is fade-only, so the copy above is correct ─────────────

def test_engine_blocks_trend_following(src):
    """If this ever changes, the ADX copy must change back."""
    engine = (Path(__file__).resolve().parents[1] / 'live_engine.py').read_text(
        encoding='utf-8')
    assert 'fade-only strategy' in engine
    assert re.search(r"_trend_follow\s*=\s*\(", engine), (
        'trend-following block is gone — revisit the ADX narrative'
    )
