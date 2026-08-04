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


# ── the engine is setup-typed, NOT fade-only ─────────────────────────────────

def test_engine_no_longer_blocks_trend_following():
    """The engine takes trend-following trades, and pays them the most.

    This test used to assert the opposite — that live_engine.py contained
    'fade-only strategy' and a `_trend_follow = (` veto — and it passed right up
    until the v80..v82 guard chain was deleted. It was passing on dead code: the
    chain sat behind `if USE_TRADER_GATE:` and had been unreachable since v83,
    so the strings were present but the veto never ran.

    What actually runs is trader_gate.py's setup-typed playbook, where
    TREND_PULLBACK carries the LARGEST risk weight of any setup and
    EXHAUSTION_REVERSAL the smallest — the opposite ordering to fade-only, and
    deliberately so (measured +0.069R vs -0.064R per trade).

    NOTE: chart.html still frames a strong ADX as a headwind on the reasoning
    that the engine fades trends. That copy and this engine now disagree; the
    wording is a product decision, which is why this test asserts the engine's
    behaviour rather than quietly rewriting the page.
    """
    from src.trading import trader_gate as TG

    weights = TG.SETUP_RISK_WEIGHT
    assert TG.SETUP_TREND_PULLBACK in weights, 'trend-following setup is gone'
    assert weights[TG.SETUP_TREND_PULLBACK] == max(weights.values()), (
        'TREND_PULLBACK is no longer the highest-weighted setup — if the engine '
        'has gone back to fading, the ADX copy on chart.html becomes correct '
        'again and this test should be inverted'
    )
    assert weights[TG.SETUP_TREND_PULLBACK] > weights[TG.SETUP_EXHAUSTION_REVERSAL]

    engine = (Path(__file__).resolve().parents[1] / 'live_engine.py').read_text(
        encoding='utf-8')
    assert 'fade-only strategy' not in engine, (
        'a fade-only veto is back in live_engine.py — it would now contradict '
        'trader_gate.py, which sizes trend-pullbacks largest'
    )
