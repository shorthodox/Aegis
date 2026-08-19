"""A signal the engine refuses must still show WHY, not go blank.

Measured on the live fleet 2026-08-19: 43 of 44 symbols rendered as NEUTRAL with
no direction, no levels and no reasoning, while the engine had a full read on
nearly all of them. 0 of 44 fired — and of those refusals, 23 were "no setup,
mid-range", 11 "exhaustion reversal refused", 4 "SELL against a strong BTC UP
tide". Every one of those is a sentence worth showing. The card showed none of
them because both refusal branches set result['side'] = 'FLAT'.

Blanking the side was never load-bearing:

  * the desk does not read result['side'] — _classify picks the side from
    range_position and the regime, and REQUIRE_MODEL_FIRE is False
  * the single caller of _open_position sets result['side'] = plan.side
    immediately before opening

So it was display state, and clearing it only destroyed information. `fire` is
what gates anything, and it still goes False in both branches.
"""
import inspect
import re

import pytest

from scripts.engine.engine import LiveEngine


@pytest.fixture(scope='module')
def commit_block():
    """The commit block with COMMENTS STRIPPED.

    The comments quote the old blanking line verbatim to explain what changed, so
    a naive substring check matches the explanation instead of the code.
    """
    src = inspect.getsource(LiveEngine._process_symbol)
    i = src.index('_q_low = (')
    block = src[i:src.index("result['tradeable'] = True", i)]
    return '\n'.join(l for l in block.splitlines() if not l.lstrip().startswith('#'))


def test_the_side_survives_a_refusal(commit_block):
    assert "result['side'] = 'FLAT'" not in commit_block, (
        'a refused signal is blanked to NEUTRAL again — the card loses its '
        'direction, levels and reasoning for no functional gain'
    )
    assert "result['side'] = _model_side" in commit_block


def test_fire_is_still_suppressed_on_both_refusal_paths(commit_block):
    """The whole point: stop the TRADE, keep the ANALYSIS."""
    hard = commit_block[commit_block.index('if _hard:'):commit_block.index('elif _q_low:')]
    low  = commit_block[commit_block.index('elif _q_low:'):commit_block.index('else:')]
    for name, branch in (('hard veto', hard), ('quality floor', low)):
        assert "result['fire'] = False" in branch, f'{name} no longer suppresses fire'


def test_the_refusal_reason_is_published(commit_block):
    """Without it the card can only say 'not tradeable' with no cause."""
    assert "result['not_tradeable']" in commit_block, (
        'the refusal reason is not published — the UI cannot explain itself'
    )
    hard = commit_block[commit_block.index('if _hard:'):commit_block.index('elif _q_low:')]
    assert '_hard' in hard, 'the hard-veto branch does not name which veto fired'


def test_the_quality_refusal_is_still_counted(commit_block):
    assert "LOW_QUALITY_REFUSED['count'] += 1" in commit_block, (
        'the counter is gone — the floor becomes untunable again'
    )


def test_only_the_gate_decides_what_opens():
    """Guards the assumption this change rests on: preserving the side must not
    let anything open a position behind the desk's back."""
    import scripts.engine.engine as eng
    src = inspect.getsource(eng)
    calls = [m for m in re.findall(r'self\._open_position\(', src)]
    assert len(calls) == 1, (
        f'_open_position now has {len(calls)} callers — preserving result["side"] '
        f'is only safe while the gate is the sole entry point'
    )
    gate = inspect.getsource(LiveEngine._run_trader_gate)
    # Whitespace-tolerant: the assignment is column-aligned with its neighbours.
    assert re.search(r"result\['side'\]\s*=\s*plan\.side", gate), (
        'the gate no longer sets the side before opening, so a preserved model '
        'side could reach _open_position'
    )
