"""TP1 is a fixed percentage of entry. The objective cap was quietly moving it.

The ladder is priced in percent of entry — 1.5, 3.0, 3.75, 5.25, 6.0 since v87 —
because
pricing it in R put the first objective two to three times further away than
the stop, so a reversal between entry and the first bank turned a profitable
position into a full loss.

The objective cap then undid that. `_ladder` may not place a rung beyond
plan.target, and it enforced that by scaling ALL FIVE rungs onto the objective.
Scaling moves TP1, and TP1 is the one rung that must not move: its entire job
is to bank something before a reversal can reach entry.

Measured on the closed signals of 2026-08-08/09, reconstructing each trade's
objective from its published TP3:

    ALGO  objective 1.64%  ->  TP1 landed at 0.23%
    AVAX            1.72%                   0.25%
    GMX             1.79%                   0.26%
    DOT             2.04%                   0.29%
    JUP             2.18%                   0.31%
    ARKM            2.28%                   0.33%
    NEAR/AR         2.84%                   0.41%
    OP              3.03%                   0.43%

Eleven of thirteen ran a compressed ladder. Those trades could not book 0.5%
because their own first rung was never there — which is why the track record is
full of wins under half a percent.

The cap truncates now instead of scaling: rungs that fit keep their published
percentage, and only the ones that do not fit are spread between the last
fitting rung and the objective.
"""
import pytest

from scripts.engine.risk import DynamicRiskEngine


TP1_PCT, TP2_PCT = DynamicRiskEngine.TP_LADDER_PCT[0], DynamicRiskEngine.TP_LADDER_PCT[1]
_N   = len(DynamicRiskEngine.TP_LADDER_PCT)
_GAP = DynamicRiskEngine.TP_MIN_GAP_PCT

# The objective needed to carry the full first rung AND still space the four
# above it. Below this TP1 has to yield — five ordered rungs simply do not fit
# inside a smaller objective — but it must yield by the gap arithmetic and not
# by scaling. v87 moved TP1 1.0 -> 1.5, which lifted this floor from 1.2 to 1.7
# and so widened the band of real objectives that land in the fallback.
TP1_FLOOR = TP1_PCT + (_N - 1) * _GAP

# every objective reconstructed from the 2026-08-08/09 track record
REAL_OBJECTIVES = [1.64, 1.72, 1.79, 2.04, 2.18, 2.28, 2.84, 2.84, 3.03, 3.09, 3.36, 3.50, 3.51]


def _pcts(objective_pct, side='BUY', price=100.0):
    tgt = price * (1 + objective_pct / 100) if side == 'BUY' else price * (1 - objective_pct / 100)
    rungs = DynamicRiskEngine._ladder(price, side, tgt)
    return [abs(r - price) / price * 100 for r in rungs]


# ── the defect ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize('objective', [o for o in REAL_OBJECTIVES if o >= TP1_FLOOR])
@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_tp1_is_the_published_percentage_on_every_real_trade(objective, side):
    """No trade with room for the rung may open with a first rung below it."""
    assert _pcts(objective, side)[0] == pytest.approx(TP1_PCT), (
        f'objective {objective}% moved TP1 — the ladder is being scaled again')


@pytest.mark.parametrize('objective', [o for o in REAL_OBJECTIVES if o < TP1_FLOOR])
@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_a_tight_objective_yields_tp1_by_the_gap_and_never_by_scaling(objective, side):
    """The rung that will not fit must give up the minimum, not a proportion.

    This is the branch v87 had to rewrite. Scaling the ladder onto the objective
    put a 1.64% trade's TP1 at 0.41% — the same "banking a fifth of a percent"
    failure the truncating cap was written to end, reached by the other door once
    TP1 moved out to 1.5%. Anchored to the gap it lands at 1.44%.
    """
    p = _pcts(objective, side)
    assert p[0] == pytest.approx(objective - (_N - 1) * _GAP), (
        f'objective {objective}% put TP1 at {p[0]:.3f}%, which is not the most '
        f'the objective can carry — the ladder is being scaled again')
    assert p[0] > TP1_PCT * (objective / DynamicRiskEngine.TP_LADDER_PCT[-1]) , (
        'TP1 is no better than the proportional scaling this replaced')


# TP2 survives only while the three rungs above it still fit at the minimum
# gap: objective >= 1.5 + 3 x TP_MIN_GAP_PCT. Below that the ladder must give
# TP2 up to keep the cap honest — TP1 is the rung that never yields.
TP2_FLOOR = TP2_PCT + 3 * DynamicRiskEngine.TP_MIN_GAP_PCT


@pytest.mark.parametrize('objective', [o for o in REAL_OBJECTIVES if o >= TP2_FLOOR])
def test_tp2_holds_too_when_the_objective_leaves_room_above_it(objective):
    assert _pcts(objective)[1] == pytest.approx(TP2_PCT)


def test_tp1_still_holds_where_tp2_cannot():
    """An objective just under the TP2 floor — TP1 must survive it anyway."""
    p = _pcts(TP2_FLOOR - 0.01)
    assert p[0] == pytest.approx(TP1_PCT)
    assert p[1] < TP2_PCT          # TP2 yielded, as it must
    assert max(p) <= TP2_FLOOR - 0.01 + 1e-9


def test_the_specific_trades_that_could_not_reach_their_own_tp1():
    """ALGO at 1.64% had TP1 at 0.23%. It booked -0.05%."""
    for objective, was in ((1.64, 0.23), (1.79, 0.26), (2.04, 0.29), (3.03, 0.43)):
        now = _pcts(objective)[0]
        assert now > was, f'objective {objective}%: TP1 still at {now:.2f}%'
        # 1.64% cannot carry the full 1.5% rung and four spaced above it, so it
        # takes the gap-anchored value; every wider objective keeps the rung.
        expect = TP1_PCT if objective >= TP1_FLOOR else objective - (_N - 1) * _GAP
        assert now == pytest.approx(expect)


# ── the cap it must not break ────────────────────────────────────────────────

@pytest.mark.parametrize('objective', [0.6, 0.8, 1.0, 1.6, 2.0, 2.5, 3.0, 3.49])
@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_no_rung_is_ever_placed_beyond_the_objective(objective, side):
    """The reason the cap exists: the payoff stage rejects unreachable targets,
    so a rung past the objective re-invents what it rejected."""
    p = _pcts(objective, side)
    assert max(p) <= objective + 1e-9, f'a rung sits past the {objective}% objective: {p}'


@pytest.mark.parametrize('objective', [0.6, 1.0, 1.6, 2.28, 3.5, 5.0])
@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_rungs_stay_strictly_ordered(objective, side):
    """Two rungs at one price makes a partial close a no-op and silently
    changes the size of the runner."""
    p = _pcts(objective, side)
    assert all(b > a for a, b in zip(p, p[1:])), p
    gaps = [b - a for a, b in zip(p, p[1:])]
    assert min(gaps) >= DynamicRiskEngine.TP_MIN_GAP_PCT - 1e-9, gaps


def test_a_far_objective_leaves_the_published_ladder_untouched():
    """Banking earlier than the objective is the point of a percent ladder.

    "Far" is derived from the top rung rather than written as a number: this
    test read 5.0 against a ladder that ended at 4.0, and v87 moved the top rung
    to 6.0 — which made the objective a CAPPING one and the assertion a
    statement about the cap instead of about the published ladder.
    """
    far = DynamicRiskEngine.TP_LADDER_PCT[-1] + 1.0
    assert _pcts(far) == pytest.approx(list(DynamicRiskEngine.TP_LADDER_PCT))


def test_an_objective_tighter_than_tp1_falls_back_rather_than_lying():
    """Nothing can hold the full TP1 inside an objective smaller than it — but
    the cap must still hold, and the rungs must still be ordered."""
    tight = TP1_PCT * 0.6
    p = _pcts(tight)
    assert p[0] < TP1_PCT
    assert max(p) <= tight + 1e-9
    assert all(b > a for a, b in zip(p, p[1:]))


# ── the two bugs together ────────────────────────────────────────────────────

def test_a_tagged_tp1_now_books_more_than_the_old_rung_was_worth():
    """The ladder fix and the give-back fix have to compound, not cancel.

    ALGO's compressed TP1 was 0.23%. With the rung uncompressed and the
    give-back booking AT it, a reversal off TP1 books the whole rung — several
    times what the compressed one was worth.
    """
    R = DynamicRiskEngine
    entry = 100.0
    tp1 = _pcts(3.0)[0]
    assert tp1 == pytest.approx(TP1_PCT)
    span = entry * tp1 / 100
    leash = min(max(span * R.TP_GIVEBACK_PCT, R.TP_GIVEBACK_MIN_ATR * entry * 0.012),
                span * R.TP_GIVEBACK_MAX_FRAC)
    booked = (span - leash) / entry * 100
    assert booked > 0.23, (
        f'a reversal off TP1 books {booked:.2f}%, no better than the compressed '
        f'rung it replaced')
