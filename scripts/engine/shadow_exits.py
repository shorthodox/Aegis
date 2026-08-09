"""Shadow exit accounting — what the alternatives WOULD have booked.

The exit study compared policies on 8,587 synthetic paths and the current rule
won on every one of them. But those entries were every bar in both directions:
a population with no directional edge by construction. The engine's entries are
selected, and if that selection carries drift the ranking could flip.

This settles it on the engine's own signals instead of on synthetic ones. Every
time a position opens, the same trade is run forward under alternative exit
policies. Nothing here touches the live book — it observes the price stream the
exit monitor is already walking and writes the comparison to disk.

Two things make it a fair test rather than a flattering one:

  * The shadows use the REAL trade's geometry — its stop, its ladder, its
    entry — not the study's fixed -1.5% stop and 0.5% first rung. Policy
    parameters are expressed as multiples of the trade's own TP1 so they scale
    when the objective cap compresses the ladder.
  * A shadow keeps running after the live position closes, up to MAX_HOLD, so
    "the live rule banked early and the runner would have kept going" is
    actually measurable. Marking a shadow flat at the live exit price would
    quietly assume the answer.

Costs are charged once per shadow trade, matching ROUND_TRIP_COST_PCT, so a
policy that exits in more pieces is not rewarded for it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Policy parameters are multiples of the trade's OWN TP1, not absolute
# percentages, so a compressed ladder scales them with it. With TP1 at 0.5%
# these reproduce the three best variants from the study:
#   lock 0.5xTP1              -> stop at +0.25%
#   trail 1.5xTP1 past 2xTP1  -> trail 0.75% once past 1.0%
#   trail 2.0xTP1 past 3xTP1  -> trail 1.00% once past 1.5%
POLICIES: Dict[str, Dict[str, Any]] = {
    'live_rule': dict(
        closes=(0.15, 0.25, 0.25, 0.15, 0.20), lock=0.0, giveback=True,
        note='the rule actually running — simulated too, as a control'),
    'bank40_lock25': dict(
        closes=(0.40, 0.20, 0.15, 0.15, 0.10), lock=0.5,
        note='bank 40% at TP1, runner stop locked at half of TP1'),
    'bank40_trail_tight': dict(
        closes=(0.40, 0.20, 0.15, 0.15, 0.10), lock=0.5,
        trail=1.5, trail_from=2.0,
        note='...and trail 1.5xTP1 once past 2xTP1'),
    'bank40_trail_wide': dict(
        closes=(0.40, 0.20, 0.15, 0.15, 0.10), lock=0.0,
        trail=2.0, trail_from=3.0,
        note='break-even stop, wider trail — the highest-ceiling variant'),
}

_EPS = 1e-9                   # see _advance: percentages are derived, not exact
GIVEBACK_FRAC = 0.20          # matches DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC
MAX_SHADOW_SECONDS = 24 * 3600
_STORE = Path(__file__).resolve().parents[2] / 'data' / 'shadow_exits.json'


@dataclass
class _Sim:
    """One policy's state on one trade. All figures are % of entry."""
    banked: float = 0.0
    left: float = 1.0
    peak: float = 0.0
    stop: float = 0.0
    hit: List[bool] = field(default_factory=lambda: [False] * 5)
    done: bool = False
    exit_pct: float = 0.0
    exit_reason: str = ''

    def settle(self, pct: float, reason: str) -> None:
        self.exit_pct = self.banked + self.left * pct
        self.exit_reason = reason
        self.left = 0.0
        self.done = True


@dataclass
class _Trade:
    trade_id: str
    symbol: str
    direction: str
    entry: float
    ladder: Tuple[float, ...]      # rung percentages, ascending
    opened_at: float
    sims: Dict[str, _Sim]
    live_pct: Optional[float] = None
    live_reason: str = ''


class ShadowBook:
    """Runs alternative exit policies alongside the live one. Read-only."""

    def __init__(self, cost_pct: float = 0.10, store: Optional[Path] = None) -> None:
        self.cost_pct = float(cost_pct)
        self.store = Path(store) if store else _STORE
        self._open: Dict[str, _Trade] = {}
        self.rows: List[Dict[str, Any]] = []
        self._load()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def open(self, trade_id: str, symbol: str, direction: str, entry: float,
             stop_loss: float, take_profits: List[float]) -> None:
        """Register a live position. Never raises into the trading path."""
        try:
            if entry <= 0 or not trade_id or trade_id in self._open:
                return
            ladder = tuple(self._pct(direction, entry, tp)
                           for tp in take_profits if tp and tp > 0)
            if len(ladder) < 2 or ladder[0] <= 0:
                return                      # no usable geometry to shadow
            stop_pct = self._pct(direction, entry, stop_loss) if stop_loss > 0 else -999.0
            sims = {}
            for name, pol in POLICIES.items():
                s = _Sim()
                s.stop = stop_pct
                s.hit = [False] * len(ladder)
                sims[name] = s
            self._open[trade_id] = _Trade(
                trade_id=trade_id, symbol=symbol, direction=direction, entry=entry,
                ladder=ladder, opened_at=time.time(), sims=sims)
        except Exception as e:
            print(f'[ShadowBook] open failed for {symbol}: {e!r}')

    def tick(self, prices: Dict[str, float]) -> None:
        """Advance every open shadow against the newest prices."""
        if not self._open:
            return
        now = time.time()
        try:
            for tid, tr in list(self._open.items()):
                px = float(prices.get(tr.symbol, 0.0) or 0.0)
                if px > 0:
                    self._advance(tr, self._pct(tr.direction, tr.entry, px))
                expired = now - tr.opened_at > MAX_SHADOW_SECONDS
                if expired:
                    last = self._pct(tr.direction, tr.entry, px) if px > 0 else 0.0
                    for s in tr.sims.values():
                        if not s.done:
                            s.settle(last, 'MAX_HOLD')
                if tr.live_pct is not None and all(s.done for s in tr.sims.values()):
                    self._finalise(tid)
        except Exception as e:
            print(f'[ShadowBook] tick error: {e!r}')

    def record_live(self, trade_id: str, pnl_pct: float, reason: str) -> None:
        """The live position closed. Shadows keep running until their own exit."""
        tr = self._open.get(trade_id)
        if tr is None:
            return
        tr.live_pct = float(pnl_pct)
        tr.live_reason = reason
        if all(s.done for s in tr.sims.values()):
            self._finalise(trade_id)

    # ── the simulation ───────────────────────────────────────────────────────

    @staticmethod
    def _pct(direction: str, entry: float, price: float) -> float:
        if entry <= 0:
            return 0.0
        d = (price - entry) if direction == 'LONG' else (entry - price)
        return d / entry * 100.0

    def _advance(self, tr: _Trade, fav: float) -> None:
        """One price observation, applied to every unfinished policy.

        The stop is checked BEFORE the rungs: when both could trigger between
        two observations there is no way to know which came first, and assuming
        the favourable one would flatter every policy that carries a stop.
        """
        tp1 = tr.ladder[0]
        for name, s in tr.sims.items():
            if s.done:
                continue
            pol = POLICIES[name]

            # Reaching the level is a trigger. Both sides of this comparison are
            # derived percentages, so a rung at 100.5 off a 100.0 entry lands on
            # 0.4999999999999432 — without the tolerance a price sitting exactly
            # on the stop misses it by one part in 1e14.
            if fav <= s.stop + _EPS:
                s.settle(s.stop, 'STOP')
                continue

            trail = pol.get('trail')
            if trail and s.peak >= pol.get('trail_from', 2.0) * tp1:
                s.stop = max(s.stop, s.peak - trail * tp1)

            for i, rung in enumerate(tr.ladder):
                if s.hit[i] or fav + _EPS < rung:
                    continue
                s.hit[i] = True
                take = min(pol['closes'][i] if i < len(pol['closes']) else 0.0, s.left)
                s.banked += take * rung
                s.left -= take
                if i == 0:
                    s.stop = max(s.stop, pol.get('lock', 0.0) * tp1)
                if pol.get('giveback'):
                    prev = 0.0 if i == 0 else tr.ladder[i - 1]
                    s.stop = max(s.stop, rung - (rung - prev) * GIVEBACK_FRAC)
                if s.left <= 1e-9:
                    s.settle(0.0, f'LADDER_TP{i + 1}')
                    break
            s.peak = max(s.peak, fav)

    # ── persistence ──────────────────────────────────────────────────────────

    def _finalise(self, trade_id: str) -> None:
        tr = self._open.pop(trade_id, None)
        if tr is None:
            return
        row = {
            'trade_id': tr.trade_id, 'symbol': tr.symbol, 'direction': tr.direction,
            'entry': tr.entry, 'tp1_pct': round(tr.ladder[0], 4),
            'opened_at': tr.opened_at, 'closed_at': time.time(),
            'live_pct': round(tr.live_pct or 0.0, 4), 'live_reason': tr.live_reason,
            'policies': {n: {'pct': round(s.exit_pct - self.cost_pct, 4),
                             'reason': s.exit_reason}
                         for n, s in tr.sims.items()},
        }
        self.rows.append(row)
        self._save()
        best = max(row['policies'], key=lambda k: row['policies'][k]['pct'])
        print(f'[ShadowBook] {tr.symbol} live {row["live_pct"]:+.2f}% | '
              + ' | '.join(f'{n} {v["pct"]:+.2f}%' for n, v in row['policies'].items())
              + f' | best={best}')

    def summary(self) -> Dict[str, Any]:
        """Mean/median per policy. n is what matters — read nothing under ~100.

        The decision number is `vs_control`: every policy against the SIMULATED
        live rule, not against what the engine actually booked. Simulated-to-
        simulated isolates the policy, because the real figure also carries fill
        slippage, model-reversal exits and manual closes that no policy here
        models. `vs_live_actual` is kept as the reality check — if the control
        drifts far from it, the simulator has stopped describing the engine and
        the whole comparison is void.
        """
        if not self.rows:
            return {'n': 0, 'policies': {}}
        out: Dict[str, Any] = {'n': len(self.rows), 'policies': {}}
        live = [r['live_pct'] for r in self.rows]
        out['live_actual'] = {
            'mean': round(sum(live) / len(live), 4),
            'win_rate': round(sum(1 for v in live if v > 0) / len(live) * 100, 1)}
        means: Dict[str, float] = {}
        for name in POLICIES:
            vals = [r['policies'][name]['pct'] for r in self.rows
                    if name in r.get('policies', {})]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            means[name] = sum(vals) / len(vals)
            out['policies'][name] = {
                'mean':     round(means[name], 4),
                'median':   round(vals_sorted[len(vals) // 2], 4),
                'win_rate': round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            }
        control = means.get('live_rule')
        for name, blk in out['policies'].items():
            if control is not None:
                blk['vs_control'] = round(means[name] - control, 4)
            blk['vs_live_actual'] = round(means[name] - out['live_actual']['mean'], 4)
        if control is not None:
            out['control_tracks_reality'] = round(
                control - out['live_actual']['mean'], 4)
        return out

    def _save(self) -> None:
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_suffix('.tmp')
            payload = {'updated_at': time.time(),
                       'note': ('observation only — no live order was ever placed from '
                                'these figures'),
                       'summary': self.summary(), 'trades': self.rows[-500:]}
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
            tmp.replace(self.store)
        except Exception as e:
            print(f'[ShadowBook] save failed: {e!r}')

    def _load(self) -> None:
        try:
            if self.store.exists():
                self.rows = json.loads(self.store.read_text(encoding='utf-8')).get('trades', [])
        except Exception:
            self.rows = []
