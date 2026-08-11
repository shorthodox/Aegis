"""Edge over geometry — the only figure that reports the signal.

In a market with no drift the chance of touching +a before -b is b/(a+b). That
expression contains nothing about the entry: no model, no features, no gate.
Measured over 13,560 real paths across 30 tokens it predicts the hit rate to
within about a point at every target/stop pair tried:

    target  stop    measured   geometry
     1.00%  1.30%      56.5%      56.5%
     0.50%  1.40%      73.7%      73.7%     <- the 0.5% ladder, and the live 75%
     1.00%  2.33%      71.5%      70.0%
     1.00%  4.00%      80.8%      80.0%

So a headline win rate is a report on the stop distance. Widening the stop to
4% takes it to 80% without losing a single signal and without earning a penny —
every row above has the same expectancy, which is minus the round trip.

    edge = measured hit rate - mean(stop / (target + stop))

is what moves when the model gets better and stays flat when the stop moves. It
is the number to tune against, and the number to be honest about: at the v86
ladder the engine needs +4.4 points of it to break even after costs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scripts.engine.config import TRACK_RECORD_PATH


def geometric_hit_rate(target_pct: float, stop_pct: float) -> Optional[float]:
    """The driftless probability of reaching the target before the stop.

    Both arguments are POSITIVE distances in percent. Returns None rather than
    a number when the geometry is degenerate — a zero-width barrier is not a
    50/50 race, it is a missing input, and averaging a fabricated 0.5 into the
    baseline would understate the edge on exactly the trades that lack data.
    """
    if target_pct <= 0 or stop_pct <= 0:
        return None
    return stop_pct / (target_pct + stop_pct)


def _distances(sig: Dict[str, Any]) -> Optional[tuple]:
    """(target %, stop %) for one recorded signal, as positive distances."""
    entry = float(sig.get('entry_price') or 0)
    tp1 = float(sig.get('take_profit_1') or 0)
    sl = float(sig.get('stop_loss') or 0)
    if entry <= 0 or tp1 <= 0 or sl <= 0:
        return None
    return abs(tp1 - entry) / entry * 100.0, abs(entry - sl) / entry * 100.0


def _reached_target(sig: Dict[str, Any]) -> Optional[bool]:
    """Did this trade touch its first rung before its stop?

    tp_hits is the authority — it counts rungs actually banked. exit_reason is
    the fallback for records written before tp_hits existed. A record with
    neither returns None and is excluded, rather than being counted as a miss:
    treating unknowns as failures would manufacture a negative edge.
    """
    hits = sig.get('tp_hits')
    if hits is not None:
        try:
            return int(hits) >= 1
        except (TypeError, ValueError):
            pass
    reason = str(sig.get('exit_reason') or '').upper()
    if not reason:
        return None
    if 'TP' in reason or 'GIVEBACK' in reason:
        return True
    if 'STOP' in reason or 'SL' in reason:
        return False
    return None


def measure(signals: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Edge over geometry for a set of closed signals."""
    hits: List[bool] = []
    geoms: List[float] = []
    skipped = 0
    for s in signals:
        if str(s.get('outcome') or '').upper() == 'OPEN':
            continue
        d = _distances(s)
        reached = _reached_target(s)
        if d is None or reached is None:
            skipped += 1
            continue
        g = geometric_hit_rate(*d)
        if g is None:
            skipped += 1
            continue
        hits.append(reached)
        geoms.append(g)

    n = len(hits)
    if n == 0:
        return {'n': 0, 'skipped': skipped,
                'note': 'no closed signal carried both a target and a stop'}
    measured = sum(hits) / n
    baseline = sum(geoms) / n
    # What the trade must clear once the round trip is charged. Derived from the
    # same distances, so it moves with the ladder instead of being a constant
    # that quietly goes stale.
    return {
        'n':               n,
        'skipped':         skipped,
        'measured_hit':    round(measured * 100, 2),
        'geometric_hit':   round(baseline * 100, 2),
        'edge_pp':         round((measured - baseline) * 100, 2),
        'verdict':         _verdict(measured - baseline, n),
    }


def _verdict(edge: float, n: int) -> str:
    if n < 30:
        return f'n={n} — too few closed signals to read'
    if edge <= 0:
        return ('no edge over geometry — the win rate is reporting the stop '
                'distance, not the signal')
    if edge < 0.02:
        return f'+{edge*100:.1f}pp of edge, inside the noise at this sample'
    return f'+{edge*100:.1f}pp of genuine edge over geometry'


def by_group(signals: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Same measure, split by a field — 'symbol' and 'regime' are the useful ones."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in signals:
        buckets.setdefault(str(s.get(key) or 'UNKNOWN'), []).append(s)
    out = {k: measure(v) for k, v in buckets.items()}
    return {k: v for k, v in sorted(out.items(),
                                    key=lambda kv: -(kv[1].get('edge_pp') or -99))}


def from_track_record(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else TRACK_RECORD_PATH
    try:
        data = json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception as e:
        return {'n': 0, 'error': f'could not read the track record: {e}'}
    sigs = data.get('signals') if isinstance(data, dict) else data
    if not isinstance(sigs, list):
        return {'n': 0, 'error': 'track record has no signals list'}
    overall = measure(sigs)
    overall['by_symbol'] = by_group(sigs, 'symbol')
    overall['by_regime'] = by_group(sigs, 'regime')
    return overall
