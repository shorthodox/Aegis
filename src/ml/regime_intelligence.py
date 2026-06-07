"""
regime_intelligence.py — Regime Confidence Modifier & Threshold System
========================================================================
Learns per-regime confidence modifiers and thresholds from historical
performance data.  Modifiers are NOT hardcoded — they are derived
statistically from the OOF dev set after HMM labeling.

Design
------
ONE global meta model is kept.  Regime intelligence adds:
  1. A per-regime confidence modifier (bonus / penalty)
  2. A per-regime threshold (learned to maximise precision × coverage)

Example output:
  TRENDING_BULL  threshold=0.62  modifier=+0.04  (better than global)
  CHOPPY         threshold=0.78  modifier=-0.08  (worse than global)
  COMPRESSION    threshold=0.70  modifier=-0.02  (slightly worse)

The RegimeConfidenceModifier is serialised inside {symbol}_aegis_state.pkl
and loaded by the live predictor.

Persistence
-----------
regime_stats.json is saved to data/regime_stats/ and auto-updated after
each retraining run.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT            = Path(__file__).resolve().parent.parent.parent
_REGIME_STATS_DIR = _ROOT / 'data' / 'regime_stats'

# Minimum bars in a regime to trust its statistics
_MIN_REGIME_BARS = 30
_MIN_FIRES       = 5


class RegimeConfidenceModifier:
    """
    Learns per-regime thresholds and confidence modifiers from OOF data.

    After training:
      • regime_thresholds[regime] → float threshold
      • regime_modifiers[regime]  → float confidence delta (positive = bonus)
      • regime_stats[regime]      → full stat block
    """

    def __init__(self, target_precision: float = 0.575) -> None:
        self.target_precision   = target_precision
        self.regime_stats:      Dict[str, Dict] = {}
        self.regime_modifiers:  Dict[str, float] = {}
        self.regime_thresholds: Dict[str, float] = {}
        self._global_precision: float = 0.0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def analyze_regimes(
        self,
        regime_labels:  np.ndarray,      # per-bar regime string
        y_prob:         np.ndarray,       # calibrated meta confidence
        y_side:         np.ndarray,       # 0=SELL, 2=BUY proposed side
        y_true:         np.ndarray,       # 0/1/2 triple-barrier label
        base_threshold: float = 0.50,
        barrier_frac:   Optional[np.ndarray] = None,
    ) -> Dict[str, Dict]:
        """
        Analyze per-regime trading performance and learn thresholds/modifiers.

        Modifiers are learned as:
            modifier = clip(regime_precision - global_precision, -0.30, +0.20)

        Thresholds are the most permissive threshold that achieves
        target_precision in each regime (with ≥ _MIN_FIRES trades).

        Parameters
        ----------
        regime_labels  : HMM state string per OOF bar
        y_prob         : calibrated meta probabilities per OOF bar
        y_side         : primary's proposed side per OOF bar (0 or 2)
        y_true         : triple-barrier label per OOF bar (0/1/2)
        base_threshold : global meta threshold
        barrier_frac   : barrier size fraction for PnL stats
        """
        valid  = ~np.isnan(y_prob)
        yp     = y_prob[valid]
        ys     = y_side[valid]
        yt     = y_true[valid]
        rl     = regime_labels[valid]
        bf     = barrier_frac[valid] if barrier_frac is not None else None

        # Global precision at base threshold
        g_fired = (yp >= base_threshold) & ((ys == 2) | (ys == 0))
        if g_fired.any():
            self._global_precision = float((ys[g_fired] == yt[g_fired]).mean())
        else:
            self._global_precision = self.target_precision

        regimes = [r for r in list(dict.fromkeys(rl)) if r != 'UNKNOWN']

        for regime in regimes:
            mask = (rl == regime)
            if mask.sum() < _MIN_REGIME_BARS:
                continue

            mp = yp[mask];  pr = ys[mask];  yt_r = yt[mask]
            bfr = bf[mask] if bf is not None else None

            # Baseline: all directional proposals at base_threshold
            fired_base = (mp >= base_threshold) & ((pr == 2) | (pr == 0))
            n_base     = int(fired_base.sum())

            if n_base > 0:
                prec_base = float((pr[fired_base] == yt_r[fired_base]).mean())
                cov_base  = float(n_base / len(mp))
            else:
                prec_base = 0.0
                cov_base  = 0.0

            # Per-side precision
            buy_m   = fired_base & (pr == 2)
            sell_m  = fired_base & (pr == 0)
            buy_p   = float((yt_r[buy_m]  == 2).mean()) if buy_m.any()  else 0.0
            sell_p  = float((yt_r[sell_m] == 0).mean()) if sell_m.any() else 0.0

            # PnL stats
            win_rate = exp_pct = pf = 0.0
            if bfr is not None and n_base >= _MIN_FIRES:
                rets = []
                for k in np.where(fired_base)[0]:
                    b   = float(bfr[k]) if np.isfinite(bfr[k]) else 0.01
                    s   = int(pr[k])
                    lbl = int(yt_r[k])
                    if lbl == 1:
                        rets.append(-0.001)
                    elif s == lbl:
                        rets.append(b - 0.001)
                    else:
                        rets.append(-b - 0.001)
                if rets:
                    ra       = np.array(rets)
                    win_rate = float((ra > 0).mean())
                    exp_pct  = float(ra.mean() * 100)
                    gw = float(ra[ra > 0].sum()) if (ra > 0).any() else 0.0
                    gl = float(abs(ra[ra < 0].sum())) if (ra < 0).any() else 1e-9
                    pf = gw / gl

            # Expectancy-aware: find the most permissive threshold for this regime
            # that still achieves the target precision, or the best precision/volume
            # tradeoff if the target is unreachable.
            opt_thr = base_threshold
            max_thr = min(0.99, float(np.nanmax(mp)) if len(mp) else 0.95)
            thresholds = np.sort(np.unique(np.concatenate([
                np.linspace(base_threshold, max_thr, 40),
                np.quantile(mp, [0.80, 0.85, 0.90, 0.95]) if len(mp) else np.array([])
            ])))
            regime_min_fires = max(_MIN_FIRES, int(len(mp) * 0.10))

            best_metric = -np.inf
            for t in thresholds:
                f = (mp >= t) & ((pr == 2) | (pr == 0))
                n = f.sum()
                if n < regime_min_fires:
                    continue
                p = float((pr[f] == yt_r[f]).mean())
                if p >= self.target_precision:
                    opt_thr = float(t)
                    break

                metric = p * np.sqrt(n)
                if metric > best_metric:
                    best_metric = metric
                    opt_thr = float(t)

            # Modifier: learned from observed vs global precision gap
            modifier = float(np.clip(prec_base - self._global_precision, -0.30, 0.20))

            self.regime_stats[regime] = {
                'trades':              n_base,
                'coverage':            round(cov_base,  4),
                'precision':           round(prec_base, 4),
                'buy_precision':       round(buy_p,     4),
                'sell_precision':      round(sell_p,    4),
                'win_rate':            round(win_rate,  4),
                'expectancy_pct':      round(exp_pct,   4),
                'profit_factor':       round(pf,        3),
                'global_precision':    round(self._global_precision, 4),
                'confidence_modifier': round(modifier,  4),
                'learned_threshold':   round(opt_thr,   4),
            }
            self.regime_modifiers[regime]  = modifier
            self.regime_thresholds[regime] = opt_thr

        if self.regime_stats:
            print(f"   [Regime] Analyzed {len(self.regime_stats)} regimes "
                  f"(global_prec={self._global_precision:.3f})")
            for r, s in sorted(self.regime_stats.items()):
                mod = s['confidence_modifier']
                print(f"      {r:<22} thr={s['learned_threshold']:.3f}  "
                      f"prec={s['precision']:.3f}  "
                      f"trades={s['trades']}  "
                      f"mod={mod:+.3f}")
        else:
            print("   [Regime] No regime stats (HMM labels not available or too few bars)")

        return self.regime_stats

    # ── Inference ────────────────────────────────────────────────────────────

    def apply_modifier(self, raw_prob: float, regime: str) -> float:
        mod = self.regime_modifiers.get(regime, 0.0)
        return float(np.clip(raw_prob + mod, 0.0, 1.0))

    def get_regime_threshold(self, regime: str, global_threshold: float) -> float:
        return self.regime_thresholds.get(regime, global_threshold)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_regime_stats(self, symbol: str) -> Path:
        """Save per-regime stats to data/regime_stats/{symbol}_regime_stats.json."""
        _REGIME_STATS_DIR.mkdir(parents=True, exist_ok=True)
        path = _REGIME_STATS_DIR / f"{symbol.replace('/', '_')}_regime_stats.json"

        payload: Dict[str, Any] = {
            'symbol':           symbol,
            'updated_at':       datetime.now().isoformat(),
            'target_precision': self.target_precision,
            'global_precision': round(self._global_precision, 4),
            'regimes':          {},
        }
        for regime, stats in self.regime_stats.items():
            payload['regimes'][regime] = {
                **stats,
                'threshold': self.regime_thresholds.get(regime, 0.50),
                'modifier':  self.regime_modifiers.get(regime, 0.0),
            }

        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"   [Regime] Stats saved → {path.name}")
        return path

    @classmethod
    def load_regime_stats(cls, symbol: str) -> Optional['RegimeConfidenceModifier']:
        path = _REGIME_STATS_DIR / f"{symbol.replace('/', '_')}_regime_stats.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            inst = cls(target_precision=data.get('target_precision', 0.575))
            inst._global_precision = data.get('global_precision', 0.0)
            for regime, s in data.get('regimes', {}).items():
                inst.regime_stats[regime]      = s
                inst.regime_thresholds[regime] = float(s.get('threshold', 0.50))
                inst.regime_modifiers[regime]  = float(s.get('modifier',  0.0))
            return inst
        except Exception:
            return None
