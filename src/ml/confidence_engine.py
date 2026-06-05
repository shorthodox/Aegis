"""
confidence_engine.py — Confidence Reliability Engine
=====================================================
Tracks prediction confidence vs. actual outcomes to answer:

  "When the model predicts 90% confidence, how often does it actually win?"
  "When the model predicts 80% confidence, how often does it actually win?"

Confidence buckets: 0.50-0.55, 0.55-0.60, ..., 0.95-1.00
(10 buckets at 5pp resolution, matching the meta-model's tradeable range)

Generates confidence_report.md with per-bucket reliability statistics.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT    = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _ROOT / 'logs' / 'confidence'

# Fixed 5pp confidence buckets starting at 0.50 (model's tradeable floor)
_BUCKET_EDGES = np.arange(0.50, 1.01, 0.05)   # [0.50, 0.55, 0.60, ..., 1.00]
_BUCKET_LABELS = [
    f"{_BUCKET_EDGES[i]:.2f}–{_BUCKET_EDGES[i+1]:.2f}"
    for i in range(len(_BUCKET_EDGES) - 1)
]


class ConfidenceReliabilityEngine:
    """
    Builds a reliability mapping from predicted confidence → actual win rate.

    Train: call build_reliability_curve(y_prob, y_true) on OOF/dev data.
    Inference: call adjust_confidence(raw_prob) to get historically-grounded value.
    Report: call generate_report(symbol) to save confidence_report.md.
    """

    def __init__(self) -> None:
        self.reliability_curve: Dict[str, Dict] = {}
        self._bucket_stats:     List[Dict]      = []
        self._mapping_x: Optional[np.ndarray]   = None
        self._mapping_y: Optional[np.ndarray]   = None

    # ── Build ────────────────────────────────────────────────────────────────

    def build_reliability_curve(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Compute per-bucket reliability using fixed 5pp confidence bins.

        Parameters
        ----------
        y_prob : predicted confidence scores (meta-calibrated)
        y_true : binary labels (1 = correct prediction, 0 = wrong)

        Returns
        -------
        Dict mapping bucket label → {predicted_midpoint, actual_win_rate, samples}
        """
        valid  = ~np.isnan(y_prob)
        yp     = y_prob[valid]
        yt     = y_true[valid]

        if len(yp) == 0:
            return {}

        bucket_stats: List[Dict] = []
        curve: Dict[str, Dict]   = {}

        for i in range(len(_BUCKET_EDGES) - 1):
            lo, hi      = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
            label       = _BUCKET_LABELS[i]
            midpoint    = float((lo + hi) / 2.0)

            if i == len(_BUCKET_EDGES) - 2:            # last bucket: include 1.0
                mask = (yp >= lo) & (yp <= hi)
            else:
                mask = (yp >= lo) & (yp < hi)

            n = int(mask.sum())
            if n == 0:
                stat = {
                    'bucket': label, 'midpoint': midpoint,
                    'samples': 0, 'actual_win_rate': None,
                    'gap': None, 'reliable': False,
                }
            else:
                win_rate = float(yt[mask].mean())
                gap      = float(midpoint - win_rate)   # positive = overconfident
                reliable = n >= 10
                stat = {
                    'bucket':          label,
                    'midpoint':        midpoint,
                    'samples':         n,
                    'actual_win_rate': round(win_rate, 4),
                    'gap':             round(gap, 4),
                    'reliable':        reliable,
                }
                if reliable:
                    curve[label] = {
                        'predicted_midpoint': midpoint,
                        'actual_win_rate':    round(win_rate, 4),
                        'samples':            n,
                    }

            bucket_stats.append(stat)

        self._bucket_stats     = bucket_stats
        self.reliability_curve = curve

        # Build monotonic interpolation function from reliable buckets
        x_pts = [0.0]
        y_pts = [0.0]
        for stat in bucket_stats:
            if stat['reliable'] and stat['actual_win_rate'] is not None:
                x_pts.append(stat['midpoint'])
                y_pts.append(stat['actual_win_rate'])
        x_pts.append(1.0)
        y_pts.append(1.0)

        # Enforce monotone-increasing (prevents the curve from bending backward)
        y_mono = [y_pts[0]]
        for y in y_pts[1:]:
            y_mono.append(max(y, y_mono[-1]))

        self._mapping_x = np.array(x_pts)
        self._mapping_y = np.array(y_mono)

        return curve

    # ── Adjust ───────────────────────────────────────────────────────────────

    def adjust_confidence(self, raw_prob: float) -> float:
        """Map a raw predicted probability to its historically reliable win rate."""
        if self._mapping_x is None:
            return raw_prob
        return float(np.interp(raw_prob, self._mapping_x, self._mapping_y))

    def adjust_confidence_array(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._mapping_x is None:
            return raw_probs.copy()
        adjusted = np.full_like(raw_probs, np.nan, dtype=float)
        valid = ~np.isnan(raw_probs)
        if valid.any():
            adjusted[valid] = np.interp(raw_probs[valid], self._mapping_x, self._mapping_y)
        return adjusted

    # ── Report ───────────────────────────────────────────────────────────────

    def generate_report(self, symbol: str) -> Path:
        """
        Save confidence_report.md answering per-bucket reliability questions.

        Key questions answered:
          When model predicts 90%, how often does it actually win?
          When model predicts 80%, how often does it actually win?
        """
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"{symbol.replace('/', '_')}_confidence_report.md"

        lines: List[str] = [
            f"# Confidence Reliability Report — {symbol}",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n",
            "## Overview\n",
            "This report answers: **when the model predicts X% confidence, "
            "how often does it actually win?**\n",
            "A perfectly calibrated model would have `actual_win_rate ≈ predicted_confidence`.\n",
            "Positive gap = overconfident. Negative gap = underconfident.\n",
        ]

        # Per-bucket table
        lines.append("\n## Per-Bucket Reliability\n")
        lines.append("| Confidence Bucket | Samples | Predicted | Actual Win Rate | Gap | Status |")
        lines.append("|-------------------|---------|-----------|-----------------|-----|--------|")

        for stat in self._bucket_stats:
            if stat['samples'] == 0:
                lines.append(
                    f"| {stat['bucket']} | 0 | {stat['midpoint']:.0%} | — | — | no data |"
                )
                continue
            win = stat['actual_win_rate']
            gap = stat['gap']
            status = "V calibrated" if abs(gap) < 0.05 else \
                     ("? overconfident" if gap > 0 else "? underconfident")
            if not stat['reliable']:
                status = "? low sample"
            lines.append(
                f"| {stat['bucket']} | {stat['samples']} | "
                f"{stat['midpoint']:.0%} | {win:.1%} | "
                f"{gap:+.3f} | {status} |"
            )

        # Key questions section
        lines.append("\n## Key Questions\n")
        question_buckets = {
            '90%': '0.90–0.95',
            '80%': '0.85–0.90',
            '70%': '0.70–0.75',
            '60%': '0.60–0.65',
        }
        for label, bucket in question_buckets.items():
            stat = next((s for s in self._bucket_stats if s['bucket'] == bucket), None)
            if stat and stat['samples'] > 0 and stat['actual_win_rate'] is not None:
                lines.append(
                    f"- **When model predicts ~{label}:** actual win rate = "
                    f"**{stat['actual_win_rate']:.1%}** "
                    f"(n={stat['samples']}, gap={stat['gap']:+.3f})"
                )
            else:
                lines.append(f"- **When model predicts ~{label}:** insufficient data")

        # Calibration quality summary
        reliable_stats = [s for s in self._bucket_stats
                          if s['reliable'] and s['actual_win_rate'] is not None]
        if reliable_stats:
            mean_gap = float(np.mean([abs(s['gap']) for s in reliable_stats]))
            lines.append(f"\n**Mean absolute calibration gap:** {mean_gap:.4f} "
                         f"({'V well calibrated' if mean_gap < 0.05 else '? needs calibration'})\n")

        path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"   [ConfidenceEngine] Report saved → {path.name}")
        return path
