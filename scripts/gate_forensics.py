#!/usr/bin/env python3
"""
gate_forensics.py — Institution-grade forensic diagnostics for meta gate profiles
==================================================================================
Generates per-token forensic reports that explain every decision made by the
gate optimizer in plain language, with quantitative evidence.

Output files per symbol
  reports/gates/{symbol}_forensic_report.json   — machine-readable
  reports/gates/{symbol}_forensic_report.txt    — human-readable

Sections (Phase-1)
  TOKEN SUMMARY        symbol, timing, gate selected, calibration selected
  MODEL QUALITY        feature importance, concentration, dominant features
  CALIBRATION          per-method ECE / Brier / coverage / PF / accepted or rejected
  ARCHITECTURE         per-architecture coverage / PF / expectancy / score / reason
  SCORE BREAKDOWN      weighted components that sum exactly to the final score
  TRUST SCORE          components, total, consistency assertion
  REGIME FORENSICS     best/worst regimes, overfitting warning
  DISABLED DIAGNOSTICS root cause tree (only when DISABLED)

Sections (Phase-2)
  EDGE FAILURE ANALYSIS    feature concentration, entropy, diversity, buy/sell balance
  ROOT CAUSE CLASSIFICATION 8-category cause + confidence (DISABLED gates only)
  ARCHITECTURE ACCOUNTING   strict Total = Accepted + Rejected (no unaccounted rows)
  MODEL HEALTH SCORE        0-100 composite from 5 sub-components
  ALPHA ATTRIBUTION         signal% + calibration% + gate% explaining PF lift
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── score weights (must match _evaluate_architecture) ────────────────────────
_W_EXPECTANCY = 0.35
_W_PF         = 0.30
_W_SHARPE     = 0.20
_W_PRECISION  = 0.10
_W_COVERAGE   = 0.05

# ── trust weights (must match _compute_trust_score) ──────────────────────────
_TW_PF         = 0.30
_TW_EXPECTANCY = 0.25
_TW_SHARPE     = 0.20
_TW_COVERAGE   = 0.15
_TW_FIRES      = 0.10

# ── root cause categories ─────────────────────────────────────────────────────
ROOT_CAUSE_LABELS = [
    'NO_ALPHA',
    'WEAK_ALPHA',
    'OVERFITTING',
    'CALIBRATION_COLLAPSE',
    'COVERAGE_COLLAPSE',
    'REGIME_INSTABILITY',
    'FEATURE_DOMINANCE',
    'DATA_QUALITY_FAILURE',
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pf_norm(pf: float) -> float:
    return _clamp((pf - 1.0) / 2.0, 0.0, 1.0)


def _exp_norm(exp: float) -> float:
    return _clamp(exp / 10.0, 0.0, 1.0)


def _sharpe_norm(sharpe: float) -> float:
    return _clamp(sharpe / 5.0, 0.0, 1.0)


def gate_grade(profit_factor: float) -> str:
    """PF-based grade identical to meta_gate_optimizer._gate_grade."""
    if profit_factor > 1.80:
        return 'A+'
    if profit_factor > 1.60:
        return 'A'
    if profit_factor > 1.40:
        return 'B+'
    if profit_factor > 1.30:
        return 'B'
    if profit_factor > 1.20:
        return 'C+'
    if profit_factor > 1.10:
        return 'C'
    if profit_factor > 1.00:
        return 'D'
    return 'F'


def compute_score_breakdown(
    profit_factor: float,
    expectancy_pct: float,
    sharpe: float,
    precision: float,
    coverage: float,
) -> Dict[str, float]:
    """Return per-component contributions and the total.  Components sum to total."""
    e_n  = _exp_norm(expectancy_pct)
    p_n  = _pf_norm(profit_factor)
    s_n  = _sharpe_norm(sharpe)
    pr_n = _clamp(precision, 0.0, 1.0)
    c_n  = _clamp(coverage, 0.0, 1.0)
    return {
        'expectancy':    round(_W_EXPECTANCY * e_n, 6),
        'profit_factor': round(_W_PF * p_n, 6),
        'sharpe':        round(_W_SHARPE * s_n, 6),
        'precision':     round(_W_PRECISION * pr_n, 6),
        'coverage':      round(_W_COVERAGE * c_n, 6),
        'total':         round(
            _W_EXPECTANCY * e_n + _W_PF * p_n + _W_SHARPE * s_n +
            _W_PRECISION * pr_n + _W_COVERAGE * c_n,
            6,
        ),
    }


def compute_trust_score(
    profit_factor: float,
    expectancy_pct: float,
    sharpe: float,
    coverage: float,
    fired_n: int,
) -> Tuple[int, Dict[str, float]]:
    """Return (trust_score_0_to_100, component_breakdown)."""
    p_n  = _pf_norm(profit_factor)
    e_n  = _exp_norm(expectancy_pct)
    s_n  = _sharpe_norm(sharpe)
    c_n  = _clamp(coverage, 0.0, 1.0)
    f_n  = _clamp(fired_n / 100.0, 0.0, 1.0)
    raw  = 100.0 * (_TW_PF * p_n + _TW_EXPECTANCY * e_n + _TW_SHARPE * s_n + _TW_COVERAGE * c_n + _TW_FIRES * f_n)
    score = int(_clamp(raw, 0.0, 100.0))
    breakdown = {
        'pf_contribution':         round(_TW_PF * p_n * 100, 2),
        'expectancy_contribution': round(_TW_EXPECTANCY * e_n * 100, 2),
        'sharpe_contribution':     round(_TW_SHARPE * s_n * 100, 2),
        'coverage_contribution':   round(_TW_COVERAGE * c_n * 100, 2),
        'fires_contribution':      round(_TW_FIRES * f_n * 100, 2),
        'total':                   score,
    }
    return score, breakdown


class GateForensicsReporter:
    """Generate and save comprehensive forensic reports for a token's gate profile."""

    REPORTS_DIR_NAME = Path('reports') / 'gates'

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.out_dir = root_dir / self.REPORTS_DIR_NAME
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def generate_report(
        self,
        symbol: str,
        profile: Dict[str, Any],
        debug_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        debug     = debug_data or {}
        selected  = profile.get('selected_profile', {})
        holdout   = profile.get('holdout', {}).get('selected', {})
        gate_type = selected.get('gate_type', 'UNKNOWN')

        pf        = float(holdout.get('profit_factor', 0.0))
        exp_pct   = float(holdout.get('expectancy_pct', 0.0))
        sharpe    = float(holdout.get('sharpe', 0.0))
        precision = float(holdout.get('precision', 0.0))
        coverage  = float(holdout.get('coverage', 0.0))
        fired_n   = int(holdout.get('fired_n', holdout.get('n', 0)))

        score_bd        = compute_score_breakdown(pf, exp_pct, sharpe, precision, coverage)
        trust, trust_bd = compute_trust_score(pf, exp_pct, sharpe, coverage, fired_n)

        # ── Phase-2 fix: trust consistency uses UNKNOWN when saved value is missing
        saved_trust_raw = selected.get('trust_score')
        if saved_trust_raw is None or int(saved_trust_raw) == -1:
            trust_consistency = 'UNKNOWN'
            trust_consistent  = True   # not a mismatch, just not recorded
        elif int(saved_trust_raw) == trust:
            trust_consistency = 'YES'
            trust_consistent  = True
        else:
            trust_consistency = 'NO'
            trust_consistent  = False

        report: Dict[str, Any] = {
            'symbol':        symbol,
            'generated_at':  datetime.now(timezone.utc).isoformat(),
            'token_summary': self._token_summary(symbol, profile, debug),
            'model_quality': self._model_quality(debug),

            # Phase-1
            'calibration_forensics':  self._calibration_forensics(debug),
            'architecture_forensics': self._architecture_forensics(debug, profile),
            'score_breakdown': {
                'final_score': score_bd['total'],
                'components':  score_bd,
                'weights': {
                    'expectancy':    _W_EXPECTANCY,
                    'profit_factor': _W_PF,
                    'sharpe':        _W_SHARPE,
                    'precision':     _W_PRECISION,
                    'coverage':      _W_COVERAGE,
                },
            },
            'trust_score_forensics': {
                'trust_score':   trust,
                'saved_trust':   int(saved_trust_raw) if saved_trust_raw is not None else -1,
                'consistent':    trust_consistent,
                'consistency':   trust_consistency,
                'components':    trust_bd,
            },
            'regime_forensics':    self._regime_forensics(debug),
            'disabled_diagnostics': self._disabled_diagnostics(profile, debug) if gate_type == 'DISABLED' else None,

            # Phase-2
            'edge_failure_analysis':      self._edge_failure_analysis(debug),
            'root_cause_classification':  self._root_cause_classification(profile, debug) if gate_type == 'DISABLED' else None,
            'architecture_accounting':    self._architecture_accounting(debug),
            'model_health_score':         self._model_health_score(debug),
            'alpha_attribution':          self._alpha_attribution(profile, debug),

            'grade':        gate_grade(pf),
            'final_verdict': self._final_verdict(profile, trust, trust_consistent),
        }
        return report

    def save_json(self, symbol: str, report: Dict[str, Any]) -> Path:
        path = self.out_dir / f"{symbol.replace('/', '_')}_forensic_report.json"
        with open(path, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
        return path

    def save_txt(self, symbol: str, report: Dict[str, Any]) -> Path:
        path = self.out_dir / f"{symbol.replace('/', '_')}_forensic_report.txt"
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(self._render_txt(report))
        return path

    def print_report(self, report: Dict[str, Any]) -> None:
        txt = self._render_txt(report)
        try:
            print(txt)
        except UnicodeEncodeError:
            print(txt.encode('ascii', errors='replace').decode('ascii'))

    # ── Phase-1 sections ──────────────────────────────────────────────────────

    def _token_summary(self, symbol: str, profile: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
        selected = profile.get('selected_profile', {})
        calib    = selected.get('calibration', {})
        return {
            'symbol':         symbol,
            'timestamp':      profile.get('updated_at', 'unknown'),
            'train_bars':     profile.get('train_bars', 0),
            'eval_bars':      profile.get('eval_bars', 0),
            'gate_selected':  selected.get('gate_type', 'UNKNOWN'),
            'calibration_selected': (
                calib.get('selected_calibrator') or
                calib.get('selected_method') or
                calib.get('method') or
                'uncalibrated'
            ),
            'score': float(selected.get('score', 0.0)),
            'grade': gate_grade(float(profile.get('holdout', {}).get('selected', {}).get('profit_factor', 0.0))),
        }

    def _model_quality(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        candidates  = debug.get('candidates', [])
        n_candidates = len(candidates)
        n_profitable = len([c for c in candidates if float(c.get('score', 0.0) or 0.0) > 0])
        sig = debug.get('signal_diagnostics', {})
        return {
            'summary': f"{n_candidates} architecture candidates evaluated, {n_profitable} profitable",
            'feature_importance_top10': sig.get('feature_importance_top10', []),
            'feature_concentration_hhi': sig.get('feature_concentration_hhi', None),
            'signal_diversity_score':    sig.get('signal_diversity_score', None),
            'top_feature_pct':           sig.get('top_feature_pct', None),
            'top3_features_pct':         sig.get('top3_features_pct', None),
        }

    def _calibration_forensics(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        cands = debug.get('calibration_candidates', [])
        if not cands:
            return {'summary': 'No calibration candidates available', 'methods': []}

        methods = []
        for c in cands:
            methods.append({
                'method':              c.get('method'),
                'ece':                 float(c.get('train_ece', 0.0)),
                'brier':               float(c.get('train_brier', 0.0)),
                'coverage_at_0_5':     float(c.get('holdout_coverage', 0.0)),
                'precision_at_0_5':    float(c.get('holdout_precision', 0.0)),
                'pf_at_0_5':           float(c.get('holdout_profit_factor', 0.0)),
                'expectancy_at_0_5':   float(c.get('holdout_expectancy', 0.0)),
                'sharpe_at_0_5':       float(c.get('holdout_sharpe', 0.0)),
                'variance_retained':   float(c.get('variance_retained_ratio', 0.0)),
                'prob_extreme_frac':   float(c.get('probability_extreme_frac', 0.0)),
                'tech_eligible':       bool(c.get('tech_eligible', c.get('eligible', True))),
                'architecture_viable': bool(c.get('architecture_viable', False)),
                'best_arch_score':     float(c.get('best_architecture_score', float('-inf'))),
                'reason':              c.get('reason', ''),
            })

        methods_sorted = sorted(methods, key=lambda x: -x['best_arch_score'])
        viable = [m for m in methods if m['architecture_viable']]

        return {
            'summary': f"Evaluated {len(cands)} calibration methods; {len(viable)} architecture-viable",
            'note': (
                'Calibrators are evaluated by architecture search outcome (percentile threshold), '
                'NOT by performance at fixed 0.50 threshold.  A calibrator appearing unprofitable '
                'at p>=0.50 can still be viable when architecture search selects the top 15-40% of signals.'
            ),
            'viable_count': len(viable),
            'methods': methods_sorted,
        }

    def _architecture_forensics(self, debug: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        candidates = debug.get('candidates', [])
        if not candidates:
            return {'summary': 'No architecture candidates', 'candidates': []}

        valid    = [c for c in candidates if c.get('precision') is not None]
        accepted = [c for c in valid if str(c.get('reason', '')).lower() in ('accepted', 'candidate') and float(c.get('score', 0.0)) > 0]
        rejected = [c for c in valid if str(c.get('reason', '')) not in ('accepted', 'candidate', '')]

        top = sorted(valid, key=lambda x: float(x.get('score', 0.0)), reverse=True)[:10]

        rejection_reasons: Dict[str, int] = {}
        for r in rejected:
            key = str(r.get('reason', 'unknown'))
            rejection_reasons[key] = rejection_reasons.get(key, 0) + 1

        rows = []
        for c in top:
            pf   = float(c.get('profit_factor', 0.0))
            exp  = float(c.get('expectancy_pct', 0.0))
            shr  = float(c.get('sharpe', 0.0))
            prec = float(c.get('precision', 0.0))
            cov  = float(c.get('coverage', 0.0))
            score = float(c.get('score', 0.0))
            rows.append({
                'gate_type':      c.get('gate_type'),
                'quantile':       float(c.get('quantile', 0.0)),
                'calibration':    c.get('calibration', 'unknown'),
                'precision':      prec,
                'profit_factor':  pf,
                'expectancy_pct': exp,
                'sharpe':         shr,
                'coverage':       cov,
                'score':          score,
                'score_breakdown': compute_score_breakdown(pf, exp, shr, prec, cov),
                'reason':         c.get('reason', ''),
                'grade':          gate_grade(pf),
            })

        return {
            'summary': f"Evaluated {len(candidates)} architectures: {len(accepted)} accepted, {len(rejected)} rejected",
            'total_count':    len(candidates),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'top_candidates': rows,
            'rejection_reasons': dict(sorted(rejection_reasons.items(), key=lambda x: -x[1])),
        }

    def _regime_forensics(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'summary': 'Regime breakdown requires ranking_df — use enrich_with_ranking_df() to populate.',
            'regimes': [],
        }

    def enrich_with_ranking_df(self, report: Dict[str, Any], ranking_df: Any) -> Dict[str, Any]:
        """Enrich regime_forensics with per-regime metrics from ranking_df (pandas DataFrame)."""
        try:
            if 'regime' not in ranking_df.columns:
                return report
            rows = []
            for regime, group in ranking_df.groupby('regime'):
                n = len(group)
                if n == 0:
                    continue
                win_rate = float(group['correct'].mean()) if 'correct' in group.columns else float((group['directional_probability'] >= 0.5).mean())
                rows.append({
                    'regime':   str(regime),
                    'n_trades': n,
                    'win_rate': round(win_rate, 4),
                    'avg_prob': round(float(group['directional_probability'].mean()), 4),
                    'avg_edge': round(float(group['edge_rank'].mean()), 4),
                })
            if rows:
                best  = max(rows, key=lambda x: x['win_rate'])
                worst = min(rows, key=lambda x: x['win_rate'])
                thin  = [r for r in rows if r['n_trades'] <= 5]
                report['regime_forensics'] = {
                    'summary': f"{len(rows)} regimes observed",
                    'best_regime':        best,
                    'worst_regime':       worst,
                    'overfitting_warning': len(thin) > 0,
                    'thin_regimes':       thin,
                    'regimes':            sorted(rows, key=lambda x: -x['win_rate']),
                }
        except Exception as exc:
            report['regime_forensics']['error'] = str(exc)
        return report

    def _disabled_diagnostics(self, profile: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an explicit root-cause tree for every DISABLED gate."""
        causes: List[str] = []
        candidates = debug.get('candidates', [])
        calib      = debug.get('calibration', {})
        baseline   = debug.get('baseline', {})

        if calib.get('arch_validation_no_viable'):
            causes.append(
                'Calibration validation: No calibrator produced a profitable architecture at any percentile threshold'
            )

        if not candidates:
            causes.append('No architecture candidates were generated (data or feature engineering issue)')
        else:
            all_rejected = [c for c in candidates if str(c.get('reason', '')) not in ('accepted', 'candidate', '')]
            if all_rejected:
                reason_counts: Dict[str, int] = {}
                for c in all_rejected:
                    r = str(c.get('reason', 'unknown'))
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
                causes.append(f"All {len(all_rejected)} candidates rejected:")
                for reason, count in top_reasons:
                    causes.append(f"  {reason}: {count} candidate(s)")

        bl_pf  = float(baseline.get('profit_factor', 0.0))
        bl_exp = float(baseline.get('expectancy_pct', 0.0))
        bl_shr = float(baseline.get('sharpe', 0.0))
        if bl_pf < 1.05:
            causes.append(f'Baseline profit factor too low: {bl_pf:.3f} (threshold 1.05)')
        if bl_exp <= 0.0:
            causes.append(f'Baseline expectancy non-positive: {bl_exp:.3f}%')
        if bl_shr <= 0.0:
            causes.append(f'Baseline Sharpe non-positive: {bl_shr:.3f}')

        root_causes_saved = profile.get('selected_profile', {}).get('root_causes', [])
        if root_causes_saved:
            causes += [c for c in root_causes_saved if c not in causes]

        if not causes:
            causes.append('Unknown — possible data quality or feature engineering issue')

        rec = [
            'Review data quality, feature engineering, and ensure at least 15% of signals '
            'have positive expectancy.  Consider extending the evaluation window.'
        ]
        if bl_pf < 1.05:
            rec.insert(0, 'Baseline profit factor is weak; favor a conservative fallback gate or wider ATR barriers instead of disabling the token.')
        return {
            'root_causes': causes,
            'baseline':    baseline,
            'recommendation': ' '.join(rec),
        }

    def _final_verdict(
        self,
        profile: Dict[str, Any],
        trust: int,
        trust_consistent: bool,
    ) -> Dict[str, Any]:
        selected  = profile.get('selected_profile', {})
        holdout   = profile.get('holdout', {}).get('selected', {})
        gate_type = selected.get('gate_type', 'UNKNOWN')
        pf        = float(holdout.get('profit_factor', 0.0))
        exp_pct   = float(holdout.get('expectancy_pct', 0.0))
        grade     = gate_grade(pf)

        if gate_type == 'DISABLED':
            status  = 'DISABLED — no profitable gate found'
            verdict = 'No architecture passed profitability thresholds'
        else:
            status = f'ENABLED — {gate_type}'
            if grade in ('A+', 'A'):
                verdict = 'HIGH CONFIDENCE — excellent profitability metrics'
            elif grade in ('B+', 'B'):
                verdict = 'GOOD — solid profitability'
            elif grade == 'C+':
                verdict = 'MODERATE — above minimum, monitor closely'
            elif grade == 'C':
                verdict = 'MARGINAL — barely profitable, monitor closely'
            elif grade == 'D':
                verdict = 'BORDERLINE — near breakeven, use with caution'
            else:
                verdict = 'WARNING — unprofitable; consider disabling'

        trust_label = 'HIGH' if trust >= 70 else ('MEDIUM' if trust >= 40 else 'LOW')

        return {
            'status':           status,
            'verdict':          verdict,
            'grade':            grade,
            'profit_factor':    pf,
            'expectancy_pct':   exp_pct,
            'trust_score':      trust,
            'trust_label':      trust_label,
            'trust_consistent': trust_consistent,
        }

    # ── Phase-2 sections ──────────────────────────────────────────────────────

    def _edge_failure_analysis(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        """Feature concentration, entropy, diversity, and signal balance diagnostics."""
        sig = debug.get('signal_diagnostics', {})
        if not sig:
            return {
                'available': False,
                'note': 'Signal diagnostics not available — re-run the optimizer to populate this section.',
            }

        ent = float(sig.get('prediction_entropy_mean', 0.0))
        hhi = float(sig.get('feature_concentration_hhi', 0.0))
        div = float(sig.get('signal_diversity_score', 0.0))

        warnings: List[str] = []
        if ent < 0.50:
            warnings.append(f'Low prediction entropy ({ent:.3f}) — model is overconfident; calibration may help')
        if ent > 0.95:
            warnings.append(f'High prediction entropy ({ent:.3f}) — predictions near random; signal quality poor')
        if hhi > 0.30:
            warnings.append(f'High feature concentration HHI={hhi:.3f} — one feature dominates; overfitting risk')
        if div < 0.30:
            warnings.append(f'Low signal diversity score={div:.3f} — portfolio of features very narrow')

        top1 = float(sig.get('top_feature_pct', 0.0))
        if top1 > 40.0:
            warnings.append(f'Top feature contributes {top1:.1f}% of gain — single-feature dependency')

        bsb = float(sig.get('buy_sell_balance', 0.5))
        if bsb > 0.80 or bsb < 0.20:
            warnings.append(f'Extreme buy/sell imbalance: {bsb:.1%} buy — regime-specific edge only')

        return {
            'available':                True,
            'n_signals':                sig.get('n_signals', 0),
            'n_buy':                    sig.get('n_buy', 0),
            'n_sell':                   sig.get('n_sell', 0),
            'buy_sell_balance':         round(bsb, 4),
            'prediction_entropy_mean':  round(ent, 4),
            'prediction_entropy_std':   round(float(sig.get('prediction_entropy_std', 0.0)), 4),
            'dir_prob_mean':            round(float(sig.get('dir_prob_mean', 0.0)), 4),
            'dir_prob_std':             round(float(sig.get('dir_prob_std', 0.0)), 4),
            'dir_prob_skew':            round(float(sig.get('dir_prob_skew', 0.0)), 4),
            'edge_rank_mean':           round(float(sig.get('edge_rank_mean', 0.0)), 4),
            'edge_rank_std':            round(float(sig.get('edge_rank_std', 0.0)), 4),
            'signal_diversity_score':   round(div, 4),
            'feature_concentration_hhi': round(hhi, 4),
            'top_feature_pct':          round(top1, 2),
            'top3_features_pct':        round(float(sig.get('top3_features_pct', 0.0)), 2),
            'feature_importance_top10': sig.get('feature_importance_top10', []),
            'warnings':                 warnings,
        }

    def _root_cause_classification(
        self,
        profile: Dict[str, Any],
        debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Classify a DISABLED gate into one of 8 root cause categories with confidence scores."""
        candidates  = debug.get('candidates', [])
        baseline    = debug.get('baseline', {})
        calib       = debug.get('calibration', {})
        calib_cands = debug.get('calibration_candidates', [])
        sig         = debug.get('signal_diagnostics', {})

        scores:  Dict[str, float] = {k: 0.0 for k in ROOT_CAUSE_LABELS}
        evidence: Dict[str, List[str]] = {k: [] for k in ROOT_CAUSE_LABELS}

        bl_pf  = float(baseline.get('profit_factor', 0.0))
        bl_exp = float(baseline.get('expectancy_pct', 0.0))
        bl_cov = float(baseline.get('coverage', 0.0))

        # ── NO_ALPHA ──────────────────────────────────────────────────────────
        if bl_pf < 0.90 and bl_exp < -5.0:
            scores['NO_ALPHA'] += 0.5
            evidence['NO_ALPHA'].append(
                f'Baseline PF={bl_pf:.3f}, Exp={bl_exp:.2f}% — raw signal quality critically poor'
            )
        all_exp = [float(c.get('expectancy_pct', 0.0)) for c in candidates]
        if candidates and all(e <= 0 for e in all_exp):
            scores['NO_ALPHA'] += 0.4
            evidence['NO_ALPHA'].append('Every architecture candidate has non-positive expectancy')

        # ── WEAK_ALPHA ────────────────────────────────────────────────────────
        best_pf = max((float(c.get('profit_factor', 0.0)) for c in candidates), default=0.0)
        if 0.90 <= best_pf < 1.10:
            scores['WEAK_ALPHA'] += 0.5
            evidence['WEAK_ALPHA'].append(
                f'Best candidate PF={best_pf:.3f} — marginally below profitability threshold'
            )
        if 0.90 <= bl_pf < 1.05:
            scores['WEAK_ALPHA'] += 0.2
            evidence['WEAK_ALPHA'].append(
                f'Baseline PF={bl_pf:.3f} — near-breakeven; insufficient edge to survive gate filters'
            )

        # ── COVERAGE_COLLAPSE ─────────────────────────────────────────────────
        cov_rejected = [c for c in candidates if 'coverage' in str(c.get('reason', '')) or 'low_signals' in str(c.get('reason', ''))]
        if candidates and len(cov_rejected) / len(candidates) > 0.50:
            scores['COVERAGE_COLLAPSE'] += 0.6
            evidence['COVERAGE_COLLAPSE'].append(
                f'{len(cov_rejected)}/{len(candidates)} candidates rejected for insufficient coverage'
            )
        if bl_cov < 0.20:
            scores['COVERAGE_COLLAPSE'] += 0.3
            evidence['COVERAGE_COLLAPSE'].append(
                f'Baseline coverage={bl_cov:.3f} — signals too rare to form a reliable gate'
            )

        # ── CALIBRATION_COLLAPSE ──────────────────────────────────────────────
        if calib_cands and all(not c.get('tech_eligible', True) for c in calib_cands):
            scores['CALIBRATION_COLLAPSE'] += 0.7
            evidence['CALIBRATION_COLLAPSE'].append(
                'All calibration candidates technically ineligible (probability collapse or variance destruction)'
            )
        elif calib.get('arch_validation_no_viable'):
            scores['CALIBRATION_COLLAPSE'] += 0.3
            evidence['CALIBRATION_COLLAPSE'].append(
                'No calibrator produced a viable architecture at any percentile threshold'
            )

        # ── OVERFITTING ───────────────────────────────────────────────────────
        train_prec   = [float(c.get('train_precision', 0.0)) for c in calib_cands if c.get('train_precision')]
        holdout_prec = [float(c.get('holdout_precision', 0.0)) for c in calib_cands if c.get('holdout_precision')]
        if train_prec and holdout_prec:
            avg_train   = float(np.mean(train_prec))
            avg_holdout = float(np.mean(holdout_prec))
            if avg_train - avg_holdout > 0.10:
                scores['OVERFITTING'] += 0.5
                evidence['OVERFITTING'].append(
                    f'Train precision {avg_train:.3f} >> holdout precision {avg_holdout:.3f} — '
                    'train/holdout divergence suggests overfitting or data leak'
                )

        # ── FEATURE_DOMINANCE ─────────────────────────────────────────────────
        hhi  = float(sig.get('feature_concentration_hhi', 0.0)) if sig else 0.0
        top1 = float(sig.get('top_feature_pct', 0.0)) if sig else 0.0
        if hhi > 0.30:
            scores['FEATURE_DOMINANCE'] += 0.4
            evidence['FEATURE_DOMINANCE'].append(
                f'Feature concentration HHI={hhi:.3f} (>0.30 indicates single-feature dominance)'
            )
        if top1 > 40.0:
            scores['FEATURE_DOMINANCE'] += 0.3
            evidence['FEATURE_DOMINANCE'].append(
                f'Top feature contributes {top1:.1f}% of total gain — brittle single-feature dependency'
            )

        # ── REGIME_INSTABILITY ────────────────────────────────────────────────
        high_pf_cands = [c for c in candidates if float(c.get('profit_factor', 0.0)) >= 1.10]
        if high_pf_cands and len(high_pf_cands) < len(candidates) * 0.15:
            scores['REGIME_INSTABILITY'] += 0.4
            evidence['REGIME_INSTABILITY'].append(
                f'{len(high_pf_cands)} candidates reached PF>=1.10 but failed global consistency — '
                'edge may be regime-specific and unstable out-of-sample'
            )

        # ── DATA_QUALITY_FAILURE ──────────────────────────────────────────────
        n_signals = int(sig.get('n_signals', 0)) if sig else 0
        if n_signals > 0 and n_signals < 200:
            scores['DATA_QUALITY_FAILURE'] += 0.5
            evidence['DATA_QUALITY_FAILURE'].append(
                f'Only {n_signals} signals in eval window (minimum 200 recommended for stable statistics)'
            )
        if not candidates:
            scores['DATA_QUALITY_FAILURE'] += 0.4
            evidence['DATA_QUALITY_FAILURE'].append(
                'No architecture candidates generated — possible data gap or feature engineering failure'
            )

        # ── Normalise and rank
        total_score = sum(scores.values()) + 1e-9
        normalised  = {k: round(v / total_score, 3) for k, v in scores.items()}
        ranked      = sorted(normalised.items(), key=lambda x: -x[1])
        primary     = ranked[0][0]

        return {
            'primary_cause':   primary,
            'confidence':      normalised[primary],
            'ranked_causes':   [{'cause': k, 'confidence': v} for k, v in ranked if v > 0.02],
            'evidence':        {k: v for k, v in evidence.items() if v},
            'all_scores':      normalised,
        }

    def _architecture_accounting(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        """Strict Total = Accepted + Coverage_rejected + Quality_rejected + Duplicate + Other."""
        candidates = debug.get('candidates', [])
        total = len(candidates)

        accepted: List[Dict] = []
        coverage_rejected: List[Dict] = []
        quality_rejected: List[Dict] = []
        duplicate_skipped: List[Dict] = []
        other_rejected: List[Dict] = []

        _QUALITY_REASONS = {
            'nonpositive_expectancy', 'nonpositive_sharpe', 'pf_below_1.10',
            'coverage_below_minimum', 'low_precision',
        }
        _COVERAGE_KEYWORDS = ('coverage', 'low_signals')

        for c in candidates:
            reason = str(c.get('reason', ''))
            score  = float(c.get('score', 0.0) or 0.0)

            if reason in ('accepted', 'candidate') and score > 0:
                accepted.append(c)
            elif reason.startswith('duplicate'):
                duplicate_skipped.append(c)
            elif any(kw in reason for kw in _COVERAGE_KEYWORDS):
                coverage_rejected.append(c)
            elif reason in _QUALITY_REASONS or reason.startswith('pf_') or reason.startswith('nonpositive'):
                quality_rejected.append(c)
            elif reason:
                other_rejected.append(c)
            else:
                # Empty reason + score <= 0 → quality failure
                quality_rejected.append(c)

        accounted   = len(accepted) + len(coverage_rejected) + len(quality_rejected) + len(duplicate_skipped) + len(other_rejected)
        unaccounted = total - accounted

        rejection_by_reason: Dict[str, int] = {}
        for c in coverage_rejected + quality_rejected + duplicate_skipped + other_rejected:
            r = str(c.get('reason', 'unknown'))
            rejection_by_reason[r] = rejection_by_reason.get(r, 0) + 1

        return {
            'total_evaluated':   total,
            'accepted':          len(accepted),
            'coverage_rejected': len(coverage_rejected),
            'quality_rejected':  len(quality_rejected),
            'duplicate_skipped': len(duplicate_skipped),
            'other_rejected':    len(other_rejected),
            'unaccounted':       unaccounted,
            'accounting_ok':     unaccounted == 0,
            'accounting_check':  (
                f"Total={total} = Accepted({len(accepted)}) + "
                f"CoverageRej({len(coverage_rejected)}) + QualityRej({len(quality_rejected)}) + "
                f"Duplicate({len(duplicate_skipped)}) + Other({len(other_rejected)})"
                + (f" + UNACCOUNTED({unaccounted})" if unaccounted else "")
            ),
            'rejection_by_reason': dict(sorted(rejection_by_reason.items(), key=lambda x: -x[1])),
        }

    def _model_health_score(self, debug: Dict[str, Any]) -> Dict[str, Any]:
        """0-100 composite health score across 5 sub-components."""
        sig         = debug.get('signal_diagnostics', {})
        calib_cands = debug.get('calibration_candidates', [])
        candidates  = debug.get('candidates', [])
        rf          = debug.get('regime_forensics', {})

        components: Dict[str, float] = {}

        # Feature diversity (0-30): based on signal_diversity_score (0-1)
        diversity = float(sig.get('signal_diversity_score', 0.5)) if sig else 0.5
        components['feature_diversity'] = round(30.0 * diversity, 1)

        # Prediction entropy (0-20): ideal = 0.60-0.95 (clear signal, not random)
        ent = float(sig.get('prediction_entropy_mean', 0.80)) if sig else 0.80
        if 0.60 <= ent <= 0.95:
            ent_score = 1.0
        elif ent < 0.60:
            ent_score = ent / 0.60  # over-confident model
        else:
            ent_score = max(0.0, 1.0 - (ent - 0.95) / 0.05)  # near-random model
        components['prediction_entropy'] = round(20.0 * ent_score, 1)

        # Calibration quality (0-25): min ECE and Brier across candidates
        if calib_cands:
            best_ece   = min(float(c.get('train_ece', 1.0)) for c in calib_cands)
            best_brier = min(float(c.get('train_brier', 1.0)) for c in calib_cands)
            ece_score   = max(0.0, 1.0 - 2.0 * best_ece)        # 0 = perfect, 0.5 = terrible
            brier_score = max(0.0, 1.0 - best_brier / 0.25)     # random baseline = 0.25
            calib_score = (ece_score + brier_score) / 2.0
        else:
            calib_score = 0.5
        components['calibration_quality'] = round(25.0 * calib_score, 1)

        # Architecture robustness (0-15): n_viable / 10 (10+ profitable = full score)
        n_viable = len([c for c in candidates if float(c.get('score', 0.0) or 0.0) > 0])
        components['architecture_robustness'] = round(15.0 * min(1.0, n_viable / 10.0), 1)

        # Regime stability (0-10): inferred from regime_forensics
        if rf.get('overfitting_warning'):
            regime_score = 0.3
        elif rf.get('regimes'):
            regime_score = 0.8
        else:
            regime_score = 0.5
        components['regime_stability'] = round(10.0 * regime_score, 1)

        total = sum(components.values())

        return {
            'total': int(round(total)),
            'max':   100,
            'components': components,
            'interpretation': (
                'EXCELLENT' if total >= 80 else
                'GOOD'      if total >= 60 else
                'MODERATE'  if total >= 40 else
                'POOR'      if total >= 20 else
                'CRITICAL'
            ),
        }

    def _alpha_attribution(
        self,
        profile: Dict[str, Any],
        debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Decompose PF lift into: signal quality %, calibration %, gate architecture %."""
        baseline    = debug.get('baseline', {})
        holdout     = profile.get('holdout', {}).get('selected', {})
        calib_cands = debug.get('calibration_candidates', [])

        bl_pf    = float(baseline.get('profit_factor', 0.0))
        final_pf = float(holdout.get('profit_factor', 0.0))

        # Neutral PF is ~0.90 (near-random after fees for binary classification)
        neutral_pf    = 0.90
        total_lift    = max(0.0, final_pf - neutral_pf)
        raw_sig_lift  = max(0.0, bl_pf - neutral_pf)

        # Calibration lift: best calibrated PF at 0.50 threshold minus uncalibrated
        uncal      = next((c for c in calib_cands if c.get('method') == 'uncalibrated'), None)
        best_calib = (
            max(calib_cands, key=lambda x: float(x.get('holdout_profit_factor', 0.0)))
            if calib_cands else None
        )
        if uncal and best_calib and best_calib.get('method') != 'uncalibrated':
            calib_lift = max(
                0.0,
                float(best_calib.get('holdout_profit_factor', 0.0)) -
                float(uncal.get('holdout_profit_factor', 0.0))
            )
        else:
            calib_lift = 0.0

        gate_lift = max(0.0, total_lift - raw_sig_lift - calib_lift)
        denom = raw_sig_lift + calib_lift + gate_lift + 1e-9

        return {
            'baseline_pf':           round(bl_pf, 3),
            'final_pf':              round(final_pf, 3),
            'neutral_reference_pf':  neutral_pf,
            'total_lift':            round(total_lift, 3),
            'signal_quality_lift':   round(raw_sig_lift, 3),
            'calibration_lift':      round(calib_lift, 3),
            'gate_architecture_lift': round(gate_lift, 3),
            'signal_quality_pct':    round(100.0 * raw_sig_lift / denom, 1),
            'calibration_pct':       round(100.0 * calib_lift  / denom, 1),
            'gate_architecture_pct': round(100.0 * gate_lift   / denom, 1),
            'explanation': (
                f"PF lift from neutral {neutral_pf:.2f} to {final_pf:.3f}: "
                f"raw signal={100.0*raw_sig_lift/denom:.0f}%, "
                f"calibration={100.0*calib_lift/denom:.0f}%, "
                f"gate architecture={100.0*gate_lift/denom:.0f}%"
            ),
        }

    # ── text renderer ─────────────────────────────────────────────────────────

    def _render_txt(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        W = 80

        def hr(char: str = '=') -> None:
            lines.append(char * W)

        def section(title: str) -> None:
            lines.append('')
            lines.append(f'[{title}]')

        symbol = report['symbol']
        grade  = report.get('grade', '?')
        hr()
        lines.append(f'FORENSIC REPORT: {symbol} | GRADE: {grade}')
        hr()

        # ── Token Summary ─────────────────────────────────────────────────────
        ts = report.get('token_summary', {})
        section('TOKEN SUMMARY')
        lines.append(f'  Symbol        : {ts.get("symbol")}')
        lines.append(f'  Timestamp     : {ts.get("timestamp")}')
        lines.append(f'  Train bars    : {ts.get("train_bars")}')
        lines.append(f'  Eval bars     : {ts.get("eval_bars")}')
        lines.append(f'  Gate selected : {ts.get("gate_selected")}')
        lines.append(f'  Calibration   : {ts.get("calibration_selected")}')
        lines.append(f'  Score         : {ts.get("score", 0.0):.4f}')
        lines.append(f'  Grade         : {ts.get("grade")}')

        # ── Edge Failure Analysis (Phase-2) ───────────────────────────────────
        efa = report.get('edge_failure_analysis', {})
        section('EDGE FAILURE ANALYSIS')
        if not efa.get('available'):
            lines.append(f'  {efa.get("note", "Not available")}')
        else:
            lines.append(f'  Signals       : {efa.get("n_signals")}  (buy={efa.get("n_buy")}  sell={efa.get("n_sell")}  balance={efa.get("buy_sell_balance", 0):.1%})')
            lines.append(f'  Pred entropy  : mean={efa.get("prediction_entropy_mean", 0):.4f}  std={efa.get("prediction_entropy_std", 0):.4f}')
            lines.append(f'  Dir prob      : mean={efa.get("dir_prob_mean", 0):.4f}  std={efa.get("dir_prob_std", 0):.4f}  skew={efa.get("dir_prob_skew", 0):+.4f}')
            lines.append(f'  Edge rank     : mean={efa.get("edge_rank_mean", 0):.4f}  std={efa.get("edge_rank_std", 0):.4f}')
            lines.append(f'  Diversity     : score={efa.get("signal_diversity_score", 0):.4f}  HHI={efa.get("feature_concentration_hhi", 0):.4f}')
            lines.append(f'  Top feature   : {efa.get("top_feature_pct", 0):.1f}%  top3={efa.get("top3_features_pct", 0):.1f}%')
            top10 = efa.get('feature_importance_top10', [])
            if top10:
                lines.append('')
                lines.append(f'  {"Feature":<35} {"Gain%":>6}')
                lines.append(f'  {"-"*43}')
                for item in top10[:10]:
                    lines.append(f'  {str(item.get("feature","")):<35} {item.get("pct", 0):>6.2f}%')
            warns = efa.get('warnings', [])
            if warns:
                lines.append('')
                for w in warns:
                    lines.append(f'  *** {w}')

        # ── Calibration ───────────────────────────────────────────────────────
        cal = report.get('calibration_forensics', {})
        section('CALIBRATION FORENSICS')
        lines.append(f'  {cal.get("summary", "")}')
        note = cal.get('note', '')
        if note:
            for ln in textwrap.wrap(note, width=W - 4):
                lines.append(f'  {ln}')
        lines.append('')
        lines.append(f'  {"Method":<12} | {"ECE":>6} | {"Brier":>6} | {"Cov@0.5":>7} | {"PF@0.5":>7} | {"Exp@0.5":>8} | {"TechOK":>6} | {"ArchOK":>6} | Reason')
        lines.append(f'  {"-"*95}')
        for m in cal.get('methods', []):
            arch_ok = 'YES' if m.get('architecture_viable') else 'NO'
            tech_ok = 'YES' if m.get('tech_eligible') else 'NO'
            lines.append(
                f'  {str(m.get("method","")):<12} | {m.get("ece",0):>6.4f} | {m.get("brier",0):>6.4f} | '
                f'{m.get("coverage_at_0_5",0):>7.3f} | {m.get("pf_at_0_5",0):>7.3f} | '
                f'{m.get("expectancy_at_0_5",0):>8.2f} | {tech_ok:>6} | {arch_ok:>6} | {m.get("reason","")}'
            )

        # ── Architecture ──────────────────────────────────────────────────────
        arch = report.get('architecture_forensics', {})
        section('ARCHITECTURE FORENSICS')
        lines.append(f'  {arch.get("summary", "")}')
        if arch.get('rejection_reasons'):
            lines.append('  Rejection reasons:')
            for reason, count in arch['rejection_reasons'].items():
                lines.append(f'    {reason:<40} {count:>4} candidate(s)')
        lines.append('')
        lines.append(f'  {"Gate":<24} | {"Q":>5} | {"PF":>6} | {"Exp%":>7} | {"Sharpe":>6} | {"Cov":>5} | {"Score":>6} | Grade | Reason')
        lines.append(f'  {"-"*95}')
        for c in arch.get('top_candidates', []):
            reason = str(c.get('reason', ''))[:20]
            lines.append(
                f'  {str(c.get("gate_type","")):<24} | {c.get("quantile",0):>5.2f} | '
                f'{c.get("profit_factor",0):>6.3f} | {c.get("expectancy_pct",0):>7.2f} | '
                f'{c.get("sharpe",0):>6.2f} | {c.get("coverage",0):>5.3f} | '
                f'{c.get("score",0):>6.4f} | {c.get("grade","?"):>5} | {reason}'
            )

        # ── Architecture Accounting (Phase-2) ─────────────────────────────────
        aa = report.get('architecture_accounting', {})
        section('ARCHITECTURE ACCOUNTING')
        lines.append(f'  {aa.get("accounting_check", "")}')
        lines.append(f'  Accounting OK : {"YES" if aa.get("accounting_ok") else "NO *** UNACCOUNTED ROWS DETECTED ***"}')
        if aa.get('rejection_by_reason'):
            lines.append('  Rejection breakdown:')
            for r, cnt in aa['rejection_by_reason'].items():
                lines.append(f'    {r:<45} {cnt:>4}')

        # ── Score Breakdown ───────────────────────────────────────────────────
        sb = report.get('score_breakdown', {})
        section('SCORE BREAKDOWN')
        comps = sb.get('components', {})
        lines.append(f'  Final score = {sb.get("final_score", 0.0):.4f}')
        lines.append('')
        lines.append(f'  {"Component":<20} {"Weight":>7}   {"Contribution":>12}')
        lines.append(f'  {"-"*43}')
        for key in ('expectancy', 'profit_factor', 'sharpe', 'precision', 'coverage'):
            w   = sb.get('weights', {}).get(key, 0.0)
            val = comps.get(key, 0.0)
            lines.append(f'  {key:<20} {w:>7.2f}   {val:>+12.6f}')
        lines.append(f'  {"-"*43}')
        lines.append(f'  {"TOTAL":<20} {"1.00":>7}   {comps.get("total", 0.0):>+12.6f}')
        computed = sum(comps.get(k, 0.0) for k in ('expectancy', 'profit_factor', 'sharpe', 'precision', 'coverage'))
        if abs(computed - comps.get('total', 0.0)) > 1e-4:
            lines.append(f'  *** WARNING: components sum to {computed:.6f} != {comps.get("total",0.0):.6f}')

        # ── Trust Score ───────────────────────────────────────────────────────
        tf = report.get('trust_score_forensics', {})
        section('TRUST SCORE FORENSICS')
        lines.append(f'  Trust score : {tf.get("trust_score", 0)}/100')
        lines.append(f'  Saved value : {tf.get("saved_trust", -1)}')
        consistency = tf.get('consistency', 'UNKNOWN')
        if consistency == 'YES':
            cons_label = 'YES'
        elif consistency == 'NO':
            cons_label = 'NO  *** MISMATCH ***'
        else:
            cons_label = 'UNKNOWN (not recorded in profile)'
        lines.append(f'  Consistent  : {cons_label}')
        bd = tf.get('components', {})
        lines.append('')
        lines.append(f'  {"Component":<30} {"Points":>8}')
        lines.append(f'  {"-"*40}')
        for key in ('pf_contribution', 'expectancy_contribution', 'sharpe_contribution', 'coverage_contribution', 'fires_contribution'):
            lines.append(f'  {key:<30} {bd.get(key, 0.0):>8.2f}')
        lines.append(f'  {"-"*40}')
        lines.append(f'  {"TOTAL":<30} {bd.get("total", 0):>8}')

        # ── Model Health Score (Phase-2) ───────────────────────────────────────
        mhs = report.get('model_health_score', {})
        section('MODEL HEALTH SCORE')
        lines.append(f'  Total : {mhs.get("total", 0)}/{mhs.get("max", 100)} — {mhs.get("interpretation", "?")}')
        lines.append('')
        comp_order = ('feature_diversity', 'prediction_entropy', 'calibration_quality', 'architecture_robustness', 'regime_stability')
        comp_max   = (30, 20, 25, 15, 10)
        lines.append(f'  {"Component":<26} {"Score":>6} {"Max":>4}')
        lines.append(f'  {"-"*38}')
        for cname, cmax in zip(comp_order, comp_max):
            val = mhs.get('components', {}).get(cname, 0.0)
            bar = '#' * int(val / cmax * 10) if cmax > 0 else ''
            lines.append(f'  {cname:<26} {val:>6.1f} /{cmax:>3}  {bar}')

        # ── Regime Forensics ──────────────────────────────────────────────────
        rf = report.get('regime_forensics', {})
        section('REGIME FORENSICS')
        lines.append(f'  {rf.get("summary", "")}')
        if rf.get('best_regime'):
            b = rf['best_regime']
            lines.append(f'  Best  regime: {b["regime"]:20}  n={b["n_trades"]:4}  win={b["win_rate"]:.3f}  avg_edge={b["avg_edge"]:.3f}')
        if rf.get('worst_regime'):
            w = rf['worst_regime']
            lines.append(f'  Worst regime: {w["regime"]:20}  n={w["n_trades"]:4}  win={w["win_rate"]:.3f}  avg_edge={w["avg_edge"]:.3f}')
        if rf.get('overfitting_warning'):
            thin = rf.get('thin_regimes', [])
            lines.append(f'  *** OVERFITTING WARNING: {len(thin)} regime(s) have <=5 trades — statistics unreliable')
        if rf.get('regimes'):
            lines.append('')
            lines.append(f'  {"Regime":<22} | {"n":>5} | {"WinRate":>7} | {"AvgProb":>7} | {"AvgEdge":>7}')
            lines.append(f'  {"-"*60}')
            for r in rf['regimes']:
                lines.append(
                    f'  {str(r["regime"]):<22} | {r["n_trades"]:>5} | {r["win_rate"]:>7.3f} | '
                    f'{r["avg_prob"]:>7.3f} | {r["avg_edge"]:>7.3f}'
                )

        # ── Alpha Attribution (Phase-2) ────────────────────────────────────────
        aa2 = report.get('alpha_attribution', {})
        section('ALPHA ATTRIBUTION')
        lines.append(f'  {aa2.get("explanation", "Not available")}')
        lines.append('')
        lines.append(f'  Baseline PF    : {aa2.get("baseline_pf", 0):.3f}')
        lines.append(f'  Final PF       : {aa2.get("final_pf", 0):.3f}')
        lines.append(f'  Total lift     : +{aa2.get("total_lift", 0):.3f} (from neutral {aa2.get("neutral_reference_pf", 0.90):.2f})')
        lines.append('')
        lines.append(f'  {"Source":<28} {"Lift":>7}  {"Share":>7}')
        lines.append(f'  {"-"*45}')
        lines.append(f'  {"Signal quality":<28} {aa2.get("signal_quality_lift",0):>+7.3f}  {aa2.get("signal_quality_pct",0):>6.1f}%')
        lines.append(f'  {"Calibration":<28} {aa2.get("calibration_lift",0):>+7.3f}  {aa2.get("calibration_pct",0):>6.1f}%')
        lines.append(f'  {"Gate architecture":<28} {aa2.get("gate_architecture_lift",0):>+7.3f}  {aa2.get("gate_architecture_pct",0):>6.1f}%')

        # ── Disabled Gate diagnostics ─────────────────────────────────────────
        dd = report.get('disabled_diagnostics')
        rcc = report.get('root_cause_classification')
        if dd:
            section('DISABLED GATE — ROOT CAUSE ANALYSIS')
            for cause in dd.get('root_causes', []):
                lines.append(f'  ├─ {cause}')
            lines.append(f'  └─ RECOMMENDATION: {dd.get("recommendation", "")}')

        if rcc:
            section('ROOT CAUSE CLASSIFICATION')
            lines.append(f'  Primary cause : {rcc.get("primary_cause")} (confidence={rcc.get("confidence", 0):.1%})')
            lines.append('')
            lines.append(f'  {"Cause":<30} {"Confidence":>11}')
            lines.append(f'  {"-"*43}')
            for item in rcc.get('ranked_causes', []):
                lines.append(f'  {item["cause"]:<30} {item["confidence"]:>10.1%}')
            evidence = rcc.get('evidence', {})
            if evidence:
                lines.append('')
                lines.append('  Evidence:')
                for cause, evs in evidence.items():
                    for ev in evs:
                        lines.append(f'    [{cause}] {ev}')

        # ── Final Verdict ─────────────────────────────────────────────────────
        fv = report.get('final_verdict', {})
        section('FINAL VERDICT')
        lines.append(f'  Status  : {fv.get("status")}')
        lines.append(f'  Verdict : {fv.get("verdict")}')
        lines.append(f'  Grade   : {fv.get("grade")}  (PF={fv.get("profit_factor", 0.0):.3f}  Exp={fv.get("expectancy_pct", 0.0):.2f}%)')
        lines.append(f'  Trust   : {fv.get("trust_score")}/100 — {fv.get("trust_label")}')
        if not fv.get('trust_consistent'):
            lines.append('  *** TRUST SCORE MISMATCH between gate selection and forensic report')

        lines.append('')
        hr()
        return '\n'.join(lines) + '\n'


# ── CLI helper ────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    import sys
    root = Path(__file__).resolve().parent.parent

    if len(sys.argv) < 2:
        print('Usage: gate_forensics.py <symbol>')
        print('Example: gate_forensics.py BTC/USDT')
        sys.exit(1)

    symbol = sys.argv[1]
    profile_dir = root / 'data' / 'meta_gate_profiles'
    profile_path = profile_dir / f"{symbol.replace('/', '_')}_gate.json"
    debug_path   = profile_dir / 'debug' / f"{symbol.replace('/', '_')}_gate_debug.json"

    if not profile_path.exists():
        print(f'Profile not found: {profile_path}')
        sys.exit(1)

    with open(profile_path) as fh:
        profile = json.load(fh)
    debug_data: Dict[str, Any] = {}
    if debug_path.exists():
        with open(debug_path) as fh:
            debug_data = json.load(fh)

    reporter = GateForensicsReporter(root)
    report   = reporter.generate_report(symbol, profile, debug_data)
    reporter.print_report(report)
    json_path = reporter.save_json(symbol, report)
    txt_path  = reporter.save_txt(symbol, report)
    print(f'JSON report: {json_path}')
    print(f'TXT  report: {txt_path}')


if __name__ == '__main__':
    _cli_main()
