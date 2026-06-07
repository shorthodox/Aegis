#!/usr/bin/env python3
"""
forensic_gate_report.py — Generate comprehensive forensic reports for gate profiles
===================================================================================
BUG 6: Creates detailed forensic diagnostics with:
- Calibration analysis per method
- Architecture evaluation per calibrator
- Regime performance breakdown
- Feature importance diagnostics
- Token grading system
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ForensicGateReporter:
    """Generate comprehensive forensic reports for meta gate profiles."""
    
    def __init__(self, profile_dir: Path):
        """Initialize reporter with profile directory."""
        self.profile_dir = profile_dir
        self.debug_dir = profile_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, symbol: str, profile: Dict[str, Any], debug_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate complete forensic report for a symbol."""
        report = {
            'symbol': symbol,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'calibration_forensics': self._calibration_forensics(debug_data or {}),
            'architecture_forensics': self._architecture_forensics(debug_data or {}, profile),
            'regime_forensics': self._regime_forensics(debug_data or {}),
            'token_grade': self._compute_token_grade(profile),
            'final_verdict': self._final_verdict(profile, debug_data or {}),
        }
        return report
    
    def _calibration_forensics(self, debug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze calibrator selection and performance."""
        calib_cands = debug_data.get('calibration_candidates', [])
        
        if not calib_cands:
            return {
                'summary': 'No calibration candidates available',
                'methods': [],
            }
        
        methods = []
        for cand in calib_cands:
            method_report = {
                'method': cand.get('method'),
                'coverage': float(cand.get('holdout_coverage', 0.0)),
                'precision': float(cand.get('holdout_precision', 0.0)),
                'profit_factor': float(cand.get('holdout_profit_factor', 0.0)),
                'expectancy': float(cand.get('holdout_expectancy', 0.0)),
                'sharpe': float(cand.get('holdout_sharpe', 0.0)),
                'ece': float(cand.get('train_ece', 0.0)),
                'brier': float(cand.get('train_brier', 0.0)),
                'variance_retained': float(cand.get('variance_retained_ratio', 0.0)),
                'expectancy_retention': float(cand.get('expectancy_retention', 0.0)),
                'pf_retention': float(cand.get('pf_retention', 0.0)),
                'precision_retention': float(cand.get('precision_retention', 0.0)),
                'eligible': bool(cand.get('eligible', False)),
                'reason': cand.get('reason', ''),
                'architecture_score': float(cand.get('best_architecture_score', -np.inf)),
            }
            methods.append(method_report)
        
        # Sort by architecture score
        methods_sorted = sorted(methods, key=lambda x: -x['architecture_score'])
        
        return {
            'summary': f"Evaluated {len(calib_cands)} calibration methods",
            'eligible_count': len([m for m in methods if m['eligible']]),
            'methods': methods_sorted[:10],  # Top 10
            'top_method': methods_sorted[0]['method'] if methods_sorted else None,
            'top_score': methods_sorted[0]['architecture_score'] if methods_sorted else None,
        }
    
    def _architecture_forensics(self, debug_data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architecture selection and alternatives."""
        candidates = debug_data.get('candidates', [])
        
        if not candidates:
            return {
                'summary': 'No architecture candidates',
                'candidates': [],
            }
        
        # Filter valid candidates
        valid = [c for c in candidates if c.get('precision', 0.0) > 0.0 and 'reason' in c]
        accepted = [c for c in valid if c['reason'] == 'accepted']
        rejected = [c for c in valid if c['reason'] not in ('accepted', 'candidate')]
        
        # Top performers
        top_candidates = sorted(valid, key=lambda x: float(x.get('score', 0.0)), reverse=True)[:5]
        
        return {
            'summary': f"Evaluated {len(candidates)} architectures: {len(accepted)} accepted, {len(rejected)} rejected",
            'total_count': len(candidates),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'top_candidates': [
                {
                    'gate_type': c.get('gate_type'),
                    'quantile': float(c.get('quantile', 0.0)),
                    'precision': float(c.get('precision', 0.0)),
                    'profit_factor': float(c.get('profit_factor', 0.0)),
                    'expectancy': float(c.get('expectancy_pct', 0.0)),
                    'sharpe': float(c.get('sharpe', 0.0)),
                    'score': float(c.get('score', 0.0)),
                }
                for c in top_candidates
            ],
            'rejection_reasons': self._rejection_reason_summary(rejected),
        }
    
    def _rejection_reason_summary(self, rejected: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize rejection reasons."""
        reasons = {}
        for r in rejected:
            reason = r.get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        return dict(sorted(reasons.items(), key=lambda x: -x[1]))
    
    def _regime_forensics(self, debug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-level performance (placeholder)."""
        return {
            'summary': 'Regime analysis not yet available',
            'regimes': [],
        }
    
    def _compute_token_grade(self, profile: Dict[str, Any]) -> str:
        """Grade gate by profit factor: A+ > 1.80, A > 1.60, B+ > 1.40, B > 1.30,
        C+ > 1.20, C > 1.10, D > 1.00, F <= 1.00.  DISABLED gates are always F."""
        if profile.get('selected_profile', {}).get('gate_type') == 'DISABLED':
            return 'F'
        metrics = profile.get('holdout', {}).get('selected', {})
        pf = float(metrics.get('profit_factor', 0.0))
        if pf > 1.80:
            return 'A+'
        if pf > 1.60:
            return 'A'
        if pf > 1.40:
            return 'B+'
        if pf > 1.30:
            return 'B'
        if pf > 1.20:
            return 'C+'
        if pf > 1.10:
            return 'C'
        if pf > 1.00:
            return 'D'
        return 'F'
    
    def _final_verdict(self, profile: Dict[str, Any], debug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verdict and recommendations."""
        gate_type = profile.get('selected_profile', {}).get('gate_type')
        metrics = profile.get('holdout', {}).get('selected', {})
        pf = float(metrics.get('profit_factor', 0.0))
        exp = float(metrics.get('expectancy_pct', 0.0))
        # trust_score is stored directly on selected_profile (set by _compute_trust_score)
        trust = float(profile.get('selected_profile', {}).get('trust_score', 0))
        
        verdict = []
        
        if gate_type == 'DISABLED':
            verdict.append(f"Status: DISABLED - No profitable gate found")
            if debug_data.get('calibration', {}).get('calibration_failed'):
                verdict.append("Cause: Calibration failure")
            else:
                verdict.append("Cause: No architecture passed profitability thresholds")
            verdict.append("Recommendation: Review data quality and feature engineering")
        else:
            verdict.append(f"Status: ENABLED - {gate_type} selected")
            
            if pf >= 1.50 and exp >= 10.0:
                verdict.append("Verdict: HIGH CONFIDENCE - Excellent profitability metrics")
            elif pf >= 1.20 and exp >= 5.0:
                verdict.append("Verdict: GOOD - Solid profitability")
            elif pf >= 1.10 and exp >= 1.0:
                verdict.append("Verdict: MARGINAL - Barely profitable, monitor closely")
            else:
                verdict.append("Verdict: WARNING - Low profitability, consider disabling")
            
            if trust >= 70:
                verdict.append(f"Trust score: {trust:.0f}/100 - HIGH")
            elif trust >= 50:
                verdict.append(f"Trust score: {trust:.0f}/100 - MEDIUM")
            else:
                verdict.append(f"Trust score: {trust:.0f}/100 - LOW")
        
        return {
            'gate_type': gate_type,
            'verdict_lines': verdict,
            'metrics_summary': {
                'profit_factor': pf,
                'expectancy_pct': exp,
                'trust_score': trust,
            }
        }
    
    def save_report(self, symbol: str, report: Dict[str, Any]) -> Path:
        """Save report to JSON file."""
        base = symbol.replace('/', '_')
        report_path = self.debug_dir / f"{base}_forensic_report.json"
        with open(report_path, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
        return report_path
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print human-readable report to console."""
        symbol = report['symbol']
        grade = report['token_grade']
        
        print(f"\n{'='*80}")
        print(f"FORENSIC REPORT: {symbol} | GRADE: {grade}")
        print(f"{'='*80}")
        
        # Calibration
        cal_fo = report['calibration_forensics']
        print(f"\n[CALIBRATION ANALYSIS]")
        print(f"  Summary: {cal_fo.get('summary')}")
        if cal_fo.get('eligible_count', 0) > 0:
            print(f"  Eligible methods: {cal_fo['eligible_count']}")
            if cal_fo.get('top_method'):
                print(f"  Top method: {cal_fo['top_method']} (score={cal_fo['top_score']:.4f})")
        
        # Architecture
        arch_fo = report['architecture_forensics']
        print(f"\n[ARCHITECTURE ANALYSIS]")
        print(f"  Summary: {arch_fo.get('summary')}")
        if arch_fo.get('top_candidates'):
            print(f"  Top candidate:")
            top = arch_fo['top_candidates'][0]
            print(f"    {top['gate_type']} @ q={top['quantile']:.2f} | " 
                  f"PF={top['profit_factor']:.3f} Exp={top['expectancy']:.1f}% Sharpe={top['sharpe']:.2f}")
        
        # Verdict
        verdict = report['final_verdict']
        print(f"\n[VERDICT]")
        for line in verdict.get('verdict_lines', []):
            print(f"  {line}")
        print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    profile_dir = Path(__file__).parent.parent / "data" / "meta_gate_profiles"
    
    if len(sys.argv) < 2:
        print("Usage: forensic_gate_report.py <symbol>")
        print("Example: forensic_gate_report.py BTC/USDT")
        sys.exit(1)
    
    symbol = sys.argv[1]
    
    # Load profile and debug data
    profile_path = profile_dir / f"{symbol.replace('/', '_')}_gate.json"
    debug_path = profile_dir / "debug" / f"{symbol.replace('/', '_')}_gate_debug.json"
    
    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        sys.exit(1)
    
    with open(profile_path) as f:
        profile = json.load(f)
    
    debug_data = {}
    if debug_path.exists():
        with open(debug_path) as f:
            debug_data = json.load(f)
    
    # Generate and print report
    reporter = ForensicGateReporter(profile_dir)
    report = reporter.generate_report(symbol, profile, debug_data)
    reporter.print_report(report)
    reporter.save_report(symbol, report)
    
    print(f"Report saved to: {profile_dir / 'debug' / f'{symbol.replace('/', '_')}_forensic_report.json'}")
