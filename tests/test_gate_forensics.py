"""
tests/test_gate_forensics.py — Unit tests for gate forensics and gate optimizer fixes

Tests prove:
  1. No calibration / architecture contradiction
  2. Trust score consistency between gate selection and forensic report
  3. Forensic score components sum to final score
  4. JSON serialisation stability (no sklearn / ndarray leakage)
  5. DISABLED gate produces explicit root cause
  6. Grade mapping is correct
  7. safe_json_serializer handles all edge cases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Make sure the project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gate_forensics import (
    GateForensicsReporter,
    compute_score_breakdown,
    compute_trust_score,
    gate_grade,
)
from scripts.meta_gate_optimizer import (
    _compute_trust_score,
    _gate_grade,
    safe_json_serializer,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_profile(
    gate_type: str = 'EDGE_CONFLUENCE_VETO',
    pf: float = 1.174,
    exp: float = 6.73,
    sharpe: float = 2.98,
    precision: float = 0.454,
    coverage: float = 0.192,
    fired_n: int = 215,
    trust_score: int = 44,
) -> Dict[str, Any]:
    return {
        'symbol': 'BTC/USDT',
        'updated_at': '2026-06-06T00:00:00+00:00',
        'train_bars': 2605,
        'eval_bars': 1117,
        'selected_profile': {
            'gate_type': gate_type,
            'score': 0.4359,
            'trust_score': trust_score,
            'calibration': {
                'method': 'uncalibrated',
                'selected_calibrator': 'uncalibrated',
                'selected_method': 'uncalibrated',
            },
        },
        'holdout': {
            'selected': {
                'profit_factor': pf,
                'expectancy_pct': exp,
                'sharpe': sharpe,
                'precision': precision,
                'coverage': coverage,
                'fired_n': fired_n,
                'n': fired_n,
            },
        },
    }


def _make_debug(
    arch_validation_no_viable: bool = False,
    gate_type: str = 'EDGE_CONFLUENCE_VETO',
    pf: float = 1.174,
) -> Dict[str, Any]:
    return {
        'calibration': {
            'arch_validation_no_viable': arch_validation_no_viable,
            'method': 'uncalibrated',
        },
        'calibration_candidates': [
            {
                'method': 'uncalibrated',
                'train_ece': 0.183,
                'train_brier': 0.233,
                'train_precision': 0.476,
                'train_coverage': 0.501,
                'holdout_coverage': 0.501,
                'holdout_precision': 0.416,
                'holdout_expectancy': -0.66,
                'holdout_profit_factor': 0.984,
                'holdout_sharpe': -0.47,
                'variance_retained_ratio': 1.0,
                'probability_extreme_frac': 0.01,
                'calibration_quality': 0.817,
                'tech_eligible': True,
                'eligible': True,
                'architecture_viable': True,
                'best_architecture_score': 0.4359,
                'reason': '',
            },
        ],
        'candidates': [
            {
                'gate_type': gate_type,
                'quantile': 0.40,
                'calibration': 'uncalibrated',
                'precision': 0.454,
                'profit_factor': pf,
                'expectancy_pct': 6.73,
                'sharpe': 2.98,
                'coverage': 0.192,
                'score': 0.4359,
                'reason': 'accepted',
            },
        ],
        'baseline': {
            'precision': 0.268,
            'profit_factor': 0.934,
            'expectancy_pct': -1.88,
            'sharpe': -2.34,
            'coverage': 1.0,
        },
    }


# ── test 1: no calibration / architecture contradiction ───────────────────────

class TestNoContradiction:
    """Verify that when arch_validation_no_viable=True but gate is ENABLED,
    the forensic report surfaces a note rather than calling it a failure."""

    def test_contradiction_note_in_report(self):
        profile = _make_profile(gate_type='EDGE_CONFLUENCE_VETO', pf=1.174)
        debug   = _make_debug(arch_validation_no_viable=True)
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)

        # Gate must be ENABLED
        assert report['token_summary']['gate_selected'] == 'EDGE_CONFLUENCE_VETO'
        # Grade must NOT be F (gate is profitable)
        assert report['grade'] != 'F'
        # Calibration note should mention threshold context
        note = report['calibration_forensics'].get('note', '')
        assert 'percentile' in note.lower() or '0.50' in note

    def test_no_false_disabled_when_gate_found(self):
        """A profile with gate_type != DISABLED must not have disabled_diagnostics."""
        profile = _make_profile(gate_type='EDGE_CONFLUENCE_VETO')
        debug   = _make_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        assert report['disabled_diagnostics'] is None


# ── test 2: trust score consistency ──────────────────────────────────────────

class TestTrustScoreConsistency:
    def test_forensic_trust_matches_saved(self):
        """Trust computed by forensic report must match what gate optimizer saved."""
        pf, exp, sharpe, cov, fired_n = 1.174, 6.73, 2.98, 0.192, 215
        # Compute via meta_gate_optimizer helper
        pf_n   = max(0.0, min(1.0, (pf - 1.0) / 2.0))
        exp_n  = max(0.0, min(1.0, exp / 10.0))
        shr_n  = max(0.0, min(1.0, sharpe / 5.0))
        cov_n  = max(0.0, min(1.0, cov))
        saved  = _compute_trust_score(pf_n, exp_n, shr_n, cov_n, fired_n)

        # Compute via gate_forensics helper
        forensic_trust, _ = compute_trust_score(pf, exp, sharpe, cov, fired_n)

        assert saved == forensic_trust, (
            f'Trust mismatch: optimizer={saved}, forensics={forensic_trust}'
        )

    def test_forensic_report_consistency_flag(self):
        pf, exp, sharpe, cov, fired_n = 1.174, 6.73, 2.98, 0.192, 215
        forensic_trust, _ = compute_trust_score(pf, exp, sharpe, cov, fired_n)
        profile = _make_profile(trust_score=forensic_trust)
        debug   = _make_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        assert report['trust_score_forensics']['consistent'] is True

    def test_inconsistency_detected(self):
        """Wrong saved trust triggers consistent=False."""
        profile = _make_profile(trust_score=99)  # clearly wrong
        debug   = _make_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        assert report['trust_score_forensics']['consistent'] is False


# ── test 3: forensic score breakdown ─────────────────────────────────────────

class TestScoreBreakdown:
    def test_components_sum_to_total(self):
        for pf, exp, sharpe, prec, cov in [
            (1.174, 6.73, 2.98, 0.454, 0.192),
            (1.50, 12.0, 3.50, 0.60, 0.30),
            (1.05, 1.0, 0.5, 0.40, 0.15),
        ]:
            bd = compute_score_breakdown(pf, exp, sharpe, prec, cov)
            component_sum = sum(bd[k] for k in ('expectancy', 'profit_factor', 'sharpe', 'precision', 'coverage'))
            assert abs(component_sum - bd['total']) < 1e-5, (
                f'Components {component_sum:.6f} != total {bd["total"]:.6f} for PF={pf}'
            )

    def test_matches_optimizer_formula(self):
        """Verify breakdown matches the weights in _evaluate_architecture."""
        pf, exp, sharpe, prec, cov = 1.174, 6.73, 2.98, 0.454, 0.192
        bd = compute_score_breakdown(pf, exp, sharpe, prec, cov)

        pf_n   = max(0.0, min(1.0, (pf - 1.0) / 2.0))
        exp_n  = max(0.0, min(1.0, exp / 10.0))
        shr_n  = max(0.0, min(1.0, sharpe / 5.0))
        prec_n = max(0.0, min(1.0, prec))
        cov_n  = max(0.0, min(1.0, cov))
        expected = (
            0.35 * exp_n + 0.30 * pf_n + 0.20 * shr_n + 0.10 * prec_n + 0.05 * cov_n
        )
        assert abs(bd['total'] - expected) < 1e-5


# ── test 4: JSON serialisation stability ─────────────────────────────────────

class TestJsonSerialisation:
    def test_numpy_types_serialised(self):
        obj = {
            'a': np.int64(42),
            'b': np.float32(3.14),
            'c': np.array([1, 2, 3]),
            'd': np.bool_(True),
        }
        result = safe_json_serializer(obj)
        serialised = json.dumps(result)  # must not raise
        parsed = json.loads(serialised)
        assert parsed['a'] == 42
        assert abs(parsed['b'] - 3.14) < 0.01
        assert parsed['c'] == [1, 2, 3]
        assert parsed['d'] is True

    def test_sklearn_model_serialised(self):
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            pytest.skip('sklearn not available')
        model = LogisticRegression()
        result = safe_json_serializer({'model': model})
        serialised = json.dumps(result)
        parsed = json.loads(serialised)
        assert parsed['model']['__type__'] == 'sklearn_model'
        assert parsed['model']['class'] == 'LogisticRegression'

    def test_nested_non_serialisable(self):
        """Complex nested object with various types must not raise."""
        obj = {
            'nested': {
                'arr': np.array([1.0, 2.0]),
                'val': np.int32(7),
            },
            'list': [np.float64(0.5), None, True, 'hello'],
        }
        result = safe_json_serializer(obj)
        json.dumps(result)  # must not raise

    def test_full_profile_serialisable(self):
        profile = _make_profile()
        debug   = _make_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        serialised = json.dumps(report)  # must not raise
        assert '"BTC/USDT"' in serialised


# ── test 5: DISABLED gate root cause ─────────────────────────────────────────

class TestDisabledGateRootCause:
    def _disabled_profile(self) -> Dict[str, Any]:
        return _make_profile(
            gate_type='DISABLED',
            pf=0.93,
            exp=-1.88,
            sharpe=-2.34,
            precision=0.268,
            coverage=1.0,
            trust_score=0,
        )

    def _disabled_debug(self) -> Dict[str, Any]:
        return {
            'calibration': {'arch_validation_no_viable': True},
            'calibration_candidates': [],
            'candidates': [
                {'gate_type': 'GLOBAL_EDGE_PERCENTILE', 'reason': 'nonpositive_expectancy',
                 'precision': 0.41, 'profit_factor': 0.88, 'expectancy_pct': -5.2, 'sharpe': -3.29,
                 'coverage': 0.4, 'score': 0.0, 'quantile': 0.4},
            ],
            'baseline': {
                'precision': 0.268, 'profit_factor': 0.934,
                'expectancy_pct': -1.88, 'sharpe': -2.34, 'coverage': 1.0,
            },
        }

    def test_disabled_has_diagnostics(self):
        profile  = self._disabled_profile()
        debug    = self._disabled_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        assert report['disabled_diagnostics'] is not None
        causes = report['disabled_diagnostics']['root_causes']
        assert len(causes) > 0, 'DISABLED gate must have at least one root cause'

    def test_disabled_never_just_no_viable_architecture(self):
        """Root cause must contain a specific explanation, not a bare 'no viable architecture'."""
        profile  = self._disabled_profile()
        debug    = self._disabled_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        causes   = report['disabled_diagnostics']['root_causes']
        # At minimum one cause must be more specific than a bare fallback
        assert any(len(c) > 30 for c in causes), (
            f'Root causes too vague: {causes}'
        )

    def test_disabled_grade_is_f(self):
        profile  = self._disabled_profile()
        debug    = self._disabled_debug()
        reporter = GateForensicsReporter(_ROOT)
        report   = reporter.generate_report('BTC/USDT', profile, debug)
        assert report['grade'] == 'F'


# ── test 6: grade mapping ─────────────────────────────────────────────────────

class TestGradeMapping:
    @pytest.mark.parametrize('pf, expected', [
        (1.85, 'A+'),
        (1.70, 'A'),
        (1.45, 'B+'),
        (1.35, 'B'),
        (1.25, 'C+'),
        (1.15, 'C'),
        (1.05, 'D'),
        (0.95, 'F'),
        (1.00, 'F'),
    ])
    def test_grade(self, pf, expected):
        assert gate_grade(pf) == expected
        assert _gate_grade(pf) == expected  # optimizer helper must match

    def test_btc_example(self):
        """BTC with PF=1.174 must grade as C (not D)."""
        assert gate_grade(1.174) == 'C'


# ── test 7: safe_json_serializer edge cases ───────────────────────────────────

class TestSafeJsonSerializer:
    def test_primitives_unchanged(self):
        obj = {'a': 1, 'b': 2.5, 'c': True, 'd': None, 'e': 'hello'}
        assert safe_json_serializer(obj) == obj

    def test_list_of_mixed(self):
        obj = [1, np.float64(2.5), 'three', None, True]
        result = safe_json_serializer(obj)
        assert result[1] == 2.5
        json.dumps(result)

    def test_deeply_nested(self):
        obj = {'a': {'b': {'c': np.int32(99)}}}
        result = safe_json_serializer(obj)
        assert result['a']['b']['c'] == 99
        json.dumps(result)

    def test_empty_structures(self):
        assert safe_json_serializer({}) == {}
        assert safe_json_serializer([]) == []

    def test_tuple_becomes_list(self):
        result = safe_json_serializer((1, 2, 3))
        assert isinstance(result, list)
        assert result == [1, 2, 3]


if __name__ == '__main__':
    import subprocess
    raise SystemExit(
        subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v'])
    )
