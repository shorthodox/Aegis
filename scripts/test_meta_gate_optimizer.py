#!/usr/bin/env python3
"""Regression check for meta_gate_optimizer calibration selection."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.meta_gate_optimizer as optimizer


class FakeCalFramework:
    def __init__(self):
        self.calibrator_type = 'uncalibrated'
        self.best_calibrator = None

    def evaluate_calibrators(self, raw_scores, correct, threshold=0.50):
        self.calibrator_type = 'platt'
        self.best_calibrator = 'platt-model'
        return {
            'uncalibrated': {
                'ece': 0.25,
                'brier': 0.15,
                'precision': 0.50,
                'model': None,
            },
            'temperature': {
                'ece': 0.12,
                'brier': 0.08,
                'precision': 0.58,
                'model': 'temp-model',
            },
            'platt': {
                'ece': 0.20,
                'brier': 0.09,
                'precision': 0.56,
                'model': 'platt-model',
            },
        }

    def calibrate(self, ev_edge):
        if self.calibrator_type == 'temperature':
            return ev_edge + 0.15
        if self.calibrator_type == 'platt':
            return ev_edge + 0.08
        return ev_edge


def fake_backtest(fire, proposed, labels, barrier_frac):
    if not fire.any():
        return {
            'precision': 0.0,
            'expectancy_pct': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
        }

    precision = float((proposed[fire] == labels[fire]).mean())
    return {
        'precision': precision,
        'expectancy_pct': 6.0 if precision >= 0.5 else 0.5,
        'profit_factor': 1.25 if precision >= 0.5 else 0.90,
        'sharpe': 1.10 if precision >= 0.5 else 0.10,
    }


def run_regression():
    optimizer.MetaCalibrationFramework = FakeCalFramework
    optimizer._backtest_holdout = fake_backtest

    raw_scores = np.array([55.0, 45.0, 60.0, 52.0, 48.0, 70.0, 40.0, 35.0, 80.0, 20.0])
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=int)
    ev_edge_raw = np.array([55.0, 45.0, 60.0, 52.0, 48.0, 70.0, 40.0, 35.0, 80.0, 20.0])
    ev_side = np.array([2, 2, 0, 0, 1, 2, 0, 0, 2, 0], dtype=int)
    ev_labels = np.array([2, 2, 0, 0, 1, 2, 0, 0, 2, 0], dtype=int)
    ev_barrier = np.ones_like(ev_side, dtype=float)

    trainer, report, candidates = optimizer._select_best_calibrator(
        raw_scores,
        correct,
        ev_edge_raw,
        ev_side,
        ev_labels,
        ev_barrier,
    )

    assert report['selected_calibrator'] == trainer.calibrator_type, (
        'Selected calibrator must match trainer.calibrator_type'
    )
    assert report['selected_calibrator'] != 'uncalibrated', (
        'A calibrated candidate should be selected for this regression scenario'
    )
    assert trainer.best_calibrator == 'temp-model', (
        'The final selected model must come from the selected calibration method'
    )
    assert any(c.get('eligible') for c in candidates), 'At least one calibration candidate must be eligible'

    print('PASS: meta_gate_optimizer calibration regression check')
    print('selected_calibrator=', report['selected_calibrator'])
    print('selected_score=', report['selected_score'])
    print('candidates=', [c['method'] for c in candidates])


if __name__ == '__main__':
    run_regression()
