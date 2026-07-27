import os
from src.ml.adaptive import AdaptiveOrchestrator


def test_adaptive_orchestrator_evaluation_returns_expected_keys():
    orchestrator = AdaptiveOrchestrator()
    signal = {
        'signal_id': 'SIG123',
        'symbol': 'BTC/USDT',
        'confidence': 0.82,
        'quality_score': 76.0,
        'fire': True,
        'direction': 'BUY',
        'is_fake_breakout': False,
    }

    evaluated = orchestrator.evaluate_signal(signal.copy())

    assert 'adaptive_evaluation' in evaluated
    assert isinstance(evaluated['adaptive_evaluation'], dict)
    assert evaluated['adaptive_evaluation']['recommendation'] in ('ACCEPT', 'CAUTION', 'REJECT', 'NO_ACTION')
    assert 0.0 <= evaluated['adaptive_evaluation']['trust_score'] <= 1.0
    assert isinstance(evaluated['adaptive_evaluation']['similar_signals'], list)
