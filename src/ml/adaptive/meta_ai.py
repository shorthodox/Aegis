from typing import Any, Dict, Optional


class MetaAIEvaluator:
    """Supervise live signals and generate trust/bias recommendations."""

    def evaluate(self, signal: Dict[str, Any], calibrated_confidence: Optional[float] = None) -> Dict[str, Any]:
        confidence = float(signal.get('confidence', signal.get('meta_confidence', 0.0)) or 0.0)
        quality = float(signal.get('quality_score', signal.get('meta_confidence', confidence) * 100.0) or 0.0)
        is_fake = bool(signal.get('is_fake_breakout'))
        side = str(signal.get('direction', 'HOLD')).upper()

        trust = calibrated_confidence if calibrated_confidence is not None else confidence
        if quality < 50.0:
            trust *= max(0.4, quality / 50.0)

        if is_fake:
            trust *= 0.5
        if side not in ('BUY', 'SELL'):
            trust *= 0.5

        trust_score = float(min(max(trust, 0.0), 1.0))

        if not signal.get('fire') or side not in ('BUY', 'SELL'):
            recommendation = 'NO_ACTION'
            reason = 'signal_not_tradeable'
        elif trust_score >= 0.70 and quality >= 60.0 and not is_fake:
            recommendation = 'ACCEPT'
            reason = 'high_trust_and_quality'
        elif trust_score >= 0.45:
            recommendation = 'CAUTION'
            reason = 'moderate_trust'
        else:
            recommendation = 'REJECT'
            reason = 'low_trust_or_fake_breakout'

        return {
            'trust_score': trust_score,
            'recommendation': recommendation,
            'reason': reason,
        }
