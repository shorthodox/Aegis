from typing import Any, Dict, List, Optional

from .trade_memory import TradeMemoryRecord, TradeMemoryStore
from .trade_investigator import TradeInvestigator
from .counterfactual import CounterfactualSimulator
from .pattern_discovery import PatternDiscoveryEngine
from .knowledge_graph import KnowledgeGraph
from .similarity_search import SimilaritySearchIndex
from .confidence_calibration import ConfidenceCalibrator
from .meta_ai import MetaAIEvaluator
from .hypothesis_engine import HypothesisEngine
from .simulation_lab import SimulationLab
from .shadow_trading import ShadowTradingManager
from .model_promotion import ModelPromotionGate
from .mistake_memory import MistakeMemory
from .drift_detector import DriftDetector
from .knowledge_base import KnowledgeBase


class AdaptiveOrchestrator:
    """Top-level adaptive intelligence orchestrator for AEGIS."""

    def __init__(self):
        self.trade_memory = TradeMemoryStore()
        self.investigator = TradeInvestigator()
        self.counterfactual = CounterfactualSimulator()
        self.pattern_discovery = PatternDiscoveryEngine()
        self.knowledge_graph = KnowledgeGraph()
        self.similarity_search = SimilaritySearchIndex()
        self.calibrator = ConfidenceCalibrator()
        self.meta_ai = MetaAIEvaluator()
        self.hypothesis_engine = HypothesisEngine()
        self.simulation_lab = SimulationLab()
        self.shadow_trading = ShadowTradingManager()
        self.model_promotion = ModelPromotionGate()
        self.mistake_memory = MistakeMemory()
        self.drift_detector = DriftDetector()
        self.knowledge_base = KnowledgeBase()

    def record_signal(self, signal: Dict[str, Any]) -> None:
        self.drift_detector.observe(signal)
        self.trade_memory.record_signal(signal)
        self.similarity_search.index_trades(self.trade_memory.load_all())
        self.knowledge_base.add_entry({
            'type': 'signal',
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'payload': signal,
        })

    def record_trade(self, trade: Dict[str, Any]) -> None:
        self.trade_memory.record_trade(TradeMemoryRecord(
            trade_id=trade.get('trade_id'),
            signal_id=trade.get('signal_id'),
            symbol=trade.get('symbol'),
            mode=trade.get('mode'),
            direction=trade.get('direction'),
            status=trade.get('outcome', 'UNKNOWN'),
            entry_time=trade.get('entry_time'),
            exit_time=trade.get('exit_time'),
            entry_price=float(trade.get('entry_price', 0.0) or 0.0),
            exit_price=float(trade.get('exit_price', 0.0) or 0.0) if trade.get('exit_price') is not None else None,
            pnl_pct=float(trade.get('pnl_pct', 0.0)) if trade.get('pnl_pct') is not None else None,
            pnl_usdt=float(trade.get('pnl_usdt', 0.0)) if trade.get('pnl_usdt') is not None else None,
            features=trade.get('signal_metadata', {}),
            signal_metadata=trade,
            metadata={
                'investigation': self.investigator.analyze_trade(trade),
                'counterfactuals': self.counterfactual.simulate_variants(trade),
            },
        ))
        self.mistake_memory.record_mistake(trade)

    def evaluate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        similarity = self.similarity_search.find_similar(signal, top_k=3)
        calibrated_confidence = self.calibrator.calibrate(float(signal.get('confidence', 0.0)))
        evaluation = self.meta_ai.evaluate(signal, calibrated_confidence=calibrated_confidence)
        signal['adaptive_evaluation'] = {
            'trust_score': evaluation['trust_score'],
            'recommendation': evaluation['recommendation'],
            'reason': evaluation['reason'],
            'calibrated_confidence': calibrated_confidence,
            'similar_signals': similarity,
        }
        return signal

    def discover_patterns(self) -> List[Dict[str, Any]]:
        trades = self.trade_memory.load_all()
        return self.pattern_discovery.discover(trades)

    def summarize(self) -> Dict[str, Any]:
        return {
            'trade_memory_size': len(self.trade_memory.load_all()),
            'pattern_count': len(self.pattern_discovery.patterns),
            'knowledge_nodes': len(self.knowledge_graph.nodes),
            'knowledge_edges': len(self.knowledge_graph.edges),
        }
