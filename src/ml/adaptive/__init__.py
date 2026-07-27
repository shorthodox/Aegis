"""Adaptive research layer for AEGIS.

This package contains the skeleton of a self-evolving intelligence layer that
sits above the existing prediction engine.
"""

from .aegis_scientist import AdaptiveOrchestrator
from .trade_memory import TradeMemoryStore
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

__all__ = [
    'AdaptiveOrchestrator',
    'TradeMemoryStore',
    'TradeInvestigator',
    'CounterfactualSimulator',
    'PatternDiscoveryEngine',
    'KnowledgeGraph',
    'SimilaritySearchIndex',
    'ConfidenceCalibrator',
    'MetaAIEvaluator',
    'HypothesisEngine',
    'SimulationLab',
    'ShadowTradingManager',
    'ModelPromotionGate',
    'MistakeMemory',
    'DriftDetector',
    'KnowledgeBase',
]
