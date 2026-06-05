"""
src/ml/lstm_inference.py
────────────────────────
Fast, pool-based inference engine for the AEGIS LSTM temporal layer.

Design
------
  • LSTMPool follows the same singleton-pool pattern as HMMPool so the
    predictors share one set of loaded models across the 60-symbol fleet.
  • Models are loaded lazily on first request per symbol (cold start ≈ 50 ms).
  • Inference is synchronous and CPU-only; called from a thread-pool executor
    inside predictor.py — no async primitives needed here.
  • Returns LSTMState, a lightweight NamedTuple mirroring HMMState.

Failure modes
-------------
  • No .pt or _lstm_meta.json for symbol → available=False, all probs = 0.5
  • PyTorch not installed              → available=False, all probs = 0.5
  • Runtime error during inference     → available=False, all probs = 0.5
  In every case the live engine receives neutral defaults and existing
  behaviour is completely unchanged.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from src.ml.lstm_models import (
    CONT_FEATURES, CONT_SEQ_LEN,
    VOL_EXP_FEATURES, VOL_EXP_SEQ_LEN,
    ContinuationLSTM, VolatilityExpansionLSTM,
)
from src.ml.lstm_trainer import _RobustScaler, _resolve_features

MODEL_STORE = Path(__file__).parent / "model_store"


# ── Public result type ─────────────────────────────────────────────────────────

class LSTMState(NamedTuple):
    """LSTM inference result attached to every predict_realtime() output."""
    continuation_prob:  float  # P(momentum continues next 12 h)  [0, 1]
    vol_expansion_prob: float  # P(ATR expands ≥ 40 % next 12 h)  [0, 1]
    exhaustion_prob:    float  # 1 − continuation_prob (exhaustion proxy)
    available:          bool   # False when no LSTM models exist for symbol


_NEUTRAL = LSTMState(
    continuation_prob=0.5,
    vol_expansion_prob=0.5,
    exhaustion_prob=0.5,
    available=False,
)


# ── Per-symbol engine ──────────────────────────────────────────────────────────

class LSTMEngine:
    """Holds the loaded models + scalers for a single symbol."""

    def __init__(self, symbol: str) -> None:
        self.symbol  = symbol
        self._ready  = False
        self._cont_model:   Optional[Any]          = None
        self._vol_model:    Optional[Any]           = None
        self._scaler_c:     Optional[_RobustScaler] = None
        self._scaler_v:     Optional[_RobustScaler] = None
        self._cont_thr:     float                   = 0.50
        self._vol_thr:      float                   = 0.50
        self._vol_available: bool                   = False
        self._load()

    def _load(self) -> None:
        if not _TORCH_OK:
            return
        base      = self.symbol.replace("/", "_")
        meta_path = MODEL_STORE / f"{base}_lstm_meta.json"
        cont_path = MODEL_STORE / f"{base}_lstm_cont.pt"
        vol_path  = MODEL_STORE / f"{base}_lstm_vol.pt"

        if not meta_path.exists() or not cont_path.exists():
            return

        try:
            meta = json.loads(meta_path.read_text())

            # Continuation model
            m_c = ContinuationLSTM(
                input_size=len(CONT_FEATURES),
                hidden_size=64, num_layers=2, dropout=0.0,
            )
            m_c.load_state_dict(
                torch.load(str(cont_path), map_location="cpu", weights_only=True)
            )
            m_c.eval()
            self._cont_model = m_c
            self._cont_thr   = float(meta.get("cont_threshold", 0.50))

            sc_c = meta.get("scaler_cont")
            if sc_c:
                self._scaler_c = _RobustScaler.from_dict(sc_c)

            # Volatility-expansion model (optional)
            if meta.get("vol_model_available") and vol_path.exists():
                m_v = VolatilityExpansionLSTM(
                    input_size=len(VOL_EXP_FEATURES),
                    hidden_size=48, num_layers=2, dropout=0.0,
                )
                m_v.load_state_dict(
                    torch.load(str(vol_path), map_location="cpu", weights_only=True)
                )
                m_v.eval()
                self._vol_model     = m_v
                self._vol_thr       = float(meta.get("vol_threshold", 0.50))
                self._vol_available = True

                sc_v = meta.get("scaler_vol")
                if sc_v:
                    self._scaler_v = _RobustScaler.from_dict(sc_v)

            self._ready = True
        except Exception:
            self._ready = False

    # ── inference ──────────────────────────────────────────────────────────────

    def infer(self, df: pd.DataFrame) -> LSTMState:
        if not self._ready or not _TORCH_OK or df is None or len(df) < CONT_SEQ_LEN:
            return _NEUTRAL

        try:
            # ── continuation ─────────────────────────────────────────────────
            feat_c = _resolve_features(df, CONT_FEATURES)
            vals_c = feat_c.tail(CONT_SEQ_LEN).values.astype(np.float32)
            if self._scaler_c is not None:
                vals_c = self._scaler_c.transform(vals_c)
            vals_c = np.where(np.isfinite(vals_c), vals_c, 0.0).astype(np.float32)

            x_c = torch.tensor(vals_c, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
            with torch.no_grad():
                cont_prob = float(self._cont_model(x_c).item())

            # ── vol expansion ─────────────────────────────────────────────────
            vol_prob = 0.5
            if self._vol_available and self._vol_model is not None and len(df) >= VOL_EXP_SEQ_LEN:
                feat_v = _resolve_features(df, VOL_EXP_FEATURES)
                vals_v = feat_v.tail(VOL_EXP_SEQ_LEN).values.astype(np.float32)
                if self._scaler_v is not None:
                    vals_v = self._scaler_v.transform(vals_v)
                vals_v = np.where(np.isfinite(vals_v), vals_v, 0.0).astype(np.float32)

                x_v = torch.tensor(vals_v, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    vol_prob = float(self._vol_model(x_v).item())

            return LSTMState(
                continuation_prob  = float(np.clip(cont_prob, 0.0, 1.0)),
                vol_expansion_prob = float(np.clip(vol_prob,  0.0, 1.0)),
                exhaustion_prob    = float(np.clip(1.0 - cont_prob, 0.0, 1.0)),
                available          = True,
            )
        except Exception:
            return _NEUTRAL


# ── Pool (singleton, thread-safe) ─────────────────────────────────────────────

class LSTMPool:
    """
    Thread-safe pool of LSTMEngine instances, one per symbol.
    Engines are instantiated lazily on first request.
    """

    def __init__(self) -> None:
        self._engines: Dict[str, LSTMEngine] = {}
        self._lock    = threading.Lock()

    def get(self, symbol: str) -> LSTMEngine:
        if symbol not in self._engines:
            with self._lock:
                if symbol not in self._engines:
                    self._engines[symbol] = LSTMEngine(symbol)
        return self._engines[symbol]

    def infer(self, symbol: str, df: pd.DataFrame) -> LSTMState:
        try:
            return self.get(symbol).infer(df)
        except Exception:
            return _NEUTRAL

    def reload(self, symbol: str) -> None:
        """Force-reload models for a symbol (e.g. after retraining)."""
        with self._lock:
            self._engines.pop(symbol, None)


_POOL: Optional[LSTMPool] = None
_POOL_LOCK = threading.Lock()


def get_lstm_pool() -> LSTMPool:
    """Return the process-global LSTMPool singleton."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = LSTMPool()
    return _POOL
