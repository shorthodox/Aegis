"""
src/ml/lstm_models.py
─────────────────────
Lightweight LSTM architectures for the AEGIS temporal intelligence layer.

Two specialised models (never used for direct trade execution):
  ContinuationLSTM        → P(current directional move continues over next 12 bars)
  VolatilityExpansionLSTM → P(ATR expands ≥ 40 % over the next 12 bars)

Both use a 2-layer residual LSTM (hidden 48–64) with LayerNorm + Dropout.
Designed for CPU inference: < 2 ms per symbol at batch = 1.

Role in the stack
-----------------
  XGBoost    → structural direction intelligence
  HMM        → market regime classification
  LSTM (this) → temporal sequence / continuation / vol-expansion intelligence
  Meta-model → execution filter (unchanged)
"""

from __future__ import annotations
from typing import List

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── Canonical feature lists ────────────────────────────────────────────────────
# Preferred column names (produced by retrain_model.py / feature_engine.py).
# lstm_trainer._resolve_features() tries aliases and zero-fills missing columns.

CONT_FEATURES: List[str] = [
    "total_confluence",      # hard confluence aggregate   [-1, +1]
    "prc_total",             # soft percentile confluence  [0,  1]
    "volume_zscore",         # volume z-score
    "volume_delta",          # directional buy/sell volume
    "rsi_14",                # RSI-14
    "macd_hist",             # MACD histogram (momentum acceleration)
    "dist_ema_21",           # price distance from EMA-21  (normalized)
    "efficiency_ratio_10",   # Kaufman ER  (trend quality, 0=noise, 1=trend)
    "funding_rate_zscore",   # funding rate z-score  (crowding sentiment)
    "oi_change_1h",          # open interest 1h delta
    "btc_corr_24h",          # BTC 24 h rolling correlation
    "sqz_on",                # Bollinger/Keltner squeeze flag  (0/1)
    "bb_pct_b",              # Bollinger %B  (band position)
    "close_position",        # close position within bar H–L range [0, 1]
    "vol_velocity",          # rolling volume acceleration
    "macd_signal",           # MACD signal line
    "rsi_7",                 # RSI-7  (fast momentum)
    "adx_14",                # ADX-14  (trend strength)
    # --- stronger temporal features added for momentum persistence ---
    "returns_1h",            # 1h return (stationarity anchor for continuation)
    "ret_4h",                # 4h return (medium-horizon momentum)
    "volatility_regime",     # ATR/ATR_mean_100 — normalized vol expansion signal
    "candle_pressure",       # buy/sell pressure ratio per bar
]
CONT_SEQ_LEN: int = 24       # extended from 20 → 24 bars for more momentum context

VOL_EXP_FEATURES: List[str] = [
    # atr_14 removed: raw ATR is an absolute price level that drifts with BTC price.
    # volatility_regime (ATR/ATR_mean_100) is stationary and measures the same thing.
    "volatility_regime",     # ATR / 100-bar ATR mean — normalized expansion signal
    "bb_pct_b",              # Bollinger %B
    "sqz_on",                # squeeze flag    (compression detector)
    "volume_zscore",         # volume activity level
    "oi_change_1h",          # OI 1 h change
    "funding_rate_zscore",   # funding z-score
    "efficiency_ratio_10",   # ER  (low ER = random walk = compression)
    "close_position",        # bar close position
    "volume_delta",          # directional volume
    "rsi_14",                # RSI-14
    "vol_velocity",          # volume acceleration
    "adx_14",                # ADX  (trend build-up before breakout)
    "bb_width_percentile",   # Bollinger bandwidth percentile [0, 1]
    "oi_chg_8h",             # OI 8 h change  (longer-horizon position build)
    # --- stronger temporal features for vol expansion ---
    "returns_1h",            # recent return direction (vol expands after moves)
    "parkinson_vol",         # Parkinson H-L estimator (true range volatility)
    "atr_pct",               # ATR as % of price — stationary vol magnitude
]
VOL_EXP_SEQ_LEN: int = 24    # 24-candle (~24 h) lookback window


if _TORCH_OK:

    class _ResidualLSTMBlock(nn.Module):
        """Single LSTM layer + LayerNorm + Dropout + residual projection."""

        def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.norm = nn.LayerNorm(hidden_size)
            self.drop = nn.Dropout(dropout)
            # project input to hidden_size for residual add (no-op if already equal)
            self.proj: nn.Module = (
                nn.Linear(input_size, hidden_size, bias=False)
                if input_size != hidden_size
                else nn.Identity()
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)           # (B, T, hidden)
            out = self.norm(out)
            out = self.drop(out)
            return out + self.proj(x)       # residual skip


    class ContinuationLSTM(nn.Module):
        """
        Predicts P(current directional momentum continues over next 12 bars).

        Input : (B, seq_len=20, n_features=18)
        Output: (B,) ∈ [0, 1]   higher → stronger continuation expected
        """

        def __init__(
            self,
            input_size:  int   = len(CONT_FEATURES),
            hidden_size: int   = 64,
            num_layers:  int   = 2,
            dropout:     float = 0.3,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.blocks = nn.ModuleList([
                _ResidualLSTMBlock(hidden_size, hidden_size, dropout)
                for _ in range(num_layers)
            ])
            self.final_norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)          # (B, T, hidden)
            for block in self.blocks:
                x = block(x)
            last = self.final_norm(x[:, -1, :])   # last timestep
            return torch.sigmoid(self.head(last)).squeeze(-1)


    class VolatilityExpansionLSTM(nn.Module):
        """
        Predicts P(ATR expands by ≥ 40 % over the next 12 bars).

        Input : (B, seq_len=24, n_features=14)
        Output: (B,) ∈ [0, 1]   higher → volatility expansion imminent
        """

        def __init__(
            self,
            input_size:  int   = len(VOL_EXP_FEATURES),
            hidden_size: int   = 48,
            num_layers:  int   = 2,
            dropout:     float = 0.3,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.blocks = nn.ModuleList([
                _ResidualLSTMBlock(hidden_size, hidden_size, dropout)
                for _ in range(num_layers)
            ])
            self.final_norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            last = self.final_norm(x[:, -1, :])
            return torch.sigmoid(self.head(last)).squeeze(-1)

else:
    # Stubs so the rest of the codebase can import without crashing when
    # PyTorch is not installed.  Training will fail gracefully; inference
    # returns neutral defaults.

    class ContinuationLSTM:        # type: ignore[no-redef]
        pass

    class VolatilityExpansionLSTM:  # type: ignore[no-redef]
        pass
