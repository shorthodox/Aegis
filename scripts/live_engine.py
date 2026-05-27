#!/usr/bin/env python3
"""
Aegis-1 Production Live Engine v4.0 – ASYMMETRIC SIGNALS + RESET PROTOCOL
- Asymmetric thresholds: STRONG_BUY>0.75, BUY>0.60, HOLD 0.20-0.60, SELL<0.20, STRONG_SELL<0.15
- Sellability brake: suppress sell signals during low volatility / choppy markets
- Wallet reset (C key) with backup and activity log purge
- Coloured terminal output for signals
"""

import asyncio
import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import uuid
import time
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import deque

import aiohttp
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import xgboost as xgb
import socket
from aiohttp import ClientTimeout
from aiohttp.resolver import AbstractResolver
from socket import AddressFamily

# Voice synthesis
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("pyttsx3 not installed. Voice notifications disabled.")

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.ml.feature_engine import prepare_features, compute_atr, compute_efficiency_ratio
from src.ml.delltandecay import adjust_threshold_by_technical_and_fundamental
from scripts.telemetry import extract_prediction_telemetry

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CONFIG_DIR = root_dir / "config"
DATA_DIR = root_dir / "data"
LOGS_DIR = root_dir / "logs"
MODEL_STORE = root_dir / "src" / "ml" / "model_store"

CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

EXCHANGE_ID = 'binance'
EXCHANGE_FEE = 0.001
SLIPPAGE = 0.001
TOTAL_COST_PCT = 2 * (EXCHANGE_FEE + SLIPPAGE)

TIMEFRAMES = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800,
    '1M': 2592000,
}

def get_limit_for_timeframe(timeframe: str, lookback_hours: int = 200) -> int:
    tf_seconds = TIMEFRAMES.get(timeframe, 3600)
    return max(50, (lookback_hours * 3600) // tf_seconds + 50)

CACHE_DURATION = {}
for tf, seconds in TIMEFRAMES.items():
    if tf in ('1h', '4h'):
        CACHE_DURATION[tf] = 300
    else:
        CACHE_DURATION[tf] = seconds

VOLATILITY_LOOKBACK = 100
VOLUME_LOOKBACK = 50
VOL_PERCENTILE_LOW = 30
VOL_PERCENTILE_HIGH = 70
VOLUME_RATIO_HIGH = 1.20
VOLUME_RATIO_LOW = 0.80
ER_STRONG_TREND = 0.50
ER_WEAK_TREND = 0.30

CONF_VERY_HIGH = 0.75
CONF_NORMAL = 1.00
CONF_WEAK = 1.25
TREND_ALIGNED_FACTOR = 0.85
TREND_MISALIGNED_FACTOR = 1.20
VOL_HIGH_FACTOR = 0.85
VOL_NORMAL_FACTOR = 1.00
VOL_LOW_FACTOR = 1.20
VOL_EXTREME_LOW_FACTOR = 1.50
VOLUME_HIGH_FACTOR = 0.90
VOLUME_NORMAL_FACTOR = 1.00
VOLUME_LOW_FACTOR = 1.10
THRESHOLD_FLOOR = 0.05
THRESHOLD_CEIL = 3.00

PROXIMITY_THRESHOLD = 0.005  # 0.5% proximity to S&R boundary triggers alert

CANDLE_CONFIRM_REQUIRED = 3    # consecutive candles needed before emitting signal
SIGNAL_EXPIRY_MOVE_PCT = 0.60  # if price moves 60% of expected TP before action → expired
REVERSAL_SCORE_THRESHOLD = 0.35  # minimum technical score to begin candle counting

MODE_PARAMS = {
    "conservative": {"entry_prob": 0.75, "risk_pct": 0.015, "atr_sl": 1.2, "atr_tp": 1.8},
    "balanced":     {"entry_prob": 0.70, "risk_pct": 0.020, "atr_sl": 1.5, "atr_tp": 2.0},
    "aggressive":   {"entry_prob": 0.65, "risk_pct": 0.030, "atr_sl": 1.8, "atr_tp": 2.5},
}

BTC_BULL_THRESHOLD = 0.55   # BTC healthy for longs
BTC_BEAR_THRESHOLD = 0.45   # BTC weak -> allow shorts
RR_RATIO = 1.5

DEFAULT_CAPITAL = 10000.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_ALPHA_RISK_PCT = 3.0
DEFAULT_MAX_POSITION = 2000.0
DEFAULT_SCAN_TIMEFRAME = '1m'
DEFAULT_ALPHA_MODE = False

CONVICTION_LONG_BAILOUT = 0.40   # close LONG if ai_prob < 0.40
CONVICTION_SHORT_BAILOUT = 0.60  # close SHORT if ai_prob > 0.60

WARMUP_DURATION = 3600

FLEET = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'HYPE/USDT', 'ASTER/USDT', 'SUI/USDT', 'TAO/USDT', 'RENDER/USDT',
    'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'TRX/USDT', 'DOT/USDT',
    'NEAR/USDT', 'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'SHIB/USDT',
    'TON/USDT', 'ICP/USDT', 'HBAR/USDT', 'APT/USDT', 'ARB/USDT',
    'OP/USDT', 'STX/USDT', 'FIL/USDT', 'AAVE/USDT', 'VET/USDT',
    'RNDR/USDT', 'INJ/USDT', 'TIA/USDT', 'SEI/USDT', 'KAS/USDT',
    'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'AKT/USDT', 'THETA/USDT',
    'GRT/USDT', 'LDO/USDT', 'PYTH/USDT', 'JUP/USDT', 'ONDO/USDT',
    'PEPE/USDT', 'DOGE/USDT', 'WIF/USDT', 'FLOKI/USDT', 'BONK/USDT',
    'WLFI/USDT', 'MNT/USDT', 'ENA/USDT', 'BGB/USDT', 'PI/USDT',
    'SKY/USDT', 'TRUMP/USDT', 'NIGHT/USDT'
]

# -------------------------------------------------------------------
# Helper: Normalize symbol (underscore -> slash)
# -------------------------------------------------------------------
def normalize_symbol(symbol: str) -> str:
    """Convert symbol format: replace '_' with '/'."""
    if not symbol:
        return symbol
    return symbol.replace('_', '/')

# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------
@dataclass
class TokenConfig:
    symbol: str
    mode: str
    base_threshold: float
    entry_prob_threshold: float
    atr_sl: float
    atr_tp: float
    risk_pct: float
    max_dd: float = 0.0
    expectancy: float = 0.0
    optimizer_thresholds: Dict[Tuple[str, str], float] = field(default_factory=dict)

@dataclass
class LiveTrade:
    symbol: str
    entry_price: float
    position_units: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    mode: str
    master_score: float
    trade_type: str = "LONG"
    closed: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    close_time: Optional[datetime] = None

# -------------------------------------------------------------------
# Clean start
# -------------------------------------------------------------------
def flush_dns_caches():
    try:
        subprocess.run(["ipconfig", "/flushdns"], capture_output=True, check=True)
        logger.info("Flushed ipconfig DNS cache")
    except Exception as e:
        logger.warning(f"ipconfig /flushdns failed: {e}")
    try:
        subprocess.run(
            ["powershell", "-Command", "Clear-DnsClientCache"],
            capture_output=True, check=True
        )
        logger.info("Flushed PowerShell DNS client cache")
    except Exception as e:
        logger.warning(f"Clear-DnsClientCache failed: {e}")

# -------------------------------------------------------------------
# Backtest file discovery & JSON loader
# -------------------------------------------------------------------
def get_latest_backtest_file(directory):
    """Return the most recently modified .json file in the directory, or None."""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Backtest directory does not exist: {dir_path}")
        return None

    # Find all .json files
    json_files = list(dir_path.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {dir_path}")
        return None

    # Return the most recent file by modification time
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Latest backtest file: {latest}")
    return latest

def load_backtest_results(json_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {json_path}")
            return {}

    # --- ENHANCED NORMALIZATION ---
    if isinstance(data, dict):
        new_data = []
        # If it's a symbol map: {"BTC/USDT": {"net_return": 10}, "version": "1.0"}
        for k, v in data.items():
            if isinstance(v, dict):
                # Only unpack if v is actually a dictionary
                new_data.append({"symbol": k, **v})
            elif k == "symbol":
                # If the dict itself is the result: {"symbol": "BTC", "mar": 2}
                new_data = [data]
                break
        
        if new_data:
            data = new_data
        else:
            # Fallback for simple flat dictionaries
            data = [data]

    if not isinstance(data, list):
        logger.error(f"Unexpected data format in {json_path.name}: {type(data)}")
        return {}
    # --- END NORMALIZATION ---

    symbol_entries: Dict[str, List[Dict]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
            
        symbol = entry.get("symbol")
        if not symbol:
            continue
            
        mode = entry.get("mode")
        if not mode:
            continue
            
        net_return = entry.get("net_return_pct", 0.0)
        max_dd = entry.get("max_drawdown_pct", 0.0)
        # SAFETY: prevent division by zero or extremely small values
        safe_dd = max(float(max_dd), 0.1) if isinstance(max_dd, (int, float)) else 0.1
        mar = float(net_return) / safe_dd if isinstance(net_return, (int, float)) else 0.0
        
        params = MODE_PARAMS.get(mode)
        if params is None:
            logger.warning(f"Unknown mode '{mode}' for {symbol}, falling back to 'balanced'")
            params = MODE_PARAMS["balanced"]
        
        symbol_entries.setdefault(symbol, []).append({
            "mode": mode,
            "mar": mar,
            "net_return_pct": net_return,
            "max_drawdown_pct": max_dd,
            "final_threshold": entry.get("final_threshold", 0.30),
            "entry_prob_threshold": params["entry_prob"],
            "atr_sl": params["atr_sl"],
            "atr_tp": params["atr_tp"],
            "risk_pct": params["risk_pct"],
        })

    best_per_token = {}
    for sym, entries in symbol_entries.items():
        best = max(entries, key=lambda x: x["mar"])
        best_per_token[sym] = best

    logger.info(f"Successfully loaded {len(best_per_token)} tokens from {json_path.name}")
    return best_per_token

# -------------------------------------------------------------------
# Regime & Threshold Engines
# -------------------------------------------------------------------
class RegimeDetector:
    @staticmethod
    def volatility(atr_series: pd.Series, idx: int) -> str:
        if idx < VOLATILITY_LOOKBACK:
            return "normal"
        window = atr_series.iloc[max(0, idx - VOLATILITY_LOOKBACK): idx + 1]
        if len(window) < VOLATILITY_LOOKBACK // 2:
            return "normal"
        pct = float((window <= atr_series.iat[idx]).mean() * 100)
        if pct <= VOL_PERCENTILE_LOW:
            return "low"
        if pct >= VOL_PERCENTILE_HIGH:
            return "high"
        return "normal"

    @staticmethod
    def volume(volume: float, volume_ma: float) -> str:
        if np.isnan(volume_ma) or volume_ma <= 0:
            return "normal"
        ratio = volume / volume_ma
        if ratio > VOLUME_RATIO_HIGH:
            return "high"
        if ratio < VOLUME_RATIO_LOW:
            return "low"
        return "normal"

    @staticmethod
    def trend(efficiency_ratio: float) -> str:
        if np.isnan(efficiency_ratio):
            return "normal"
        if efficiency_ratio > ER_STRONG_TREND:
            return "strong"
        if efficiency_ratio < ER_WEAK_TREND:
            return "weak"
        return "normal"

    @staticmethod
    def is_trend_aligned(direction: str, trend_strength: str) -> bool:
        return direction == "long" and trend_strength == "strong"

class ThresholdEngine:
    @staticmethod
    def confidence_factor(prob: float) -> Tuple[float, str]:
        if prob > 0.80 or prob < 0.20:
            return CONF_VERY_HIGH, "very_high"
        if 0.60 <= prob <= 0.80:
            return CONF_NORMAL, "normal"
        return CONF_WEAK, "weak"

    @staticmethod
    def vol_factor(vol_regime: str, atr: float, price: float) -> float:
        atr_ratio = atr / price if price > 0 else 0.0
        if vol_regime == "low" and atr_ratio < 0.003:
            return VOL_EXTREME_LOW_FACTOR
        return {"high": VOL_HIGH_FACTOR, "normal": VOL_NORMAL_FACTOR, "low": VOL_LOW_FACTOR}.get(vol_regime, VOL_NORMAL_FACTOR)

    @staticmethod
    def volume_factor(vol_cond: str) -> float:
        return {"high": VOLUME_HIGH_FACTOR, "normal": VOLUME_NORMAL_FACTOR, "low": VOLUME_LOW_FACTOR}.get(vol_cond, VOLUME_NORMAL_FACTOR)

    @staticmethod
    def trend_factor(trend_aligned: bool) -> float:
        return TREND_ALIGNED_FACTOR if trend_aligned else TREND_MISALIGNED_FACTOR

    @classmethod
    def compute(cls, base: float, prob: float, vol_regime: str, volume_condition: str,
                trend_aligned: bool, atr: float, price: float) -> Tuple[float, str]:
        cf, conf_label = cls.confidence_factor(prob)
        adjusted = base * cf
        adjusted *= cls.vol_factor(vol_regime, atr, price)
        adjusted *= cls.volume_factor(volume_condition)
        adjusted *= cls.trend_factor(trend_aligned)
        adjusted = float(np.clip(adjusted, THRESHOLD_FLOOR, THRESHOLD_CEIL))
        return adjusted, conf_label

# -------------------------------------------------------------------
# LiveWallet (with reset feature)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Reversal Anticipation Engine
# -------------------------------------------------------------------
class ReversalDetector:
    """Detects early signs of trend reversal using RSI, candlestick patterns, and S/R proximity."""

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @classmethod
    def detect_bearish_reversal_setup(
        cls, df: pd.DataFrame, resistance: float = 0.0
    ) -> tuple:
        """Return (score 0-1, [signal_tags]) indicating how ripe a bearish reversal is."""
        if len(df) < 20:
            return 0.0, []
        close = df['close']
        signals: list = []
        score = 0.0

        rsi_s = cls._rsi(close)
        rsi_now = float(rsi_s.iloc[-1])
        if rsi_now > 70:
            signals.append(f"RSI_OB_{rsi_now:.0f}")
            score += 0.25
        elif rsi_now > 65:
            score += 0.10

        # Bearish divergence: price higher high, RSI lower high
        if len(df) >= 10:
            rsi_window = rsi_s.iloc[-6:-1]
            close_window = close.iloc[-6:-1]
            if not rsi_window.empty and close.iloc[-1] >= close_window.max() and rsi_now <= float(rsi_window.max()):
                signals.append("BEARISH_RSI_DIV")
                score += 0.30

        # Exhaustion candle: small body + long upper wick
        c = df.iloc[-1]
        body = abs(float(c['close']) - float(c['open']))
        up_wick = float(c['high']) - max(float(c['close']), float(c['open']))
        total = float(c['high']) - float(c['low'])
        if total > 0 and up_wick / total > 0.50 and body / total < 0.35:
            signals.append("SHOOTING_STAR")
            score += 0.20

        # Bearish engulfing
        if len(df) >= 2:
            prev = df.iloc[-2]
            if (float(c['close']) < float(c['open']) and float(prev['close']) > float(prev['open'])
                    and float(c['open']) >= float(prev['close']) * 0.997
                    and float(c['close']) < float(prev['open'])):
                signals.append("BEARISH_ENGULF")
                score += 0.25

        # Near resistance
        cp = float(close.iloc[-1])
        if resistance > 0:
            dist = (resistance - cp) / cp
            if 0 <= dist <= 0.005:
                signals.append("AT_RESISTANCE")
                score += 0.20

        # Price overextended above EMA-20
        ema20 = close.ewm(span=20, adjust=False).mean()
        if float(ema20.iloc[-1]) > 0:
            ext = (cp - float(ema20.iloc[-1])) / float(ema20.iloc[-1])
            if ext > 0.03:
                signals.append(f"OVEREXT_{ext*100:.1f}%")
                score += 0.15

        return min(score, 1.0), signals

    @classmethod
    def detect_bullish_reversal_setup(
        cls, df: pd.DataFrame, support: float = 0.0
    ) -> tuple:
        """Return (score 0-1, [signal_tags]) indicating how ripe a bullish reversal is."""
        if len(df) < 20:
            return 0.0, []
        close = df['close']
        signals: list = []
        score = 0.0

        rsi_s = cls._rsi(close)
        rsi_now = float(rsi_s.iloc[-1])
        if rsi_now < 30:
            signals.append(f"RSI_OS_{rsi_now:.0f}")
            score += 0.25
        elif rsi_now < 35:
            score += 0.10

        # Bullish divergence: price lower low, RSI higher low
        if len(df) >= 10:
            rsi_window = rsi_s.iloc[-6:-1]
            close_window = close.iloc[-6:-1]
            if not rsi_window.empty and close.iloc[-1] <= close_window.min() and rsi_now >= float(rsi_window.min()):
                signals.append("BULLISH_RSI_DIV")
                score += 0.30

        # Hammer candle: small body + long lower wick
        c = df.iloc[-1]
        body = abs(float(c['close']) - float(c['open']))
        low_wick = min(float(c['close']), float(c['open'])) - float(c['low'])
        total = float(c['high']) - float(c['low'])
        if total > 0 and low_wick / total > 0.50 and body / total < 0.35:
            signals.append("HAMMER")
            score += 0.20

        # Bullish engulfing
        if len(df) >= 2:
            prev = df.iloc[-2]
            if (float(c['close']) > float(c['open']) and float(prev['close']) < float(prev['open'])
                    and float(c['open']) <= float(prev['close']) * 1.003
                    and float(c['close']) > float(prev['open'])):
                signals.append("BULLISH_ENGULF")
                score += 0.25

        # Near support
        cp = float(close.iloc[-1])
        if support > 0:
            dist = (cp - support) / cp
            if 0 <= dist <= 0.005:
                signals.append("AT_SUPPORT")
                score += 0.20

        # Price overextended below EMA-20
        ema20 = close.ewm(span=20, adjust=False).mean()
        if float(ema20.iloc[-1]) > 0:
            ext = (cp - float(ema20.iloc[-1])) / float(ema20.iloc[-1])
            if ext < -0.03:
                signals.append(f"OVEREXT_{ext*100:.1f}%")
                score += 0.15

        return min(score, 1.0), signals

    @staticmethod
    def count_confirming_candles(df: pd.DataFrame, direction: str, max_look: int = 5) -> int:
        if len(df) < 5:
            return 0
        # Access contiguous underlying numpy matrix locations directly to prevent IndexErrors
        closes = df['close'].to_numpy()
        opens = df['open'].to_numpy()

        count = 0
        # Loop backwards starting from the last completed candle (-2) down to lookback limits
        for i in range(2, min(max_look + 2, len(df))):
            idx = -i
            if direction == "LONG" and closes[idx] > opens[idx]:
                count += 1
            elif direction == "SHORT" and closes[idx] < opens[idx]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def is_support_broken(current_close: float, support: float, buffer_pct: float = 0.001) -> bool:
        """True when close is below support by more than buffer."""
        if support <= 0:
            return False
        return current_close < support * (1 - buffer_pct)

    @staticmethod
    def is_resistance_broken(current_close: float, resistance: float, buffer_pct: float = 0.001) -> bool:
        """True when close is above resistance by more than buffer."""
        if resistance <= 0:
            return False
        return current_close > resistance * (1 + buffer_pct)


class LiveWallet:
    def __init__(self, initial_balance: float, max_position_usdt: float):
        self.initial_balance = initial_balance   # store for reset
        self.balance = initial_balance
        self.max_position_usdt = max_position_usdt
        self.open_trades: Dict[str, LiveTrade] = {}
        self.trade_history: List[LiveTrade] = []
        self._load_state()

    def _load_state(self):
        wallet_file = CONFIG_DIR / "live_wallet.json"
        if wallet_file.exists():
            try:
                with open(wallet_file, 'r') as f:
                    data = json.load(f)
                self.balance = data.get("balance", self.balance)
                for tdata in data.get("open_trades", []):
                    trade = LiveTrade(**tdata)
                    trade.entry_time = datetime.fromisoformat(tdata["entry_time"])
                    if tdata.get("close_time"):
                        trade.close_time = datetime.fromisoformat(tdata["close_time"])
                    self.open_trades[trade.symbol] = trade
                logger.info(f"Loaded wallet: balance={self.balance:.2f}, open trades={len(self.open_trades)}")
            except Exception as e:
                logger.warning(f"Could not load wallet state: {e}")

    def _save_state(self):
        wallet_file = CONFIG_DIR / "live_wallet.json"
        data = {
            "balance": self.balance,
            "open_trades": [
                {
                    "symbol": t.symbol,
                    "entry_price": t.entry_price,
                    "position_units": t.position_units,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "entry_time": t.entry_time.isoformat(),
                    "mode": t.mode,
                    "master_score": t.master_score,
                    "trade_type": t.trade_type,
                    "closed": t.closed,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "close_time": t.close_time.isoformat() if t.close_time else None,
                }
                for t in self.open_trades.values()
            ],
            "trade_history": [
                {
                    "symbol": t.symbol,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "entry_time": t.entry_time.isoformat(),
                    "close_time": t.close_time.isoformat() if t.close_time else None,
                }
                for t in self.trade_history[-100:]
            ]
        }
        with open(wallet_file, 'w') as f:
            json.dump(data, f, indent=2)

    def reset_wallet(self) -> bool:
        """Backup current wallet, clear trades, reset balance to initial."""
        wallet_file = CONFIG_DIR / "live_wallet.json"
        backup_file = CONFIG_DIR / "live_wallet_backup.json"
        try:
            if wallet_file.exists():
                shutil.copy2(wallet_file, backup_file)
                logger.info(f"Wallet backed up to {backup_file}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

        self.open_trades.clear()
        self.trade_history.clear()
        self.balance = self.initial_balance
        self._save_state()
        logger.info("Wallet reset: balance restored, all trades purged.")
        return True

    def can_open_trade(self, symbol: str, entry_price: float) -> bool:
        if symbol in self.open_trades:
            return False
        position_value = self.max_position_usdt
        if position_value > self.balance:
            position_value = self.balance
        return position_value > 0 and self.balance >= position_value * 0.01

    def open_trade(self, trade: LiveTrade):
        cost = trade.position_units * trade.entry_price
        fee = cost * EXCHANGE_FEE
        self.balance -= (cost + fee)
        self.open_trades[trade.symbol] = trade
        self._save_state()
        logger.info(f"🟢 OPEN {trade.symbol} ({trade.trade_type}) | Units={trade.position_units:.6f} | Entry={trade.entry_price:.4f} | "
                    f"SL={trade.stop_loss:.4f} TP={trade.take_profit:.4f} | Balance=${self.balance:.2f}")

    def close_trade(self, symbol: str, exit_price: float, reason: str):
        trade = self.open_trades.pop(symbol, None)
        if not trade:
            return
        if trade.trade_type == "LONG":
            raw_pnl = trade.position_units * (exit_price - trade.entry_price)
        else:  # SHORT
            raw_pnl = trade.position_units * (trade.entry_price - exit_price)
        exit_fee = trade.position_units * exit_price * EXCHANGE_FEE
        pnl = raw_pnl - exit_fee
        self.balance += (trade.position_units * exit_price - exit_fee)
        trade.pnl = pnl
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.closed = True
        trade.close_time = datetime.now()
        self.trade_history.append(trade)
        self._save_state()
        logger.info(f"🔴 CLOSE {symbol} | Exit={exit_price:.4f} | Reason={reason} | PnL={pnl:+.2f} | Balance=${self.balance:.2f}")

    def get_position_units(self, symbol: str, entry_price: float, sl_price: float, risk_pct: float, confidence_scalar: float = 1.0) -> float:
        risk_amount = self.balance * (risk_pct / 100) * confidence_scalar
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 0.0
        units_by_risk = risk_amount / sl_distance
        max_units_by_cap = self.max_position_usdt / entry_price
        units = min(units_by_risk, max_units_by_cap)
        max_units_by_balance = self.balance / entry_price
        return min(units, max_units_by_balance)

# -------------------------------------------------------------------
# Hardcoded DNS resolver (unchanged)
# -------------------------------------------------------------------
class HardcodedResolver(AbstractResolver):
    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping

    async def resolve(self, hostname: str, port: int = 443, family: AddressFamily = AddressFamily.AF_UNSPEC):
        ip = self._mapping.get(hostname)
        if ip is None:
            from aiohttp.resolver import DefaultResolver
            default_resolver = DefaultResolver()
            return await default_resolver.resolve(hostname, port, family)
        return [{'hostname': hostname, 'host': ip, 'port': port, 'family': family.value, 'proto': 0, 'flags': 0}]

    async def close(self):
        pass

# -------------------------------------------------------------------
# MarketDataFetcher (unchanged)
# -------------------------------------------------------------------
class MarketDataFetcher:
    def __init__(self, proxy_url: Optional[str] = None, activity_log: Optional[deque] = None):
        self.proxy_url = proxy_url
        self.exchange: Optional[Any] = None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.activity_log = activity_log

    def _log_activity(self, msg: str):
        if self.activity_log is not None:
            self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(self.activity_log) > 12:
                self.activity_log.pop()
        logger.info(msg)

    async def start(self):
        self._log_activity("🌐 Starting exchange connection...")
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as temp_session:
                async with temp_session.get('https://api.ipify.org') as resp:
                    if resp.status == 200:
                        ip = await resp.text()
                        self._log_activity(f"🌐 Public IP detected: {ip.strip()}")
                    else:
                        self._log_activity("⚠️ Could not determine public IP (HTTP error)")
        except Exception as e:
            self._log_activity(f"⚠️ Public IP detection skipped: {e}")

        endpoints = [
            'https://api.binance.com',
            'https://api1.binance.com',
            'https://api2.binance.com',
            'https://api3.binance.com',
            'https://api-gcp.binance.com',
        ]

        custom = os.getenv("BINANCE_ENDPOINT")
        if custom and custom.startswith("http"):
            endpoints.insert(0, custom)

        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        dns_mapping = {ep.split('/')[-1]: '99.86.16.124' for ep in endpoints if 'api' in ep}
        resolver = HardcodedResolver(dns_mapping)

        self._connector = aiohttp.TCPConnector(
            resolver=resolver,
            family=socket.AF_INET,
            ssl=False,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )

        session_kwargs = {
            'connector': self._connector,
            'timeout': ClientTimeout(total=60),
            'headers': {'User-Agent': user_agent},
            'trust_env': True,
        }
        if self.proxy_url:
            session_kwargs['proxy'] = self.proxy_url
            self._log_activity(f"🔌 Using proxy: {self.proxy_url}")

        self._session = aiohttp.ClientSession(**session_kwargs)

        exchange_cls = getattr(ccxt_async, EXCHANGE_ID, None)
        if exchange_cls is None:
            self._log_activity(f"❌ Exchange '{EXCHANGE_ID}' not found")
            await self.stop()
            raise RuntimeError(f"CCXT exchange {EXCHANGE_ID} not available")

        max_attempts_per_endpoint = 2
        last_error = None
        connection_success = False

        try:
            for base_url in endpoints:
                for attempt in range(1, max_attempts_per_endpoint + 1):
                    if self.exchange:
                        try:
                            await self.exchange.close()
                        except Exception:
                            pass
                        self.exchange = None

                    try:
                        self._log_activity(f"🔌 Connecting to {base_url} (attempt {attempt})")
                        exchange_config = {
                            'enableRateLimit': True,
                            'aiohttp_trust_env': True,
                            'connector': self._connector,
                            'session': self._session,
                            'options': {'defaultType': 'spot'},
                            'timeout': 30000,
                            'urls': {
                                'api': {
                                    'public': f'{base_url}/api/v3',
                                    'private': f'{base_url}/api/v3',
                                }
                            },
                            'headers': {'User-Agent': user_agent},
                        }
                        if self.proxy_url:
                            exchange_config['aiohttp_proxy'] = self.proxy_url
                        self.exchange = exchange_cls(exchange_config)
                        if self.exchange is None:
                            raise ConnectionError(f"Failed to create exchange instance for {base_url}")
                        await self.exchange.load_markets()
                        self._log_activity(f"✅ Connected to {base_url}")
                        connection_success = True
                        return
                    except Exception as e:
                        last_error = e
                        self._log_activity(f"⚠️ Failed {base_url} attempt {attempt}: {e}")
                        if self.exchange:
                            try:
                                await self.exchange.close()
                            except Exception:
                                pass
                            self.exchange = None
                        if attempt == max_attempts_per_endpoint:
                            break
                        await asyncio.sleep(2 ** attempt)

            if not connection_success:
                self._log_activity("❌ CRITICAL: Hardcoded IP unreachable. VPN may not be routing raw socket traffic.")
                raise ConnectionError(f"All Binance endpoints failed. Last error: {last_error}")
        except Exception as e:
            await self.stop()
            raise

    async def stop(self):
        if self.exchange:
            try:
                await self.exchange.close()
                self.exchange = None
            except Exception as e:
                self._log_activity(f"Exchange close error: {e}")
        if self._session:
            try:
                await self._session.close()
                self._session = None
            except Exception as e:
                self._log_activity(f"Session close error: {e}")
        if self._connector:
            try:
                await self._connector.close()
                self._connector = None
            except Exception as e:
                self._log_activity(f"Connector close error: {e}")

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        if not self.exchange:
            self._log_activity(f"⚠️ No exchange instance for {symbol} {timeframe}")
            return pd.DataFrame()

        for attempt in range(1, 4):
            try:
                raw = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                if attempt > 1:
                    self._log_activity(f"✅ Successfully fetched {symbol} {timeframe} (after retry)")
                return df
            except Exception as e:
                error_str = str(e)
                if '429' in error_str:
                    self._log_activity(f"⚠️ Rate limit (429) for {symbol} {timeframe} (attempt {attempt}/3)")
                elif 'timeout' in error_str.lower():
                    self._log_activity(f"⏱️ Timeout for {symbol} {timeframe} (attempt {attempt}/3)")
                else:
                    self._log_activity(f"❌ Fetch failed for {symbol} {timeframe}: {error_str} (attempt {attempt}/3)")
                if attempt < 3:
                    sleep_seconds = attempt * 2
                    self._log_activity(f"⏳ Waiting {sleep_seconds}s before retry...")
                    await asyncio.sleep(sleep_seconds)
                else:
                    self._log_activity(f"❌ All 3 attempts failed for {symbol} {timeframe}")
        return pd.DataFrame()

    async def get_data(self, symbol: str, timeframe: str, lookback_hours: int) -> pd.DataFrame:
        now = time.time()
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if now - cached['timestamp'] < CACHE_DURATION.get(timeframe, 60):
                return cached['df'].copy()
        limit = get_limit_for_timeframe(timeframe, lookback_hours)
        self._log_activity(f"📥 Fetching {symbol} {timeframe} (limit={limit})")
        df = await self.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not df.empty:
            self.cache[cache_key] = {'df': df, 'timestamp': now}
            self._log_activity(f"✓ Cached {symbol} {timeframe}")
        return df

# -------------------------------------------------------------------
# SignalGenerator – ASYMMETRIC THRESHOLDS + SELLABILITY BRAKE
# -------------------------------------------------------------------
class SignalGenerator:
    def __init__(self, token_configs: List[TokenConfig], btc_model: Optional[xgb.XGBClassifier], alpha_risk_pct: float):
        # Normalize symbols in configs (replace underscore with slash)
        self.configs = {}
        for cfg in token_configs:
            norm_sym = normalize_symbol(cfg.symbol)
            self.configs[norm_sym] = cfg
        self.btc_model = btc_model
        self.alpha_risk_pct = alpha_risk_pct
        self.regime_detector = RegimeDetector()
        self.threshold_engine = ThresholdEngine()
        self.score_history: Dict[str, Deque[float]] = {norm_sym: deque(maxlen=5) for norm_sym in self.configs.keys()}
        self.macro_cache: Dict[str, Dict[str, float]] = {}
        # Tracks reversal candidates per "sym_tf_DIR" key until candle confirmation
        self.reversal_candidates: Dict[str, dict] = {}

    def _align_features(self, df: pd.DataFrame, expected_features: List[str]) -> Optional[pd.DataFrame]:
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            logger.critical(f"Missing expected features: {missing}")
            return None
        aligned = df[expected_features].copy()
        return aligned

    async def compute_macro_trend(self, symbol: str, fetcher: MarketDataFetcher) -> Tuple[str, Dict]:
        norm_sym = normalize_symbol(symbol)
        try:
            # Request an expanded daily history pool (4500 hours / ~180 days) to construct a valid weekly matrix
            df_1d = await fetcher.get_data(norm_sym, '1d', lookback_hours=4500)
            if df_1d.empty or len(df_1d) < 50:
                return 'DIVERGENT', {}

            # Synthetically resample the clean daily dataframe into chronological weekly blocks (ending Sundays)
            df_1w = df_1d.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            if df_1w.empty or len(df_1w) < 2:
                return 'DIVERGENT', {}

            # Calculate 50-period Exponential Moving Averages across both horizons safely
            ema50_1d = df_1d['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            ema50_1w = df_1w['close'].ewm(span=50, adjust=False).mean().iloc[-1]

            close_1d = float(df_1d['close'].iloc[-1])
            close_1w = float(df_1w['close'].iloc[-1])

            bull_1d = close_1d > ema50_1d
            bull_1w = close_1w > ema50_1w

            macro_state = {
                'trend_1d': 1.0 if bull_1d else -1.0,
                'trend_1w': 1.0 if bull_1w else -1.0,
                'confluence_score': float((1.0 if bull_1d else -1.0) + (1.0 if bull_1w else -1.0))
            }
            self.macro_cache[norm_sym] = macro_state

            if bull_1d and bull_1w:
                return 'BULL', { 'ema50_1d': float(ema50_1d), 'ema50_1w': float(ema50_1w), 'close_1d': close_1d, 'close_1w': close_1w }
            if not bull_1d and not bull_1w:
                return 'BEAR', { 'ema50_1d': float(ema50_1d), 'ema50_1w': float(ema50_1w), 'close_1d': close_1d, 'close_1w': close_1w }

            return 'DIVERGENT', { 'ema50_1d': float(ema50_1d), 'ema50_1w': float(ema50_1w), 'close_1d': close_1d, 'close_1w': close_1w }
        except Exception as e:
            logger.debug(f"Synthetic macro trend compute failed for {symbol}: {e}")
            return 'DIVERGENT', {}

    async def compute_signal(self, symbol: str, fetcher: MarketDataFetcher, alpha_mode: bool, timeframe: str = '1h', btc_healthy: bool = True, macro_trend: Optional[str] = None) -> Optional[Dict]:
        # Normalize incoming symbol (e.g., BTC_USDT -> BTC/USDT)
        norm_sym = normalize_symbol(symbol)
        cfg = self.configs.get(norm_sym)
        if not cfg:
            logger.debug(f"No config for symbol {norm_sym} (original {symbol})")
            return None

        current_risk_pct = self.alpha_risk_pct if alpha_mode else cfg.risk_pct

        # Primary timeframe data (allows generating signals per timeframe)
        tf_lookbacks = {
            '1m': 2, '5m': 6, '15m': 24, '30m': 48,
            '1h': 200, '4h': 400, '1d': 800, '1w': 4000
        }
        lookback = tf_lookbacks.get(timeframe, 200)
        df_tf = await fetcher.get_data(norm_sym, timeframe, lookback_hours=lookback)
        if df_tf.empty or len(df_tf) < 20:
            logger.debug(f"{norm_sym} {timeframe}: insufficient data (len={len(df_tf)})")
            return None
        current_price = float(df_tf['close'].iloc[-1])

        # Context: keep 1h features for the core model preprocessing (backward compatible)
        df_1h = await fetcher.get_data(norm_sym, '1h', lookback_hours=200)
        if df_1h.empty or len(df_1h) < 20:
            df_1h = df_tf.copy()

        df_1h['atr'] = compute_atr(df_1h, 14)
        df_1h['volume_ma'] = df_1h['volume'].rolling(VOLUME_LOOKBACK).mean()
        df_1h['efficiency_ratio'] = compute_efficiency_ratio(df_1h['close'], period=10)

        last_idx = len(df_1h) - 1
        vol_regime = RegimeDetector.volatility(df_1h['atr'], last_idx)
        volume_cond = RegimeDetector.volume(df_1h['volume'].iloc[-1], df_1h['volume_ma'].iloc[-1])
        er = df_1h['efficiency_ratio'].iloc[-1]
        trend_str = RegimeDetector.trend(er)
        trend_aligned = RegimeDetector.is_trend_aligned("long", trend_str)

        btc_df = await fetcher.get_data('BTC/USDT', '1h', lookback_hours=200)
        news_score = 0.0
        news_file = DATA_DIR / "news_data.json"
        if news_file.exists():
            try:
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
                if isinstance(news_data, list) and news_data:
                    recent = news_data[-20:]
                    news_score = sum(item.get('sentiment', 0.0) for item in recent) / len(recent)
            except:
                pass

        full_1h = df_1h.reset_index()
        macro_state = self.macro_cache.get(norm_sym)
        features_df = prepare_features(
            full_1h,
            btc_df.reset_index() if not btc_df.empty else None,
            pd.DataFrame([{'timestamp': datetime.now(), 'sentiment': news_score}]),
            add_target_flag=False,
            forward_hours=1,
            df_1d=None,
            df_1w=None,
            macro_state=macro_state
        )
        if features_df is None or features_df.empty:
            return None

        _current_row = features_df.iloc[-1]
        latest_features = features_df.iloc[-1:].drop(columns=['timestamp', 'target'], errors='ignore')

        try:
            dist_to_support = float(_current_row["pct_dist_to_support"])
            dist_to_resistance = float(_current_row["pct_dist_to_resistance"])
            # Only flag NEAR_SUPPORT when price is *above* support (positive dist) within threshold.
            # A negative dist_to_support means price is already below support (broken).
            if 0 <= dist_to_support <= PROXIMITY_THRESHOLD:
                sr_alert_state = "NEAR_SUPPORT"
            elif 0 <= dist_to_resistance <= PROXIMITY_THRESHOLD:
                sr_alert_state = "NEAR_RESISTANCE"
            else:
                sr_alert_state = "NONE"
            sr_telemetry = {
                "support_line": float(_current_row["rolling_support"]),
                "resistance_line": float(_current_row["rolling_resistance"]),
                "dist_to_support_pct": round(dist_to_support * 100, 4),
                "dist_to_resistance_pct": round(dist_to_resistance * 100, 4),
                "alert_state": sr_alert_state,
            }
        except (KeyError, TypeError, ValueError):
            sr_telemetry = {
                "support_line": None, "resistance_line": None,
                "dist_to_support_pct": None, "dist_to_resistance_pct": None,
                "alert_state": "NONE",
            }

        macro_regime = {
            "confluence_score": float(_current_row.get("macro_confluence_score", 0.0)),
            "trend_1d": float(_current_row.get("macro_trend_1d", 0.0)),
        }

        model_path = MODEL_STORE / f"{norm_sym.replace('/', '_')}_model.json"
        if not model_path.exists():
            logger.warning(f"No model for {norm_sym}")
            return None
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        try:
            expected_feature_names = model.get_booster().feature_names
            expected_features = list(expected_feature_names) if expected_feature_names else []
        except Exception as e:
            logger.critical(f"Could not extract feature names from model for {norm_sym}: {e}")
            return None
        if not expected_features:
            logger.critical(f"No expected features extracted from model for {norm_sym}")
            return None

        aligned_features = self._align_features(latest_features, expected_features)
        if aligned_features is None:
            logger.critical(f"Feature mismatch for {norm_sym}: cannot predict")
            return None

        try:
            probs = model.predict_proba(aligned_features)[0]
            
            # --- NEW TELEMETRY ---
            telemetry_data = extract_prediction_telemetry(model, aligned_features, expected_features)
            shap_contribs = telemetry_data.get("shap_contributions", [])
            raw_probs_dict = telemetry_data.get("probabilities", {})
            # --- END TELEMETRY ---

            if len(probs) >= 3:
                prob_short = float(probs[0])
                prob_hold = float(probs[1])
                prob_long = float(probs[2])
                predicted_class = int(np.argmax(probs))
                raw_prob = float(probs[predicted_class])
                
                # Calibrate probabilities to realistic confidence ranges (avoids 99% overconfidence)
                if raw_prob > 0.8:
                    ai_prob = 0.75 + (raw_prob - 0.8) * 0.4
                elif raw_prob > 0.5:
                    ai_prob = 0.55 + (raw_prob - 0.5) * 0.6
                else:
                    ai_prob = raw_prob
            else:
                raw_prob = float(probs[1]) if len(probs) > 1 else 0.0
                predicted_class = 2 if raw_prob >= 0.5 else 0
                
                # Calibrate probabilities to enhance signal generation
                base_prob = raw_prob if predicted_class == 2 else (1 - raw_prob)
                if base_prob > 0.8:
                    ai_prob = 0.75 + (base_prob - 0.8) * 0.4
                elif base_prob > 0.5:
                    ai_prob = 0.55 + (base_prob - 0.5) * 0.6
                else:
                    ai_prob = base_prob
                    
                prob_short, prob_hold, prob_long = (1-ai_prob, 0.0, ai_prob) if predicted_class == 2 else (ai_prob, 0.0, 1-ai_prob)
        except Exception as e:
            ai_prob = 0.0
            prob_short, prob_hold, prob_long = 0.0, 1.0, 0.0
            predicted_class = 1
            shap_contribs = []
            raw_probs_dict = {}

        atr_val = df_1h['atr'].iloc[-1] if not pd.isna(df_1h['atr'].iloc[-1]) else current_price * 0.01

        # Alpha Risk Monitor logic
        atr_series = df_1h['atr'].dropna()
        atr_mean = atr_series.mean() if len(atr_series) > 0 else atr_val
        market_regime = vol_regime
        alpha_risk_level = "NORMAL"
        if atr_val > 3 * atr_mean:
            market_regime = "CRITICAL_VOLATILITY"
            alpha_risk_level = "CRITICAL"
        elif atr_val > 2 * atr_mean:
            market_regime = "HIGH_VOLATILITY"
            alpha_risk_level = "HIGH"

        # Minimum confidence gate — use the per-token threshold set by the mode
        # config (balanced=0.70, aggressive=0.65, conservative=0.75). Raise it by
        # 0.10 during elevated volatility so noisy markets don't flood signals.
        req_confidence = (
            min(0.90, cfg.entry_prob_threshold + 0.10)
            if market_regime in ["CRITICAL_VOLATILITY", "HIGH_VOLATILITY"]
            else cfg.entry_prob_threshold
        )

        # Conviction Spread
        conviction_spread = ai_prob - prob_hold

        # Direction logic: Class 2 = LONG, Class 0 = SHORT, Class 1 = NEUTRAL.
        base_signal = "NEUTRAL"
        if market_regime == "CRITICAL_VOLATILITY":
            base_signal = "NEUTRAL"
        elif predicted_class == 2 and ai_prob >= req_confidence and conviction_spread > 0.05:
            base_signal = "LONG"
        elif predicted_class == 0 and ai_prob >= req_confidence and conviction_spread > 0.05:
            base_signal = "SHORT"

        # BTC safety downgrade for LONG signals (only when Alpha OFF)
        if not alpha_mode and not btc_healthy and base_signal == "LONG":
            base_signal = "NEUTRAL"

        # ──────────────────────────────────────────────────────────────
        # REVERSAL ANTICIPATION ENGINE
        # Signals fire when the trend is *about to* reverse, not after it
        # has already moved ≥50%.  Logic:
        #   1. Detect technical reversal setup (RSI divergence, patterns, S/R)
        #   2. Count 3 consecutive confirming candles in the new direction
        #   3. For SHORT near support: wait for actual support break first
        #   4. Mark signal_status: AWAITING_CONFIRMATION / AWAITING_SR_BREAK / CONFIRMED
        # ──────────────────────────────────────────────────────────────
        rev_key = f"{norm_sym}_{timeframe}"
        signal_status = "ACTIVE"
        candles_confirmed = 0
        reversal_score_val = 0.0
        reversal_signals_list: list = []

        support_lvl = float(sr_telemetry.get("support_line") or 0.0)
        resistance_lvl = float(sr_telemetry.get("resistance_line") or 0.0)

        if base_signal == "LONG":
            bull_score, bull_sigs = ReversalDetector.detect_bullish_reversal_setup(
                df_1h, support=support_lvl
            )
            reversal_score_val = bull_score
            reversal_signals_list = bull_sigs
            candles_confirmed = ReversalDetector.count_confirming_candles(df_1h, "LONG")

            cand_key = f"{rev_key}_LONG"
            if bull_score >= REVERSAL_SCORE_THRESHOLD:
                if cand_key not in self.reversal_candidates:
                    self.reversal_candidates[cand_key] = {
                        "direction": "LONG",
                        "first_price": current_price,
                        "candles_confirmed": candles_confirmed,
                        "score": bull_score,
                    }
                else:
                    self.reversal_candidates[cand_key]["candles_confirmed"] = candles_confirmed

                if candles_confirmed >= CANDLE_CONFIRM_REQUIRED:
                    signal_status = "CONFIRMED"
                    self.reversal_candidates.pop(cand_key, None)
                else:
                    # Preserve signal payload for dashboard tracking; entries blocked via signal_status
                    signal_status = "AWAITING_CONFIRMATION"
            else:
                # Reversal setup not strong enough — clear candidate and neutralise
                self.reversal_candidates.pop(cand_key, None)
                base_signal = "NEUTRAL"
                signal_status = "ACTIVE"

        elif base_signal == "SHORT":
            # ── Support-break gate: SELL near support → wait for break ──
            if sr_alert_state == "NEAR_SUPPORT" and support_lvl > 0:
                support_broken = ReversalDetector.is_support_broken(current_price, support_lvl)
                if not support_broken:
                    # Preserve SHORT payload for dashboard tracking; entries blocked via signal_status
                    signal_status = "AWAITING_SR_BREAK"
                    self.reversal_candidates[f"{rev_key}_SR_BREAK"] = {
                        "support_level": support_lvl,
                        "direction": "SHORT",
                    }
                else:
                    # Support actually broken — allow SHORT to proceed
                    signal_status = "SR_BREAK_CONFIRMED"
                    self.reversal_candidates.pop(f"{rev_key}_SR_BREAK", None)
            else:
                # Not near support; apply standard candle-confirmation
                bear_score, bear_sigs = ReversalDetector.detect_bearish_reversal_setup(
                    df_1h, resistance=resistance_lvl
                )
                reversal_score_val = bear_score
                reversal_signals_list = bear_sigs
                candles_confirmed = ReversalDetector.count_confirming_candles(df_1h, "SHORT")

                cand_key = f"{rev_key}_SHORT"
                if bear_score >= REVERSAL_SCORE_THRESHOLD:
                    if cand_key not in self.reversal_candidates:
                        self.reversal_candidates[cand_key] = {
                            "direction": "SHORT",
                            "first_price": current_price,
                            "candles_confirmed": candles_confirmed,
                            "score": bear_score,
                        }
                    else:
                        self.reversal_candidates[cand_key]["candles_confirmed"] = candles_confirmed

                    if candles_confirmed >= CANDLE_CONFIRM_REQUIRED:
                        signal_status = "CONFIRMED"
                        self.reversal_candidates.pop(cand_key, None)
                    else:
                        # Preserve signal payload for dashboard tracking; entries blocked via signal_status
                        signal_status = "AWAITING_CONFIRMATION"
                else:
                    self.reversal_candidates.pop(cand_key, None)
                    base_signal = "NEUTRAL"
                    signal_status = "ACTIVE"

        # SL/TP Logic based on asset classification
        heavy_caps = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        atr_multiplier = 1.2 if norm_sym in heavy_caps else 1.8
        
        suggested_sl_distance = atr_multiplier * atr_val
        suggested_tp_distance = RR_RATIO * suggested_sl_distance

        if base_signal == "LONG":
            suggested_sl = current_price - suggested_sl_distance
            suggested_tp = current_price + suggested_tp_distance
            final_signal = "BUY"
            signal_strength = "STRONG_BUY" if ai_prob > 0.8 else "BUY"
        elif base_signal == "SHORT":
            suggested_sl = current_price + suggested_sl_distance
            suggested_tp = current_price - suggested_tp_distance
            final_signal = "SELL"
            signal_strength = "STRONG_SELL" if ai_prob > 0.8 else "SELL"
        else:
            suggested_sl = current_price
            suggested_tp = current_price
            final_signal = "HOLD"
            signal_strength = "HOLD"

        # ------------------------------------------------------------------
        # Legacy calculations (for completeness and compatibility)
        # ------------------------------------------------------------------
        base = cfg.base_threshold
        adj_thresh, _ = self.threshold_engine.compute(
            base=base, prob=ai_prob, vol_regime=vol_regime, volume_condition=volume_cond,
            trend_aligned=trend_aligned, atr=atr_val, price=current_price,
        )
        # Further adjust threshold with fundamentals & anchor signals
        _btc_dist = float(btc_df['btc_dist_ema200'].iloc[-1]) if (btc_df is not None and not btc_df.empty and 'btc_dist_ema200' in btc_df.columns) else 0.0
        try:
            adj_thresh = adjust_threshold_by_technical_and_fundamental(
                adj_thresh,
                vol_regime=vol_regime,
                news_score=news_score,
                efficiency_ratio=er,
                btc_anchor=_btc_dist
            )
        except Exception:
            pass
        if alpha_mode:
            adj_thresh = adj_thresh * 0.85

        expected_net_pct = 0.0
        if not alpha_mode:
            atr_sl_dist = cfg.atr_sl * atr_val
            atr_tp_dist = RR_RATIO * atr_sl_dist
            tp_move_pct = atr_tp_dist / current_price
            sl_move_pct = atr_sl_dist / current_price
            expected_net = ai_prob * tp_move_pct - (1 - ai_prob) * sl_move_pct - TOTAL_COST_PCT
            expected_net_pct = expected_net * 100

        btc_ai = 0.5
        if self.btc_model:
            btc_1h = await fetcher.get_data('BTC/USDT', '1h', lookback_hours=200)
            if not btc_1h.empty:
                btc_features = prepare_features(btc_1h.reset_index(), None, None)
                if btc_features is not None and not btc_features.empty:
                    btc_latest = btc_features.iloc[-1:].drop(columns=['timestamp', 'target'], errors='ignore')
                    try:
                        btc_expected = self.btc_model.get_booster().feature_names
                        btc_expected = list(btc_expected) if btc_expected else []
                        btc_aligned = self._align_features(btc_latest, btc_expected)
                        if btc_aligned is not None:
                            btc_ai = self.btc_model.predict_proba(btc_aligned)[0, 1]
                    except Exception as e:
                        logger.warning(f"BTC model prediction error: {e}")

        self.score_history[norm_sym].appendleft(ai_prob)

        confluence_scorecards = {
            "volatility": vol_regime,
            "volume": volume_cond,
            "trend": "Aligned" if trend_aligned else "Misaligned",
            "btc_anchor": "Healthy" if btc_healthy else "Shaky",
            "efficiency": round(float(er), 2),
            "news_sentiment": round(float(news_score), 2)
        }

        visual_zone_tracking = {
            "support_sl": suggested_sl,
            "resistance_tp": suggested_tp,
            "market_regime": market_regime,
            "alpha_risk_level": alpha_risk_level
        }

        expectancy_matrix = {
            "expected_net_pct": round(expected_net_pct, 2) if expected_net_pct else 0.0,
            "profitability_index": round(expected_net_pct / 100, 4) if expected_net_pct else 0.0,
            "max_dd_pct": round(cfg.max_dd, 2) if hasattr(cfg, 'max_dd') else 0.0,
            "historical_expectancy": round(cfg.expectancy, 2) if hasattr(cfg, 'expectancy') else 0.0
        }

        return {
            "symbol": norm_sym,
            "price": current_price,
            "ai_prob": ai_prob,
            "raw_probabilities": raw_probs_dict,
            "shap_contributions": shap_contribs,
            "confluence_scorecards": confluence_scorecards,
            "visual_zone_tracking": visual_zone_tracking,
            "expectancy_matrix": expectancy_matrix,
            "threshold": adj_thresh,
            "expected_net_pct": expected_net_pct,
            "signal": "HOLD" if signal_status in ("AWAITING_CONFIRMATION", "AWAITING_SR_BREAK") else final_signal,
            "signal_strength": signal_strength,
            # Reversal anticipation fields
            "signal_status": signal_status,
            "candles_confirmed": candles_confirmed,
            "reversal_score": round(reversal_score_val, 3),
            "reversal_signals": reversal_signals_list,
            "btc_ai": btc_ai,
            "btc_filter_ok": not (not alpha_mode and not btc_healthy and final_signal in ("STRONG_BUY", "BUY")),
            "vol_regime": vol_regime,
            "volume_cond": volume_cond,
            "trend_aligned": trend_aligned,
            "atr": atr_val,
            "mode": cfg.mode,
            "risk_pct": current_risk_pct,
            "atr_sl": cfg.atr_sl,
            "atr_tp": RR_RATIO * cfg.atr_sl,
            "efficiency_ratio": er,               # added for UI / debugging
            "direction": "NEUTRAL" if signal_status in ("AWAITING_CONFIRMATION", "AWAITING_SR_BREAK") else base_signal,
            "entry_price": current_price,
            "sl": suggested_sl,
            "tp": suggested_tp,
            "suggested_sl_distance": suggested_sl_distance,
            "suggested_tp_distance": suggested_tp_distance,
            "alpha_risk_level": alpha_risk_level,
            "signal_id": str(uuid.uuid4()),
            "trading_accuracy": ai_prob,  # Using ai_prob as trading accuracy
            "profitability_index": expected_net_pct / 100 if expected_net_pct else 0.0,
            "btc_dist_ema200": _btc_dist,
            "data_timestamp": datetime.now().isoformat(),
            "sr_telemetry": sr_telemetry,
            "macro_regime": macro_regime,
        }

# -------------------------------------------------------------------
# Confirmation Engine – four-gate signal validation guardrail
# -------------------------------------------------------------------
_CRITICAL_NUMERIC_FIELDS = ("price", "atr", "ai_prob")

async def confirm_live_signal(
    symbol: str,
    raw_prediction: dict,
    df_row: pd.Series,
) -> Tuple[bool, str]:
    """
    Async guardrail executed before every signal transmission.
    Returns (True, 'OK') when all gates pass, (False, REJECTION_CODE) on first failure.
    """
    # Gate 1a — Time sync: reject if signal data is older than 10 seconds
    try:
        data_ts_str = raw_prediction.get("data_timestamp", "")
        if data_ts_str:
            data_ts = datetime.fromisoformat(data_ts_str)
            if (datetime.now() - data_ts).total_seconds() > 10:
                return False, "STALE_DATA_TIMEOUT"
    except Exception:
        pass

    # Gate 1b — Feature stream integrity: critical numeric fields must be finite and non-zero
    for field in _CRITICAL_NUMERIC_FIELDS:
        val = raw_prediction.get(field)
        if val is None:
            return False, "CORRUPTED_FEATURE_STREAM"
        try:
            fval = float(val)
            if np.isnan(fval) or (fval == 0.0 and field in ("price", "atr")):
                return False, "CORRUPTED_FEATURE_STREAM"
        except (TypeError, ValueError):
            return False, "CORRUPTED_FEATURE_STREAM"

    # Gate 2 — BTC macro confluence: suppress LONG when BTC is deeply below EMA-200
    direction = raw_prediction.get("direction", "NEUTRAL")
    logger.debug(f"[ConfirmationEngine] {symbol} direction={direction}")
    if direction == "LONG":
        btc_dist = raw_prediction.get("btc_dist_ema200")
        if btc_dist is None and "btc_dist_ema200" in df_row.index:
            btc_dist = df_row["btc_dist_ema200"]
        btc_dist = btc_dist if btc_dist is not None else 0.0
        try:
            if float(btc_dist) < -0.02:
                return False, "MACRO_TREND_DIVERGENT"
        except (TypeError, ValueError):
            pass

    # Gate 3 — Liquidity / slippage risk: high volatility paired with thin volume
    vol_regime = raw_prediction.get("vol_regime", "normal")
    volume_cond = raw_prediction.get("volume_cond", "normal")
    if vol_regime == "high" and volume_cond == "low":
        return False, "EXCESSIVE_SLIPPAGE_RISK"

    # Gate 4 — S/R context: suppress counter-directional signals at key levels
    # SHORT at support and LONG at resistance are high-failure setups
    sr_data = raw_prediction.get("sr_telemetry", {})
    sr_state = sr_data.get("alert_state", "NONE") if isinstance(sr_data, dict) else "NONE"
    if direction == "SHORT" and sr_state == "NEAR_SUPPORT":
        return False, "SR_CONTEXT_MISMATCH"
    if direction == "LONG" and sr_state == "NEAR_RESISTANCE":
        return False, "SR_CONTEXT_MISMATCH"

    return True, "OK"

# -------------------------------------------------------------------
# LiveEngine – with RESET (C key) and colour‑coded dashboard
# -------------------------------------------------------------------
class LiveEngine:
    def __init__(self, token_configs: List[TokenConfig], capital: float, max_position_usdt: float,
                 scan_interval_seconds: int, alpha_mode: bool, alpha_risk_pct: float, proxy_url: Optional[str] = None):
        self.token_configs = token_configs
        self.wallet = LiveWallet(capital, max_position_usdt)
        self.activity_log = deque(maxlen=12)
        self.fetcher = MarketDataFetcher(proxy_url, self.activity_log)
        self.btc_model = None
        self.signal_gen = None
        self.scan_interval = scan_interval_seconds
        self.alpha_mode = alpha_mode
        self.alpha_risk_pct = alpha_risk_pct
        self.trading_accuracies = {}

        # Load backtest summary for trading accuracy
        summary_file = LOGS_DIR / "backtests" / "signal_analysis_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                for item in summary_data.get("best_by_accuracy", []):
                    if len(item) >= 3:
                        self.trading_accuracies[item[0]] = float(item[2])
            except Exception as e:
                logger.error(f"Error loading accuracy data: {e}")

        # Warm‑up timer (only for Alpha OFF)
        self.engine_start_time = time.time()
        self.warmup_complete = False

        self.bootstrap_total = len(token_configs) * 2
        self.bootstrap_done = 0
        self.bootstrap_complete = False
        self.last_context_refresh = 0

        self.live_prices: Dict[str, float] = {}
        self.prev_prices: Dict[str, float] = {}
        self.last_signals: Dict[str, Optional[Dict]] = {}
        # Tracks issued signals per "sym_tf" to detect when they expire (>60% move before action)
        self.signal_expiry_tracker: Dict[str, dict] = {}
        self._ticker_task: Optional[asyncio.Task] = None
        self._signal_task: Optional[asyncio.Task] = None
        self._keyboard_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    # --------------------------------------------------------------
    # BTC Safety Check (unchanged)
    # --------------------------------------------------------------
    async def _is_btc_safe(self) -> bool:
        try:
            btc_df = await self.fetcher.get_data('BTC/USDT', '1h', lookback_hours=50)
            if btc_df.empty or len(btc_df) < 30:
                logger.warning("Insufficient BTC data, assuming unsafe.")
                return False

            last_close = float(btc_df['close'].iloc[-1])
            prev_close = float(btc_df['close'].iloc[-2]) if len(btc_df) >= 2 else last_close
            pct_change = (last_close - prev_close) / prev_close * 100.0
            if pct_change >= 0:
                return True

            ema20 = btc_df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            if last_close > ema20:
                return True

            return False
        except Exception as e:
            logger.error(f"BTC safety check failed: {e}")
            return False

    async def announce(self, text: str):
        if not VOICE_AVAILABLE:
            return
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 175)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.warning(f"Voice announcement failed: {e}")
        await asyncio.to_thread(_speak)

    def sync_to_dashboard(self, signals_list):
        output_file = root_dir / "web" / "current_signals.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_file, 'w') as f:
                json.dump({"signals": signals_list}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to sync dashboard: {e}")

    # --------------------------------------------------------------
    # Keyboard listener – added 'C' for reset
    # --------------------------------------------------------------
    async def _keyboard_listener(self):
        loop = asyncio.get_event_loop()
        while not self._shutdown_event.is_set():
            try:
                inp = await loop.run_in_executor(None, sys.stdin.readline)
                if not inp:
                    continue
                inp = inp.strip().lower()
                if inp == 'a':
                    self.alpha_mode = not self.alpha_mode
                    status = "ON" if self.alpha_mode else "OFF"
                    await self.announce(f"Alpha mode {status}")
                    self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Alpha Mode toggled: {status}")
                    logger.info(f"Alpha Mode toggled: {status}")
                elif inp == 'c':
                    print("\n⚠️  DATA PURGE CONFIRMATION ⚠️")
                    confirm = await loop.run_in_executor(None, lambda: input("Confirm Data Purge? [Y/N]: ").strip().upper())
                    if confirm == 'Y':
                        self.wallet.reset_wallet()
                        self.activity_log.clear()
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Wallet reset and activity log purged.")
                        logger.info("Wallet reset and activity log cleared.")
                        await self.announce("Data purge complete")
                    else:
                        print("Reset cancelled.")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Reset cancelled by user.")
                elif inp.startswith('risk'):
                    parts = inp.split()
                    if len(parts) == 2:
                        try:
                            new_risk = float(parts[1])
                            if 0.5 <= new_risk <= 10:
                                self.alpha_risk_pct = new_risk
                                await self.announce(f"Alpha risk set to {new_risk} percent")
                                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Alpha risk updated to {new_risk}%")
                                logger.info(f"Alpha risk updated to {new_risk}%")
                            else:
                                logger.warning("Risk must be between 0.5 and 10")
                        except ValueError:
                            logger.warning("Invalid risk value")
                    else:
                        logger.info(f"Current alpha risk: {self.alpha_risk_pct}% (use 'risk X' to change)")
                elif inp == 'q':
                    logger.info("Quit command received, shutting down...")
                    self._shutdown_event.set()
                    break
            except Exception as e:
                logger.warning(f"Keyboard listener error: {e}")

    async def initialize(self):
        await self.fetcher.start()
        asyncio.create_task(self.announce("Aegis-1 System Online and Scanning"))

        logger.info("Loading BTC model...")
        btc_model_path = MODEL_STORE / "BTC_USDT_model.json"
        if btc_model_path.exists():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(lambda: xgb.XGBClassifier().load_model(str(btc_model_path))),
                    timeout=15.0
                )
                self.btc_model = xgb.XGBClassifier()
                self.btc_model.load_model(str(btc_model_path))
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] BTC model loaded")
                logger.info("BTC model loaded")
            except asyncio.TimeoutError:
                logger.critical("BTC model loading timed out (15s) – continuing without BTC filter")
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ BTC model timeout – filter disabled")
                self.btc_model = None
            except Exception as e:
                logger.critical(f"BTC model load error: {e}")
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ BTC model load error")
                self.btc_model = None
        else:
            logger.warning("BTC model not found – BTC filter disabled")
            self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ BTC model not found – filter disabled")

        logger.info("Starting Signal Generator...")
        self.signal_gen = SignalGenerator(self.token_configs, self.btc_model, self.alpha_risk_pct)
        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Signal Generator ready")
        logger.info("Live engine initialized – starting lazy data load")

        try:
            test_btc = await self.fetcher.get_data('BTC/USDT', '1h', lookback_hours=2)
            if test_btc.empty:
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Data fetch test FAILED – no data")
                logger.critical("ERROR: Markets loaded but price data is empty. Check VPN/DNS.")
                raise ConnectionError("No price data after exchange connection")
            self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Data fetch test passed")
            logger.info("Data fetch test passed")
        except Exception as e:
            logger.critical(f"Data fetch test failed: {e}")
            raise

    async def _fetch_all_tickers(self):
        if not self.fetcher.exchange:
            return
        symbols = [cfg.symbol for cfg in self.token_configs]
        try:
            # First try fetching only our symbols (more efficient but fails if one is invalid)
            tickers = await self.fetcher.exchange.fetch_tickers(symbols)
        except Exception as e:
            logger.warning(f"Ticker fetch error with specific symbols ({e}), falling back to fetch all...")
            try:
                # Fallback: fetch all tickers and filter
                tickers = await self.fetcher.exchange.fetch_tickers()
            except Exception as e_all:
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Ticker fetch fallback error: {e_all}")
                logger.warning(f"Ticker fetch fallback error: {e_all}")
                return

        for sym in symbols:
            ticker = tickers.get(sym)
            if ticker and 'last' in ticker and ticker['last'] is not None:
                price = float(ticker['last'])
                if sym in self.live_prices:
                    self.prev_prices[sym] = self.live_prices[sym]
                self.live_prices[sym] = price

    async def _run_ticker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                await self._fetch_all_tickers()
            except Exception as e:
                logger.warning(f"Ticker loop error: {e}")
            await asyncio.sleep(0.1)
            await asyncio.sleep(max(0, 1 - 0.1))

    async def _refresh_context(self):
        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Refreshing 1h/4h context data...")
        symbols = [cfg.symbol for cfg in self.token_configs] + ['BTC/USDT']
        tasks = []
        for sym in symbols:
            tasks.append(self.fetcher.get_data(sym, '1h', lookback_hours=200))
            tasks.append(self.fetcher.get_data(sym, '4h', lookback_hours=200))
        await asyncio.gather(*tasks)
        self.last_context_refresh = time.time()
        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Context data refresh complete")
        logger.debug("Context data refreshed (1h/4h)")

    async def _run_signal_loop(self):
        token_count = 0
        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting bootstrap data load...")
        for cfg in self.token_configs:
            for tf in ['1h', '4h']:
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 {cfg.symbol} {tf}...")
                try:
                    df = await self.fetcher.get_data(cfg.symbol, tf, lookback_hours=200)
                    if not df.empty:
                        self.bootstrap_done += 1
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {cfg.symbol} {tf} cached ({self.bootstrap_done}/{self.bootstrap_total})")
                        logger.debug(f"Cached {cfg.symbol} {tf} history")
                    else:
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Empty data for {cfg.symbol} {tf}")
                except Exception as e:
                    self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error fetching {cfg.symbol} {tf}: {e}")
                await asyncio.sleep(random.uniform(0.6, 1.2))
            token_count += 1
            if token_count % 10 == 0:
                self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 💤 Cool‑down: paused 5s after {token_count} tokens")
                await asyncio.sleep(5)
            if self.signal_gen:
                try:
                    sig = await self.signal_gen.compute_signal(cfg.symbol, self.fetcher, self.alpha_mode, btc_healthy=True)  # initial bootstrap, assume BTC safe
                    if sig:
                        sig['timeframe'] = '1h'
                        self.last_signals[cfg.symbol] = {
                            tf: dict(sig, timeframe=tf) for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
                        }
                except Exception as e:
                    logger.warning(f"Initial signal compute error for {cfg.symbol}: {e}")
        self.bootstrap_complete = True
        self.last_context_refresh = time.time()
        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Bootstrap complete! Loaded {self.bootstrap_done}/{self.bootstrap_total} datasets")
        logger.info(f"Bootstrap complete: loaded {self.bootstrap_done}/{self.bootstrap_total} context datasets")

        # ====== MAIN SIGNAL LOOP ======
        while not self._shutdown_event.is_set():
            loop_start = time.time()

            # 1. Warm‑up check (only when Alpha Mode OFF)
            if not self.alpha_mode:
                elapsed = time.time() - self.engine_start_time
                remaining = max(0, WARMUP_DURATION - elapsed)
                if elapsed < WARMUP_DURATION:
                    if int(remaining) % 60 == 0 or remaining < 60:
                        mins = int(remaining // 60)
                        secs = int(remaining % 60)
                        logger.info(f"[SENTINEL] Engine warming up... {mins}m {secs}s remaining")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔵 WARMUP: {mins}m {secs}s left")
                    await asyncio.sleep(self.scan_interval)
                    continue
                else:
                    if not self.warmup_complete:
                        self.warmup_complete = True
                        logger.info("[SENTINEL] Warm‑up complete – signal generation enabled.")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Warm‑up complete.")
            else:
                # Alpha Mode ON – skip warm‑up entirely
                pass

            # 2. Refresh context every hour
            if time.time() - self.last_context_refresh > 3600:
                await self._refresh_context()

            # 3. Check BTC safety (BRAKE) – ONLY if Alpha OFF
            btc_safe = True
            if not self.alpha_mode:
                btc_safe = await self._is_btc_safe()
                if not btc_safe:
                    logger.warning("BTC is shaky – all BUY/STRONG_BUY signals will be downgraded to HOLD (BTC_SHAKY).")
                    self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🛑 BTC BRAKE ACTIVE – Buy signals suppressed.")

            # 4. Compute signals for all tokens (pass btc_safe flag)
            if self.signal_gen is None:
                logger.warning("Signal generator not initialized, skipping signal computation")
                await asyncio.sleep(self.scan_interval)
                continue

            self.signal_gen.alpha_risk_pct = self.alpha_risk_pct

            # Macro-first multi-timeframe signal computation
            # Remove '1w' from live prediction blocks; weekly context is handled synthetically via compute_macro_trend
            tf_blocks = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

            # 1) Macro trend computation (parallel)
            macro_tasks = {cfg.symbol: asyncio.create_task(self.signal_gen.compute_macro_trend(cfg.symbol, self.fetcher)) for cfg in self.token_configs}
            macro_results: Dict[str, str] = {}
            for sym, task in macro_tasks.items():
                try:
                    trend, _tele = await task
                    macro_results[sym] = trend
                except Exception:
                    macro_results[sym] = 'DIVERGENT'

            # 2) Per-timeframe compute tasks
            compute_tasks = []
            for cfg in self.token_configs:
                sym = cfg.symbol
                for tf in tf_blocks:
                    compute_tasks.append((sym, tf, asyncio.create_task(self.signal_gen.compute_signal(sym, self.fetcher, self.alpha_mode, timeframe=tf, btc_healthy=btc_safe, macro_trend=macro_results.get(sym)))))

            # 3) Await per-timeframe tasks and apply macro filtering rules
            nested_signals: Dict[str, Dict[str, Optional[Dict]]] = {cfg.symbol: {} for cfg in self.token_configs}
            for sym, tf, task in compute_tasks:
                sig = None
                try:
                    sig = await task
                except Exception:
                    sig = None

                macro = macro_results.get(sym, 'DIVERGENT')
                lower_tfs = ['1m','5m','15m','30m','1h','4h']
                if sig and tf in lower_tfs:
                    direction = sig.get('direction', 'NEUTRAL')
                    if macro == 'BULL' and direction == 'SHORT':
                        sig['signal'] = 'HOLD'
                        sig['signal_strength'] = 'HOLD'
                        sig['direction'] = 'NEUTRAL'
                    elif macro == 'BEAR' and direction == 'LONG':
                        sig['signal'] = 'HOLD'
                        sig['signal_strength'] = 'HOLD'
                        sig['direction'] = 'NEUTRAL'
                    elif macro == 'DIVERGENT':
                        sig['signal'] = 'HOLD'
                        sig['signal_strength'] = 'HOLD'
                        sig['direction'] = 'NEUTRAL'

                if isinstance(sig, dict):
                    sig['timeframe'] = tf

                nested_signals[sym][tf] = sig

            # 4) Confirmation engine per-timeframe (suppress signals if confirmation rejects)
            for sym, tf_map in nested_signals.items():
                for tf, s in tf_map.items():
                    if not s or s.get('direction', 'NEUTRAL') == 'NEUTRAL':
                        continue
                    try:
                        ok, reason = await confirm_live_signal(s['symbol'], s, pd.Series(s))
                        if not ok:
                            logger.warning(f"⚠️ Signal for {s['symbol']} {tf} suppressed by Confirmation Engine. Reason: {reason}")
                            nested_signals[sym][tf] = None
                    except Exception as e:
                        logger.debug(f"Confirmation check failed for {sym} {tf}: {e}")

            # 4b) Signal expiry: if price has moved ≥60% of expected TP from entry without
            #     action, mark signal_status = "EXPIRED" so the dashboard can display it.
            for sym, tf_map in nested_signals.items():
                for tf, sig in tf_map.items():
                    if sig is None:
                        continue
                    direction = sig.get('direction', 'NEUTRAL')
                    current_price_now = sig.get('price', 0.0)
                    sig_key = f"{sym}_{tf}"

                    if direction in ('LONG', 'SHORT') and current_price_now > 0:
                        entry_px = float(sig.get('entry_price', current_price_now))
                        tp_px = float(sig.get('tp', 0.0))
                        expected_move = abs(tp_px - entry_px) if tp_px else 0.0

                        tracked = self.signal_expiry_tracker.get(sig_key)
                        if tracked is None or tracked.get('direction') != direction:
                            # New or direction-changed signal — start fresh tracker
                            self.signal_expiry_tracker[sig_key] = {
                                'entry_price': entry_px,
                                'direction': direction,
                                'expected_move': expected_move,
                            }
                        elif expected_move > 0:
                            orig_entry = tracked['entry_price']
                            if direction == 'LONG':
                                completion = (current_price_now - orig_entry) / expected_move
                            else:
                                completion = (orig_entry - current_price_now) / expected_move

                            if completion >= SIGNAL_EXPIRY_MOVE_PCT:
                                sig['signal_status'] = 'EXPIRED'
                                sig['signal'] = 'EXPIRED'
                                self.signal_expiry_tracker.pop(sig_key, None)
                                logger.info(f"⏰ Signal EXPIRED: {sym} {tf} moved {completion*100:.0f}% of expected range before action")
                    else:
                        # NEUTRAL — clear any stale expiry tracker for this pair/tf
                        self.signal_expiry_tracker.pop(sig_key, None)

            # 5) Persist nested results into last_signals (symbol -> timeframe -> signal)
            for sym, tf_map in nested_signals.items():
                self.last_signals[sym] = tf_map

            # 6) Build a flattened frontend_signals list used by older UI paths: pick 1h as summary if present
            signals = []
            for sym, tf_map in nested_signals.items():
                chosen = tf_map.get('1h')
                if not chosen:
                    # Pick first available non-null timeframe in order
                    for tf in tf_blocks:
                        if tf_map.get(tf):
                            chosen = tf_map.get(tf)
                            break
                if chosen:
                    if 'signal_id' not in chosen:
                        chosen['signal_id'] = str(uuid.uuid4())
                    signals.append(chosen)

            frontend_signals = []
            for sig in signals:
                if sig:
                    if "signal_id" not in sig:
                        sig["signal_id"] = str(uuid.uuid4())

                    sym = sig["symbol"]
                    acc = self.trading_accuracies.get(sym, 0.65)
                    tp_dist = sig.get("suggested_tp_distance", 0)
                    sl_dist = sig.get("suggested_sl_distance", 0)
                    profitability_index = (acc * tp_dist) - ((1 - acc) * sl_dist)

                    frontend_signals.append({
                        "signal_id": sig["signal_id"],
                        "symbol": sym,
                        "timestamp": datetime.now().isoformat(),
                        "direction": sig.get("direction", "NEUTRAL"),
                        "confidence": sig["ai_prob"],
                        "entry_price": sig["price"],
                        "suggested_sl": sig.get("suggested_sl"),
                        "suggested_tp": sig.get("suggested_tp"),
                        "alpha_risk_level": sig.get("alpha_risk_level", "NORMAL"),
                        "trading_accuracy": acc,
                        "profitability_index": profitability_index,
                        "raw_probabilities": sig.get("raw_probabilities", {}),
                        "shap_contributions": sig.get("shap_contributions", []),
                        "confluence_scorecards": sig.get("confluence_scorecards", {}),
                        "visual_zone_tracking": sig.get("visual_zone_tracking", {}),
                        "expectancy_matrix": sig.get("expectancy_matrix", {}),
                        "sr_telemetry": sig.get("sr_telemetry", {}),
                    })

            # Push signals to dashboard
            if frontend_signals:
                self.sync_to_dashboard(frontend_signals)

            # CONVICTION-BASED EXITS
            for sig in signals:
                if not sig:
                    continue
                symbol = sig["symbol"]
                trade = self.wallet.open_trades.get(symbol)
                if not trade:
                    continue

                current_price = self.live_prices.get(symbol)
                if current_price is None:
                    current_price = sig.get("price") if sig and "price" in sig else trade.entry_price
                try:
                    current_price = float(current_price) if current_price is not None else float(trade.entry_price)
                except (TypeError, ValueError):
                    current_price = float(trade.entry_price)

                ai_prob = float(sig.get("ai_prob", 0.0))

                if trade.trade_type == "LONG" and ai_prob < CONVICTION_LONG_BAILOUT:
                    self.wallet.close_trade(symbol, current_price, "BEARISH_CONVICTION")
                    await self.announce(f"Bearish conviction on {symbol}, exiting long")
                    self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 BEARISH_CONVICTION on {symbol} @ {current_price:.4f} (AI prob {ai_prob:.2f})")
                elif trade.trade_type == "SHORT" and ai_prob > CONVICTION_SHORT_BAILOUT:
                    self.wallet.close_trade(symbol, current_price, "BULLISH_CONVICTION")
                    await self.announce(f"Bullish conviction on {symbol}, exiting short")
                    self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 BULLISH_CONVICTION on {symbol} @ {current_price:.4f} (AI prob {ai_prob:.2f})")

            # STOP LOSS / TAKE PROFIT
            for sig in signals:
                if not sig:
                    continue
                symbol = sig["symbol"]
                trade = self.wallet.open_trades.get(symbol)
                if not trade:
                    continue

                current_price = self.live_prices.get(symbol)
                if current_price is None:
                    current_price = sig.get("price") if sig and "price" in sig else trade.entry_price
                try:
                    current_price = float(current_price) if current_price is not None else float(trade.entry_price)
                except (TypeError, ValueError):
                    current_price = float(trade.entry_price)

                if trade.trade_type == "LONG":
                    if trade.stop_loss is not None and current_price <= trade.stop_loss:
                        self.wallet.close_trade(symbol, trade.stop_loss, "STOP_LOSS")
                        await self.announce(f"Stop loss triggered on {symbol}")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 STOP LOSS on {symbol} @ {current_price:.4f}")
                    elif trade.take_profit is not None and current_price >= trade.take_profit:
                        self.wallet.close_trade(symbol, trade.take_profit, "TAKE_PROFIT")
                        await self.announce(f"Take profit secured on {symbol}")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 TAKE PROFIT on {symbol} @ {current_price:.4f}")
                else:  # SHORT
                    if trade.stop_loss is not None and current_price >= trade.stop_loss:
                        self.wallet.close_trade(symbol, trade.stop_loss, "STOP_LOSS")
                        await self.announce(f"Stop loss triggered on {symbol} (short)")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 STOP LOSS (short) on {symbol} @ {current_price:.4f}")
                    elif trade.take_profit is not None and current_price <= trade.take_profit:
                        self.wallet.close_trade(symbol, trade.take_profit, "TAKE_PROFIT")
                        await self.announce(f"Take profit secured on {symbol} (short)")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 TAKE PROFIT (short) on {symbol} @ {current_price:.4f}")

            # OPEN NEW TRADES (BUY or SELL) – only if signal not suppressed
            for sig in signals:
                if not sig:
                    continue
                symbol = sig["symbol"]
                if symbol in self.wallet.open_trades:
                    continue

                signal_type = sig["signal"]
                # Skip suppressed signals (HOLD variants) for entry
                if "HOLD" in signal_type:
                    continue

                price = sig["price"]
                atr_val = sig["atr"]
                risk_pct = sig["risk_pct"]
                atr_sl = sig["atr_sl"]
                atr_tp = sig["atr_tp"]

                if signal_type in ("STRONG_BUY", "BUY") and self.wallet.can_open_trade(symbol, price):
                    sl_price = price - atr_sl * atr_val
                    tp_price = price + atr_tp * atr_val
                    confidence = 1.0 if signal_type == "STRONG_BUY" else 0.6
                    units = self.wallet.get_position_units(symbol, price, sl_price, risk_pct, confidence)
                    if units > 0:
                        trade = LiveTrade(
                            symbol=symbol, entry_price=price, position_units=units,
                            stop_loss=sl_price, take_profit=tp_price, entry_time=datetime.now(),
                            mode=sig["mode"], master_score=sig["ai_prob"], trade_type="LONG"
                        )
                        self.wallet.open_trade(trade)
                        action = "Strong Buy" if signal_type == "STRONG_BUY" else "Buy"
                        await self.announce(f"{action} {symbol}")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 {action.upper()} {symbol} @ {price:.4f} (risk {risk_pct}%)")

                elif signal_type in ("STRONG_SELL", "SELL") and self.wallet.can_open_trade(symbol, price):
                    sl_price = price + atr_sl * atr_val
                    tp_price = price - atr_tp * atr_val
                    confidence = 1.0 if signal_type == "STRONG_SELL" else 0.6
                    units = self.wallet.get_position_units(symbol, price, sl_price, risk_pct, confidence)
                    if units > 0:
                        trade = LiveTrade(
                            symbol=symbol, entry_price=price, position_units=units,
                            stop_loss=sl_price, take_profit=tp_price, entry_time=datetime.now(),
                            mode=sig["mode"], master_score=sig["ai_prob"], trade_type="SHORT"
                        )
                        self.wallet.open_trade(trade)
                        action = "Strong Sell" if signal_type == "STRONG_SELL" else "Sell"
                        await self.announce(f"{action} {symbol}")
                        self.activity_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 {action.upper()} {symbol} @ {price:.4f} (risk {risk_pct}%)")

            elapsed = time.time() - loop_start
            sleep_time = max(0, self.scan_interval - elapsed)
            await asyncio.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Colour‑coded dashboard display
    # ------------------------------------------------------------------
    def _colorize_signal(self, signal: str) -> str:
        """Return ANSI coloured signal string."""
        if signal.startswith("STRONG_BUY") or signal.startswith("BUY"):
            return f"\033[36m{signal}\033[0m"          # Cyan
        elif signal.startswith("STRONG_SELL") or signal.startswith("SELL"):
            return f"\033[31m{signal}\033[0m"          # Red
        else:
            return f"\033[90m{signal}\033[0m"          # Dim grey

    async def _display_dashboard(self):
        while not self._shutdown_event.is_set():
            os.system('cls' if os.name == 'nt' else 'clear')
            now = datetime.now()
            total_unrealized_usd = 0.0
            for sym, trade in self.wallet.open_trades.items():
                current = self.live_prices.get(sym, trade.entry_price)
                if trade.trade_type == "LONG":
                    pnl_raw = trade.position_units * (current - trade.entry_price)
                else:
                    pnl_raw = trade.position_units * (trade.entry_price - current)
                pnl_usd = pnl_raw - (trade.position_units * current * EXCHANGE_FEE)
                total_unrealized_usd += pnl_usd

            total_equity = self.wallet.balance + total_unrealized_usd

            if not self.bootstrap_complete:
                ai_status = f"WARMING UP - {self.bootstrap_done}/{self.bootstrap_total}"
            else:
                ai_status = "ACTIVE"

            print("\n" + "="*140)
            print(f"AEGIS‑1 DASHBOARD – {now.strftime('%Y-%m-%d %H:%M:%S')}  |  ALPHA: {'ON' if self.alpha_mode else 'OFF'}  |  Alpha Risk: {self.alpha_risk_pct}%")
            print(f"Balance: ${self.wallet.balance:.2f} | Unrealized PnL: ${total_unrealized_usd:+.2f} | Equity: ${total_equity:.2f}")
            print(f"Max Position: ${self.wallet.max_position_usdt:.0f} | Scan Interval: {self.scan_interval}s")
            print(f"Open Trades: {len(self.wallet.open_trades)} | Total History: {len(self.wallet.trade_history)}")
            print(f"AI STATUS: [{ai_status}]")
            print("Keys: [A] Toggle Alpha Mode  |  [C] RESET Wallet (purge trades)  |  [risk X] Set Alpha Risk  |  [Q] Quit")
            print("="*140)
            print(f"{'Symbol':<12} | {'Mode':<10} | {'Live Price':>10} | {'Δ':<2} | {'AI Prob':>6} | {'Thresh':>6} | {'Signal':<20} | {'Risk':<12} | {'PnL %':>8} | {'Status':<10}")
            print("-"*140)

            for cfg in self.token_configs:
                sym = cfg.symbol
                price = self.live_prices.get(sym, 0.0)
                if price == 0.0:
                    continue
                prev = self.prev_prices.get(sym, price)
                delta_symbol = "▲" if price > prev else ("▼" if price < prev else " ")
                sig = self.last_signals.get(sym)
                # Resolve nested timeframe dictionary to summary signal
                if sig is not None and isinstance(sig, dict) and any(tf in sig for tf in ('1m','5m','15m','30m','1h','4h','1d','1w')):
                    tf_sig = sig.get('1h')
                    if tf_sig is None:
                        for tf in ('4h','15m','30m','1m','5m','1d','1w'):
                            if sig.get(tf) is not None:
                                tf_sig = sig[tf]
                                break
                    sig = tf_sig

                if sig is None:
                    ai_str, thresh_str, signal_str, risk_str = "0.000", "0.00", "WAITING", "-"
                else:
                    ai_str = f"{sig['ai_prob']:.3f}"
                    thresh_str = f"{sig['threshold']:.3f}"
                    signal_str = self._colorize_signal(sig.get("signal", "UNKNOWN")[:20])
                    risk_str = f"{sig['risk_pct']:.1f}%" if 'risk_pct' in sig else "-"
                mode_display = cfg.mode.capitalize()[:10]
                trade = self.wallet.open_trades.get(sym)
                if trade:
                    if trade.trade_type == "LONG":
                        pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
                    else:
                        pnl_pct = (trade.entry_price - price) / trade.entry_price * 100
                    pnl_str = f"{pnl_pct:+.2f}%"
                else:
                    pnl_str = "   -   "
                status = f"{trade.trade_type[:4]}" if trade else "SCANNING"
                print(f"{sym:<12} | {mode_display:<10} | {price:10.4f} | {delta_symbol:2} | {ai_str:>6} | {thresh_str:>6} | {signal_str:<20} | {risk_str:<12} | {pnl_str:>8} | {status:<10}")

            if self.wallet.open_trades:
                print("\n" + "*"*140)
                print("OPEN POSITIONS DETAILS:")
                for sym, trade in self.wallet.open_trades.items():
                    current = self.live_prices.get(sym, trade.entry_price)
                    if trade.trade_type == "LONG":
                        pnl_pct = (current - trade.entry_price) / trade.entry_price * 100
                        unreal_pnl_usd = trade.position_units * (current - trade.entry_price) - (trade.position_units * current * EXCHANGE_FEE)
                    else:
                        pnl_pct = (trade.entry_price - current) / trade.entry_price * 100
                        unreal_pnl_usd = trade.position_units * (trade.entry_price - current) - (trade.position_units * current * EXCHANGE_FEE)
                    print(f"  {sym:<12} | {trade.trade_type:<5} | Entry: {trade.entry_price:.4f} | Units: {trade.position_units:.4f} | Current: {current:.4f} | PnL: {pnl_pct:+.2f}% (${unreal_pnl_usd:+.2f}) | SL: {trade.stop_loss:.4f} | TP: {trade.take_profit:.4f}")
                print("*"*140)

            if self.activity_log:
                print("\n" + "─"*140)
                print("📋 RECENT ACTIVITY:")
                for line in list(self.activity_log)[:8]:
                    print(f"  {line}")
                print("─"*140)

            await asyncio.sleep(1)

    async def run(self):
        # Start fetching live prices immediately so the dashboard shows real prices
        # during the bootstrap phase rather than waiting for initialize() to complete.
        self._ticker_task = asyncio.create_task(self._run_ticker_loop())
        await self.initialize()
        logger.info("Entering Main Loop Heartbeat...")

        guidelines = """
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TRADER GUIDELINES – READ CAREFULLY                                      │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. PAPER TRADE FIRST – Run in simulation mode for at least 48 hours before real funds.              │
│ 2. STOP LOSS IS HOLY – Never manually remove or move an AI‑set stop loss.                           │
│ 3. THE 3% RULE – Never risk more than 3% of total capital on a single trade.                        │
│ 4. INTERNET STABILITY – Ensure VPN is stable; a disconnection could prevent closing trades.         │
│ 5. MONITOR DASHBOARD – Watch for abnormal PnL or connectivity warnings.                             │
│ 6. LOGS ARE YOUR FRIEND – Check logs/ live_engine.log after any unexpected behavior.               │
│ 7. ALPHA MODE – When active, BTC filter and warm‑up are bypassed.                                  │
│ 8. ALPHA RISK – Use 'risk X' to change risk percentage for Alpha Mode (e.g., risk 4.5).            │
│ 9. BI-DIRECTIONAL TRADING – BUY and SELL signals are asymmetric (sell thresholds stricter).        │
│10. CONVICTION EXITS – Longs exit if AI < 40%, Shorts exit if AI > 60%.                             │
│11. SIGNAL GRID – Asymmetric thresholds: SELL<0.20, STRONG_SELL<0.15, expanded HOLD 0.20-0.60.      │
│12. SELLABILITY BRAKE – Sell signals suppressed during low volatility or choppy markets.            │
│13. DATA PURGE – Press 'C' to reset wallet (backup created) and clear activity log.                │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
        print(guidelines)
        self._signal_task = asyncio.create_task(self._run_signal_loop())
        self._keyboard_task = asyncio.create_task(self._keyboard_listener())
        await self._display_dashboard()

    async def shutdown(self):
        self._shutdown_event.set()
        if self._ticker_task:
            self._ticker_task.cancel()
        if self._signal_task:
            self._signal_task.cancel()
        if self._keyboard_task:
            self._keyboard_task.cancel()
        await self.fetcher.stop()
        logger.info("Engine stopped")

# -------------------------------------------------------------------
# Automated setup (with fallback to default configs)
# -------------------------------------------------------------------
def automated_setup(backtest_dir: Path, args: argparse.Namespace) -> Tuple[List[TokenConfig], float, float, int, bool, float, Optional[str]]:
    # Ensure backtest directory exists (create if missing)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    json_path = get_latest_backtest_file(backtest_dir)
    if not json_path:
        logger.critical("⚠️ No backtest JSON found. Switching to DEFAULT configuration for all tokens.")
        # Fallback: create default TokenConfig for every symbol in FLEET using balanced mode
        configs = []
        # Safely retrieve balanced mode parameters, fallback to hardcoded values
        balanced_params = MODE_PARAMS.get("balanced", {"entry_prob": 0.70, "risk_pct": 0.020, "atr_sl": 1.5, "atr_tp": 2.0})
        for symbol in FLEET:
            configs.append(TokenConfig(
                symbol=symbol,
                mode="balanced",
                base_threshold=0.30,
                entry_prob_threshold=balanced_params["entry_prob"],
                atr_sl=balanced_params["atr_sl"],
                atr_tp=balanced_params["atr_tp"],
                risk_pct=float(balanced_params["risk_pct"] * 100),   # explicit float conversion
                optimizer_thresholds={},
            ))
        capital = args.capital if args.capital else DEFAULT_CAPITAL
        risk_pct = args.risk if args.risk else DEFAULT_RISK_PCT
        max_pos = args.max_position if args.max_position else DEFAULT_MAX_POSITION
        if max_pos <= 0 or max_pos > capital:
            max_pos = capital
        tf_choice = args.timeframe if args.timeframe else DEFAULT_SCAN_TIMEFRAME
        tf_map = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '1d': 86400, '1w': 604800, '1M': 2592000}
        scan_seconds = tf_map.get(tf_choice, 60)
        alpha_mode = args.alpha_mode if args.alpha_mode else DEFAULT_ALPHA_MODE
        alpha_risk = args.alpha_risk if args.alpha_risk else DEFAULT_ALPHA_RISK_PCT
        proxy_url = args.proxy if hasattr(args, 'proxy') else None
        logger.info(f"Using default configs for {len(configs)} tokens (no backtest file)")
        return configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy_url

    logger.info(f"Using backtest file: {json_path.name}")
    best_per_token = load_backtest_results(json_path)

    if not best_per_token:
        logger.critical("⚠️ No valid tokens found in backtest JSON. Falling back to default configuration.")
        configs = []
        balanced_params = MODE_PARAMS.get("balanced", {"entry_prob": 0.70, "risk_pct": 0.020, "atr_sl": 1.5, "atr_tp": 2.0})
        for symbol in FLEET:
            configs.append(TokenConfig(
                symbol=symbol,
                mode="balanced",
                base_threshold=0.30,
                entry_prob_threshold=balanced_params["entry_prob"],
                atr_sl=balanced_params["atr_sl"],
                atr_tp=balanced_params["atr_tp"],
                risk_pct=float(balanced_params["risk_pct"] * 100),
                max_dd=0.0,
                expectancy=0.0,
                optimizer_thresholds={},
            ))
        capital = args.capital if args.capital else DEFAULT_CAPITAL
        risk_pct = args.risk if args.risk else DEFAULT_RISK_PCT
        max_pos = args.max_position if args.max_position else DEFAULT_MAX_POSITION
        if max_pos <= 0 or max_pos > capital:
            max_pos = capital
        tf_choice = args.timeframe if args.timeframe else DEFAULT_SCAN_TIMEFRAME
        tf_map = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '1d': 86400, '1w': 604800, '1M': 2592000}
        scan_seconds = tf_map.get(tf_choice, 60)
        alpha_mode = args.alpha_mode if args.alpha_mode else DEFAULT_ALPHA_MODE
        alpha_risk = args.alpha_risk if args.alpha_risk else DEFAULT_ALPHA_RISK_PCT
        proxy_url = args.proxy if hasattr(args, 'proxy') else None
        logger.info(f"Using default configs for {len(configs)} tokens (fallback from empty backtest)")
        return configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy_url

    configs = []
    for symbol, info in best_per_token.items():
        configs.append(TokenConfig(
            symbol=symbol,
            mode=info["mode"],
            base_threshold=info["final_threshold"],
            entry_prob_threshold=info["entry_prob_threshold"],
            atr_sl=info["atr_sl"],
            atr_tp=info["atr_tp"],
            risk_pct=info["risk_pct"],
            max_dd=info.get("max_drawdown_pct", 0.0),
            expectancy=info.get("net_return_pct", 0.0),
            optimizer_thresholds={},
        ))

    capital = args.capital if args.capital else DEFAULT_CAPITAL
    risk_pct = args.risk if args.risk else DEFAULT_RISK_PCT
    max_pos = args.max_position if args.max_position else DEFAULT_MAX_POSITION
    if max_pos <= 0 or max_pos > capital:
        max_pos = capital

    tf_choice = args.timeframe if args.timeframe else DEFAULT_SCAN_TIMEFRAME
    tf_map = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '1d': 86400, '1w': 604800, '1M': 2592000}
    scan_seconds = tf_map.get(tf_choice, 60)

    alpha_mode = args.alpha_mode if args.alpha_mode else DEFAULT_ALPHA_MODE
    alpha_risk = args.alpha_risk if args.alpha_risk else DEFAULT_ALPHA_RISK_PCT
    proxy_url = args.proxy if hasattr(args, 'proxy') else None

    logger.info(f"Automated config: capital=${capital:.2f}, risk={risk_pct}%, max_pos=${max_pos:.0f}, "
                f"timeframe={tf_choice}, alpha_mode={alpha_mode}, alpha_risk={alpha_risk}%, proxy={proxy_url}")
    return configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy_url

# -------------------------------------------------------------------
# Command line argument parsing
# -------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Aegis-1 Automated Live Trading Engine")
    parser.add_argument("--capital", type=float, help="Starting USDT balance")
    parser.add_argument("--risk", type=float, help="Base risk per trade (%) [1-5]")
    parser.add_argument("--alpha-risk", type=float, help="Risk % when Alpha Mode is ON (default 3.0)")
    parser.add_argument("--max-position", type=float, help="Max USDT per single trade")
    parser.add_argument("--timeframe", choices=['1m','5m','15m','30m','1h','1d','1w','1M'],
                        help="Scanning timeframe (price update frequency)")
    parser.add_argument("--alpha-mode", action="store_true", help="Start with Alpha Mode ON")
    parser.add_argument("--proxy", type=str, help="Proxy URL (e.g., http://127.0.0.1:7890)")
    return parser.parse_args()

# -------------------------------------------------------------------
# Main entry point (environment-aware path)
# -------------------------------------------------------------------
async def main():
    flush_dns_caches()

    args = parse_arguments()
    # Use dynamic path based on root_dir
    backtest_dir = root_dir / "logs" / "backtests"

    configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy_url = automated_setup(backtest_dir, args)
    if not configs:
        logger.error("No token configurations loaded.")
        return

    engine = LiveEngine(configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy_url)
    try:
        await engine.initialize()
    except ConnectionError as e:
        logger.critical(f"Connection failed: {e}")
        print("\n❌ Could not connect to Binance. Check your network or VPN.")
        return
    except Exception as e:
        logger.critical(f"Initialization error: {e}")
        return

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Shutdown by user")
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    import socket
    asyncio.run(main())
