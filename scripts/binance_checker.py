#!/usr/bin/env python3
"""
Aegis-1 Live Signal Engine v1.0
Dynamic backtest file discovery, flat list JSON parsing, MAR‑based mode selection.
Uses final_threshold from backtest for live threshold.
Robust network layer: proxy support, URL rotation, system DNS, user‑agent masking.
"""

import asyncio
import json
import logging
import os
import sys
import time
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
from aiohttp import ClientTimeout

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.ml.feature_engine import prepare_features, compute_atr, compute_efficiency_ratio

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

TIMEFRAMES = {'1m': 60, '1h': 3600, '4h': 14400}
CACHE_DURATION = {'1m': 60, '1h': 3600, '4h': 14400}

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

MODE_PARAMS = {
    "conservative": {"entry_prob": 0.75, "risk_pct": 0.015, "atr_sl": 1.2, "atr_tp": 1.8},
    "balanced":     {"entry_prob": 0.70, "risk_pct": 0.020, "atr_sl": 1.5, "atr_tp": 2.0},
    "aggressive":   {"entry_prob": 0.65, "risk_pct": 0.030, "atr_sl": 1.8, "atr_tp": 2.5},
}

BTC_FILTER_THRESHOLD = 0.45

# Global proxy setting (can be overwritten by environment or user input)
GLOBAL_PROXY = os.getenv("PROXY_URL", None)  # Example: "http://127.0.0.1:7890"

# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------
@dataclass
class TokenConfig:
    symbol: str
    mode: str
    base_threshold: float          # final_threshold from backtest
    entry_prob_threshold: float
    atr_sl: float
    atr_tp: float
    risk_pct: float
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
# Regime & Threshold Engines (unchanged)
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
# Dynamic file discovery
# -------------------------------------------------------------------
def get_latest_backtest_results(directory: Path) -> Optional[Path]:
    """Return the newest .json file in directory (by modification time)."""
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return None
    json_files = list(directory.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {directory}")
        return None
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Latest backtest file: {latest.name} (modified: {datetime.fromtimestamp(latest.stat().st_mtime)})")
    return latest

# -------------------------------------------------------------------
# Robust JSON loader for flat list format
# -------------------------------------------------------------------
def load_backtest_results(json_path: Path, min_trades: int = 5) -> Dict[str, Dict[str, Any]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Expected JSON list, got {type(data)}")

    symbol_entries: Dict[str, List[Dict]] = {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        total_trades = entry.get("total_trades", 0)
        if total_trades < min_trades:
            continue
        mode = entry.get("mode")
        if not mode:
            continue
        net_return = entry.get("net_return_pct", 0.0)
        max_dd = entry.get("max_drawdown_pct", 0.0)
        safe_dd = max(max_dd, 0.1)
        mar = net_return / safe_dd

        params = MODE_PARAMS.get(mode, MODE_PARAMS["balanced"])
        symbol_entries.setdefault(symbol, []).append({
            "mode": mode,
            "mar": mar,
            "net_return_pct": net_return,
            "max_drawdown_pct": max_dd,
            "total_trades": total_trades,
            "win_rate": entry.get("win_rate", 0.0),
            "profit_factor": entry.get("profit_factor"),
            "final_threshold": entry.get("final_threshold", 0.30),
            "entry_prob_threshold": params["entry_prob"],
            "atr_sl": params["atr_sl"],
            "atr_tp": params["atr_tp"],
            "risk_pct": params["risk_pct"],
        })

    best_per_token = {}
    for symbol, entries in symbol_entries.items():
        best = max(entries, key=lambda x: x["mar"])
        best_per_token[symbol] = best

    logger.info(f"Loaded {len(best_per_token)} tokens with ≥{min_trades} trades")
    return best_per_token

def load_optimizer_thresholds(json_path: Path) -> Dict[str, Dict[Tuple[str, str], float]]:
    opt_path = json_path.parent / "adaptive_thresholds.json"
    if not opt_path.exists():
        return {}
    with open(opt_path, 'r') as f:
        data = json.load(f)
    out = {}
    for res in data.get("results", []):
        sym = res["symbol"]
        sym_map = {}
        for regime in res.get("regime_thresholds", []):
            key = (regime["volatility"], regime["volume"])
            sym_map[key] = float(regime["optimal_threshold_pct"])
        if sym_map:
            out[sym] = sym_map
    return out

# -------------------------------------------------------------------
# LiveWallet (unchanged)
# -------------------------------------------------------------------
class LiveWallet:
    def __init__(self, initial_balance: float, max_position_usdt: float):
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
        logger.info(f"🟢 OPEN {trade.symbol} | Units={trade.position_units:.6f} | Entry={trade.entry_price:.4f} | "
                    f"SL={trade.stop_loss:.4f} TP={trade.take_profit:.4f} | Balance=${self.balance:.2f}")

    def close_trade(self, symbol: str, exit_price: float, reason: str):
        trade = self.open_trades.pop(symbol, None)
        if not trade:
            return
        if trade.trade_type == "LONG":
            raw_pnl = trade.position_units * (exit_price - trade.entry_price)
        else:
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
# REFACTORED MarketDataFetcher – Regional Resilience
# -------------------------------------------------------------------
class MarketDataFetcher:
    def __init__(self, proxy_url: Optional[str] = None):
        # Use passed proxy or fallback to global GLOBAL_PROXY
        self.proxy = proxy_url or GLOBAL_PROXY
        self.exchange: Optional[Any] = None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        # List of Binance API endpoints to rotate
        endpoints = [
            'https://api.binance.com',
            'https://api1.binance.com',
            'https://api2.binance.com',
            'https://api3.binance.com',
        ]
        # Override with custom endpoint from env if set
        custom = os.getenv("BINANCE_ENDPOINT")
        if custom and custom.startswith("http"):
            endpoints.insert(0, custom)

        # Browser‑like User‑Agent to avoid simple packet inspection
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        # Create a reusable aiohttp session with connector
        connector_args = {
            "resolver": aiohttp.ThreadedResolver(),   # Use system DNS (no aiodns)
            "verify_ssl": False,                      # Avoid SSL certificate issues (optional)
            "enable_cleanup_closed": True,
            "ttl_dns_cache": 300,
        }
        if self.proxy:
            connector_args["proxy"] = self.proxy
            logger.info(f"Using proxy: {self.proxy}")
        self._connector = aiohttp.TCPConnector(**connector_args)

        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=ClientTimeout(total=30),
            headers={"User-Agent": user_agent}
        )

        exchange_cls = getattr(ccxt_async, EXCHANGE_ID, None)
        if exchange_cls is None:
            logger.error(f"Exchange '{EXCHANGE_ID}' not found in ccxt.async_support")
            return

        # Exponential backoff with endpoint rotation
        max_attempts_per_endpoint = 2
        total_endpoints = len(endpoints)
        last_error = None

        for ep_idx, base_url in enumerate(endpoints):
            for attempt in range(1, max_attempts_per_endpoint + 1):
                try:
                    logger.info(f"Connecting to {base_url} (attempt {attempt}/{max_attempts_per_endpoint})")
                    exchange_config = {
                        'enableRateLimit': True,
                        'aiohttp_proxy': self.proxy,   # CCXT's proxy parameter
                        'connector': self._connector,
                        'session': self._session,
                        'aiohttp_trust_env': True,
                        'options': {'defaultType': 'spot'},
                        'urls': {
                            'api': {
                                'public': f'{base_url}/api/v3',
                                'private': f'{base_url}/api/v3',
                            }
                        },
                        'headers': {'User-Agent': user_agent},
                    }
                    self.exchange = exchange_cls(exchange_config)

                    if self.exchange is not None:
                        await self.exchange.load_markets()
                        logger.info(f"✅ Connected to {base_url} – markets loaded")
                        return

                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed with {base_url} (attempt {attempt}/{max_attempts_per_endpoint}): {e}")
                    # Close any partially-created exchange to release aiohttp sessions
                    if self.exchange:
                        try:
                            await self.exchange.close()
                        except Exception as cerr:
                            logger.warning(f"Error closing exchange after failure: {cerr}")
                        finally:
                            self.exchange = None
                    if attempt == max_attempts_per_endpoint:
                        logger.warning(f"Giving up on {base_url}, trying next endpoint...")
                        break
                    await asyncio.sleep(2 ** attempt)  # 2, 4 seconds

        # All endpoints failed
        logger.error("ERROR: All Binance endpoints unreachable. DNS/proxy issue.")
        logger.error(f"Last error: {last_error}")
        raise ConnectionError("Unable to connect to Binance after multiple attempts and endpoint rotations")

    async def stop(self):
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info("Exchange connection closed")
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        try:
            if self.exchange is None:
                return pd.DataFrame()
            raw = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    async def get_data(self, symbol: str, timeframe: str, lookback_hours: int) -> pd.DataFrame:
        now = time.time()
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if now - cached['timestamp'] < CACHE_DURATION.get(timeframe, 60):
                return cached['df'].copy()
        tf_minutes = {'1m': 1, '1h': 60, '4h': 240}[timeframe]
        limit = max(200, (lookback_hours * 60) // tf_minutes + 100)
        df = await self.fetch_ohlcv(symbol, timeframe, limit=limit)
        if df.empty:
            return df
        self.cache[cache_key] = {'df': df, 'timestamp': now}
        return df.copy()

# -------------------------------------------------------------------
# SignalGenerator (uses base_threshold from TokenConfig)
# -------------------------------------------------------------------
class SignalGenerator:
    def __init__(self, token_configs: List[TokenConfig], btc_model: Optional[xgb.XGBClassifier]):
        self.configs = {cfg.symbol: cfg for cfg in token_configs}
        self.btc_model = btc_model
        self.regime_detector = RegimeDetector()
        self.threshold_engine = ThresholdEngine()
        self.score_history: Dict[str, Deque[float]] = {cfg.symbol: deque(maxlen=5) for cfg in token_configs}

    async def compute_signal(self, symbol: str, fetcher: MarketDataFetcher) -> Optional[Dict]:
        cfg = self.configs.get(symbol)
        if not cfg:
            return None

        df_1m = await fetcher.get_data(symbol, '1m', lookback_hours=2)
        if df_1m.empty or len(df_1m) < 50:
            logger.warning(f"{symbol}: insufficient 1m data")
            return None

        current_price = float(df_1m['close'].iloc[-1])

        df_1h = await fetcher.get_data(symbol, '1h', lookback_hours=200)
        if df_1h.empty or len(df_1h) < 100:
            logger.warning(f"{symbol}: insufficient 1h data")
            return None

        df_1h['atr'] = compute_atr(df_1h, 14)
        df_1h['volume_ma'] = df_1h['volume'].rolling(VOLUME_LOOKBACK).mean()
        df_1h['efficiency_ratio'] = compute_efficiency_ratio(df_1h['close'], period=10)

        last_idx = len(df_1h) - 1
        vol_regime = self.regime_detector.volatility(df_1h['atr'], last_idx)
        volume_cond = self.regime_detector.volume(df_1h['volume'].iloc[-1], df_1h['volume_ma'].iloc[-1])
        er = df_1h['efficiency_ratio'].iloc[-1]
        trend_str = self.regime_detector.trend(er)
        trend_aligned = self.regime_detector.is_trend_aligned("long", trend_str)

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
        features_df = prepare_features(
            full_1h,
            btc_df.reset_index() if not btc_df.empty else None,
            pd.DataFrame([{'timestamp': datetime.now(), 'sentiment': news_score}])
        )
        if features_df is None or features_df.empty:
            return None

        latest_features = features_df.iloc[-1:].drop(columns=['timestamp', 'target'], errors='ignore')
        model_path = MODEL_STORE / f"{symbol.replace('/', '_')}_model.json"
        if not model_path.exists():
            logger.warning(f"No model for {symbol}")
            return None
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        ai_prob = model.predict_proba(latest_features)[0, 1]
        ai_prob = float(ai_prob)

        base = cfg.base_threshold
        atr_val = df_1h['atr'].iloc[-1] if not pd.isna(df_1h['atr'].iloc[-1]) else current_price * 0.01
        adj_thresh, _ = self.threshold_engine.compute(
            base=base, prob=ai_prob, vol_regime=vol_regime, volume_condition=volume_cond,
            trend_aligned=trend_aligned, atr=atr_val, price=current_price,
        )

        btc_ai = 0.5
        if self.btc_model:
            btc_1h = await fetcher.get_data('BTC/USDT', '1h', lookback_hours=200)
            if not btc_1h.empty:
                btc_features = prepare_features(btc_1h.reset_index(), None, None)
                if btc_features is not None and not btc_features.empty:
                    btc_last = btc_features.iloc[-1:].drop(columns=['timestamp', 'target'], errors='ignore')
                    btc_ai = self.btc_model.predict_proba(btc_last)[0, 1]

        btc_filter_ok = btc_ai >= BTC_FILTER_THRESHOLD if symbol != 'BTC/USDT' else True

        atr_sl_dist = cfg.atr_sl * atr_val
        atr_tp_dist = cfg.atr_tp * atr_val
        tp_move_pct = atr_tp_dist / current_price
        sl_move_pct = atr_sl_dist / current_price
        expected_net = ai_prob * tp_move_pct - (1 - ai_prob) * sl_move_pct - TOTAL_COST_PCT
        expected_net_pct = expected_net * 100
        signal = "NEUTRAL"
        if expected_net_pct >= adj_thresh and ai_prob > cfg.entry_prob_threshold and btc_filter_ok:
            signal = "BUY"
        elif expected_net_pct >= adj_thresh * 0.8 and ai_prob > cfg.entry_prob_threshold and btc_filter_ok:
            signal = "WEAK_BUY"

        self.score_history[symbol].appendleft(ai_prob)

        return {
            "symbol": symbol, "price": current_price, "ai_prob": ai_prob, "threshold": adj_thresh,
            "expected_net_pct": expected_net_pct, "signal": signal, "btc_ai": btc_ai,
            "btc_filter_ok": btc_filter_ok, "vol_regime": vol_regime, "volume_cond": volume_cond,
            "trend_aligned": trend_aligned, "atr": atr_val, "mode": cfg.mode,
            "risk_pct": cfg.risk_pct, "atr_sl": cfg.atr_sl, "atr_tp": cfg.atr_tp,
        }

# -------------------------------------------------------------------
# LiveEngine
# -------------------------------------------------------------------
class LiveEngine:
    def __init__(self, token_configs: List[TokenConfig], initial_balance: float, max_position_usdt: float, proxy_url: Optional[str] = None):
        self.token_configs = token_configs
        self.wallet = LiveWallet(initial_balance, max_position_usdt)
        self.fetcher = MarketDataFetcher(proxy_url)
        self.btc_model = None
        self.signal_gen = None

    async def initialize(self):
        await self.fetcher.start()
        btc_model_path = MODEL_STORE / "BTC_USDT_model.json"
        if btc_model_path.exists():
            self.btc_model = xgb.XGBClassifier()
            self.btc_model.load_model(str(btc_model_path))
            logger.info("BTC model loaded")
        self.signal_gen = SignalGenerator(self.token_configs, self.btc_model)
        logger.info("Live engine initialized")

    async def run(self):
        await self.initialize()
        logger.info("Starting main loop (60s pulse)")
        context_refresh = 0
        while True:
            start_cycle = time.time()
            if context_refresh == 0 or (time.time() - context_refresh) > 3600:
                logger.info("Refreshing 1h/4h context data...")
                symbols = [cfg.symbol for cfg in self.token_configs] + ['BTC/USDT']
                tasks = []
                for sym in symbols:
                    tasks.append(self.fetcher.get_data(sym, '1h', lookback_hours=200))
                    tasks.append(self.fetcher.get_data(sym, '4h', lookback_hours=200))
                await asyncio.gather(*tasks)
                context_refresh = time.time()
                logger.info("Context data refreshed")

            assert self.signal_gen is not None
            tasks = [self.signal_gen.compute_signal(cfg.symbol, self.fetcher) for cfg in self.token_configs]
            signals = await asyncio.gather(*tasks)
            for sig in signals:
                if sig is None:
                    continue
                symbol       = sig["symbol"]
                current_price = sig["price"]
                signal_type  = sig.get("signal", "HOLD")
                existing     = self.wallet.open_trades.get(symbol)

                if existing:
                    # ── Dynamic TP: exit when the model fires the opposite direction ──
                    # This is the trend-reversal TP: we entered on one reversal signal;
                    # we exit when the model detects the next reversal in the other direction.
                    opposite_fired = (
                        (existing.trade_type == "LONG"  and signal_type in ("SELL", "STRONG_SELL")) or
                        (existing.trade_type == "SHORT" and signal_type in ("BUY", "WEAK_BUY", "STRONG_BUY"))
                    )

                    if existing.trade_type == "LONG":
                        if current_price <= existing.stop_loss:
                            self.wallet.close_trade(symbol, existing.stop_loss, "STOP_LOSS")
                        elif opposite_fired:
                            self.wallet.close_trade(symbol, current_price, "MODEL_REVERSAL_TP")
                        elif current_price >= existing.take_profit > 0:
                            # Hard ceiling fallback (wide ATR multiple) — only if no reversal signal yet
                            self.wallet.close_trade(symbol, existing.take_profit, "TAKE_PROFIT")

                    elif existing.trade_type == "SHORT":
                        if current_price >= existing.stop_loss:
                            self.wallet.close_trade(symbol, existing.stop_loss, "STOP_LOSS")
                        elif opposite_fired:
                            self.wallet.close_trade(symbol, current_price, "MODEL_REVERSAL_TP")
                        elif current_price <= existing.take_profit > 0:
                            self.wallet.close_trade(symbol, existing.take_profit, "TAKE_PROFIT")

                else:
                    if signal_type in ("BUY", "WEAK_BUY") and self.wallet.can_open_trade(symbol, sig["price"]):
                        risk_pct = sig["risk_pct"]
                        sl_price = sig["price"] - sig["atr_sl"] * sig["atr"]
                        tp_price = sig["price"] + sig["atr_tp"] * sig["atr"]
                        confidence = 1.0 if signal_type == "BUY" else 0.6
                        units = self.wallet.get_position_units(symbol, sig["price"], sl_price, risk_pct, confidence)
                        if units > 0:
                            trade = LiveTrade(
                                symbol=symbol, entry_price=sig["price"], position_units=units,
                                stop_loss=sl_price, take_profit=tp_price, entry_time=datetime.now(),
                                mode=sig["mode"], master_score=sig["ai_prob"],
                            )
                            self.wallet.open_trade(trade)
                status = "IN_TRADE" if symbol in self.wallet.open_trades else "SCANNING"
                btc_filter = f"BTC:{sig['btc_ai']:.2f}" if not sig['btc_filter_ok'] else "OK"
                print(f"{symbol:12} | Price {sig['price']:8.4f} | AI {sig['ai_prob']:.3f} | Thresh {sig['threshold']:.3f} | {btc_filter} | {status}")
            elapsed = time.time() - start_cycle
            sleep_time = max(0, 60 - elapsed)
            await asyncio.sleep(sleep_time)

    async def shutdown(self):
        await self.fetcher.stop()
        logger.info("Engine stopped")

# -------------------------------------------------------------------
# Interactive setup (with dynamic file discovery and zero-trade filter)
# -------------------------------------------------------------------
def interactive_setup(backtest_dir: Path) -> Tuple[List[TokenConfig], float, float]:
    print("\n" + "="*60)
    print("🚀 AEGIS‑1 LIVE ENGINE v1.0 – PRE‑FLIGHT SETUP")
    print("="*60)

    json_path = get_latest_backtest_results(backtest_dir)
    if json_path is None:
        print("❌ No backtest JSON file found. Run cost_aware_backtester.py first.")
        sys.exit(1)
    print(f"📂 Using backtest file: {json_path.name}")

    while True:
        try:
            capital = float(input("💰 Starting USDT balance: "))
            if capital <= 0:
                raise ValueError
            break
        except:
            print("Invalid amount. Enter positive number.")

    while True:
        try:
            risk_pct = float(input("🎯 Risk per trade (%) [1-5]: "))
            if 1 <= risk_pct <= 5:
                break
            print("Enter between 1 and 5")
        except:
            pass

    temp_best = load_backtest_results(json_path, min_trades=1)
    if not temp_best:
        print("❌ No tokens found in backtest JSON. Exiting.")
        sys.exit(1)

    print("\n📊 Available tokens from backtest (with trade counts):")
    sorted_temp = sorted(temp_best.items(), key=lambda x: x[1]["total_trades"], reverse=True)
    for sym, info in sorted_temp:
        if info["total_trades"] == 0:
            print(f"  {sym:12} | Trades: {info['total_trades']:3d} | ⚠️ ZERO TRADES (will be filtered out)")
        else:
            print(f"  {sym:12} | Trades: {info['total_trades']:3d} | Return: {info['net_return_pct']:+6.2f}% | Mode: {info['mode']}")

    while True:
        try:
            min_trades_input = input(f"\n🔢 Minimum number of backtest trades required for a token (default 5): ").strip()
            min_trades = int(min_trades_input) if min_trades_input else 5
            if min_trades >= 1:
                break
            print("Enter at least 1")
        except:
            print("Invalid number, using 5")
            min_trades = 5
            break

    best_per_token = load_backtest_results(json_path, min_trades=min_trades)
    if not best_per_token:
        print(f"❌ No tokens with ≥{min_trades} trades. Try lowering the minimum.")
        sys.exit(1)

    sorted_tokens = sorted(best_per_token.items(), key=lambda x: x[1]["net_return_pct"], reverse=True)
    print(f"\n🏆 Tokens with ≥{min_trades} trades (ranked by net return):")
    for i, (sym, info) in enumerate(sorted_tokens, 1):
        print(f"  {i:2}. {sym:12} | Return {info['net_return_pct']:+6.2f}% | MAR {info['mar']:.2f} | Trades {info['total_trades']} | Mode {info['mode']}")

    print("\n📌 Selection options:")
    print("   'all'          – use all tokens")
    print("   'stable'       – top 10 by MAR")
    print("   comma list     – e.g., 1,3,5")
    choice = input("Your choice: ").strip().lower()
    selected_symbols = []
    if choice == 'all':
        selected_symbols = [sym for sym, _ in sorted_tokens]
    elif choice == 'stable':
        mar_sorted = sorted(sorted_tokens, key=lambda x: x[1]["mar"], reverse=True)[:10]
        selected_symbols = [sym for sym, _ in mar_sorted]
    else:
        indices = [int(x.strip())-1 for x in choice.split(',') if x.strip().isdigit()]
        selected_symbols = [sorted_tokens[i][0] for i in indices if 0 <= i < len(sorted_tokens)]

    if not selected_symbols:
        print("No valid selection, using top 10 by MAR.")
        mar_sorted = sorted(sorted_tokens, key=lambda x: x[1]["mar"], reverse=True)[:10]
        selected_symbols = [sym for sym, _ in mar_sorted]

    print(f"✅ Selected {len(selected_symbols)} tokens: {', '.join(selected_symbols[:5])}{'...' if len(selected_symbols)>5 else ''}")

    while True:
        try:
            max_pos = float(input(f"💰 Max USDT per single trade (max {capital:.0f}): "))
            if max_pos <= 0:
                max_pos = capital
            if max_pos > capital:
                max_pos = capital
            break
        except:
            max_pos = capital
            break

    configs = []
    for sym in selected_symbols:
        info = best_per_token[sym]
        configs.append(TokenConfig(
            symbol=sym,
            mode=info["mode"],
            base_threshold=info["final_threshold"],
            entry_prob_threshold=info["entry_prob_threshold"],
            atr_sl=info["atr_sl"],
            atr_tp=info["atr_tp"],
            risk_pct=info["risk_pct"],
            optimizer_thresholds={},
        ))
    return configs, capital, max_pos

# -------------------------------------------------------------------
# Main entry point with proxy fallback
# -------------------------------------------------------------------
async def main():
    backtest_dir = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
    if not backtest_dir.exists():
        logger.error(f"Backtest directory not found: {backtest_dir}")
        return

    configs, capital, max_pos = interactive_setup(backtest_dir)
    if not configs:
        logger.error("No token configurations loaded.")
        return

    proxy = GLOBAL_PROXY
    # First attempt without proxy (or with env variable)
    engine = LiveEngine(configs, capital, max_pos, proxy_url=proxy)
    try:
        await engine.initialize()
    except (ConnectionError, aiohttp.ClientError, Exception) as e:
        logger.warning(f"Initial connection failed: {e}")
        print("\n⚠️  Connection to Binance failed. This is likely due to ISP blocking.")
        print("You can use a proxy (HTTP/HTTPS) to bypass restrictions.")
        proxy_input = input("Enter proxy URL (e.g., http://127.0.0.1:7890) or press Enter to exit: ").strip()
        if proxy_input:
            proxy = proxy_input
            engine = LiveEngine(configs, capital, max_pos, proxy_url=proxy)
            try:
                await engine.initialize()
            except Exception as e2:
                logger.error(f"Still cannot connect with proxy: {e2}")
                print("❌ Proxy connection failed. Exiting.")
                return
        else:
            print("❌ No proxy provided. Exiting.")
            return

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Shutdown by user")
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())