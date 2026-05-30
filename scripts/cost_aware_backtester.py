#!/usr/bin/env python3
"""
adaptive_intelligent_backtester.py – Production-Grade Adaptive Backtester v2.1

Runs backtests for all symbols and modes, saves results to JSON.
No input JSON required – it fetches live data and computes everything.
"""

from __future__ import annotations

import os
import sys
import csv
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── path fix ──────────────────────────────────────────────────────────────────
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import (
    prepare_features,
    compute_atr,
    compute_efficiency_ratio,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION  (single source of truth)
# ============================================================

BACKTEST_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# Capital & execution
INITIAL_BALANCE: float = 1_000.0
EXCHANGE_FEE_PCT: float = 0.001        # 0.1 % per leg
SLIPPAGE_PCT: float = 0.001            # 0.1 % per leg
TOTAL_COST_PCT: float = 2 * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)   # 0.4 %

# Trade management
MAX_HOLD_CANDLES: int = 48
MIN_SL_DISTANCE_PCT: float = 0.005
COST_MULTIPLIER: float = 1.5           # expected_net must beat 1.5× cost

# Confidence floor
MIN_TRADES_REQUIRED: int = 30

# Rolling window sizes
VOLATILITY_LOOKBACK: int = 100
VOLUME_LOOKBACK: int = 50

# Regime percentile cut-offs (for ATR-based vol)
VOL_PERCENTILE_LOW: int = 30
VOL_PERCENTILE_HIGH: int = 70

# Volume regime thresholds (ratio vs rolling mean)
VOLUME_RATIO_HIGH: float = 1.20
VOLUME_RATIO_LOW: float = 0.80

# Efficiency-ratio thresholds (trend strength)
ER_STRONG_TREND: float = 0.50
ER_WEAK_TREND: float = 0.30

# ── Adjustment multipliers ────────────────────────────────────────────────────
CONF_VERY_HIGH: float = 0.75
CONF_NORMAL: float = 1.00
CONF_WEAK: float = 1.25

TREND_ALIGNED_FACTOR: float = 0.85
TREND_MISALIGNED_FACTOR: float = 1.20

VOL_HIGH_FACTOR: float = 0.85
VOL_NORMAL_FACTOR: float = 1.00
VOL_LOW_FACTOR: float = 1.20
VOL_EXTREME_LOW_FACTOR: float = 1.50

VOLUME_HIGH_FACTOR: float = 0.90
VOLUME_NORMAL_FACTOR: float = 1.00
VOLUME_LOW_FACTOR: float = 1.10

# ── Mode base thresholds (%) ──────────────────────────────────────────────────
MODE_BASE_THRESHOLDS: Dict[str, float] = {
    "conservative": 0.45,
    "balanced":     0.30,
    "aggressive":   0.20,
}

# ── Mode trading parameters ───────────────────────────────────────────────────
MODE_PARAMS: Dict[str, Dict[str, float]] = {
    "conservative": {"entry_prob": 0.75, "risk_pct": 0.015, "atr_sl": 1.2, "atr_tp": 1.8},
    "balanced":     {"entry_prob": 0.70, "risk_pct": 0.020, "atr_sl": 1.5, "atr_tp": 2.0},
    "aggressive":   {"entry_prob": 0.65, "risk_pct": 0.030, "atr_sl": 1.8, "atr_tp": 2.5},
}

# ── Multi-tier position sizing ────────────────────────────────────────────────
TIER_FULL: float    = 1.00
TIER_HALF: float    = 0.50
TIER_QUARTER: float = 0.25

# ── Adaptive safety controls ──────────────────────────────────────────────────
MAX_SKIP_RATE: float = 0.80
SAFETY_CHECK_EVERY: int = 100
MIN_TRADES_TARGET: int = 30
MAX_TRADES_TARGET: int = 100
THRESHOLD_RELAX_STEP: float = 0.95
THRESHOLD_TIGHTEN_STEP: float = 1.05

# ── Absolute bounds for adaptive threshold ────────────────────────────────────
THRESHOLD_FLOOR: float = 0.05
THRESHOLD_CEIL: float = 3.00

# ── Starvation recovery ───────────────────────────────────────────────────────
STARVATION_START_HOUR: int = 1000
STARVATION_MIN_TRADES: int = 3
STARVATION_TARGET_TRADES: int = 5
STARVATION_DECAY_STEP: float = 0.90
STARVATION_DECAY_INTERVAL: int = 500

# ── Fleet ────────────────────────────────────────────────────────────────────
FLEET: List[str] = [
    "BTC/USDT",  "ETH/USDT",  "SOL/USDT",  "BNB/USDT",  "XRP/USDT",
    "ADA/USDT",  "AVAX/USDT", "LINK/USDT", "DOT/USDT",  "NEAR/USDT",
    "MATIC/USDT","LTC/USDT",  "BCH/USDT",  "SHIB/USDT", "TON/USDT",
    "ICP/USDT",  "HBAR/USDT", "APT/USDT",  "ARB/USDT",  "OP/USDT",
    "STX/USDT",  "FIL/USDT",  "AAVE/USDT", "VET/USDT",  "INJ/USDT",
]

MIN_HOURLY_BARS: int = 500
DATA_HOURS: int = 3_000

# Live config export (optional)
LIVE_CONFIG_PATH = Path("config/live_fleet_strategy.json")
LIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA-CLASSES
# ============================================================
@dataclass
class TradeRecord:
    symbol: str
    mode: str
    candle_index: int
    timestamp: str
    direction: str
    prob: float
    base_threshold: float
    adjusted_threshold: float
    expected_net_pct: float
    confidence_level: str
    vol_regime: str
    trend_strength: str
    trend_aligned: bool
    volume_condition: str
    size_tier: str
    size_factor: float
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    decision: str


@dataclass
class BacktestResult:
    symbol: str
    mode: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: Optional[float] = None
    skip_rate: float = 0.0
    total_signals: int = 0
    skipped_below_threshold: int = 0
    skipped_cost: int = 0
    low_confidence: bool = True
    over_filtered: bool = False
    final_threshold: float = 0.0
    error: Optional[str] = None
    trades: List[TradeRecord] = field(default_factory=list)


# ============================================================
# REGIME DETECTOR & THRESHOLD ENGINE (identical to v2.1)
# ============================================================
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


# ============================================================
# POSITION SIZER
# ============================================================
class PositionSizer:
    @staticmethod
    def tier(expected_net_pct: float, threshold_pct: float) -> Tuple[str, float]:
        if expected_net_pct >= threshold_pct:
            return "full", TIER_FULL
        if expected_net_pct >= threshold_pct * 0.80:
            return "half", TIER_HALF
        if expected_net_pct >= threshold_pct * 0.60:
            return "quarter", TIER_QUARTER
        return "skipped", 0.0

    @staticmethod
    def units(capital: float, risk_pct: float, size_factor: float, sl_distance: float, price: float) -> float:
        if sl_distance <= 0 or size_factor <= 0:
            return 0.0
        risk_amount = capital * risk_pct * size_factor
        units = risk_amount / sl_distance
        max_units = capital / price
        return float(min(units, max_units))


# ============================================================
# SAFETY CONTROLLER
# ============================================================
class SafetyController:
    def __init__(self, initial_threshold: float, symbol: str, mode: str):
        self.threshold = float(initial_threshold)
        self._window: List[int] = []
        self._window_size = 100

    def record_signal(self, trade_taken: bool) -> None:
        self._window.append(1 if trade_taken else 0)
        if len(self._window) > self._window_size:
            self._window.pop(0)

    def check_and_adjust(self, total_signals: int, total_skipped: int) -> float:
        if total_signals == 0:
            return self.threshold
        skip_rate = total_skipped / total_signals
        density = sum(self._window)
        old = self.threshold
        if skip_rate > MAX_SKIP_RATE:
            self.threshold = max(THRESHOLD_FLOOR, self.threshold * THRESHOLD_RELAX_STEP)
        elif density < MIN_TRADES_TARGET and len(self._window) == self._window_size:
            self.threshold = max(THRESHOLD_FLOOR, self.threshold * THRESHOLD_RELAX_STEP)
        elif density > MAX_TRADES_TARGET:
            self.threshold = min(THRESHOLD_CEIL, self.threshold * THRESHOLD_TIGHTEN_STEP)
        return self.threshold


# ============================================================
# METRICS HELPERS
# ============================================================
def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr))
    denom = max(std, 0.01)
    return float(np.mean(arr) / denom * np.sqrt(252 * 24))

def _cost_adjusted_expected_return(prob: float, atr: float, entry: float, atr_sl: float, atr_tp: float) -> float:
    tp_move = (atr_tp * atr) / entry
    sl_move = (atr_sl * atr) / entry
    raw = prob * tp_move - (1.0 - prob) * sl_move
    return raw - TOTAL_COST_PCT


# ============================================================
# ADAPTIVE BACKTESTER (simplified, no external JSON load)
# ============================================================
class AdaptiveBacktester:
    def __init__(self, symbol: str, mode: str, initial_balance: float = INITIAL_BALANCE):
        self.symbol = symbol
        self.mode = mode
        self.initial_balance = initial_balance
        self.params = MODE_PARAMS[mode]
        self.mode_base = MODE_BASE_THRESHOLDS[mode]

    def run(self, hours: int = DATA_HOURS) -> BacktestResult:
        result = BacktestResult(symbol=self.symbol, mode=self.mode)
        df = self._fetch_and_prepare(hours)
        if df is None or len(df) < MIN_HOURLY_BARS:
            result.error = "Insufficient data"
            return result

        df["volume_ma"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()
        if "efficiency_ratio_10" not in df.columns:
            df["efficiency_ratio_10"] = compute_efficiency_ratio(df["close"], period=10)

        p = self.params
        entry_prob = p["entry_prob"]
        risk_pct = p["risk_pct"]
        atr_sl = p["atr_sl"]
        atr_tp = p["atr_tp"]

        capital = self.initial_balance
        peak_capital = capital
        position_units = 0.0
        entry_price = 0.0
        entry_atr = 0.0
        direction: Optional[str] = None
        trade_returns: List[float] = []
        max_drawdown = 0.0
        total_signals = 0
        skipped_below_threshold = 0
        skipped_cost = 0
        safety = SafetyController(self.mode_base, self.symbol, self.mode)
        pending_rec: Optional[TradeRecord] = None
        trades_count = 0
        n = len(df)

        for i in range(n - MAX_HOLD_CANDLES - 1):
            row = df.iloc[i]
            prob = float(row["prob"]) if not pd.isna(row["prob"]) else np.nan
            if np.isnan(prob):
                continue

            # Exit logic
            if position_units != 0:
                exit_px = self._scan_exit(df, i, entry_price, entry_atr, atr_sl, atr_tp, direction)
                if exit_px is not None:
                    exit_slipped = exit_px * (1.0 - SLIPPAGE_PCT)
                    gross_pnl = position_units * (exit_slipped - entry_price)
                    exit_cost = abs(position_units) * exit_slipped * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
                    net_pnl = gross_pnl - exit_cost
                    capital += net_pnl
                    ret_pct = (net_pnl / (entry_price * position_units) * 100) if entry_price * position_units > 0 else 0.0
                    trade_returns.append(ret_pct / 100.0)
                    if net_pnl > 0:
                        result.wins += 1
                        result.gross_profit += net_pnl
                    else:
                        result.losses += 1
                        result.gross_loss += abs(net_pnl)
                    result.total_trades += 1
                    trades_count += 1
                    if pending_rec is not None:
                        pending_rec.exit_price = round(exit_slipped, 6)
                        pending_rec.pnl = round(net_pnl, 4)
                        pending_rec.return_pct = round(ret_pct, 4)
                        pending_rec = None
                    peak_capital = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0.0
                    max_drawdown = max(max_drawdown, dd)
                    position_units = 0.0
                continue

            # Signal filter
            if prob <= entry_prob:
                continue
            total_signals += 1
            direction = "long"

            # Regime detection
            vol_regime = RegimeDetector.volatility(df["atr_14"], i)
            vol_cond = RegimeDetector.volume(float(row["volume"]), float(row["volume_ma"]))
            er = float(row["efficiency_ratio_10"]) if not pd.isna(row["efficiency_ratio_10"]) else 0.5
            trend_str = RegimeDetector.trend(er)
            trend_ok = RegimeDetector.is_trend_aligned(direction, trend_str)

            # Adaptive threshold (no optimizer file)
            base_thresh = self.mode_base
            adj_thresh, conf_label = ThresholdEngine.compute(
                base=base_thresh, prob=prob, vol_regime=vol_regime,
                volume_condition=vol_cond, trend_aligned=trend_ok,
                atr=float(row["atr_14"]) if float(row["atr_14"]) > 0 else float(row["close"]) * 0.001,
                price=float(row["close"])
            )

            cur_atr = float(row["atr_14"]) if float(row["atr_14"]) > 0 else float(row["close"]) * 0.001
            exp_net_frac = _cost_adjusted_expected_return(prob, cur_atr, float(row["close"]), atr_sl, atr_tp)
            exp_net_pct = exp_net_frac * 100.0
            min_cost_bar = COST_MULTIPLIER * TOTAL_COST_PCT * 100.0
            if exp_net_pct <= min_cost_bar:
                skipped_cost += 1
                safety.record_signal(False)
                continue

            tier_name, size_factor = PositionSizer.tier(exp_net_pct, adj_thresh)
            if size_factor == 0.0:
                skipped_below_threshold += 1
                safety.record_signal(False)
                continue

            sl_dist = max(atr_sl * cur_atr, float(row["close"]) * MIN_SL_DISTANCE_PCT)
            pos_units = PositionSizer.units(capital, risk_pct, size_factor, sl_dist, float(row["close"]))
            if pos_units <= 0:
                continue

            entry_price = float(row["close"])
            entry_atr = cur_atr
            position_units = pos_units
            entry_cost = position_units * entry_price * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
            capital -= entry_cost
            safety.record_signal(True)

        # Final metrics
        if result.total_trades == 0:
            result.error = "No trades executed"
            result.skip_rate = 1.0
            return result

        result.net_return_pct = round((capital - self.initial_balance) / self.initial_balance * 100, 2)
        result.max_drawdown_pct = round(max_drawdown, 2)
        result.win_rate = round(result.wins / result.total_trades * 100, 2)
        if result.gross_loss > 0:
            result.profit_factor = round(result.gross_profit / result.gross_loss, 3)
        else:
            result.profit_factor = round(result.net_return_pct, 3)
        result.avg_trade_return_pct = round(float(np.mean(trade_returns)) * 100, 4) if trade_returns else 0.0
        result.sharpe_ratio = round(_sharpe(trade_returns), 3)
        total_skipped = skipped_below_threshold + skipped_cost
        result.skip_rate = round(total_skipped / total_signals, 3) if total_signals > 0 else 1.0
        result.total_signals = total_signals
        result.skipped_below_threshold = skipped_below_threshold
        result.skipped_cost = skipped_cost
        result.low_confidence = result.total_trades < MIN_TRADES_REQUIRED
        result.over_filtered = result.skip_rate > 0.80
        result.final_threshold = round(safety.threshold, 4)
        return result

    def _scan_exit(self, df: pd.DataFrame, i: int, entry_price: float, entry_atr: float,
                   atr_sl: float, atr_tp: float, direction: Optional[str]) -> Optional[float]:
        tp = entry_price + atr_tp * entry_atr
        sl = entry_price - atr_sl * entry_atr
        for j in range(1, MAX_HOLD_CANDLES + 1):
            future = df.iloc[i + j]
            if direction == "long":
                both = future["high"] >= tp and future["low"] <= sl
                if both:
                    return sl
                if future["high"] >= tp:
                    return tp
                if future["low"] <= sl:
                    return sl
        return None

    def _fetch_and_prepare(self, hours: int) -> Optional[pd.DataFrame]:
        predictor = Predictor(self.symbol)
        df = predictor.fetch_live_data(limit=hours)
        if df is None or df.empty:
            return None
        btc_df = predictor.fetch_btc_data(limit=hours)
        news_df = predictor.load_news_data()
        if btc_df is None or btc_df.empty:
            btc_df = pd.DataFrame({"timestamp": df["timestamp"], "close": 0.0})
            btc_df["open"] = btc_df["high"] = btc_df["low"] = btc_df["close"]
            btc_df["volume"] = 0
        if news_df is None or news_df.empty:
            news_df = pd.DataFrame({"timestamp": df["timestamp"], "sentiment": 0.0})

        df_1d = None
        try:
            df_1d = predictor.fetch_live_data(timeframe='1d', limit=max(1000, int(hours / 24) + 50))
        except Exception:
            df_1d = None

        df = prepare_features(
            df,
            btc_df=btc_df,
            news_df=news_df,
            add_target_flag=False,
            forward_hours=MAX_HOLD_CANDLES,
            df_1d=df_1d,
            df_1w=None,
            macro_state=None
        )
        if df is None or df.empty:
            return None
        df = df.sort_values('timestamp').reset_index(drop=True)

        probabilities = predictor.predict_proba(df)
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
            df['prob_sell'] = [row[0] for row in probabilities]
            df['prob_hold'] = [row[1] for row in probabilities]
            df['prob_buy'] = [row[2] for row in probabilities]
            df['prob'] = df['prob_buy']
        else:
            prob_values = [float(x) for x in list(probabilities)] if hasattr(probabilities, '__iter__') else [float(probabilities)]
            df['prob'] = prob_values
            df['prob_buy'] = df['prob']
            df['prob_hold'] = 0.0
            df['prob_sell'] = 0.0

        df['atr_14'] = compute_atr(df, 14)
        return df


# ============================================================
# FLEET RUNNER
# ============================================================
def run_backtest_fleet():
    print("=" * 80)
    print("🧠 ADAPTIVE BACKTESTER v2.1 – Generating results for live engine")
    print(f"   Balance ${INITIAL_BALANCE:,.0f} | Data {DATA_HOURS} hrs | {len(FLEET)} symbols")
    print("=" * 80)

    all_results = []
    for symbol in FLEET:
        print(f"\n📊 {symbol}")
        for mode in MODE_BASE_THRESHOLDS:
            bt = AdaptiveBacktester(symbol, mode)
            res = bt.run(DATA_HOURS)
            if res.error:
                print(f"   {mode.capitalize():<12} ❌ {res.error}")
            else:
                print(f"   {mode.capitalize():<12} | Trades {res.total_trades:3d} | Return {res.net_return_pct:+6.2f}% | PF {res.profit_factor or 0:.2f}")
                # Convert to dict without trades to save space
                res_dict = asdict(res)
                res_dict.pop("trades", None)
                all_results.append(res_dict)

    # Save results as a list (not dict) for easy loading
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = BACKTEST_DIR / f"cost_aware_backtest_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {out_file}")
    print(f"   Total backtest entries: {len(all_results)}")


if __name__ == "__main__":
    run_backtest_fleet()           