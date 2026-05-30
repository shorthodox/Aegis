#!/usr/bin/env python3
"""
cost_aware_backtester.py – Aegis-1 Production Backtester v3.0

Aligned with retrain_model.py meta-labeling strategy:
  - Signal gating via per-symbol meta_threshold_buy / meta_threshold_sell
  - tradeable_buy / tradeable_sell flags honoured
  - S&R filter and trend filter mirror predict_signal() in live engine
  - ATR multiplier read from trained model JSON (not hardcoded)
  - Position sizing scales with meta_conf excess above threshold
  - 3 modes differ only in risk_pct and TP R:R ratio
  - Data and Predictor cached per symbol (one model load per symbol)
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features, compute_atr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_BALANCE:    float = 1_000.0
EXCHANGE_FEE_PCT:   float = 0.001          # 0.10 % per leg
SLIPPAGE_PCT:       float = 0.001          # 0.10 % per leg
TOTAL_COST_PCT:     float = 2 * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)   # 0.40 %

MAX_HOLD_CANDLES:   int   = 48             # time-based exit after 48 h
MIN_SL_DISTANCE_PCT: float = 0.005         # minimum SL = 0.5 % of price
COST_MULTIPLIER:    float = 1.5            # EV must beat 1.5× round-trip cost
MIN_TRADES_REQUIRED: int  = 10

DATA_HOURS:         int   = 3_000
MIN_HOURLY_BARS:    int   = 500

# ── Modes: risk per trade and TP R:R relative to model SL distance ───────────
# The model's atr_multiplier is the SL distance.
# TP = model_atr_mult × rr_ratio × ATR.
MODE_PARAMS: Dict[str, Dict[str, float]] = {
    "conservative": {"risk_pct": 0.010, "rr_ratio": 1.5},
    "balanced":     {"risk_pct": 0.020, "rr_ratio": 2.0},
    "aggressive":   {"risk_pct": 0.030, "rr_ratio": 2.5},
}

# ── Fleet — symbols that have trained models ──────────────────────────────────
# Includes tradeable and non-tradeable; the backtester skips non-tradeable ones
# automatically via the tradeable_buy / tradeable_sell flags from model JSON.
FLEET: List[str] = [
    "BTC/USDT",  "ETH/USDT",  "SOL/USDT",  "BNB/USDT",  "XRP/USDT",
    "ADA/USDT",  "AVAX/USDT", "LINK/USDT", "DOT/USDT",  "TON/USDT",
    "ICP/USDT",  "SUI/USDT",  "INJ/USDT",  "ATOM/USDT", "UNI/USDT",
    "ALGO/USDT", "CHZ/USDT",  "ICX/USDT",  "GMX/USDT",  "ONDO/USDT",
    "JUP/USDT",  "POLYX/USDT","ORDI/USDT", "RSR/USDT",  "CRV/USDT",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA-CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol:       str
    mode:         str
    bar_index:    int
    timestamp:    str
    direction:    str
    meta_conf:    float
    thr:          float
    size_factor:  float
    entry_price:  float
    exit_price:   float = 0.0
    pnl:          float = 0.0
    return_pct:   float = 0.0
    exit_reason:  str   = ""


@dataclass
class BacktestResult:
    symbol:              str
    mode:                str
    total_trades:        int   = 0
    wins:                int   = 0
    losses:              int   = 0
    gross_profit:        float = 0.0
    gross_loss:          float = 0.0
    net_return_pct:      float = 0.0
    max_drawdown_pct:    float = 0.0
    avg_trade_return_pct: float = 0.0
    sharpe_ratio:        float = 0.0
    win_rate:            float = 0.0
    profit_factor: Optional[float] = None
    total_signals:       int   = 0
    skipped_cost:        int   = 0
    skipped_sr_trend:    int   = 0
    low_confidence:      bool  = True
    atr_multiplier:      float = 1.5
    error:   Optional[str]    = None
    trades:  List[TradeRecord] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    return float(np.mean(arr) / max(np.std(arr), 1e-8) * np.sqrt(252 * 24))


def _ev(meta_conf: float, atr: float, entry: float,
        atr_sl: float, atr_tp: float) -> float:
    """Cost-adjusted expected value using meta_conf as win probability."""
    tp_frac = (atr_tp * atr) / entry
    sl_frac = (atr_sl * atr) / entry
    return meta_conf * tp_frac - (1.0 - meta_conf) * sl_frac - TOTAL_COST_PCT


def _meta_size_factor(meta_conf: float, thr: float) -> float:
    """Scale position between 0.5 and 1.0 based on how far above threshold."""
    excess = meta_conf - thr
    if excess >= 0.10:
        return 1.00
    if excess >= 0.05:
        return 0.75
    return 0.50


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveBacktester:
    # Class-level caches — shared across all modes for a given symbol so
    # model weights are loaded once and feature engineering runs once.
    _df_cache:        Dict[str, pd.DataFrame] = {}
    _predictor_cache: Dict[str, Predictor]    = {}

    def __init__(self, symbol: str, mode: str,
                 initial_balance: float = INITIAL_BALANCE):
        self.symbol          = symbol
        self.mode            = mode
        self.initial_balance = initial_balance
        self.params          = MODE_PARAMS[mode]

        # Predictor: load once per symbol, reuse across modes.
        if symbol not in AdaptiveBacktester._predictor_cache:
            AdaptiveBacktester._predictor_cache[symbol] = Predictor(symbol)
        self.predictor = AdaptiveBacktester._predictor_cache[symbol]

        # Per-symbol gates from trained model JSON.
        meta = self.predictor.meta
        self.thr_buy        = float(meta.get("meta_threshold_buy",
                                    meta.get("meta_threshold", 0.62)))
        self.thr_sell       = float(meta.get("meta_threshold_sell",
                                    meta.get("meta_threshold", 0.62)))
        self.tradeable_buy  = bool(meta.get("tradeable_buy",
                                   meta.get("tradeable", True)))
        self.tradeable_sell = bool(meta.get("tradeable_sell",
                                   meta.get("tradeable", True)))
        # override_thr: signals above this are "high conviction" and bypass S&R/trend filter
        self.override_thr   = float(meta.get("meta_override_confidence", 1.0))
        # ATR multiplier = SL distance (same value used to label training data)
        self.atr_mult       = float(meta.get("atr_multiplier", 1.5))

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, hours: int = DATA_HOURS) -> BacktestResult:
        result = BacktestResult(symbol=self.symbol, mode=self.mode,
                                atr_multiplier=self.atr_mult)
        df = self._fetch_and_prepare(hours)
        if df is None or len(df) < MIN_HOURLY_BARS:
            result.error = "Insufficient data"
            return result

        p        = self.params
        risk_pct = p["risk_pct"]
        atr_sl   = self.atr_mult                        # SL = model barrier width
        atr_tp   = self.atr_mult * p["rr_ratio"]        # TP = SL × mode R:R

        capital       = self.initial_balance
        peak_capital  = capital
        pos_units     = 0.0
        entry_price   = 0.0
        entry_atr_val = 0.0
        entry_bar     = -1
        direction: Optional[str] = None
        trade_returns: List[float] = []
        max_dd        = 0.0
        total_signals = 0
        skipped_cost  = 0
        skipped_srt   = 0
        n             = len(df)

        for i in range(n - MAX_HOLD_CANDLES - 1):
            row = df.iloc[i]

            # ── EXIT ─────────────────────────────────────────────────────────
            if pos_units != 0.0:
                bars_held = i - entry_bar
                exit_px   = self._check_exit(row, entry_price, entry_atr_val,
                                             atr_sl, atr_tp, direction)
                exit_reason = ""
                if exit_px is not None:
                    exit_reason = "tp_sl"
                elif bars_held >= MAX_HOLD_CANDLES:
                    exit_px     = float(row["close"])
                    exit_reason = "timeout"

                if exit_px is not None:
                    if direction == "long":
                        exit_slipped = exit_px * (1.0 - SLIPPAGE_PCT)
                        gross_pnl    = pos_units * (exit_slipped - entry_price)
                    else:
                        exit_slipped = exit_px * (1.0 + SLIPPAGE_PCT)
                        gross_pnl    = pos_units * (entry_price - exit_slipped)

                    exit_cost = abs(pos_units) * exit_slipped * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
                    net_pnl   = gross_pnl - exit_cost
                    capital  += net_pnl

                    pos_val  = entry_price * pos_units
                    ret_pct  = (net_pnl / pos_val * 100) if pos_val > 0 else 0.0
                    trade_returns.append(ret_pct / 100.0)

                    if net_pnl >= 0:
                        result.wins         += 1
                        result.gross_profit += net_pnl
                    else:
                        result.losses      += 1
                        result.gross_loss  += abs(net_pnl)

                    result.total_trades += 1
                    result.trades.append(TradeRecord(
                        symbol=self.symbol, mode=self.mode, bar_index=entry_bar,
                        timestamp=str(df.iloc[entry_bar]["timestamp"]),
                        direction=direction or "",
                        meta_conf=float(df.iloc[entry_bar].get("meta_conf", 0.0)),
                        thr=self.thr_buy if direction == "long" else self.thr_sell,
                        size_factor=round(pos_units * entry_price / capital, 4),
                        entry_price=round(entry_price, 6),
                        exit_price=round(exit_slipped, 6),
                        pnl=round(net_pnl, 4),
                        return_pct=round(ret_pct, 4),
                        exit_reason=exit_reason,
                    ))

                    peak_capital = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0.0
                    max_dd = max(max_dd, dd)
                    pos_units = 0.0
                continue   # do not enter a new trade on the same bar as exit

            # ── SIGNAL GATE ──────────────────────────────────────────────────
            meta_conf = float(row.get("meta_conf", 0.0))
            meta_dir  = int(row.get("meta_direction", 1))   # 2=BUY, 0=SELL, 1=HOLD

            if meta_dir == 2:
                if not self.tradeable_buy  or meta_conf < self.thr_buy:
                    continue
                direction = "long"
                thr = self.thr_buy
            elif meta_dir == 0:
                if not self.tradeable_sell or meta_conf < self.thr_sell:
                    continue
                direction = "short"
                thr = self.thr_sell
            else:
                continue    # HOLD — never trade

            total_signals += 1

            # ── S&R + TREND FILTER (mirrors predict_signal in live engine) ───
            is_high_conviction = meta_conf >= self.override_thr
            if not is_high_conviction:
                at_res   = bool(row.get("is_at_resistance", 0))
                at_sup   = bool(row.get("is_at_support",    0))
                if (direction == "long"  and at_res) or \
                   (direction == "short" and at_sup):
                    skipped_srt += 1
                    continue

                trend_1d = float(row.get("macro_trend_1d", 0.0))
                if (direction == "long"  and trend_1d < -0.2) or \
                   (direction == "short" and trend_1d >  0.2):
                    skipped_srt += 1
                    continue

            # ── EXPECTED VALUE GATE ───────────────────────────────────────────
            cur_atr = float(row["atr_14"]) if float(row["atr_14"]) > 0 \
                      else float(row["close"]) * 0.01
            ev = _ev(meta_conf, cur_atr, float(row["close"]), atr_sl, atr_tp)
            if ev <= COST_MULTIPLIER * TOTAL_COST_PCT:
                skipped_cost += 1
                continue

            # ── POSITION SIZING ───────────────────────────────────────────────
            size_factor = _meta_size_factor(meta_conf, thr)
            sl_dist     = max(atr_sl * cur_atr, float(row["close"]) * MIN_SL_DISTANCE_PCT)
            risk_amt    = capital * risk_pct * size_factor
            units       = min(risk_amt / sl_dist, capital / float(row["close"]))
            if units <= 0:
                continue

            entry_price   = float(row["close"])
            entry_atr_val = cur_atr
            pos_units     = units
            entry_bar     = i                             # ← track entry bar
            entry_cost    = pos_units * entry_price * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
            capital      -= entry_cost

        # ── METRICS ──────────────────────────────────────────────────────────
        if result.total_trades == 0:
            result.error = "No trades executed"
            return result

        result.net_return_pct      = round((capital - self.initial_balance) / self.initial_balance * 100, 2)
        result.max_drawdown_pct    = round(max_dd, 2)
        result.win_rate            = round(result.wins / result.total_trades * 100, 2)
        result.avg_trade_return_pct = round(float(np.mean(trade_returns)) * 100, 4) if trade_returns else 0.0
        result.sharpe_ratio        = round(_sharpe(trade_returns), 3)
        result.total_signals       = total_signals
        result.skipped_cost        = skipped_cost
        result.skipped_sr_trend    = skipped_srt
        result.low_confidence      = result.total_trades < MIN_TRADES_REQUIRED

        if result.gross_loss > 0:
            result.profit_factor = round(result.gross_profit / result.gross_loss, 3)
        elif result.gross_profit > 0:
            result.profit_factor = None   # all wins, no losses
        else:
            result.profit_factor = 0.0
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_exit(self, row: pd.Series, entry_price: float, entry_atr: float,
                    atr_sl: float, atr_tp: float,
                    direction: Optional[str]) -> Optional[float]:
        """Check current bar's high/low for TP or SL. No lookahead."""
        if direction == "long":
            tp = entry_price + atr_tp * entry_atr
            sl = entry_price - atr_sl * entry_atr
            if row["high"] >= tp and row["low"] <= sl:
                return sl   # gap-through: assume SL first (conservative)
            if row["high"] >= tp:
                return tp
            if row["low"]  <= sl:
                return sl
        else:  # short
            tp = entry_price - atr_tp * entry_atr
            sl = entry_price + atr_sl * entry_atr
            if row["low"] <= tp and row["high"] >= sl:
                return sl
            if row["low"]  <= tp:
                return tp
            if row["high"] >= sl:
                return sl
        return None

    def _fetch_and_prepare(self, hours: int) -> Optional[pd.DataFrame]:
        if self.symbol in AdaptiveBacktester._df_cache:
            return AdaptiveBacktester._df_cache[self.symbol]

        df = self.predictor.fetch_live_data(limit=hours)
        if df is None or df.empty:
            return None

        btc_df   = self.predictor.fetch_btc_data(limit=hours)
        news_df  = self.predictor.load_news_data()

        if btc_df is None or btc_df.empty:
            btc_df = pd.DataFrame({"timestamp": df["timestamp"], "close": 0.0})
            btc_df["open"] = btc_df["high"] = btc_df["low"] = btc_df["close"]
            btc_df["volume"] = 0
        if news_df is None or news_df.empty:
            news_df = pd.DataFrame({"timestamp": df["timestamp"], "sentiment": 0.0})

        df_1d = None
        try:
            df_1d = self.predictor.fetch_live_data(
                timeframe="1d", limit=max(1000, int(hours / 24) + 50))
        except Exception:
            pass

        df = prepare_features(
            df, btc_df=btc_df, news_df=news_df,
            add_target_flag=False, forward_hours=MAX_HOLD_CANDLES,
            df_1d=df_1d, df_1w=None, macro_state=None,
        )
        if df is None or df.empty:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Batch primary + meta inference — exact same gating as live engine.
        proba, meta_conf = self.predictor.predict_meta_batch(df)
        df["prob_sell"]  = proba[:, 0]
        df["prob_hold"]  = proba[:, 1]
        df["prob_buy"]   = proba[:, 2]
        df["meta_conf"]  = meta_conf

        # meta_direction: 2=BUY, 0=SELL, 1=HOLD (when hold prob dominates)
        dir_raw = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
        df["meta_direction"] = np.where(
            np.maximum(proba[:, 2], proba[:, 0]) > proba[:, 1],
            dir_raw, 1
        )
        df["atr_14"] = compute_atr(df, 14)

        AdaptiveBacktester._df_cache[self.symbol] = df
        return df


# ══════════════════════════════════════════════════════════════════════════════
# FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_fleet():
    AdaptiveBacktester._df_cache.clear()
    AdaptiveBacktester._predictor_cache.clear()

    print("=" * 80)
    print("🧠 AEGIS-1 BACKTESTER v3.0 — meta-labeling aligned")
    print(f"   Balance ${INITIAL_BALANCE:,.0f} | Data {DATA_HOURS} h | {len(FLEET)} symbols")
    print(f"   Gate: per-symbol meta_threshold_buy/sell + S&R/trend filter")
    print("=" * 80)

    all_results = []
    for symbol in FLEET:
        print(f"\n📊 {symbol}")
        for mode in MODE_PARAMS:
            bt  = AdaptiveBacktester(symbol, mode)
            res = bt.run(DATA_HOURS)
            if res.error:
                print(f"   {mode.capitalize():<12} ❌  {res.error}")
            else:
                pf_str = f"{res.profit_factor:.2f}" if res.profit_factor is not None else "∞"
                print(
                    f"   {mode.capitalize():<12} | "
                    f"Trades {res.total_trades:3d} | "
                    f"Return {res.net_return_pct:+6.2f}% | "
                    f"WR {res.win_rate:5.1f}% | "
                    f"PF {pf_str} | "
                    f"Signals {res.total_signals} "
                    f"(skipped cost {res.skipped_cost}, S&R/trend {res.skipped_sr_trend})"
                )
                res_dict = asdict(res)
                res_dict.pop("trades", None)   # keep JSON compact
                all_results.append(res_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = BACKTEST_DIR / f"backtest_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ {len(all_results)} results saved → {out_file}")


if __name__ == "__main__":
    run_backtest_fleet()
