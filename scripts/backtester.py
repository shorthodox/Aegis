#!/usr/bin/env python3
# signal_backtester.py — Aegis-1 Signal Quality Evaluator
# No trade simulation, only signal metrics: accuracy, MFE, MAE, future returns.

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# --- Path fix: ensure repository root on sys.path so `src` imports resolve ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
BACKTEST_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# Signal thresholds
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4

# Lookahead window for metrics (number of candles after signal)
LOOKAHEAD_CANDLES = 24   # 24 hours (1 day) – adjustable

# Data period (hours) – 365 days * 24h
DATA_HOURS = 365 * 24

# Fleet of symbols (full 58 as per original script)
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

MIN_HOURLY_BARS = 252   # Minimum bars required for meaningful analysis

# -------------------------------------------------------------------
# Signal Analyzer Class
# -------------------------------------------------------------------
class SignalAnalyzer:
    def __init__(self, symbol: str, lookahead: int = LOOKAHEAD_CANDLES):
        self.symbol = symbol
        self.lookahead = lookahead
        self.predictor = Predictor(symbol)

    def fetch_and_prepare_data(self, hours: int = DATA_HOURS) -> Optional[pd.DataFrame]:
        """Fetch OHLCV, BTC data, news, apply feature engineering, return DataFrame with 'prob'."""
        print(f"   Fetching data for {self.symbol}...")
        df = self.predictor.fetch_live_data(limit=hours)
        if df is None or df.empty:
            print(f"   ⚠️ No OHLCV data for {self.symbol}")
            return None

        btc_df = self.predictor.fetch_btc_data(limit=hours)
        news_df = self.predictor.load_news_data()

        if btc_df is None or btc_df.empty:
            print(f"   ⚠️ BTC data missing for {self.symbol}, creating dummy.")
            btc_df = pd.DataFrame({'timestamp': df['timestamp'], 'close': 0.0})
            btc_df['open'] = btc_df['high'] = btc_df['low'] = btc_df['close']
            btc_df['volume'] = 0

        if news_df is None or news_df.empty:
            print(f"   ⚠️ News data missing for {self.symbol}, using neutral sentiment.")
            news_df = pd.DataFrame({'timestamp': df['timestamp'], 'sentiment': 0.0})

        print(f"   Applying feature engineering...")
        df = prepare_features(df, btc_df=btc_df, news_df=news_df)
        if df is None or df.empty:
            return None

        # Generate predictions
        print(f"   Generating predictions...")
        df['prob'] = self.predictor.predict(df)
        return df

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Iterate over DataFrame, generate signals, compute metrics."""
        if df is None or len(df) < self.lookahead + 1:
            return {}

        stats = {
            "symbol": self.symbol,
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "correct_buy": 0,
            "correct_sell": 0,
            "sum_return_buy": 0.0,
            "sum_return_sell": 0.0,
            "sum_mfe_buy": 0.0,
            "sum_mfe_sell": 0.0,
            "sum_mae_buy": 0.0,
            "sum_mae_sell": 0.0,
            "sum_prob_buy": 0.0,
            "sum_prob_sell": 0.0,
            # Detailed list of signals (optional, for debugging)
            "signals": []   # each entry: index, timestamp, direction, prob, future_return, mfe_pct, mae_pct, correct
        }

        for i in range(len(df) - self.lookahead):
            row = df.iloc[i]
            prob = row['prob']
            if pd.isna(prob):
                continue

            direction = None
            if prob > BUY_THRESHOLD:
                direction = 'BUY'
            elif prob < SELL_THRESHOLD:
                direction = 'SELL'
            else:
                continue

            # Entry price = current close
            entry = row['close']
            if entry <= 0:
                continue

            # Look ahead window
            window = df.iloc[i+1 : i+self.lookahead+1]
            if len(window) < self.lookahead:
                continue   # insufficient data

            future_close = window['close'].iloc[-1]
            future_return = (future_close - entry) / entry

            # MFE and MAE (as percentages)
            if direction == 'BUY':
                max_high = window['high'].max()
                min_low = window['low'].min()
                mfe_pct = (max_high - entry) / entry
                mae_pct = (entry - min_low) / entry
                correct = future_return > 0
            else:  # SELL
                max_high = window['high'].max()
                min_low = window['low'].min()
                mfe_pct = (entry - min_low) / entry   # favorable move down
                mae_pct = (max_high - entry) / entry  # adverse move up
                correct = future_return < 0

            # Accumulate stats
            stats["total_signals"] += 1
            if direction == 'BUY':
                stats["buy_signals"] += 1
                stats["sum_return_buy"] += future_return
                stats["sum_mfe_buy"] += mfe_pct
                stats["sum_mae_buy"] += mae_pct
                stats["sum_prob_buy"] += prob
                if correct:
                    stats["correct_buy"] += 1
            else:
                stats["sell_signals"] += 1
                stats["sum_return_sell"] += future_return
                stats["sum_mfe_sell"] += mfe_pct
                stats["sum_mae_sell"] += mae_pct
                stats["sum_prob_sell"] += prob
                if correct:
                    stats["correct_sell"] += 1

            # Optional: store individual signal (reduce memory if many signals)
            if stats["total_signals"] <= 1000:   # limit for JSON size
                stats["signals"].append({
                    "timestamp": str(row['timestamp']),
                    "direction": direction,
                    "prob": round(prob, 4),
                    "future_return_pct": round(future_return * 100, 2),
                    "mfe_pct": round(mfe_pct * 100, 2),
                    "mae_pct": round(mae_pct * 100, 2),
                    "correct": correct
                })

        # Compute aggregated metrics
        if stats["total_signals"] == 0:
            return stats

        # Overall accuracy
        total_correct = stats["correct_buy"] + stats["correct_sell"]
        stats["accuracy"] = round(total_correct / stats["total_signals"], 4) if stats["total_signals"] > 0 else 0.0

        # Buy accuracy
        if stats["buy_signals"] > 0:
            stats["buy_accuracy"] = round(stats["correct_buy"] / stats["buy_signals"], 4)
            stats["avg_return_buy"] = round(stats["sum_return_buy"] / stats["buy_signals"], 6)
            stats["avg_mfe_buy"] = round(stats["sum_mfe_buy"] / stats["buy_signals"], 6)
            stats["avg_mae_buy"] = round(stats["sum_mae_buy"] / stats["buy_signals"], 6)
            stats["avg_prob_buy"] = round(stats["sum_prob_buy"] / stats["buy_signals"], 4)
        else:
            stats["buy_accuracy"] = 0.0
            stats["avg_return_buy"] = 0.0
            stats["avg_mfe_buy"] = 0.0
            stats["avg_mae_buy"] = 0.0
            stats["avg_prob_buy"] = 0.0

        if stats["sell_signals"] > 0:
            stats["sell_accuracy"] = round(stats["correct_sell"] / stats["sell_signals"], 4)
            stats["avg_return_sell"] = round(stats["sum_return_sell"] / stats["sell_signals"], 6)
            stats["avg_mfe_sell"] = round(stats["sum_mfe_sell"] / stats["sell_signals"], 6)
            stats["avg_mae_sell"] = round(stats["sum_mae_sell"] / stats["sell_signals"], 6)
            stats["avg_prob_sell"] = round(stats["sum_prob_sell"] / stats["sell_signals"], 4)
        else:
            stats["sell_accuracy"] = 0.0
            stats["avg_return_sell"] = 0.0
            stats["avg_mfe_sell"] = 0.0
            stats["avg_mae_sell"] = 0.0
            stats["avg_prob_sell"] = 0.0

        # Bullish strength vs bearish strength: average probability of buy vs sell signals
        stats["bullish_strength"] = stats["avg_prob_buy"]
        stats["bearish_strength"] = stats["avg_prob_sell"]

        # Remove detailed signals list from final JSON if too large (optional)
        # We keep it for now but it may be trimmed later.
        return stats

# -------------------------------------------------------------------
# Fleet analysis
# -------------------------------------------------------------------
def run_signal_analysis(lookahead: int = LOOKAHEAD_CANDLES, hours: int = DATA_HOURS):
    print("=" * 80)
    print("📊 AEGIS‑1 SIGNAL QUALITY ANALYZER")
    print(f"   Lookahead window: {lookahead} candles")
    print(f"   Buy threshold: >{BUY_THRESHOLD} | Sell threshold: <{SELL_THRESHOLD}")
    print("   Metrics: Direction accuracy, Future return, MFE, MAE (no trade simulation)")
    print("=" * 80)

    all_results = []
    valid_symbols = []
    # Simple symbol validation (skip if Predictor fails later)
    for sym in FLEET:
        # Quick test: try to create predictor and fetch a small sample
        try:
            test_pred = Predictor(sym)
            test_df = test_pred.fetch_live_data(limit=100)
            if test_df is not None and len(test_df) > 50:
                valid_symbols.append(sym)
            else:
                print(f"⚠️ {sym} – insufficient data, skipping.")
        except Exception as e:
            print(f"⚠️ {sym} – error during validation: {e}, skipping.")
    print(f"\n✅ {len(valid_symbols)} symbols passed initial validation out of {len(FLEET)}.")

    for idx, symbol in enumerate(valid_symbols, 1):
        print(f"\n[{idx}/{len(valid_symbols)}] Analyzing {symbol}...")
        analyzer = SignalAnalyzer(symbol, lookahead=lookahead)
        df = analyzer.fetch_and_prepare_data(hours=hours)
        if df is None:
            print(f"   ❌ {symbol} – data preparation failed.")
            continue
        stats = analyzer.analyze_signals(df)
        if stats["total_signals"] == 0:
            print(f"   ⚠️ {symbol} – no signals generated.")
            continue
        all_results.append(stats)
        print(f"   ✅ {symbol} – Signals: {stats['total_signals']} (Buy: {stats['buy_signals']}, Sell: {stats['sell_signals']}) | Accuracy: {stats['accuracy']*100:.1f}%")

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = BACKTEST_DIR / f"signal_analysis_{timestamp}.json"
    # Remove the detailed 'signals' list from each result to keep file size manageable
    output_results = []
    for res in all_results:
        res_copy = res.copy()
        if "signals" in res_copy:
            # Keep only first 100 signals as sample
            res_copy["signals_sample"] = res_copy["signals"][:100]
            del res_copy["signals"]
        output_results.append(res_copy)
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\n📁 Full signal analysis saved to {results_path}")

    # Print summary
    if not all_results:
        print("❌ No valid results.")
        return

    # Sort by accuracy
    sorted_by_acc = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    # Sort by average return (buy)
    sorted_by_return = sorted(all_results, key=lambda x: x.get("avg_return_buy", 0), reverse=True)

    print("\n" + "=" * 80)
    print("🏆 BEST SYMBOLS BY ACCURACY (overall)")
    print("=" * 80)
    for i, res in enumerate(sorted_by_acc[:10], 1):
        print(f"{i:2}. {res['symbol']:12} | Accuracy: {res['accuracy']*100:5.1f}% | Signals: {res['total_signals']:4} | Buy acc: {res['buy_accuracy']*100:5.1f}% | Sell acc: {res['sell_accuracy']*100:5.1f}%")

    print("\n" + "=" * 80)
    print("📈 BEST SYMBOLS BY AVERAGE BUY RETURN")
    print("=" * 80)
    for i, res in enumerate(sorted_by_return[:10], 1):
        if res["buy_signals"] > 0:
            print(f"{i:2}. {res['symbol']:12} | Avg Return: {res['avg_return_buy']*100:6.2f}% | MFE: {res['avg_mfe_buy']*100:6.2f}% | MAE: {res['avg_mae_buy']*100:6.2f}%")

    print("\n" + "=" * 80)
    print("📉 WORST SYMBOLS BY ACCURACY (bottom 5)")
    print("=" * 80)
    for i, res in enumerate(sorted_by_acc[-5:][::-1], 1):
        print(f"{i:2}. {res['symbol']:12} | Accuracy: {res['accuracy']*100:5.1f}% | Signals: {res['total_signals']:4}")

    # Additional: symbols with strong bullish vs bearish signal strength
    print("\n" + "=" * 80)
    print("💪 BULLISH VS BEARISH STRENGTH (avg probability)")
    print("=" * 80)
    for res in all_results[:10]:
        print(f"{res['symbol']:12} | Bullish strength: {res['bullish_strength']:.3f} | Bearish strength: {res['bearish_strength']:.3f}")

    # Save a simplified best/worst summary
    summary = {
        "timestamp": timestamp,
        "lookahead": lookahead,
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "best_by_accuracy": [(r["symbol"], r["accuracy"]) for r in sorted_by_acc[:10]],
        "best_by_buy_return": [(r["symbol"], r["avg_return_buy"]) for r in sorted_by_return[:10] if r["buy_signals"] > 0],
        "worst_by_accuracy": [(r["symbol"], r["accuracy"]) for r in sorted_by_acc[-5:]]
    }
    summary_path = BACKTEST_DIR / "signal_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Summary saved to {summary_path}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        look_input = input(f"Enter lookahead candles (default {LOOKAHEAD_CANDLES}): ").strip()
        lookahead = int(look_input) if look_input else LOOKAHEAD_CANDLES
        hours_input = input(f"Enter data hours (default {DATA_HOURS}): ").strip()
        hours = int(hours_input) if hours_input else DATA_HOURS
    except:
        lookahead = LOOKAHEAD_CANDLES
        hours = DATA_HOURS

    run_signal_analysis(lookahead=lookahead, hours=hours)