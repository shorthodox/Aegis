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
from src.ml.feature_engine import prepare_features, compute_atr
from src.ml.delltandecay import adjust_threshold_by_technical_and_fundamental

# -------------------------------------------------------------------
# ATR Multiplier for Triple Barriers
# -------------------------------------------------------------------
def get_atr_multiplier(symbol: str) -> float:
    """
    ATR multipliers for triple barriers (aligned with retrain_model.py).
    Heavy caps use 1.2x ATR.
    Low-accuracy assets (like SUI, DOT) get wider barriers (1.8x) to avoid stop-outs.
    Others use 1.5x.
    """
    high_cap = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    low_accuracy = ['SUI/USDT', 'DOT/USDT', 'SEI/USDT', 'KAS/USDT']
    if symbol in high_cap:
        return 1.2
    elif symbol in low_accuracy:
        return 1.8
    else:
        return 1.5

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
        self.btc_predictor = Predictor('BTC/USDT')
        self.multiplier = get_atr_multiplier(symbol)

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
        df = prepare_features(
            df,
            btc_df=btc_df,
            news_df=news_df,
            add_target_flag=True,
            forward_hours=self.lookahead
        )
        if df is None or df.empty:
            return None

        # Generate predictions
        print(f"   Generating predictions...")
        predictions = self.predictor.predict(df)
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            df['prob'] = [row.tolist() for row in predictions]
        else:
            df['prob'] = predictions

        # Generate BTC predictions for correlation guard
        print(f"   Generating BTC predictions...")
        # Filter btc_df to match df's timestamps to ensure same length
        btc_df_filtered = btc_df[btc_df['timestamp'].isin(df['timestamp'])].copy()
        btc_features = prepare_features(
            btc_df_filtered,
            btc_df=btc_df_filtered,
            news_df=news_df,
            add_target_flag=False,
            forward_hours=self.lookahead
        )
        if btc_features is not None and not btc_features.empty:
            # Ensure btc_features matches df's timestamps exactly
            btc_features = btc_features[btc_features['timestamp'].isin(df['timestamp'])]
            btc_predictions = self.btc_predictor.predict(btc_features)
            # Map btc_predictions to df by timestamp to handle any order differences
            btc_prob_series = pd.Series(index=btc_features['timestamp'], data=[row.tolist() for row in btc_predictions])
            df['btc_prob'] = df['timestamp'].map(btc_prob_series)
        else:
            df['btc_prob'] = [[0.0, 1.0, 0.0]] * len(df)  # neutral

        return df

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Iterate over DataFrame, generate signals, compute metrics using triple-barrier evaluation."""
        if df is None or len(df) < self.lookahead + 1:
            return {}

        df = df.copy()
        df['atr_ratio'] = df['atr_14'] / df['close']
        atr_mean = df['atr_ratio'].rolling(100, min_periods=1).mean()
        atr_std = df['atr_ratio'].rolling(100, min_periods=1).std().fillna(0)
        df['alpha_risk_flag'] = np.where(
            df['atr_ratio'] > atr_mean + 2 * atr_std,
            'HIGH_RISK_VOLATILITY',
            'NORMAL'
        )

        stats = {
            "symbol": self.symbol,
            "total_signals": 0,
            "predicted_buy": 0,
            "predicted_sell": 0,
            "predicted_hold": 0,
            "actual_buy": 0,
            "actual_sell": 0,
            "actual_hold": 0,
            "correct_buy": 0,
            "correct_sell": 0,
            "correct_hold": 0,
            "sum_return_buy": 0.0,
            "sum_return_sell": 0.0,
            "sum_return_hold": 0.0,
            "sum_mfe_buy": 0.0,
            "sum_mfe_sell": 0.0,
            "sum_mfe_hold": 0.0,
            "sum_mae_buy": 0.0,
            "sum_mae_sell": 0.0,
            "sum_mae_hold": 0.0,
            "sum_prob_buy": 0.0,
            "sum_prob_sell": 0.0,
            "sum_prob_hold": 0.0,
            "alpha_risk_signals": 0,
            "time_to_upper": [],
            "time_to_lower": [],
            "signals": []
        }

        for i in range(len(df) - self.lookahead):
            row = df.iloc[i]
            prob = row['prob']
            if prob is None or (isinstance(prob, float) and np.isnan(prob)):
                continue

            if isinstance(prob, np.ndarray):
                proba = prob.tolist()
            elif isinstance(prob, (list, tuple)):
                proba = list(prob)
            else:
                proba = [1 - float(prob), 0.0, float(prob)]

            if len(proba) < 3:
                proba = (proba + [0.0, 0.0, 0.0])[:3]

            sell_prob, hold_prob, buy_prob = proba[0], proba[1], proba[2]

            # Dynamic threshold based on volatility -> further adjusted by fundamentals and anchor
            base_thresh = 0.75 if row['alpha_risk_flag'] == 'HIGH_RISK_VOLATILITY' else 0.65
            threshold = adjust_threshold_by_technical_and_fundamental(
                base_thresh,
                vol_regime=row.get('volatility_regime', None),
                news_score=row.get('news_score', 0.0),
                efficiency_ratio=row.get('efficiency_ratio', row.get('efficiency_ratio_10', 0.0)),
                btc_anchor=(row.get('btc_dist_ema200', 0.0))
            )

            # BTC correlation guard
            btc_prob = row.get('btc_prob', [0.0, 1.0, 0.0])
            if isinstance(btc_prob, np.ndarray):
                btc_proba = btc_prob.tolist()
            elif isinstance(btc_prob, (list, tuple)):
                btc_proba = list(btc_prob)
            else:
                btc_proba = [1 - float(btc_prob), 0.0, float(btc_prob)]
            if len(btc_proba) < 3:
                btc_proba = (btc_proba + [0.0, 0.0, 0.0])[:3]
            btc_sell_prob, btc_hold_prob, btc_buy_prob = btc_proba

            # Signal trigger with conviction spread
            predicted = 'HOLD'
            if buy_prob > threshold and (buy_prob - hold_prob) >= 0.20 and btc_buy_prob > 0.55:
                predicted = 'BUY'
            elif sell_prob > threshold and btc_sell_prob < 0.45:
                predicted = 'SELL'

            # Compute actual outcome using triple barriers
            entry = row['close']
            atr_val = row['atr_14']
            if atr_val == 0 or np.isnan(atr_val):
                atr_val = entry * 0.001
            upper = entry + self.multiplier * atr_val
            lower = entry - self.multiplier * atr_val

            window = df.iloc[i+1 : i+self.lookahead+1]
            if len(window) < self.lookahead:
                continue

            hit_upper = False
            hit_lower = False
            time_to_hit = None
            for j in range(1, min(self.lookahead, len(df) - i)):
                wrow = df.iloc[i + j]
                if wrow['high'] >= upper:
                    hit_upper = True
                    time_to_hit = j
                    break
                if wrow['low'] <= lower:
                    hit_lower = True
                    time_to_hit = j
                    break

            if hit_upper and not hit_lower:
                actual = 'BUY'
                stats['time_to_upper'].append(time_to_hit)
            elif hit_lower and not hit_upper:
                actual = 'SELL'
                stats['time_to_lower'].append(time_to_hit)
            else:
                actual = 'HOLD'

            if actual == 'BUY':
                stats['actual_buy'] += 1
            elif actual == 'SELL':
                stats['actual_sell'] += 1
            else:
                stats['actual_hold'] += 1

            if predicted == 'BUY':
                stats['predicted_buy'] += 1
            elif predicted == 'SELL':
                stats['predicted_sell'] += 1
            else:
                stats['predicted_hold'] += 1

            entry = row['close']
            if entry <= 0:
                continue

            future_close = window['close'].iloc[-1]
            future_return = (future_close - entry) / entry
            max_high = window['high'].max()
            min_low = window['low'].min()

            if actual == 'BUY':
                mfe_pct = (max_high - entry) / entry
                mae_pct = (entry - min_low) / entry
            elif actual == 'SELL':
                mfe_pct = (entry - min_low) / entry
                mae_pct = (max_high - entry) / entry
            else:
                mfe_pct = (max_high - entry) / entry
                mae_pct = (entry - min_low) / entry

            correct = predicted == actual
            stats['total_signals'] += 1
            if predicted == 'BUY':
                stats['correct_buy'] += int(correct)
                stats['sum_return_buy'] += future_return
                stats['sum_mfe_buy'] += mfe_pct
                stats['sum_mae_buy'] += mae_pct
                stats['sum_prob_buy'] += buy_prob
            elif predicted == 'SELL':
                stats['correct_sell'] += int(correct)
                stats['sum_return_sell'] += future_return
                stats['sum_mfe_sell'] += mfe_pct
                stats['sum_mae_sell'] += mae_pct
                stats['sum_prob_sell'] += sell_prob
            else:
                stats['correct_hold'] += int(correct)
                stats['sum_return_hold'] += future_return
                stats['sum_mfe_hold'] += mfe_pct
                stats['sum_mae_hold'] += mae_pct
                stats['sum_prob_hold'] += hold_prob

            if row['alpha_risk_flag'] == 'HIGH_RISK_VOLATILITY':
                stats['alpha_risk_signals'] += 1

            if stats['total_signals'] <= 1000:
                stats['signals'].append({
                    'timestamp': str(row['timestamp']),
                    'predicted': predicted,
                    'actual': actual,
                    'prob_sell': round(sell_prob, 4),
                    'prob_hold': round(hold_prob, 4),
                    'prob_buy': round(buy_prob, 4),
                    'future_return_pct': round(future_return * 100, 2),
                    'mfe_pct': round(mfe_pct * 100, 2),
                    'mae_pct': round(mae_pct * 100, 2),
                    'correct': correct,
                    'alpha_risk': row['alpha_risk_flag']
                })

        if stats['total_signals'] == 0:
            return stats

        stats['accuracy'] = round(
            (stats['correct_buy'] + stats['correct_sell'] + stats['correct_hold']) / stats['total_signals'],
            4
        )
        stats['hold_frequency'] = round(stats['predicted_hold'] / stats['total_signals'], 4)

        for label in ['buy', 'sell', 'hold']:
            count = stats[f'predicted_{label}']
            correct = stats[f'correct_{label}']
            stats[f'{label}_accuracy'] = round(correct / count, 4) if count > 0 else 0.0
            stats[f'avg_return_{label}'] = round(stats[f'sum_return_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_mfe_{label}'] = round(stats[f'sum_mfe_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_mae_{label}'] = round(stats[f'sum_mae_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_prob_{label}'] = round(stats[f'sum_prob_{label}'] / count, 4) if count > 0 else 0.0

        stats['bullish_strength'] = stats['avg_prob_buy']
        stats['bearish_strength'] = stats['avg_prob_sell']
        stats['actual_hold_ratio'] = round(stats['actual_hold'] / stats['total_signals'], 4)
        stats['high_risk_ratio'] = round(stats['alpha_risk_signals'] / stats['total_signals'], 4)

        # Trading accuracy (BUY + SELL signals only)
        trading_signals = stats['predicted_buy'] + stats['predicted_sell']
        correct_trading = stats['correct_buy'] + stats['correct_sell']
        stats['trading_accuracy'] = round(correct_trading / trading_signals, 4) if trading_signals > 0 else 0.0

        # Exit metrics
        stats['avg_time_to_upper'] = round(np.mean(stats['time_to_upper']), 2) if stats['time_to_upper'] else 0.0
        stats['avg_time_to_lower'] = round(np.mean(stats['time_to_lower']), 2) if stats['time_to_lower'] else 0.0

        # Safety score
        stats['safety_score'] = round(stats['avg_mae_buy'] / stats['avg_mfe_buy'], 4) if stats['avg_mfe_buy'] > 0 else 0.0
        stats['alpha_stability'] = 'STABLE' if stats['safety_score'] <= 2.0 else 'UNSTABLE_ALPHA'

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
        print(
            f"   ✅ {symbol} – Total: {stats['total_signals']} | Buy: {stats['predicted_buy']} | Sell: {stats['predicted_sell']} | Hold: {stats['predicted_hold']} | Accuracy: {stats['accuracy']*100:.1f}% | Trading Acc: {stats.get('trading_accuracy', 0)*100:.1f}% | Stability: {stats.get('alpha_stability', 'UNKNOWN')}"
        )

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
    sorted_by_acc = sorted([r for r in all_results if r.get('alpha_stability') != 'UNSTABLE_ALPHA'], key=lambda x: x["accuracy"], reverse=True)
    # Sort by average return (buy)
    sorted_by_return = sorted([r for r in all_results if r.get('alpha_stability') != 'UNSTABLE_ALPHA'], key=lambda x: x.get("avg_return_buy", 0), reverse=True)

    print("\n" + "=" * 80)
    print("🏆 BEST SYMBOLS BY ACCURACY (overall) - STABLE ALPHA ONLY")
    print("=" * 80)
    for i, res in enumerate(sorted_by_acc[:10], 1):
        print(
            f"{i:2}. {res['symbol']:12} | Accuracy: {res['accuracy']*100:5.1f}% | Trading Acc: {res.get('trading_accuracy', 0)*100:5.1f}% | Total: {res['total_signals']:4} | Buy acc: {res['buy_accuracy']*100:5.1f}% | Sell acc: {res['sell_accuracy']*100:5.1f}% | Hold acc: {res['hold_accuracy']*100:5.1f}% | Safety: {res.get('safety_score', 0):.2f}"
        )

    print("\n" + "=" * 80)
    print("📈 BEST SYMBOLS BY AVERAGE BUY RETURN")
    print("=" * 80)
    for i, res in enumerate(sorted_by_return[:10], 1):
        if res.get("predicted_buy", 0) > 0:
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

    # Unstable alpha warning
    unstable = [r for r in all_results if r.get('alpha_stability') == 'UNSTABLE_ALPHA']
    if unstable:
        print("\n" + "=" * 80)
        print("⚠️ UNSTABLE ALPHA TOKENS (SUPPRESSED FROM RANKINGS)")
        print("=" * 80)
        for res in unstable:
            print(f"{res['symbol']:12} | Safety Score: {res.get('safety_score', 0):.2f} | MAE/MFE Ratio > 2.0")

    # Save a simplified best/worst summary
    summary = {
        "timestamp": timestamp,
        "lookahead": lookahead,
        "buy_threshold": "dynamic (0.65-0.75)",
        "sell_threshold": "dynamic (0.65-0.75)",
        "conviction_spread": 0.20,
        "btc_guard_buy": 0.55,
        "btc_guard_sell": 0.45,
        "best_by_accuracy": [(r["symbol"], r["accuracy"], r.get("trading_accuracy", 0), r.get("safety_score", 0), r.get("avg_time_to_upper", 0), r.get("avg_time_to_lower", 0)) for r in sorted_by_acc[:10]],
        "best_by_buy_return": [(r["symbol"], r["avg_return_buy"]) for r in sorted_by_return[:10] if r.get("predicted_buy", 0) > 0],
        "worst_by_accuracy": [(r["symbol"], r["accuracy"]) for r in sorted_by_acc[-5:]],
        "unstable_alpha": [r["symbol"] for r in all_results if r.get('alpha_stability') == 'UNSTABLE_ALPHA']
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