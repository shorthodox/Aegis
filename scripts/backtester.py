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

# Strategy tuning
DYNAMIC_THRESHOLD_BASE = 0.68
CONVICTION_SPREAD = 0.18
BTC_GUARD_BUY = 0.55
BTC_GUARD_SELL = 0.55
MIN_CONFIDENCE_FOR_TRADE = 0.62


def normalize_probability(prob: Any) -> List[float]:
    if isinstance(prob, np.ndarray):
        proba = prob.tolist()
    elif isinstance(prob, (list, tuple)):
        proba = list(prob)
    else:
        try:
            value = float(prob)
            return [max(0.0, 1.0 - value), 0.0, min(1.0, value)]
        except Exception:
            return [0.0, 1.0, 0.0]

    if len(proba) < 3:
        proba = (proba + [0.0, 0.0, 0.0])[:3]
    return [float(x) for x in proba]


def get_dynamic_barrier_multiplier(vol_regime: Any, alpha_flag: str) -> float:
    adjustment = 1.0
    if vol_regime is not None:
        try:
            if isinstance(vol_regime, str):
                vr = vol_regime.lower()
                if 'high' in vr:
                    adjustment += 0.12
                elif 'low' in vr:
                    adjustment -= 0.08
            else:
                # vol_regime may be various types; convert safely to float
                try:
                    vnum = float(vol_regime)
                except Exception:
                    # fallback: convert to string then to float
                    vnum = float(str(vol_regime))
                if vnum > 1.2:
                    adjustment += 0.10
                elif vnum < 0.9:
                    adjustment -= 0.08
        except Exception:
            pass

    if alpha_flag == 'HIGH_RISK_VOLATILITY':
        adjustment += 0.10

    return float(np.clip(adjustment, 0.80, 1.35))


def select_signal(buy_prob: float, sell_prob: float, hold_prob: float,
                  btc_buy_prob: float, btc_sell_prob: float,
                  threshold: float, row: pd.Series) -> str:
    if hold_prob >= max(buy_prob, sell_prob) and hold_prob > 0.55:
        return 'HOLD'

    buy_strength = buy_prob - max(hold_prob, sell_prob)
    sell_strength = sell_prob - max(hold_prob, buy_prob)
    momentum = float(row.get('close_delta_4', 0.0))

    if buy_prob >= threshold and buy_strength >= CONVICTION_SPREAD and btc_buy_prob >= BTC_GUARD_BUY:
        if momentum >= 0 or buy_prob > 0.75:
            return 'BUY'

    if sell_prob >= threshold and sell_strength >= CONVICTION_SPREAD and btc_sell_prob >= BTC_GUARD_SELL:
        if momentum <= 0 or sell_prob > 0.75:
            return 'SELL'

    return 'HOLD'


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

        print(f"   Fetching higher-timeframe macro context...")
        df_1d = None
        try:
            df_1d = self.predictor.fetch_live_data(timeframe='1d', limit=max(1000, int(hours / 24) + 50))
        except Exception:
            df_1d = None

        print(f"   Applying feature engineering...")
        df = prepare_features(
            df,
            btc_df=btc_df,
            news_df=news_df,
            add_target_flag=False,
            forward_hours=self.lookahead,
            df_1d=df_1d,
            df_1w=None,
            macro_state=None
        )
        if df is None or df.empty:
            return None
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Generate predictions
        print(f"   Generating predictions...")
        predictions = self.predictor.predict(df)
        if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
            df['prob_sell'] = [row[0] for row in predictions]
            df['prob_hold'] = [row[1] for row in predictions]
            df['prob_buy'] = [row[2] for row in predictions]
            df['prob'] = [row.tolist() for row in predictions]
        else:
            df['prob'] = predictions
            df['prob_sell'] = 0.0
            df['prob_hold'] = 0.0
            df['prob_buy'] = 0.0

        # Generate BTC predictions for correlation guard
        print(f"   Generating BTC predictions...")
        btc_df_filtered = btc_df[btc_df['timestamp'].isin(df['timestamp'])].copy()
        btc_df_filtered = btc_df_filtered.sort_values('timestamp').reset_index(drop=True)
        btc_features = prepare_features(
            btc_df_filtered,
            btc_df=btc_df_filtered,
            news_df=news_df,
            add_target_flag=False,
            forward_hours=self.lookahead,
            df_1d=None,
            df_1w=None,
            macro_state=None
        )
        if btc_features is not None and not btc_features.empty:
            btc_features = btc_features[btc_features['timestamp'].isin(df['timestamp'])].sort_values('timestamp').reset_index(drop=True)
            btc_predictions = self.btc_predictor.predict(btc_features)
            if isinstance(btc_predictions, np.ndarray) and btc_predictions.ndim == 2:
                btc_prob_series = pd.Series(index=btc_features['timestamp'], data=[row.tolist() for row in btc_predictions])
            else:
                btc_prob_series = pd.Series(index=btc_features['timestamp'], data=list(btc_predictions))
            df['btc_prob'] = df['timestamp'].map(btc_prob_series)
            missing_mask = df['btc_prob'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'btc_prob'] = [[0.0, 1.0, 0.0]] * missing_mask.sum()
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
            "sum_realized_return_buy": 0.0,
            "sum_realized_return_sell": 0.0,
            "sum_realized_return_hold": 0.0,
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

            sell_prob, hold_prob, buy_prob = normalize_probability(prob)
            btc_sell_prob, btc_hold_prob, btc_buy_prob = normalize_probability(row.get('btc_prob', [0.0, 1.0, 0.0]))

            base_thresh = 0.72 if row['alpha_risk_flag'] == 'HIGH_RISK_VOLATILITY' else DYNAMIC_THRESHOLD_BASE
            threshold = adjust_threshold_by_technical_and_fundamental(
                base_thresh,
                vol_regime=row.get('volatility_regime', None),
                news_score=row.get('news_score', 0.0),
                efficiency_ratio=row.get('efficiency_ratio', row.get('efficiency_ratio_10', 0.0)),
                btc_anchor=row.get('btc_dist_ema200', 0.0)
            )
            threshold = max(threshold, MIN_CONFIDENCE_FOR_TRADE)

            predicted = select_signal(
                buy_prob, sell_prob, hold_prob,
                btc_buy_prob, btc_sell_prob,
                threshold,
                row
            )

            entry = row['close']
            atr_val = row['atr_14']
            if atr_val == 0 or np.isnan(atr_val):
                atr_val = max(entry * 0.001, 1e-8)

            barrier_multiplier = self.multiplier * get_dynamic_barrier_multiplier(row.get('volatility_regime', None), row['alpha_risk_flag'])
            upper = entry + barrier_multiplier * atr_val
            lower = entry - barrier_multiplier * atr_val

            window = df.iloc[i + 1 : i + self.lookahead + 1]
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

            if entry <= 0:
                continue

            future_close = window['close'].iloc[-1]
            future_return = (future_close - entry) / entry
            max_high = window['high'].max()
            min_low = window['low'].min()

            if actual == 'BUY':
                realized_return = (upper - entry) / entry
                mfe_pct = (max_high - entry) / entry
                mae_pct = max(0.0, (entry - min_low) / entry)
            elif actual == 'SELL':
                realized_return = (entry - lower) / entry
                mfe_pct = (entry - min_low) / entry
                mae_pct = max(0.0, (max_high - entry) / entry)
            else:
                realized_return = future_return
                mfe_pct = (max_high - entry) / entry
                mae_pct = max(0.0, (entry - min_low) / entry)

            correct = predicted == actual
            stats['total_signals'] += 1
            if predicted == 'BUY':
                stats['correct_buy'] += int(correct)
                stats['sum_return_buy'] += future_return
                stats['sum_realized_return_buy'] += realized_return
                stats['sum_mfe_buy'] += mfe_pct
                stats['sum_mae_buy'] += mae_pct
                stats['sum_prob_buy'] += buy_prob
            elif predicted == 'SELL':
                stats['correct_sell'] += int(correct)
                stats['sum_return_sell'] += future_return
                stats['sum_realized_return_sell'] += realized_return
                stats['sum_mfe_sell'] += mfe_pct
                stats['sum_mae_sell'] += mae_pct
                stats['sum_prob_sell'] += sell_prob
            else:
                stats['correct_hold'] += int(correct)
                stats['sum_return_hold'] += future_return
                stats['sum_realized_return_hold'] += realized_return
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
                    'threshold': round(threshold, 4),
                    'btc_buy_prob': round(btc_buy_prob, 4),
                    'btc_sell_prob': round(btc_sell_prob, 4),
                    'future_return_pct': round(future_return * 100, 2),
                    'realized_return_pct': round(realized_return * 100, 2),
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
            stats[f'avg_realized_return_{label}'] = round(stats[f'sum_realized_return_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_mfe_{label}'] = round(stats[f'sum_mfe_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_mae_{label}'] = round(stats[f'sum_mae_{label}'] / count, 6) if count > 0 else 0.0
            stats[f'avg_prob_{label}'] = round(stats[f'sum_prob_{label}'] / count, 4) if count > 0 else 0.0
            stats[f'{label}_precision'] = round(correct / (stats[f'predicted_{label}'] or 1), 4)

        stats['bullish_strength'] = stats['avg_prob_buy']
        stats['bearish_strength'] = stats['avg_prob_sell']
        stats['actual_hold_ratio'] = round(stats['actual_hold'] / stats['total_signals'], 4)
        stats['high_risk_ratio'] = round(stats['alpha_risk_signals'] / stats['total_signals'], 4)

        trading_signals = stats['predicted_buy'] + stats['predicted_sell']
        correct_trading = stats['correct_buy'] + stats['correct_sell']
        stats['trading_accuracy'] = round(correct_trading / trading_signals, 4) if trading_signals > 0 else 0.0

        stats['avg_time_to_upper'] = round(np.mean(stats['time_to_upper']), 2) if stats['time_to_upper'] else 0.0
        stats['avg_time_to_lower'] = round(np.mean(stats['time_to_lower']), 2) if stats['time_to_lower'] else 0.0

        stats['avg_realized_return_buy'] = round(stats['sum_realized_return_buy'] / max(stats['predicted_buy'], 1), 6)
        stats['avg_realized_return_sell'] = round(stats['sum_realized_return_sell'] / max(stats['predicted_sell'], 1), 6)
        stats['avg_realized_return_hold'] = round(stats['sum_realized_return_hold'] / max(stats['predicted_hold'], 1), 6)

        stats['buy_risk_adjusted'] = round(stats['avg_realized_return_buy'] / stats['avg_mae_buy'], 4) if stats['avg_mae_buy'] > 0 else 0.0
        stats['sell_risk_adjusted'] = round(stats['avg_realized_return_sell'] / stats['avg_mae_sell'], 4) if stats['avg_mae_sell'] > 0 else 0.0
        stats['avg_confidence'] = round((stats['avg_prob_buy'] + stats['avg_prob_sell'] + stats['avg_prob_hold']) / 3.0, 4)
        stats['signal_density'] = round(stats['total_signals'] / max(len(df), 1), 4)

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
    print(f"   Base threshold: ~{DYNAMIC_THRESHOLD_BASE:.2f} | Conviction spread: {CONVICTION_SPREAD:.2f}")
    print(f"   BTC guard: buy≥{BTC_GUARD_BUY:.2f}, sell≥{BTC_GUARD_SELL:.2f} | Min confidence: {MIN_CONFIDENCE_FOR_TRADE:.2f}")
    print("   Metrics: Direction accuracy, realized returns, risk-adjusted return, MFE, MAE")
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
    # Sort by average buy return
    sorted_by_return = sorted([r for r in all_results if r.get('alpha_stability') != 'UNSTABLE_ALPHA'], key=lambda x: x.get("avg_return_buy", 0), reverse=True)
    # Sort by risk-adjusted realized buy return
    sorted_by_risk_adjusted = sorted([r for r in all_results if r.get('alpha_stability') != 'UNSTABLE_ALPHA'], key=lambda x: x.get("buy_risk_adjusted", 0), reverse=True)

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
            print(f"{i:2}. {res['symbol']:12} | Avg Return: {res['avg_return_buy']*100:6.2f}% | Realized: {res.get('avg_realized_return_buy',0)*100:6.2f}% | MFE: {res['avg_mfe_buy']*100:6.2f}% | MAE: {res['avg_mae_buy']*100:6.2f}%")

    print("\n" + "=" * 80)
    print("🏅 BEST SYMBOLS BY RISK-ADJUSTED BUY RETURN")
    print("=" * 80)
    for i, res in enumerate(sorted_by_risk_adjusted[:10], 1):
        if res.get("predicted_buy", 0) > 0:
            print(f"{i:2}. {res['symbol']:12} | RiskAdj: {res.get('buy_risk_adjusted',0):.2f} | Realized: {res.get('avg_realized_return_buy',0)*100:6.2f}% | MAE: {res.get('avg_mae_buy',0)*100:6.2f}%")

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
        print(f"{res['symbol']:12} | Bullish strength: {res['bullish_strength']:.3f} | Bearish strength: {res['bearish_strength']:.3f} | Avg confidence: {res.get('avg_confidence', 0):.3f}")

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
        "buy_threshold": "dynamic (0.62-0.75)",
        "sell_threshold": "dynamic (0.62-0.75)",
        "conviction_spread": CONVICTION_SPREAD,
        "btc_guard_buy": BTC_GUARD_BUY,
        "btc_guard_sell": BTC_GUARD_SELL,
        "best_by_accuracy": [(r["symbol"], r["accuracy"], r.get("trading_accuracy", 0), r.get("safety_score", 0), r.get("avg_time_to_upper", 0), r.get("avg_time_to_lower", 0)) for r in sorted_by_acc[:10]],
        "best_by_buy_return": [(r["symbol"], r["avg_return_buy"], r.get("avg_realized_return_buy", 0)) for r in sorted_by_return[:10] if r.get("predicted_buy", 0) > 0],
        "best_by_risk_adjusted": [(r["symbol"], r.get("buy_risk_adjusted", 0), r.get("avg_realized_return_buy", 0)) for r in sorted_by_risk_adjusted[:10] if r.get("predicted_buy", 0) > 0],
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