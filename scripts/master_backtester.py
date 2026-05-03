import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# --- THE PATH FIX ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.ml.predictor import Predictor
    from src.data.signals.news_engine import NewsEngine
    from src.ml.feature_engine import prepare_features
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class MasterBacktester:
    def __init__(self, symbol="BTC/USDT"):
        self.symbol = symbol
        self.predictor = Predictor(symbol)
        self.news_engine = NewsEngine()
        # Increased fee slightly to account for 2026 slippage
        self.fee = 0.002 # 0.2% total (Entry + Exit + Slippage)

    def run_test(self, days=30):
        print(f"🕵️ Running Tuned Master Strategy...")
        
        # 1. Fetch and Prepare
        df = self.predictor.fetch_live_data(timeframe='1h', limit=24*days)
        if df is None:
            raise RuntimeError("Failed to fetch live data for backtest.")
        df = prepare_features(df)
        
        # 2. Get AI Probabilities
        X = df.drop(columns=['timestamp', 'target'], errors='ignore')
        if self.predictor.model is None:
            raise RuntimeError("Predictor model is not loaded.")
        df['ai_prob'] = self.predictor.model.predict_proba(X)[:, 1]
        
        # 3. Get News Sentiment & Safety Switch
        # We fetch current sentiment to weigh against the recent price action
        news_score = self.news_engine.get_sentiment("Bitcoin")
        print(f"📰 News Sentiment Sensor: {news_score:.2f}")

        # 4. Master Score (80% AI / 20% News)
        df['ai_score'] = (df['ai_prob'] - 0.5) * 2
        df['master_score'] = (df['ai_score'] * 0.80) + (news_score * 0.20)

        # 5. Tuned Strategy Logic (Anti-Overtrading)
        # Entry Threshold: 0.65 | Exit Threshold: 0.30
        df['signal'] = 0
        current_pos = 0
        
        for i in range(len(df)):
            score = df.iloc[i]['master_score']
            
            # Logic: Only enter if confidence is VERY high
            if current_pos == 0:
                if score > 0.65:
                    current_pos = 1
                elif score < -0.65:
                    current_pos = -1
            
            # Logic: Exit if confidence drops below the "Weak" threshold
            else:
                if current_pos == 1 and score < 0.30:
                    current_pos = 0
                elif current_pos == -1 and score > -0.30:
                    current_pos = 0
            
            df.at[df.index[i], 'signal'] = current_pos

        # 6. Calculate Results
        df['market_return'] = df['close'].pct_change()
        # We shift signal by 1 because we enter at the NEXT hour's price
        df['strategy_return'] = df['signal'].shift(1) * df['market_return']
        
        # Calculate Fees only when signal changes (Actual Trades)
        df['trade_occurred'] = df['signal'].diff().fillna(0).abs()
        df['strategy_return'] -= (df['trade_occurred'] * self.fee)

        df['cum_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()
        
        self.print_summary(df)

    def print_summary(self, df):
        final_return = (df['cum_strategy'].iloc[-1] - 1) * 100
        max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min() * 100
        total_trades = df['trade_occurred'].sum()
        
        print("\n--- 🏆 MASTER STRATEGY SUMMARY (TUNED) ---")
        print(f"Total Return: {final_return:.2f}%")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Total Trades: {int(total_trades)}")
        print(f"Avg Trades/Day: {total_trades/30:.1f}")
        
        if final_return > 0:
            print("Status: ✅ PROFITABLE (Ready for Dry Run)")
        else:
            print("Status: ⚠️ UNDERPERFORMING (Needs further tuning)")
        print("------------------------------------------")

if __name__ == "__main__":
    tester = MasterBacktester()
    tester.run_test(days=30)