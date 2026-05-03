import os
import pandas as pd
from src.data.market_data import MarketData
from src.ml.feature_engine import prepare_features
from src.ml.trainer import ModelTrainer

def train_now():
    # 1. Ensure the model_store folder exists
    os.makedirs('src/ml/model_store', exist_ok=True)
    
    symbol = "BTC/USDT"
    
    # 2. Fetch 1 year of data
    data_fetcher = MarketData()
    raw_df = data_fetcher.fetch_historical_data(symbol, timeframe='1h', days=365)
    
    # 3. Process indicators (EMA, RSI, etc.)
    processed_df = prepare_features(raw_df)
    
    # 4. Train and Save the model
    trainer = ModelTrainer(symbol)
    trainer.train(processed_df)
    
    print(f"\n✅ Success! The 'Brain' has been saved to src/ml/model_store/")

if __name__ == "__main__":
    train_now()