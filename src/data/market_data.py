import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

class MarketData:
    def __init__(self):
        self.exchange = ccxt.binance()

    def fetch_historical_data(self, symbol, timeframe='1h', days=365):
        """Fetches 1 year of data for training."""
        print(f"Fetching {days} days of historical data for {symbol}...")
        
        since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        if since is None:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            symbol_data = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            if not symbol_data:
                break
            since = symbol_data[-1][0] + 1
            all_ohlcv += symbol_data
            time.sleep(self.exchange.rateLimit / 1000) # Respect rate limits

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_live_price(self, symbol):
        """Fetches the current price for real-time signal generation."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']