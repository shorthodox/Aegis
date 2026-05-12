# src/ml/predictor.py
# Refactored for pandas 2.2.0+, Binance timeframe compatibility, and pagination.

import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, cast

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Path fix: use model_store instead of backtest_models
# -------------------------------------------------------------------
root_dir = Path(__file__).parent.parent.parent
model_store = root_dir / "src" / "ml" / "model_store"

# -------------------------------------------------------------------
# TIMEFRAME MAPPING (for Binance compatibility)
# -------------------------------------------------------------------
def map_timeframe_to_ccxt(tf: str) -> str:
    """
    Convert any user-friendly timeframe to Binance/CCXT format.
    Supports:
        '1min', '1m' -> '1m'
        '5min', '5m' -> '5m'
        '15min', '15m' -> '15m'
        '30min', '30m' -> '30m'
        '1h' -> '1h'
        '4h' -> '4h'
        '1d' -> '1d'
    Defaults to '1h' and logs a warning.
    """
    tf = tf.lower().strip()
    mapping = {
        '1min': '1m', '1m': '1m',
        '5min': '5m', '5m': '5m',
        '15min': '15m', '15m': '15m',
        '30min': '30m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d'
    }
    if tf in mapping:
        return mapping[tf]
    else:
        logger.warning(f"Unsupported timeframe '{tf}'. Defaulting to '1h'.")
        return '1h'

class Predictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.exchange = ccxt.binance()
        self.exchange.enableRateLimit = True
        options = cast(Dict[str, Any], self.exchange.options or {})
        options['defaultType'] = 'spot'
        self.exchange.options = options
        self.model = None
        self.load_model()

    def load_model(self):
        """Load XGBoost model from JSON or BIN file in model_store."""
        model_filename_json = f"{self.symbol.replace('/', '_')}_model.json"
        model_filename_bin = f"{self.symbol.replace('/', '_')}_model.bin"
        model_path_json = model_store / model_filename_json
        model_path_bin = model_store / model_filename_bin

        if model_path_json.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path_json))
            logger.info(f"✅ Model loaded: {model_path_json}")
        elif model_path_bin.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path_bin))
            logger.info(f"✅ Model loaded: {model_path_bin}")
        else:
            logger.warning(f"⚠️ Model not found: {model_path_json} or {model_path_bin}")

    def fetch_live_data(self, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with pagination (Binance max 1000 per request).
        Uses a loop to accumulate up to 'limit' candles.
        Timeframe is automatically mapped to CCXT format.
        """
        ccxt_tf = map_timeframe_to_ccxt(timeframe)
        logger.info(f"Fetching {self.symbol} with CCXT timeframe: {ccxt_tf}, target limit: {limit}")

        all_bars = []
        remaining = limit
        chunk_size = 1000
        # Calculate initial 'since' timestamp: go back 'limit' candles in milliseconds.
        # Since we don't know the exact candle duration in ms, we approximate using the timeframe.
        # For safety, we start from 'limit' * 60*60*1000 = limit hours ago if timeframe is '1h'.
        # For sub‑hourly, we use a conservative estimate.
        if 'm' in ccxt_tf:
            minutes = int(ccxt_tf.replace('m', ''))
            ms_per_candle = minutes * 60 * 1000
        elif 'h' in ccxt_tf:
            hours = int(ccxt_tf.replace('h', ''))
            ms_per_candle = hours * 60 * 60 * 1000
        elif 'd' in ccxt_tf:
            days = int(ccxt_tf.replace('d', ''))
            ms_per_candle = days * 24 * 60 * 60 * 1000
        else:
            ms_per_candle = 60 * 60 * 1000  # fallback 1 hour

        since = self.exchange.milliseconds() - (limit * ms_per_candle)

        while remaining > 0:
            try:
                fetch_limit = min(chunk_size, remaining)
                logger.debug(f"Fetching up to {fetch_limit} candles from {since}")
                bars = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=ccxt_tf,
                    since=since,
                    limit=fetch_limit
                )
                if not bars:
                    logger.warning(f"No more data returned for {self.symbol} at {since}. Stopping pagination.")
                    break

                all_bars.extend(bars)
                remaining -= len(bars)
                # Update 'since' to the timestamp of the last candle + 1 ms to avoid overlap
                if bars:
                    last_ts = bars[-1][0]
                    since = last_ts + 1  # add 1 millisecond to avoid fetching the same candle again
                else:
                    break

                # Polite delay to respect rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Pagination error for {self.symbol}: {e}")
                break

        if not all_bars:
            logger.warning(f"No data fetched for {self.symbol}")
            return None

        # Create DataFrame
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Remove duplicate timestamps (possible at pagination boundaries)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        # Trim to exact limit if we overshot
        if len(df) > limit:
            df = df.iloc[-limit:]

        logger.info(f"✅ Fetched {len(df)} candles for {self.symbol}")
        return df

    def fetch_btc_data(self, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        """Fetch BTC/USDT data for market context using pagination."""
        ccxt_tf = map_timeframe_to_ccxt(timeframe)
        try:
            # Reuse the pagination logic by temporarily swapping symbol
            temp_predictor = Predictor('BTC/USDT')
            return temp_predictor.fetch_live_data(timeframe=timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")
            return None

    @staticmethod
    def load_news_data(news_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Load news_data.json and return a DataFrame with timestamps and sentiment scores."""
        if news_path is None:
            news_path = Path("data/news_data.json")
        if not news_path.exists():
            logger.warning("News file not found. Sentiment features will be zero.")
            return None
        try:
            with open(news_path, "r") as f:
                data = json.load(f)
            df_news = pd.DataFrame(data)
            df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])
            df_news = df_news.sort_values('timestamp')
            if 'sentiment' not in df_news.columns:
                # Try VADER
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    df_news['sentiment'] = df_news['headline'].apply(
                        lambda x: analyzer.polarity_scores(x)['compound']
                    )
                except ImportError:
                    logger.warning("vaderSentiment not installed; defaulting news sentiment to 0.")
                    df_news['sentiment'] = 0.0
            return df_news
        except Exception as e:
            logger.error(f"Failed to load news: {e}")
            return None

    def get_features_with_context(self, hours: int = 5000) -> Optional[pd.DataFrame]:
        """
        Fetch symbol data, BTC data, and news, then run feature engineering.
        Returns a DataFrame ready for prediction or training.
        """
        df = self.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            return None

        btc_df = self.fetch_btc_data(timeframe='1h', limit=hours)
        news_df = self.load_news_data()

        from src.ml.feature_engine import prepare_features
        df_features = prepare_features(df, btc_df=btc_df, news_df=news_df)
        return df_features

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for the given feature DataFrame."""
        if self.model is None:
            raise ValueError("Model not loaded. Train first.")
        X = df_features.drop(columns=['timestamp', 'target'], errors='ignore')
        # Ensure columns match the model's expected feature names (XGBoost requires exact match)
        try:
            booster = self.model.get_booster()
            expected = booster.feature_names
        except Exception:
            expected = None

        if expected is not None:
            X = X.reindex(columns=list(expected), fill_value=0)

        y_proba = self.model.predict_proba(X)
        if y_proba.ndim == 1:
            return y_proba
        if y_proba.shape[1] == 2:
            return y_proba[:, 1]
        return y_proba

    def predict_realtime(self) -> float:
        """Convenience method: fetch latest data and return current signal probability."""
        df = self.get_features_with_context(hours=500)  # last 500 hours
        if df is None or df.empty:
            return 0.5
        last_row = df.iloc[[-1]]
        prediction = self.predict(last_row)
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 2 and prediction.shape[1] >= 3:
                return float(prediction[0, 2])
            return float(prediction[0])
        return float(prediction)