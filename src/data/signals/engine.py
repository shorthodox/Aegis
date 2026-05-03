import time
import logging
from src.data.signals.whale_tracker import WhaleTracker
from src.ml.predictor import Predictor
from src.data.signals.news_engine import NewsEngine
from src.data.signals.whale_tracker import WhaleTracker # Our placeholder

class MasterSignalEngine:
    def __init__(self):
        self.predictor = Predictor("BTC/USDT")
        self.news = NewsEngine()
        self.whales = WhaleTracker()
        
    def generate_signal(self):
        # 1. Get AI Probability (0 to 1)
        ai_prob = self.predictor.get_probability()
        ai_score = (ai_prob - 0.5) * 2  # Scale to -1 to 1
        
        # 2. Get News Sentiment (-1 to 1)
        news_score = self.news.get_sentiment("Bitcoin")
        
        # 3. Get Whale Sentiment (currently 0.0)
        whale_score = self.whales.get_whale_sentiment("BTC")
        
        # 4. Final Weighted Calculation
        # Weights: AI (80%), News (20%), Whales (0% for now)
        master_score = (ai_score * 0.80) + (news_score * 0.20) + (whale_score * 0.0)
        
        return {
            "score": master_score,
            "ai": ai_prob,
            "news": news_score,
            "timestamp": time.time()
        }