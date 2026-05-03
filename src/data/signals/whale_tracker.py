import requests
import os
from dotenv import load_dotenv

load_dotenv()

class WhaleTracker:
    def __init__(self):
        # We keep this here so the 'Predictor' doesn't crash
        self.api_key = None

    def get_whale_sentiment(self, symbol="BTC"):
        """
        PLACEHOLDER: Returns 0.0 (Neutral).
        Ready to be connected to Whale-Alert.io API later.
        """
        # print("🐳 Whale Tracker: Running in 'Budget Mode' (Neutral)")
        return 0.0
            
        try:
            # Check last 1 hour of big moves
            url = f"{self.base_url}/transactions?api_key={self.api_key}&min_value=5000000"
            response = requests.get(url).json()
            
            score = 0
            if response.get('count', 0) > 0:
                for tx in response['transactions']:
                    if tx['symbol'].upper() == symbol:
                        # Logic: Inflow to Exchange = Sell Pressure (-1)
                        if tx.get('to_hash') and "exchange" in tx['to_hash']:
                            score -= 0.5
                        # Logic: Outflow to Wallet = Buy Pressure (+1)
                        elif tx.get('from_hash') and "exchange" in tx['from_hash']:
                            score += 0.5
            
            # Clip score between -1 and 1
            return max(min(score, 1.0), -1.0)
        except Exception as e:
            print(f"Whale API Error: {e}")
            return 0.0