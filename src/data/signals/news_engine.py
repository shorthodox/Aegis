import requests
import os
try:
    from vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

class NewsEngine:
    # 🚨 DANGER KEYWORDS: If these appear, we kill all 'Buy' signals immediately.
    DANGER_KEYWORDS = [
        "crash", "hack", "scam", "dump", "bearish", "liquidated", 
        "sec lawsuit", "hacked", "bankrupt", "suspends withdrawals"
    ]

    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.analyzer = SentimentIntensityAnalyzer()
        self.base_url = "https://newsapi.org/v2/everything"

    def get_sentiment(self, query="Bitcoin"):
        """
        Fetches headlines and returns a score from -1.0 to 1.0.
        Triggers a -1.0 'Emergency Brake' if danger keywords are found.
        """
        if not self.api_key:
            return 0.0

        try:
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20, 
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params).json()
            articles = response.get('articles', [])
            
            if not articles:
                return 0.0

            total_vader_score = 0
            danger_detected = False

            for article in articles:
                # 1. Prepare text
                title = str(article.get('title', ""))
                desc = str(article.get('description', ""))
                full_text = f"{title} {desc}".lower()

                # 2. THE EMERGENCY BRAKE (Danger Check)
                if any(word in full_text for word in self.DANGER_KEYWORDS):
                    danger_detected = True
                    # We found a major red flag, no need to keep checking other articles
                    break

                # 3. NORMAL VIBE CHECK (VADER)
                sentiment = self.analyzer.polarity_scores(full_text)
                total_vader_score += sentiment['compound']

            # --- FINAL LOGIC ---
            if danger_detected:
                print(f"🛑 EMERGENCY BRAKE: Danger keyword detected in '{query}' news!")
                return -1.0  # Force a strong bearish signal

            # Return the average sentiment of the headlines
            avg_score = total_vader_score / len(articles)
            return avg_score

        except Exception as e:
            print(f"⚠️ NewsAPI Error: {e}")
            return 0.0

if __name__ == "__main__":
    # Quick Test
    engine = NewsEngine()
    score = engine.get_sentiment("Bitcoin")
    print(f"Current Market Sentiment Score: {score:.2f}")