#!/usr/bin/env python3
# scripts/news_scraper.py
# Aegis-1 News Scraper – Real-time sentiment with VADER, deduplication, and pruning.
# Runs autonomously; designed to be called by live_engine.py or manually.

import os
import sys
import json
import time
import logging
import requests
import feedparser
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
# Paths (relative to project root)
root_dir = Path(__file__).parent.parent
DATA_DIR = root_dir / "data"
LOGS_DIR = root_dir / "logs"
NEWS_FILE = DATA_DIR / "news_data.json"
LOG_FILE = LOGS_DIR / "news_scraper.log"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Scraping settings
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_AGE_HOURS = 48  # keep only last 48 hours of news

# Fleet symbols for priority weighting
FLEET_SYMBOLS = [
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP',
    'HYPE', 'ASTER', 'SUI', 'TAO', 'RENDER',
    'ADA', 'AVAX', 'LINK', 'TRX', 'DOT',
    'NEAR', 'MATIC', 'LTC', 'BCH', 'SHIB',
    'TON', 'ICP', 'HBAR', 'APT', 'ARB',
    'OP', 'STX', 'FIL', 'AAVE', 'VET',
    'RNDR', 'INJ', 'TIA', 'SEI', 'KAS',
    'FET', 'AGIX', 'OCEAN', 'AKT', 'THETA',
    'GRT', 'LDO', 'PYTH', 'JUP', 'ONDO',
    'PEPE', 'DOGE', 'WIF', 'FLOKI', 'BONK',
    'WLFI', 'MNT', 'ENA', 'BGB', 'PI',
    'SKY', 'TRUMP', 'NIGHT'
]

# RSS feed sources (free, no API key)
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://decrypt.co/feed",
]

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Sentiment analyzer (VADER)
# -------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def compute_vader_sentiment(text: str) -> float:
    """Return VADER compound score (range -1 to 1)."""
    return analyzer.polarity_scores(text)['compound']

# -------------------------------------------------------------------
# Symbol weighting: boost importance if headline mentions a fleet symbol
# -------------------------------------------------------------------
def compute_importance(headline: str) -> float:
    """
    Compute importance factor (1.0 baseline).
    +0.5 for each fleet symbol mentioned (max 2.0).
    """
    headline_upper = headline.upper()
    mentioned = 0
    for sym in FLEET_SYMBOLS:
        if sym in headline_upper:
            mentioned += 1
    # Cap at 2.0 extra (max 3.0 total? but keep reasonable)
    importance = 1.0 + min(mentioned * 0.5, 1.0)
    return importance

# -------------------------------------------------------------------
# Fetch news from RSS feeds
# -------------------------------------------------------------------
def fetch_rss_news() -> List[Dict[str, Any]]:
    """Fetch news items from RSS feeds, return list of raw dicts."""
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Fetching RSS: {feed_url}")
            feed = feedparser.parse(feed_url)
            if feed.bozo:  # parsing error
                logger.warning(f"Feedparser error for {feed_url}: {feed.bozo_exception}")
            for entry in feed.entries[:20]:  # limit per feed
                # Extract timestamp
                published = entry.get('published_parsed', entry.get('updated_parsed', None))
                if published and isinstance(published, time.struct_time):
                    # feedparser returns a time.struct_time; convert to datetime
                    try:
                        dt = datetime(*published[:6])
                    except Exception:
                        # fallback to current time
                        dt = datetime.now()
                else:
                    dt = datetime.now()
                article = {
                    "headline": entry.title,
                    "link": entry.link,
                    "timestamp": dt,
                    "source": feed_url.split('/')[2]
                }
                articles.append(article)
            time.sleep(1)  # be polite
        except Exception as e:
            logger.error(f"Failed to fetch RSS {feed_url}: {e}")
    return articles

# -------------------------------------------------------------------
# (Optional) CryptoPanic API – if you have an API key, you can enable it.
# For now, we use RSS only. To enable, uncomment and set CRYPTOPANIC_API_KEY.
# -------------------------------------------------------------------
# CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")
# def fetch_cryptopanic_news() -> List[Dict]:
#     if not CRYPTOPANIC_API_KEY:
#         return []
#     url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&public=true"
#     try:
#         resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
#         resp.raise_for_status()
#         data = resp.json()
#         articles = []
#         for post in data.get('results', []):
#             articles.append({
#                 "headline": post['title'],
#                 "link": post['url'],
#                 "timestamp": datetime.strptime(post['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ"),
#                 "source": "cryptopanic",
#                 "importance": post.get('votes', {}).get('positive', 0) + post.get('votes', {}).get('negative', 0)  # votes as importance
#             })
#         return articles
#     except Exception as e:
#         logger.error(f"CryptoPanic API error: {e}")
#         return []

# -------------------------------------------------------------------
# Deduplication & pruning
# -------------------------------------------------------------------
def load_existing_news() -> List[Dict[str, Any]]:
    """Load existing news from JSON file, return list (empty if file missing)."""
    if not NEWS_FILE.exists():
        return []
    try:
        with open(NEWS_FILE, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load existing news: {e}")
        return []

def save_news(news_list: List[Dict[str, Any]]):
    """Save news list to JSON file (pretty print)."""
    try:
        with open(NEWS_FILE, 'w') as f:
            json.dump(news_list, f, indent=2)
        logger.info(f"Saved {len(news_list)} news items to {NEWS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save news: {e}")

def prune_old_news(news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only entries from last MAX_AGE_HOURS hours."""
    cutoff = datetime.now() - timedelta(hours=MAX_AGE_HOURS)
    pruned = [item for item in news_list if datetime.fromisoformat(item['timestamp']) > cutoff]
    if len(pruned) != len(news_list):
        logger.info(f"Pruned {len(news_list) - len(pruned)} old news items (>{MAX_AGE_HOURS}h)")
    return pruned

def deduplicate(news_list: List[Dict[str, Any]], existing_headlines: set) -> List[Dict[str, Any]]:
    """Return only items whose headline is not in existing_headlines."""
    unique = []
    for item in news_list:
        if item['headline'] not in existing_headlines:
            unique.append(item)
            existing_headlines.add(item['headline'])
    return unique

# -------------------------------------------------------------------
# Main scraping pipeline
# -------------------------------------------------------------------
def scrape_news() -> List[Dict[str, Any]]:
    """
    Fetch news from all sources, compute sentiment and importance,
    deduplicate against existing file, prune old entries, and save.
    Returns the final list of news items (including existing + new).
    """
    logger.info("Starting news scraping cycle...")
    
    # Load existing news
    existing_news = load_existing_news()
    existing_headlines = {item['headline'] for item in existing_news}
    
    # Fetch raw articles
    raw_articles = fetch_rss_news()
    # Optionally add CryptoPanic: raw_articles.extend(fetch_cryptopanic_news())
    
    if not raw_articles:
        logger.warning("No new articles fetched. Keeping existing data.")
        # Still prune existing
        pruned_existing = prune_old_news(existing_news)
        save_news(pruned_existing)
        return pruned_existing
    
    # Process new articles: compute sentiment, importance, timestamp format
    new_items = []
    for art in raw_articles:
        headline = art['headline']
        # Skip duplicates already in existing file
        if headline in existing_headlines:
            continue
        
        # Compute sentiment using VADER
        sentiment = compute_vader_sentiment(headline)
        # Compute importance based on symbol mentions
        importance = compute_importance(headline)
        # Weighted sentiment (for potential future use; we store both)
        weighted_sentiment = sentiment * importance
        
        # Format timestamp as ISO string
        ts = art['timestamp']
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = datetime.now().isoformat()
        
        new_items.append({
            "timestamp": ts_str,
            "headline": headline,
            "link": art.get('link', ''),
            "source": art.get('source', 'rss'),
            "sentiment": round(sentiment, 4),
            "importance": round(importance, 2),
            "weighted_sentiment": round(weighted_sentiment, 4)
        })
    
    if not new_items:
        logger.info("No new unique headlines found.")
        # Still prune existing
        pruned_existing = prune_old_news(existing_news)
        save_news(pruned_existing)
        return pruned_existing
    
    logger.info(f"Fetched {len(raw_articles)} raw articles, {len(new_items)} new unique items.")
    
    # Combine existing (already pruned) and new, then prune overall
    combined = existing_news + new_items
    pruned_combined = prune_old_news(combined)
    
    # Save final list
    save_news(pruned_combined)
    
    # Compute aggregate sentiment for logging
    if pruned_combined:
        avg_sentiment = sum(item['sentiment'] for item in pruned_combined[-50:]) / min(50, len(pruned_combined))
        logger.info(f"Scraping complete. Total stored: {len(pruned_combined)} items. Avg sentiment (last 50): {avg_sentiment:.4f}")
    else:
        logger.info("Scraping complete. No news stored.")
    
    return pruned_combined

# -------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------
def main():
    """Run scraper once."""
    try:
        scrape_news()
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
