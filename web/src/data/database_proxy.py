import firebase_admin
from firebase_admin import credentials, db

# 1. Path to your service account key
# Suggestion: Put the .json file in your 'config/' folder
service_key_path = "config/serviceAccountKey.json" 

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(service_key_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://gatekeeper-sbs-default-rtdb.firebaseio.com'
        })
        print("✅ Firebase initialized successfully.")
    except Exception as e:
        print(f"❌ Firebase Init Error: {e}")

def push_signal_to_web(symbol, ai_score, news_score, master_score, signal):
    try:
        # Sanitize symbol (No '/' allowed in Firebase keys)
        clean_symbol = symbol.replace("/", "_")
        ref = db.reference(f'signals/{clean_symbol}')
        
        ref.set({
            'ai_score': float(ai_score),
            'news_score': float(news_score),
            'master_score': float(master_score),
            'signal': str(signal),
            'timestamp': {'.sv': 'timestamp'}
        })
        print(f"🚀 Signal Broadcasted: {symbol} -> {signal}")
        
    except Exception as e:
        print(f"⚠️ Web Sync Failed: {e}")