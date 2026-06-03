from src.ml.predictor import Predictor
p = Predictor('HBAR/USDT')
df = p.get_features_with_context(hours=350)
if df is not None:
    res = p.predict_signal(df)
    last_row = df.iloc[-1]
    print("Raw total_confluence:", last_row.get('total_confluence'))
    print("Raw smart_money_confluence:", last_row.get('smart_money_confluence'))
    print("VETO_HARD (0.25) -> -0.25")
    print("Veto triggered by total?", last_row.get('total_confluence') < -0.25)
    
    ctx = p._extract_market_context(df, df['close'].iloc[-1], df['_atr'].iloc[-1] if '_atr' in df else 0.015, 1.5)
    print("UI Confluence:", ctx.get('confluence'))
