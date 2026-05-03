def determine_signal(self, ai_prob, whale_score, news_score):
    """
    ai_prob: 0 to 1 (from XGBoost)
    whale_score: -1 to 1 (sentiment)
    news_score: -1 to 1 (sentiment)
    """
    # Calculate Master Score (-1 to 1)
    # Convert ai_prob (0.5 center) to -1 to 1 range
    ai_momentum = (ai_prob - 0.5) * 2 
    
    master_score = (
        (self.weights["ai"] * ai_momentum) +
        (self.weights["whale"] * whale_score) +
        (self.weights["news"] * news_score)
    )

    # Directional Logic
    if master_score > 0.7:
        return "🟢 STRONG BUY", master_score
    elif master_score < -0.7:
        return "🔴 STRONG SELL / SHORT", master_score
    else:
        return "⚪️ NEUTRAL / WAIT", master_score