import numpy as np

def extract_prediction_telemetry(model, feature_row, feature_names):
    """
    Extracts raw probabilities and SHAP feature contributions for a specific prediction.
    
    Args:
        model: Trained XGBoost model (must support predict_proba).
        feature_row: A 2D numpy array or pandas DataFrame of shape (1, n_features).
        feature_names: List of strings corresponding to the columns in feature_row.
        
    Returns:
        dict: JSON-serializable dictionary with probabilities and SHAP data.
    """
    try:
        import shap
    except ImportError:
        return {
            "probabilities": {"SHORT": 33.3, "HOLD": 33.4, "LONG": 33.3},
            "shap_contributions": [],
            "error": "SHAP library not installed"
        }
        
    try:
        # Extract Probabilities
        raw_probs = model.predict_proba(feature_row)[0]
        # Assuming order is [SHORT, HOLD, LONG]
        short_prob = float(raw_probs[0] * 100)
        hold_prob = float(raw_probs[1] * 100) if len(raw_probs) > 1 else 0.0
        long_prob = float(raw_probs[2] * 100) if len(raw_probs) > 2 else 0.0
        
        # Normalize just in case
        total = short_prob + hold_prob + long_prob
        if total > 0:
            short_prob = (short_prob / total) * 100
            hold_prob = (hold_prob / total) * 100
            long_prob = (long_prob / total) * 100
        
        probabilities = {
            "SHORT": round(short_prob, 2),
            "HOLD": round(hold_prob, 2),
            "LONG": round(long_prob, 2)
        }
        
        # Extract SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_row)
        
        # In multiclass, shap_values is a list of arrays.
        # We'll focus on the LONG class (usually index 2) or aggregate if needed.
        # For a single prediction, it's (num_classes, 1, num_features) or list of (1, num_features).
        if isinstance(shap_values, list):
            target_shap = shap_values[-1][0]  # Last class (e.g. LONG)
        elif len(shap_values.shape) == 3:
            target_shap = shap_values[:, 0, :][-1] # Last class
        else:
            target_shap = shap_values[0] # Binary or generic
            
        # Map to feature names
        feature_impacts = []
        for i, name in enumerate(feature_names):
            if i < len(target_shap):
                val = float(target_shap[i])
                if abs(val) > 0.0001:  # filter noise
                    feature_impacts.append({
                        "feature": str(name),
                        "impact": val
                    })
                    
        # Sort by absolute impact descending
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_features = feature_impacts[:3]
        
        return {
            "probabilities": probabilities,
            "shap_contributions": top_features
        }
        
    except Exception as e:
        print(f"Error in extract_prediction_telemetry: {e}")
        # Safe fallback
        return {
            "probabilities": {"SHORT": 33.3, "HOLD": 33.4, "LONG": 33.3},
            "shap_contributions": [],
            "error": str(e)
        }
