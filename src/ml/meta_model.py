from typing import Optional, Any
from pathlib import Path
import pickle
import numpy as np

def train_logistic_meta(X, y_binary, save_path: Optional[str] = None, C: float = 1.0):
    """Train a lightweight logistic meta-model (sklearn) and optionally save it.

    X: DataFrame or 2D array
    y_binary: 1D array of 0/1 labels indicating profitable trades
    Returns trained model object.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        raise RuntimeError('sklearn required for logistic meta model')
    lr = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
    lr.fit(X, y_binary)
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(lr, f)
    return lr


def load_meta_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, 'rb') as f:
        return pickle.load(f)


def predict_proba(model: Any, X):
    """Return probability of positive class (1).
    Works for sklearn LogisticRegression or any model with predict_proba.
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    # fallback: raw predict
    return model.predict(X)
