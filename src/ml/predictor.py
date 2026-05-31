# src/ml/predictor.py
# Patched for the meta-labeling trainer: loads the primary as a raw Booster,
# loads the meta model + gate from the sidecar, and returns a fire/no-fire signal.

import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, cast

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent.parent
model_store = root_dir / "src" / "ml" / "model_store"


def map_timeframe_to_ccxt(tf: str) -> str:
    tf = tf.lower().strip()
    mapping = {
        '1min': '1m', '1m': '1m', '5min': '5m', '5m': '5m',
        '15min': '15m', '15m': '15m', '30min': '30m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d',
    }
    if tf in mapping:
        return mapping[tf]
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

        self.model: Optional[xgb.Booster] = None          # primary (Booster)
        self.meta_model: Optional[xgb.Booster] = None      # meta gate (Booster)
        self.meta: Dict[str, Any] = {}                     # sidecar contents
        self._token_params: Optional[Dict[str, Any]] = None  # optimizer output
        self.load_model()
        self._token_params = self._load_token_params()

    # -------------------------------------------------------------
    # Model loading (Booster format + sidecar)
    # -------------------------------------------------------------
    def load_model(self):
        base = self.symbol.replace('/', '_')
        primary_path = model_store / f"{base}_model.json"
        sidecar_path = model_store / f"{base}_meta.json"

        if not primary_path.exists():
            logger.warning(f"Model not found: {primary_path}")
            return

        # Load primary as a raw Booster (the trainer saves xgb.train output).
        self.model = xgb.Booster()
        self.model.load_model(str(primary_path))
        logger.info(f"Primary model loaded: {primary_path.name}")

        if sidecar_path.exists():
            try:
                self.meta = json.loads(sidecar_path.read_text())
            except Exception as e:
                logger.warning(f"Could not read sidecar {sidecar_path.name}: {e}")
                self.meta = {}

        meta_file = self.meta.get("meta_model_file")
        if meta_file:
            meta_path = model_store / meta_file
            if meta_path.exists():
                self.meta_model = xgb.Booster()
                self.meta_model.load_model(str(meta_path))
                logger.info(f"Meta model loaded: {meta_path.name}")

    def _load_token_params(self) -> Optional[Dict[str, Any]]:
        """Load per-token optimizer output from data/token_params/ if present."""
        path = (root_dir / "data" / "token_params"
                / f"{self.symbol.replace('/', '_')}_params.json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _detect_regime(self, df: pd.DataFrame) -> Optional[str]:
        """
        Classify the current bar into one of 9 regime buckets using the
        percentile boundaries stored in the token params JSON.

        Returns a string like 'high_low' or None if params are not available.
        """
        if self._token_params is None:
            return None
        bounds = self._token_params.get("regime_boundaries")
        if not bounds:
            return None
        try:
            vol_avg   = float(df["volume"].iloc[-24:].mean()) if "volume" in df.columns else None
            atr_val   = float(df["_atr"].iloc[-1])            if "_atr"   in df.columns else None
            close_val = float(df["close"].iloc[-1])            if "close"  in df.columns else None

            if vol_avg is None or atr_val is None or not close_val:
                return None

            atr_pct = atr_val / close_val

            def _tier(val: float, p33: float, p67: float) -> str:
                if val <= p33:
                    return "low"
                if val <= p67:
                    return "med"
                return "high"

            v = _tier(vol_avg, bounds["vol_p33"],     bounds["vol_p67"])
            a = _tier(atr_pct, bounds["atr_pct_p33"], bounds["atr_pct_p67"])
            return f"{v}_{a}"
        except Exception:
            return None

    # -------------------------------------------------------------
    # Data fetch (unchanged logic)
    # -------------------------------------------------------------
    def fetch_live_data(self, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        ccxt_tf = map_timeframe_to_ccxt(timeframe)
        all_bars, remaining, chunk_size = [], limit, 1000
        if 'm' in ccxt_tf:
            ms_per_candle = int(ccxt_tf.replace('m', '')) * 60 * 1000
        elif 'h' in ccxt_tf:
            ms_per_candle = int(ccxt_tf.replace('h', '')) * 60 * 60 * 1000
        elif 'd' in ccxt_tf:
            ms_per_candle = int(ccxt_tf.replace('d', '')) * 24 * 60 * 60 * 1000
        else:
            ms_per_candle = 60 * 60 * 1000

        since = self.exchange.milliseconds() - (limit * ms_per_candle)
        while remaining > 0:
            try:
                bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=ccxt_tf,
                                                 since=since, limit=min(chunk_size, remaining))
                if not bars:
                    break
                all_bars.extend(bars)
                remaining -= len(bars)
                since = bars[-1][0] + 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Pagination error for {self.symbol}: {e}")
                break

        if not all_bars:
            return None
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        if len(df) > limit:
            df = df.iloc[-limit:]
        return df

    def fetch_btc_data(self, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        try:
            return Predictor('BTC/USDT').fetch_live_data(timeframe=timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")
            return None

    @staticmethod
    def load_news_data(news_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        if news_path is None:
            news_path = root_dir / "data" / "news_data.json"
        if not news_path.exists():
            return None
        try:
            df_news = pd.DataFrame(json.loads(news_path.read_text()))
            df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])
            df_news = df_news.sort_values('timestamp')
            if 'sentiment' not in df_news.columns:
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    a = SentimentIntensityAnalyzer()
                    df_news['sentiment'] = df_news['headline'].apply(lambda x: a.polarity_scores(x)['compound'])
                except ImportError:
                    df_news['sentiment'] = 0.0
            return df_news
        except Exception as e:
            logger.error(f"Failed to load news: {e}")
            return None

    # Symbols confirmed to have no perpetual futures market — skip without API call.
    _NO_PERP_SYMBOLS: set = set()

    _NO_PERP_PHRASES = (
        'does not have market symbol',
        'invalid symbol',
        'symbol not found',
        'no data',
        'does not exist',
        'not support',
        'market symbol',
        'no market',
    )

    def _fetch_futures_data(self, df: pd.DataFrame):
        """Fetch funding rate and OI from Binance perpetual futures. Returns (funding_df, oi_df)."""
        if self.symbol in Predictor._NO_PERP_SYMBOLS:
            return None, None
        try:
            import ccxt as _ccxt
            ex = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 10000})  # type: ignore[arg-type]
            ts = pd.to_datetime(df['timestamp'])
            since_ms = int(ts.iloc[0].timestamp() * 1000)
            n = len(df)
            futures_sym = self.symbol.replace('/USDT', '/USDT:USDT')

            def _is_no_perp(err: Exception) -> bool:
                msg = str(err).lower()
                return any(p in msg for p in Predictor._NO_PERP_PHRASES)

            # Fetch Funding Rate (paginated)
            all_fr = []
            fr_target = (n // 8) + 50
            current_since = since_ms

            while len(all_fr) < fr_target:
                try:
                    chunk = ex.fetch_funding_rate_history(futures_sym, since=current_since, limit=1000)
                    if not chunk:
                        break
                    all_fr.extend(chunk)
                    last_ts = int(chunk[-1].get('timestamp', 0))
                    if last_ts <= current_since or len(chunk) < 1000:
                        break
                    current_since = last_ts + 1
                    time.sleep(0.3)
                except Exception as e:
                    if _is_no_perp(e):
                        Predictor._NO_PERP_SYMBOLS.add(self.symbol)
                        return None, None
                    logger.warning(f"Funding rate fetch stopped: {e}")
                    break

            funding_df = None
            if all_fr:
                funding_df = (pd.DataFrame(all_fr)[['timestamp', 'fundingRate']]
                              .rename(columns={'fundingRate': 'funding_rate'}))
                funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                funding_df = funding_df.drop_duplicates('timestamp').sort_values('timestamp')

            # Fetch Open Interest (paginated; Binance caps OI history to ~30 days)
            all_oi = []
            safe_oi_since = int(time.time() * 1000) - (29 * 24 * 60 * 60 * 1000)
            current_since = max(since_ms, safe_oi_since)
            oi_target = min(n, 29 * 24)  # cap to what Binance actually serves

            while len(all_oi) < oi_target:
                try:
                    chunk = ex.fetch_open_interest_history(futures_sym, '1h', since=current_since, limit=500)
                    if not chunk:
                        break
                    all_oi.extend(chunk)
                    last_ts = int(chunk[-1].get('timestamp', 0))
                    if last_ts <= current_since or len(chunk) < 500:
                        break
                    current_since = last_ts + 1
                    time.sleep(0.3)
                except Exception as e:
                    if _is_no_perp(e):
                        Predictor._NO_PERP_SYMBOLS.add(self.symbol)
                        break
                    logger.warning(f"OI fetch stopped: {e}")
                    break

            oi_df = None
            if all_oi:
                raw = pd.DataFrame(all_oi)
                oi_col = next((c for c in ('openInterestAmount', 'openInterest') if c in raw.columns), None)
                if oi_col:
                    oi_df = raw[['timestamp', oi_col]].rename(columns={oi_col: 'open_interest'})
                    oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                    oi_df = oi_df.drop_duplicates('timestamp').sort_values('timestamp')

            return funding_df, oi_df
        except Exception as e:
            logger.error(f"Failed to fetch futures data for {self.symbol}: {e}")
            return None, None

    @staticmethod
    def _fetch_fear_greed(days: int = 700) -> Optional[pd.DataFrame]:
        """Fetch Fear & Greed Index from alternative.me (free, no key required)."""
        import urllib.request, json as _json
        try:
            url = f"https://api.alternative.me/fng/?limit={days}&date_format=world"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read().decode())
            records = data.get('data', [])
            if not records:
                return None
            fg = pd.DataFrame(records)[['value', 'timestamp']]
            fg.columns = ['fear_greed_value', 'timestamp']
            fg['fear_greed_value'] = pd.to_numeric(fg['fear_greed_value'], errors='coerce')
            fg['timestamp'] = pd.to_datetime(fg['timestamp'], format='%d-%m-%Y')
            return fg.sort_values('timestamp').reset_index(drop=True)
        except Exception:
            return None

    def get_features_with_context(self, hours: int = 5000) -> Optional[pd.DataFrame]:
        df = self.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            return None
        btc_df = self.fetch_btc_data(timeframe='1h', limit=hours)
        news_df = self.load_news_data()
        try:
            df_1d = self.fetch_live_data(timeframe='1d', limit=max(300, int(hours / 24) + 10))
        except Exception:
            df_1d = None
        funding_df, oi_df = self._fetch_futures_data(df)
        fg_df = self._fetch_fear_greed()
        from src.ml.feature_engine import prepare_features
        return prepare_features(df, btc_df=btc_df, news_df=news_df, df_1d=df_1d, df_1w=None,
                                funding_df=funding_df, oi_df=oi_df, fg_df=fg_df)

    # -------------------------------------------------------------
    # Prediction with the meta gate
    # -------------------------------------------------------------
    def _align(self, df_features: pd.DataFrame, cols) -> pd.DataFrame:
        X = df_features.drop(columns=['timestamp', 'target'], errors='ignore')
        return X.reindex(columns=list(cols), fill_value=0)

    @staticmethod
    def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
        if T <= 0 or abs(T - 1.0) < 1e-6:
            return probs
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """3-class primary probabilities (n, 3), temperature-calibrated."""
        if self.model is None:
            raise ValueError("Model not loaded. Train first.")
        feat_cols = self.meta.get("feature_cols")
        if feat_cols is None:
            feat_cols = self.model.feature_names
        X = self._align(df_features, feat_cols)
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        proba = self.model.predict(dm)
        if proba.ndim == 1:  # safety, shouldn't happen for multi:softprob
            proba = np.column_stack([1 - proba, np.zeros_like(proba), proba])
        T = float(self.meta.get("calibration_temperature", 1.0))
        return self._apply_temperature(proba, T)

    def predict_meta_batch(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Run primary + meta models on the full dataframe (batch, no lookahead).

        Returns
        -------
        proba     : ndarray (n, 3)  — primary [p_sell, p_hold, p_buy]
        meta_conf : ndarray (n,)    — meta model confidence per row
                    (falls back to max primary prob when meta model absent)
        """
        proba = self.predict_proba(df_features)          # (n, 3)
        if self.meta_model is None:
            return proba, proba.max(axis=1)

        mcols     = self.meta.get("meta_feature_cols")
        feat_cols = self.meta.get("feature_cols") or (
            self.model.feature_names if self.model else []
        )
        base = self._align(df_features, feat_cols).copy()

        if mcols and any(c.startswith("_p_") for c in mcols):
            base["_p_sell"]    = proba[:, 0]
            base["_p_hold"]    = proba[:, 1]
            base["_p_buy"]     = proba[:, 2]
            base["_p_max"]     = proba.max(axis=1)
            base["_p_dir_gap"] = np.abs(proba[:, 2] - proba[:, 0])

        if mcols:
            base = base.reindex(columns=mcols, fill_value=0)

        meta_conf = self.meta_model.predict(
            xgb.DMatrix(base, feature_names=list(base.columns))
        ).astype(float)
        return proba, meta_conf

    def predict_signal(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Return the gated trading decision for the LAST row of df_features.
        This is what the bot should emit. Honours the meta gate + threshold so
        production reproduces the trainer's operating point."""
        proba = self.predict_proba(df_features)
        last = proba[-1]
        side = 2 if last[2] >= last[0] else 0          # BUY vs SELL proposal
        side_name = "BUY" if side == 2 else "SELL"

        # meta confidence for the last bar (score last row only — no need for full history)
        if self.meta_model is not None:
            mcols = self.meta.get("meta_feature_cols")
            feat_cols = self.meta.get("feature_cols") or (
                self.model.feature_names if self.model else []
            )
            base = self._align(df_features.iloc[[-1]], feat_cols)
            if mcols and any(c.startswith('_p_') for c in mcols):
                # backward compat: old models that include primary probs as meta features
                base = base.copy()
                base['_p_sell'] = float(last[0])
                base['_p_hold'] = float(last[1])
                base['_p_buy'] = float(last[2])
                base['_p_max'] = float(last.max())
                base['_p_dir_gap'] = float(abs(last[2] - last[0]))
            if mcols:
                base = base.reindex(columns=mcols, fill_value=0)
            meta_conf = float(self.meta_model.predict(
                xgb.DMatrix(base, feature_names=list(base.columns)))[0])
        else:
            meta_conf = float(last.max())

        # Per-side thresholds: BUY and SELL are evaluated independently during
        # training. Use the side-specific gate so a token that only has SELL alpha
        # doesn't stay silent just because its combined precision is too low.
        if side == 2:  # BUY proposal
            thr       = float(self.meta.get("meta_threshold_buy",
                                             self.meta.get("meta_threshold", 0.6)))
            tradeable = bool(self.meta.get("tradeable_buy",
                                           self.meta.get("tradeable", True)))
        else:          # SELL proposal
            thr       = float(self.meta.get("meta_threshold_sell",
                                             self.meta.get("meta_threshold", 0.6)))
            tradeable = bool(self.meta.get("tradeable_sell",
                                           self.meta.get("tradeable", True)))

        # ── Regime-specific threshold override ───────────────────────────────
        # If threshold_optimizer.py has run, swap in the optimised threshold for
        # the current volume × volatility regime (only when that regime's result
        # passed the precision target, i.e. ok=True).
        regime = self._detect_regime(df_features)
        if regime and self._token_params:
            reg = self._token_params.get("regimes", {}).get(regime, {})
            if reg and not reg.get("skipped"):
                if side == 2 and reg.get("buy_ok") and "buy_threshold" in reg:
                    thr = float(reg["buy_threshold"])
                elif side == 0 and reg.get("sell_ok") and "sell_threshold" in reg:
                    thr = float(reg["sell_threshold"])

        fire = tradeable and (meta_conf >= thr)

        # ── S&R + trend alignment filters (mirrors training holdout logic) ───
        # These suppress weak signals (below top-25% confidence) that fight a
        # confirmed structure or macro trend. High-conviction signals pass through.
        if fire:
            last_row = df_features.iloc[-1]
            override_thr = float(self.meta.get("meta_override_confidence", 1.0))
            is_high_conviction = meta_conf >= override_thr

            if not is_high_conviction:
                # S&R filter: weak BUY at resistance or weak SELL at support
                at_res = bool(last_row.get('is_at_resistance', 0))
                at_sup = bool(last_row.get('is_at_support', 0))
                if (side == 2 and at_res) or (side == 0 and at_sup):
                    fire = False

            if fire and not is_high_conviction:
                # Trend filter: weak BUY against strong downtrend or weak SELL against uptrend
                trend_1d = float(last_row.get('macro_trend_1d', 0.0))
                if (side == 2 and trend_1d < -0.2) or (side == 0 and trend_1d > 0.2):
                    fire = False

        # Overall tradeable = either side is live (for dashboard indicator)
        either_tradeable = bool(
            self.meta.get("tradeable_buy", self.meta.get("tradeable", True)) or
            self.meta.get("tradeable_sell", self.meta.get("tradeable", True))
        )
        return {
            "symbol":         self.symbol,
            "fire":           bool(fire),
            "side":           side_name if fire else "FLAT",
            "tradeable":      either_tradeable,
            "tradeable_buy":  bool(self.meta.get("tradeable_buy",  either_tradeable)),
            "tradeable_sell": bool(self.meta.get("tradeable_sell", either_tradeable)),
            "meta_confidence": meta_conf,
            "threshold":       thr,
            "p_sell": float(last[0]), "p_hold": float(last[1]), "p_buy": float(last[2]),
            "expected_signal_precision": self.meta.get("dev_estimate", {}).get("precision"),
        }

    def predict_realtime(self) -> Dict[str, Any]:
        df = self.get_features_with_context(hours=350)
        if df is None or df.empty:
            return {"symbol": self.symbol, "fire": False, "side": "FLAT", "meta_confidence": 0.0}
        result = self.predict_signal(df)
        # Attach live price and ATR so the engine can set SL without an extra API call
        result['price'] = float(df['close'].iloc[-1])
        result['atr']   = (float(df['_atr'].iloc[-1])
                           if '_atr' in df.columns
                           else float(df['close'].iloc[-1]) * 0.015)
        result['atr_multiplier'] = float(self.meta.get('atr_multiplier', 1.5))
        return result