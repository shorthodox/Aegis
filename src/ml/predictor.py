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
import threading
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
    _SHARED_SPOT_EX = None
    _SHARED_FUTURES_EX = None
    _CCXT_LOCK = threading.Lock()
    _NO_PERP_SYMBOLS: set = set()

    # OI updates once per hour, funding rate every 8h — caching for 55 min eliminates
    # 120+ serial Binance API calls per scan cycle (60 symbols × 2 calls each) that
    # previously stalled signal generation for the entire 5-minute window.
    _FUTURES_CACHE: Dict[str, Any] = {}
    _FUTURES_CACHE_TTL: int = 3300  # 55 minutes

    # Fear & Greed is daily data — one HTTP call shared across all 60 symbols per cycle
    _FG_CACHE: Dict[str, Any] = {}   # keys: 'df', 'fetched_at'
    _FG_CACHE_TTL: int = 21600       # 6 hours

    def __init__(self, symbol: str):
        self.symbol = symbol
        
        if Predictor._SHARED_SPOT_EX is None:
            Predictor._SHARED_SPOT_EX = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})
            options = cast(Dict[str, Any], Predictor._SHARED_SPOT_EX.options or {})
            options['defaultType'] = 'spot'
            Predictor._SHARED_SPOT_EX.options = options
            
        self.exchange = Predictor._SHARED_SPOT_EX

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
        Classify the current bar into one of 27 regime buckets
        (3 vol × 3 vola × 3 trend) using the percentile boundaries stored
        in the sidecar _meta.json (embedded by retrain_model.py) or, as a
        fallback, the standalone optimizer JSON.

        Returns a string like 'high_low_up' or None if unavailable.
        """
        # Prefer sidecar (single file, always in sync with the trained model)
        bounds = (self.meta.get("regime_boundaries")
                  or (self._token_params or {}).get("regime_boundaries"))
        if not bounds:
            return None
        try:
            vol_avg   = float(df["volume"].iloc[-24:].mean()) if "volume" in df.columns else None
            atr_val   = float(df["_atr"].iloc[-1])            if "_atr"   in df.columns else None
            close_val = float(df["close"].iloc[-1])            if "close"  in df.columns else None

            if vol_avg is None or atr_val is None or not close_val:
                return None

            atr_pct = atr_val / close_val

            # 24-bar momentum (same window used during optimisation)
            if "close" in df.columns and len(df) >= 25:
                momentum = float(df["close"].iloc[-1] / df["close"].iloc[-25] - 1)
            else:
                momentum = 0.0

            def _tier(val: float, p33: float, p67: float) -> str:
                if val <= p33:
                    return "low"
                if val <= p67:
                    return "med"
                return "high"

            def _trend_tier(val: float, p33: float, p67: float) -> str:
                if val <= p33:
                    return "down"
                if val <= p67:
                    return "flat"
                return "up"

            v = _tier(vol_avg, bounds["vol_p33"],          bounds["vol_p67"])
            a = _tier(atr_pct, bounds["atr_pct_p33"],      bounds["atr_pct_p67"])
            t = _trend_tier(momentum,
                            bounds.get("momentum_p33", -0.02),
                            bounds.get("momentum_p67",  0.02))
            return f"{v}_{a}_{t}"
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
                with Predictor._CCXT_LOCK:
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

        # Return cached data when fresh — OI updates hourly, funding every 8h, so
        # re-fetching 300 bars from Binance every 5-min scan cycle is pure waste and
        # serialises 120+ API calls that stall signal generation for the whole fleet.
        _cached = Predictor._FUTURES_CACHE.get(self.symbol)
        if _cached and (time.time() - _cached['fetched_at']) < Predictor._FUTURES_CACHE_TTL:
            return _cached['funding_df'], _cached['oi_df']

        try:
            if Predictor._SHARED_FUTURES_EX is None:
                import ccxt as _ccxt
                Predictor._SHARED_FUTURES_EX = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 10000})
            ex = Predictor._SHARED_FUTURES_EX
            
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
                    with Predictor._CCXT_LOCK:
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
                    with Predictor._CCXT_LOCK:
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

            Predictor._FUTURES_CACHE[self.symbol] = {
                'funding_df': funding_df,
                'oi_df':      oi_df,
                'fetched_at': time.time(),
            }
            return funding_df, oi_df
        except Exception as e:
            logger.error(f"Failed to fetch futures data for {self.symbol}: {e}")
            return None, None

    @staticmethod
    def _fetch_fear_greed(days: int = 700) -> Optional[pd.DataFrame]:
        """Fetch Fear & Greed Index from alternative.me. Cached for 6 h — data is daily."""
        import urllib.request, json as _json
        _c = Predictor._FG_CACHE
        if _c.get('df') is not None and (time.time() - _c.get('fetched_at', 0)) < Predictor._FG_CACHE_TTL:
            return _c['df']
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
            result = fg.sort_values('timestamp').reset_index(drop=True)
            Predictor._FG_CACHE = {'df': result, 'fetched_at': time.time()}
            return result
        except Exception:
            return _c.get('df')  # return stale cache on error rather than None

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

    def predict_signal(self, df_features: pd.DataFrame,
                       risk_tier: str = "balanced") -> Dict[str, Any]:
        """Return the gated trading decision for the LAST row of df_features.

        risk_tier controls which quality bar to apply:
          "conservative" — per-side holdout confirmed at or above fee breakeven
          "balanced"     — combined holdout ≥ breakeven (default)
          "aggressive"   — dev target met + positive timeout-adjusted expectancy
        """
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

        # ── Risk tier gate ────────────────────────────────────────────────────
        # Check whether this token qualifies under the caller's chosen tier.
        # Falls back to legacy tradeable flags when risk_tier data is absent
        # (older sidecar files trained before the tier system was added).
        _tiers = self.meta.get("risk_tier", {})
        if _tiers:
            tier_ok = bool(_tiers.get(risk_tier, False))
        else:
            # Legacy fallback: map tier names to old tradeable flags
            if risk_tier == "conservative":
                tier_ok = bool(self.meta.get("tradeable_buy", False) or
                               self.meta.get("tradeable_sell", False))
            elif risk_tier == "aggressive":
                tier_ok = bool(self.meta.get("tradeable", False))
            else:  # balanced
                tier_ok = bool(self.meta.get("tradeable", False))

        if not tier_ok:
            return {
                "symbol": self.symbol, "fire": False, "side": "FLAT",
                "tradeable": False, "risk_tier": risk_tier,
                "meta_confidence": meta_conf, "threshold": 0.0,
                "p_sell": float(last[0]), "p_hold": float(last[1]), "p_buy": float(last[2]),
            }

        # Per-side thresholds. Aggressive mode uses the per-side best threshold
        # found during training (even when that side didn't meet the quality bar),
        # giving more signals at the cost of lower per-signal precision.
        if side == 2:  # BUY proposal
            if risk_tier == "aggressive":
                thr = float(self.meta.get("meta_threshold_buy_aggressive",
                            self.meta.get("meta_threshold_buy",
                            self.meta.get("meta_threshold", 0.6))))
            else:
                thr = float(self.meta.get("meta_threshold_buy",
                                          self.meta.get("meta_threshold", 0.6)))
        else:          # SELL proposal
            if risk_tier == "aggressive":
                thr = float(self.meta.get("meta_threshold_sell_aggressive",
                            self.meta.get("meta_threshold_sell",
                            self.meta.get("meta_threshold", 0.6))))
            else:
                thr = float(self.meta.get("meta_threshold_sell",
                                          self.meta.get("meta_threshold", 0.6)))

        fire = meta_conf >= thr

        # ── Regime-specific directional-probability filter ────────────────────
        # threshold_optimizer.py finds per-regime thresholds on p_buy / p_sell.
        # These are embedded in the sidecar by retrain_model.py (preferred) or
        # loaded from the standalone optimizer JSON (fallback).
        # Applied as an additional suppressor: if the production primary model's
        # directional probability is below the independently-derived floor for
        # this regime, the signal is suppressed regardless of meta_conf.
        if fire:
            regime_thresholds = (self.meta.get("regime_thresholds")
                                 or (self._token_params or {}).get("regimes", {}))
            if regime_thresholds:
                regime = self._detect_regime(df_features)
                if regime:
                    reg = regime_thresholds.get(regime, {})
                    if reg and not reg.get("skipped"):
                        p_sell, p_hold, p_buy = float(last[0]), float(last[1]), float(last[2])
                        if side == 2:
                            if (not reg.get("buy_ok")
                                    or p_buy < reg.get("buy_threshold", 0.0)
                                    or (p_buy - p_sell) < reg.get("buy_margin", 0.0)
                                    or p_hold > reg.get("buy_max_hold", 1.0)):
                                fire = False
                        elif side == 0:
                            if (not reg.get("sell_ok")
                                    or p_sell < reg.get("sell_threshold", 0.0)
                                    or (p_sell - p_buy) < reg.get("sell_margin", 0.0)
                                    or p_hold > reg.get("sell_max_hold", 1.0)):
                                fire = False

        # ── S&R + trend + technical confluence veto gate ──────────────────────
        # Mirrors the training holdout logic AND adds explicit confluence checks
        # so the engine never trades against the broader technical picture.
        if fire:
            last_row = df_features.iloc[-1]
            override_thr = float(self.meta.get("meta_override_confidence", 1.0))
            is_high_conviction = meta_conf >= override_thr

            def _fv(col: str, default: float = 0.0) -> float:
                v = last_row.get(col, default)
                try:
                    f = float(v)
                    return default if (f != f) else f
                except Exception:
                    return default

            # ── 1. S&R filter ─────────────────────────────────────────────────
            # Weak signals at the wrong structural level are noise, not edge.
            if not is_high_conviction:
                at_res = bool(last_row.get('is_at_resistance', 0))
                at_sup = bool(last_row.get('is_at_support', 0))
                if (side == 2 and at_res) or (side == 0 and at_sup):
                    fire = False

            # ── 2. Macro trend filter ─────────────────────────────────────────
            # A weak signal fighting the daily trend is typically a counter-trend
            # scalp with poor expectancy — suppressed unless high conviction.
            if fire and not is_high_conviction:
                trend_1d = _fv('macro_trend_1d')
                if (side == 2 and trend_1d < -0.2) or (side == 0 and trend_1d > 0.2):
                    fire = False

            # ── 3. Technical confluence veto gate ─────────────────────────────
            # total_confluence is the weighted aggregate of ALL indicator families
            # (momentum, trend, volume, bands, smart money, candles) scaled -1→+1.
            # Block signals where the aggregate technical picture strongly contradicts
            # the proposed direction, regardless of meta confidence.
            #
            # High-conviction signals (>= override threshold) bypass VETO_HARD but
            # still respect VETO_EXTREME — an 80% bearish technical consensus should
            # never be traded long, even if the model is highly confident.
            VETO_HARD    = 0.05   # strong contradiction → block all signals
            VETO_EXTREME = 0.35   # extreme contradiction → block even high-conviction

            if fire:
                total_conf  = _fv('total_confluence')
                smart_conf  = _fv('smart_money_confluence')
                candle_conf = _fv('candle_confluence')
                vol_zscore  = _fv('volume_zscore')

                if side == 2:   # ── BUY proposed ─────────────────────────────
                    # Veto if overall technicals are strongly bearish
                    if total_conf < -VETO_EXTREME:
                        fire = False
                    elif not is_high_conviction and total_conf < -VETO_HARD:
                        fire = False
                    # Veto if smart money (S/R zones, BOS, CHoCH) is strongly bearish
                    elif not is_high_conviction and smart_conf < -VETO_HARD:
                        fire = False
                    # Veto if a meaningful bearish candle just printed
                    elif candle_conf < -0.4:
                        fire = False
                    # Veto if volume is significantly below average — no conviction
                    elif vol_zscore < -1.2:
                        fire = False

                elif side == 0:  # ── SELL proposed ────────────────────────────
                    if total_conf > VETO_EXTREME:
                        fire = False
                    elif not is_high_conviction and total_conf > VETO_HARD:
                        fire = False
                    elif not is_high_conviction and smart_conf > VETO_HARD:
                        fire = False
                    elif candle_conf > 0.4:
                        fire = False
                    elif vol_zscore < -1.2:
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
            "risk_tier":  risk_tier,
            "risk_tiers": self.meta.get("risk_tier", {}),
        }

    def predict_realtime(self, risk_tier: str = "balanced") -> Dict[str, Any]:
        df = self.get_features_with_context(hours=350)
        if df is None or df.empty:
            return {"symbol": self.symbol, "fire": False, "side": "FLAT", "meta_confidence": 0.0}
        result = self.predict_signal(df, risk_tier=risk_tier)
        price = float(df['close'].iloc[-1])
        atr   = (float(df['_atr'].iloc[-1])
                 if '_atr' in df.columns
                 else price * 0.015)
        atr_mult = float(self.meta.get('atr_multiplier', 1.5))
        result['price']          = price
        result['atr']            = atr
        result['atr_multiplier'] = atr_mult
        # Rich market context for all trader types
        result.update(self._extract_market_context(df, price, atr, atr_mult))

        # ── HMM regime intelligence ───────────────────────────────────────────
        # Runs AFTER the XGBoost signal so it never blocks direction prediction.
        # The HMM state is attached as hmm_* fields to the result dict.
        # live_engine.py reads these fields to modulate thresholds and stops.
        # If no HMM model exists for this symbol, all hmm_* fields default to
        # neutral values and existing behavior is completely unchanged.
        try:
            from src.ml.hmm_regime import get_pool as _hmm_pool
            hmm_state = _hmm_pool().infer(self.symbol, df)
            result['hmm_regime']             = hmm_state.regime
            result['hmm_confidence']         = hmm_state.confidence
            result['hmm_state_id']           = hmm_state.state_id
            result['hmm_state_probs']        = hmm_state.state_probs
            result['hmm_transition_risk']    = hmm_state.transition_risk
            result['hmm_stability']          = hmm_state.regime_stability
            result['hmm_conf_adjustment']    = hmm_state.confidence_adjustment
            result['hmm_atr_mult']           = hmm_state.atr_multiplier
            result['hmm_position_scale']     = hmm_state.position_scale
            result['hmm_trade_allowed']      = hmm_state.trade_allowed
            result['hmm_available']          = hmm_state.hmm_available
            # Early warning: imminent regime transition
            engine   = _hmm_pool().get(self.symbol)
            result['hmm_transition_warning'] = engine.get_transition_warning()
        except Exception as _hmm_err:
            result['hmm_regime']          = 'UNKNOWN'
            result['hmm_available']       = False
            result['hmm_conf_adjustment'] = 0.0
            result['hmm_atr_mult']        = 1.0
            result['hmm_position_scale']  = 1.0
            result['hmm_trade_allowed']   = True

        return result

    @staticmethod
    def _extract_market_context(df: pd.DataFrame, price: float, atr: float, atr_mult: float) -> dict:
        """Extract rich market context from the last bar for multi-trader view."""
        from datetime import datetime, timezone as _tz
        if df is None or df.empty or price <= 0:
            return {}

        row = df.iloc[-1]

        def _f(col: str, default: float = 0.0) -> float:
            v = row.get(col, default)
            try:
                f = float(v)
                return default if (f != f) else f  # NaN guard
            except Exception:
                return default

        # ── Trend & bias ──────────────────────────────────────────────────────
        macro_1d   = _f('macro_trend_1d')
        macro_1w   = _f('macro_trend_1w')
        adx        = _f('adx_14', 20.0)

        # ── Soft percentile-rank confluence (display-only, ML model unchanged) ─
        # Binary sign-based scores saturate at 100% in strong trends and default
        # to 50% when a column is missing.  Percentile ranks against the
        # historical window (typically 350 bars) give meaningful gradients:
        # RSI-80 in a token that usually sits at 55 → 92nd pct → 9.2/10, not 10.
        # Missing columns return 0.5 (neutral) rather than corrupting the average.

        def _prank(col: str, higher_bullish: bool = True) -> float:
            """Rolling percentile rank of last row: 0=most bearish ever, 1=most bullish."""
            if col not in df.columns:
                return 0.5
            vals = df[col].dropna()
            if len(vals) < 10:
                return 0.5
            last = float(vals.iloc[-1])
            if last != last:  # NaN
                return 0.5
            rank = float((vals < last).sum()) / len(vals)
            return rank if higher_bullish else (1.0 - rank)

        def _cat(*entries) -> float:
            """Mean pct-rank for a category; entries = (col, higher_bullish)."""
            ranks = [_prank(c, b) for c, b in entries if c in df.columns]
            return float(np.mean(ranks)) if ranks else 0.5

        def _s10(v: float) -> float:
            """Map [0,1] pct rank to [0,10] display scale."""
            return round(v * 10.0, 1)

        # ── Category scores (0-1, later mapped to 0-10) ───────────────────────
        _trend_01 = _cat(
            ('dist_ema_50',       True),  ('dist_ema_200',      True),
            ('dist_ema_100',      True),  ('dist_hma20',        True),
            ('dist_kama',         True),  ('dist_rolling_vwap', True),
            ('dist_vwap',         True),  ('supertrend_dist',   True),
            ('sar_dist',          True),  ('linreg_slope_14',   True),
            ('structure_bias',    True),  ('macro_trend_1d',    True),
            ('macro_trend_1w',    True),
        )

        _mom_01 = _cat(
            ('rsi_14',       True),   ('rsi_7',        True),
            ('rsi_21',       True),   ('stoch_k',      True),
            ('williams_r',   False),  ('macd_hist',    True),
            ('cci_20',       True),   ('tsi',          True),
            ('cmo_14',       True),   ('awesome_osc',  True),
            ('bop',          True),   ('roc_14',       True),
            ('ppo',          True),   ('dpo_20',       True),
            ('fisher',       True),
        )

        _vol_01 = _cat(
            ('volume_delta',    True),  ('volume_delta_14', True),
            ('cmf_20',          True),  ('mfi_14',          True),
            ('eom_14',          True),  ('relative_volume', True),
            ('vol_velocity',    True),  ('kvo',             True),
        )

        _bands_01 = _cat(
            ('bb_pct_b',          True),  ('atr_band_position', True),
            ('donchian_position', True),  ('close_position',    True),
            ('quantile_position', True),  ('se_position',       True),
            ('starc_position',    True),  ('gaussian_position', True),
        )

        # Smart money: structure_bias + range position are percentile-ranked;
        # BOS/CHoCH events are sparse (0/1) so we use recent rolling sums instead.
        _struct_01 = _cat(('structure_bias', True), ('range_position_score', True))
        _bos_bull  = float(df['bos_up'].tail(5).sum())   if 'bos_up'   in df.columns else 0.0
        _bos_bear  = float(df['bos_down'].tail(5).sum()) if 'bos_down' in df.columns else 0.0
        _cho_bull  = float(df['choch_bull'].tail(5).sum()) if 'choch_bull' in df.columns else 0.0
        _cho_bear  = float(df['choch_bear'].tail(5).sum()) if 'choch_bear' in df.columns else 0.0
        _at_sup    = _f('is_at_support',    0.0)
        _at_res    = _f('is_at_resistance', 0.0)
        _smc_bull_pts = _bos_bull * 0.8 + _cho_bull * 1.2 + _at_sup * 0.6
        _smc_bear_pts = _bos_bear * 0.8 + _cho_bear * 1.2 + _at_res * 0.6
        _smc_event_bias = np.clip((_smc_bull_pts - _smc_bear_pts) / 3.0, -0.35, 0.35)
        _smart_01  = float(np.clip(_struct_01 + _smc_event_bias, 0.0, 1.0))

        # Candle patterns: binary events (0/1) on the last bar
        _bull_cdl = sum(_f(p, 0.0) for p in ('CDL_HAMMER', 'CDL_BULL_ENGULFING', 'CDL_MORNINGSTAR'))
        _bear_cdl = sum(_f(p, 0.0) for p in ('CDL_SHOOTINGSTAR', 'CDL_BEAR_ENGULFING', 'CDL_EVENINGSTAR'))
        _bar_dir  = float(np.clip(_f('bar_direction', 0.0) * 0.3, -0.3, 0.3))
        _cdl_raw  = float(np.clip(_bull_cdl - _bear_cdl, -1.0, 1.0)) + _bar_dir
        _candle_01 = float(np.clip((_cdl_raw + 1.0) / 2.0, 0.0, 1.0))

        # Weighted total (same weights as compute_category_confluence)
        _W = {'tr': 2.0, 'mo': 1.5, 'vo': 1.5, 'sm': 1.5, 'bd': 1.0, 'cd': 0.5}
        _Wsum = sum(_W.values())  # 8.0
        _total_01 = (
            _trend_01  * _W['tr'] + _mom_01    * _W['mo'] +
            _vol_01    * _W['vo'] + _smart_01  * _W['sm'] +
            _bands_01  * _W['bd'] + _candle_01 * _W['cd']
        ) / _Wsum

        # Map to [0,10] display scale
        trend_conf  = _s10(_trend_01)
        mom_conf    = _s10(_mom_01)
        vol_conf    = _s10(_vol_01)
        smart_conf  = _s10(_smart_01)
        bands_conf  = _s10(_bands_01)
        candle_conf = _s10(_candle_01)
        total_conf  = _s10(_total_01)

        # Keep raw [-1,+1] values for the bias score calculation
        # (bias uses macro + pct-rank based trend/mom converted back)
        trend_conf_raw  = _trend_01  * 2.0 - 1.0
        mom_conf_raw    = _mom_01    * 2.0 - 1.0
        smart_conf_raw  = _smart_01  * 2.0 - 1.0
        vol_conf_raw    = _vol_01    * 2.0 - 1.0
        candle_conf_raw = _candle_01 * 2.0 - 1.0
        bands_conf_raw  = _bands_01  * 2.0 - 1.0
        total_conf_raw  = _total_01  * 2.0 - 1.0

        # Weighted composite bias score (-1 → +1), using raw centered values directly
        bias_score = (
            macro_1d       * 0.30
            + macro_1w     * 0.15
            + trend_conf_raw  * 0.25
            + mom_conf_raw    * 0.20
            + smart_conf_raw  * 0.10
        )
        if   bias_score >  0.15: market_bias = "BULLISH"
        elif bias_score < -0.15: market_bias = "BEARISH"
        else:                    market_bias = "NEUTRAL"

        # Trend regime
        if   adx > 25 and macro_1d >  0.1: trend_regime = "TRENDING_UP"
        elif adx > 25 and macro_1d < -0.1: trend_regime = "TRENDING_DOWN"
        elif adx > 25:                      trend_regime = "TRENDING"
        else:                               trend_regime = "RANGING"

        # Volatility regime
        atr_pct = atr / price
        if   atr_pct > 0.025: vol_regime = "HIGH"
        elif atr_pct > 0.012: vol_regime = "MEDIUM"
        else:                  vol_regime = "LOW"

        # ── S/R levels ────────────────────────────────────────────────────────
        support    = _f('rolling_support',    price * 0.97)
        resistance = _f('rolling_resistance', price * 1.03)
        pivot      = _f('pivot',  price)
        r1 = _f('r1', price * 1.010); r2 = _f('r2', price * 1.020)
        s1 = _f('s1', price * 0.990); s2 = _f('s2', price * 0.980)

        # ── Price targets (ATR-projected) ─────────────────────────────────────
        step = atr * atr_mult
        bull_tp1 = round(price + 1.0 * step, 8)
        bull_tp2 = round(price + 2.0 * step, 8)
        bull_tp3 = round(price + 3.5 * step, 8)
        bear_tp1 = round(price - 1.0 * step, 8)
        bear_tp2 = round(price - 2.0 * step, 8)
        bear_tp3 = round(price - 3.5 * step, 8)

        # ── Confluence summary label (total_conf is now [0, 10]) ─────────────
        # Strength is the distance from neutral (5.0): 0–4 bearish, 6–10 bullish
        conf_magnitude = abs(total_conf - 5.0)  # 0 = fully neutral, 5 = max signal
        if   conf_magnitude >= 3.0: conf_tier = "Strong"
        elif conf_magnitude >= 1.75: conf_tier = "Moderate"
        elif conf_magnitude >= 0.75: conf_tier = "Weak"
        else:                        conf_tier = "Very Weak"
        if   market_bias == "BULLISH": conf_summary = f"{conf_tier} Bullish"
        elif market_bias == "BEARISH": conf_summary = f"{conf_tier} Bearish"
        else:                          conf_summary = f"{conf_tier} / Neutral"

        # ── Momentum indicators ───────────────────────────────────────────────
        rsi = _f('rsi_14') or _f('rsi_7', 50.0)
        macd_hist  = _f('macd_hist')
        cci        = _f('cci_20')
        supertrend_dir = int(_f('supertrend_dir', 0))
        if   supertrend_dir > 0: st_label = "BULLISH"
        elif supertrend_dir < 0: st_label = "BEARISH"
        else:                    st_label = "NEUTRAL"

        # ── Volume ────────────────────────────────────────────────────────────
        vol_zscore = _f('volume_zscore')
        if   vol_zscore >  1.0: vol_strength = "ABOVE_AVERAGE"
        elif vol_zscore < -0.5: vol_strength = "BELOW_AVERAGE"
        else:                   vol_strength = "AVERAGE"

        # ── Derivatives ───────────────────────────────────────────────────────
        funding_rate  = _f('funding_rate')
        oi_change_1h  = _f('oi_change_1h')
        oi_zscore     = _f('oi_zscore')

        if   funding_rate >  0.0001: funding_bias = "LONGS_PAYING"
        elif funding_rate < -0.0001: funding_bias = "SHORTS_PAYING"
        else:                        funding_bias = "NEUTRAL"

        if   oi_change_1h >  0.01: oi_trend = "INCREASING"
        elif oi_change_1h < -0.01: oi_trend = "DECREASING"
        else:                      oi_trend = "STABLE"

        # ── Fear & Greed ──────────────────────────────────────────────────────
        fg_val = _f('fear_greed_value', -1.0)
        if fg_val >= 0:
            if   fg_val >= 75: fg_label = "Extreme Greed"
            elif fg_val >= 55: fg_label = "Greed"
            elif fg_val >= 45: fg_label = "Neutral"
            elif fg_val >= 25: fg_label = "Fear"
            else:              fg_label = "Extreme Fear"
            fear_greed = {"value": round(fg_val, 1), "label": fg_label}
        else:
            fear_greed = None

        # ── Trading session ───────────────────────────────────────────────────
        hour = datetime.now(_tz.utc).hour
        if   0  <= hour <  8: session, session_note = "ASIA",       "Lower volatility; watch for false breakouts near key levels"
        elif 8  <= hour < 12: session, session_note = "LONDON",     "High volatility; trend-following setups tend to work well"
        elif 12 <= hour < 16: session, session_note = "NY_OVERLAP", "Peak liquidity; cleanest breakouts and reversals"
        elif 16 <= hour < 21: session, session_note = "NEW_YORK",   "Strong directional moves; good for momentum entries"
        else:                 session, session_note = "LATE_NY",    "Consolidation likely; reduce size, tighter stops"

        # ── Trader-type views ─────────────────────────────────────────────────
        # Scalper (5m–1h): half-ATR targets, needs volume confirmation
        scalper_bias = market_bias if vol_strength == "ABOVE_AVERAGE" else "NEUTRAL"
        scalper = {
            "bias":        scalper_bias,
            "entry_zone":  round(price, 8),
            "bull_target": round(price + 0.5 * atr, 8),
            "bear_target": round(price - 0.5 * atr, 8),
            "bull_stop":   round(price - 0.5 * atr, 8),
            "bear_stop":   round(price + 0.5 * atr, 8),
            "range_pct":   round(atr * 0.5 / price * 100, 3),
            "session_ok":  session in ("NY_OVERLAP", "NEW_YORK", "LONDON"),
        }

        # Day trader (1h–4h): full ATR targets, session-aware
        day_trader = {
            "bias":          market_bias,
            "bull_target":   bull_tp1,
            "bear_target":   bear_tp1,
            "key_support":   round(s1, 8),
            "key_resistance":round(r1, 8),
            "atr_daily_pct": round(atr_pct * 100, 2),
            "trend":         trend_regime,
        }

        # Swing (4h–1w): weekly trend + wide levels
        swing_bias = ("BULLISH" if macro_1w > 0.10 else ("BEARISH" if macro_1w < -0.10 else "NEUTRAL"))
        swing = {
            "bias":           swing_bias,
            "weekly_trend":   "UP" if macro_1w > 0.10 else ("DOWN" if macro_1w < -0.10 else "FLAT"),
            "key_support":    round(s2, 8),
            "key_resistance": round(r2, 8),
            "bull_target":    bull_tp2,
            "bear_target":    bear_tp2,
            "extended_bull":  bull_tp3,
            "extended_bear":  bear_tp3,
        }

        return {
            # Market overview
            "market_bias":     market_bias,
            "bias_strength":   round(abs(bias_score), 3),
            "trend_regime":    trend_regime,
            "volatility_regime": vol_regime,
            "atr_pct":         round(atr_pct * 100, 3),

            # S/R levels (for "view logic" button)
            "support":    round(support, 8),
            "resistance": round(resistance, 8),
            "pivot":      round(pivot, 8),
            "r1": round(r1, 8), "r2": round(r2, 8),
            "s1": round(s1, 8), "s2": round(s2, 8),

            # Price targets
            "bull_tp1": bull_tp1, "bull_tp2": bull_tp2, "bull_tp3": bull_tp3,
            "bear_tp1": bear_tp1, "bear_tp2": bear_tp2, "bear_tp3": bear_tp3,

            # Confluence breakdown (key "view logic" data)
            "confluence": {
                "total":       round(total_conf, 1),
                "momentum":    round(mom_conf, 1),
                "trend":       round(trend_conf, 1),
                "volume":      round(vol_conf, 1),
                "smart_money": round(smart_conf, 1),
                "bands":       round(bands_conf, 1),   # price position / BB / ATR
                "candle":      round(candle_conf, 1),
                "summary":     conf_summary,
            },

            # Indicators
            "rsi":            round(rsi, 1),
            "macd_signal":    "BULLISH" if macd_hist > 0 else ("BEARISH" if macd_hist < 0 else "NEUTRAL"),
            "cci":            round(cci, 1),
            "adx":            round(adx, 1),
            "supertrend":     st_label,
            "macro_daily":    round(macro_1d, 3),
            "macro_weekly":   round(macro_1w, 3),

            # Volume & derivatives
            "volume_strength": vol_strength,
            "volume_zscore":   round(vol_zscore, 2),
            "funding_rate":    round(funding_rate * 100, 4),
            "funding_bias":    funding_bias,
            "oi_trend":        oi_trend,
            "oi_change_1h_pct":round(oi_change_1h * 100, 3),
            "oi_zscore":       round(oi_zscore, 2),

            # Session & fear/greed
            "session":       session,
            "session_note":  session_note,
            "fear_greed":    fear_greed,

            # Trader-type views
            "scalper_view":   scalper,
            "day_trader_view":day_trader,
            "swing_view":     swing,
        }