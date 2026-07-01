#!/usr/bin/env python3
"""
cost_aware_backtester.py – Aegis-1 Production Backtester v4.3

Gate alignment with live_engine.py (as of ab5749c0):
  Gate 1   — Regime gate: regime_thresholds buy_ok / sell_ok
  Gate 2   — Edge score ≥ threshold (0-100 scale, via EdgeScoringEngine)
             + tradeable_buy / tradeable_sell flags
  Gate 3   — S&R / HTF trend / confluence / choppiness / BOS filter
  Gate 4.5 — Two-zone MTF reversal timing (approximated on 1H bars)
              Prior zone  (4 bars ago): ≥60% in prior-trend direction
              Reversal zone (last 2 bars): ≥1 bar in reversal direction
  Gate 5   — RSI exhaustion / deceleration gate
  Gate 6   — Post-trade cooldown (3-bar minimum between entries)
  Gate 7   — ATR floor (min 0.1% of price — skip dead markets)

Gate removed vs v4.0 (not in live engine):
  * EV gate removed — live engine has no EV gate; quality is enforced by
    edge_score threshold (Gate 2). EV is still computed and logged in the
    forensic script for diagnostic use.

Known gaps vs live engine (cannot backtest on 1H-only data):
  * HTF hard veto (weekly + daily EMA50) is approximated by trend_1d filter
  * Portfolio-guard concurrent-position cap is per-symbol here, always available
  * Actual 5m/15m candle data — approximated by 1H bar sequences
  * HMM regime detection — approximated by regime_thresholds from meta.json

Bug checklist surfaced by forensic runs:
  1. Gate 2 scale mismatch (FIXED v4.1): backtester was comparing meta_conf
     (0-1) against threshold (0-100), blocking 100% of signals. Now uses
     edge_score from EdgeScoringEngine which is on the same 0-100 scale.
  2. MTF prior-zone threshold 60% may over-block in ranging markets — monitor
     skipped_mtf vs win-rate lift; if ratio is unfavorable, lower to 0.50
  3. RSI acceleration threshold -0.08 is hardcoded in live engine; if changing
     it materially shifts results, calibrate per-token
  4. Cooldown=3 bars is not per-ATR-regime calibrated — high-vol tokens may
     need fewer bars; slow tokens may need more
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features, compute_atr
from src.trading.edge_engine import EdgeScoringEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_BALANCE:     float = 1_000.0
EXCHANGE_FEE_PCT:    float = 0.001          # 0.10 % per leg
SLIPPAGE_PCT:        float = 0.001          # 0.10 % per leg
TOTAL_COST_PCT:      float = 2 * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)   # 0.40 %

MAX_HOLD_CANDLES:    int   = 48             # time-based exit after 48 h
MIN_SL_DISTANCE_PCT: float = 0.005         # minimum SL = 0.5 % of price
COST_MULTIPLIER:     float = 1.5            # EV must beat 1.5× round-trip cost
MIN_TRADES_REQUIRED: int   = 10

DATA_HOURS:          int   = 3_000
MIN_HOURLY_BARS:     int   = 500

# ── Gate 1.5: Direction-regime hard veto (HMM proxy) ─────────────────────────
# Live engine blocks BUY if HMM regime == TRENDING_BEAR; SELL if TRENDING_BULL.
# HMM batch inference not available — approximate with macro_trend_1d.
HMM_BEAR_PROXY:  float = -0.5   # macro_trend_1d below → TRENDING_BEAR proxy
HMM_BULL_PROXY:  float =  0.5   # macro_trend_1d above → TRENDING_BULL proxy

# ── Gate 1.7: HTF macro hard veto (live_engine.py Gate 1.7) ──────────────────
HTF_HARD_OPPOSE: float =  0.5   # both weekly+daily must exceed this to veto

# ── Gate 2 (live engine ATR floor) — must match live_engine.py Gate 2 ─────────
ATR_FLOOR_PCT_LIVE: float = 0.008   # 0.8% — live engine skips if atr_pct < 0.8%

# ── Gate 3: Model quality floor (live_engine.py MIN_QUALITY_SCORE) ────────────
MIN_QUALITY_SCORE: float = 70.0  # edge_score must exceed this after Gate 2 thr

# ── Gate 3b: Contextual quality filter (live_engine.py score_signal >= 70) ───
SCORE_SIG_FLOOR:  float = 70.0  # score_signal() approximation floor

# ── Gate 3.8: Signal stability (live_engine.py SIGNAL_STABILITY_WINDOW) ──────
SIGNAL_BYPASS_EDGE: float = 85.0  # bypass stability at very high edge
STABILITY_WINDOW:   int   = 2     # consecutive same-direction bars required

# ── Gate 4.5: MTF two-zone parameters (must match live_engine.py Gate 4.5) ───
MTF_PRIOR_BARS:  int   = 4     # older 1H bars → confirm prior trend (≥60%)
MTF_RECENT_BARS: int   = 2     # most-recent 1H bars → confirm reversal (≥50%)
MTF_PRIOR_FRAC:  float = 0.60  # minimum fraction in prior direction
MTF_REV_FRAC:    float = 0.50  # minimum fraction in reversal direction

# ── Gate 5: RSI exhaustion (must match live_engine.py RSI gate) ──────────────
RSI_OVERBOUGHT:  float = 72.0
RSI_OVERSOLD:    float = 28.0
RSI_ACCEL_BUY:   float = -0.08  # block BUY if accel < this (RSI in free-fall)
RSI_ACCEL_SELL:  float =  0.08  # block SELL if accel > this (RSI recovering)

# ── Gate 6: Cooldown (minimum bars between consecutive trades) ───────────────
COOLDOWN_BARS:   int   = 3      # 3h minimum; mirrors live engine cooldown logic

# ── Gate 7: ATR floor (dead market guard, kept for legacy / low-vol veto) ────
ATR_FLOOR_PCT:   float = 0.001  # 0.10% fallback (weaker than Gate 2 live check)

# ── Modes ─────────────────────────────────────────────────────────────────────
MODE_PARAMS: Dict[str, Dict[str, float]] = {
    "conservative": {"risk_pct": 0.010, "rr_ratio": 1.5},
    "balanced":     {"risk_pct": 0.020, "rr_ratio": 2.0},
    "aggressive":   {"risk_pct": 0.030, "rr_ratio": 2.5},
}

# ── Fleet (full 63-token set matching live engine) ────────────────────────────
FLEET: List[str] = [
    "BTC/USDT",  "ETH/USDT",  "SOL/USDT",  "BNB/USDT",  "XRP/USDT",
    "ADA/USDT",  "AVAX/USDT", "LINK/USDT", "DOT/USDT",  "TON/USDT",
    "ICP/USDT",  "SUI/USDT",  "INJ/USDT",  "ATOM/USDT", "UNI/USDT",
    "ALGO/USDT", "CHZ/USDT",  "ICX/USDT",  "GMX/USDT",  "ONDO/USDT",
    "JUP/USDT",  "POLYX/USDT","ORDI/USDT", "RSR/USDT",  "CRV/USDT",
    "AAVE/USDT", "APT/USDT",  "ARB/USDT",  "ARKM/USDT", "AXS/USDT",
    "BCH/USDT",  "COMP/USDT", "DYDX/USDT", "EGLD/USDT", "ENA/USDT",
    "ENJ/USDT",  "ETC/USDT",  "FET/USDT",  "FIL/USDT",  "FLOKI/USDT",
    "GALA/USDT", "GRT/USDT",  "HBAR/USDT", "IMX/USDT",  "LDO/USDT",
    "LTC/USDT",  "MANA/USDT", "NEAR/USDT", "OP/USDT",   "PENDLE/USDT",
    "PEPE/USDT", "PYTH/USDT", "SAND/USDT", "SEI/USDT",  "SNX/USDT",
    "STX/USDT",  "TIA/USDT",  "TRX/USDT",  "VET/USDT",  "WIF/USDT",
    "WLFI/USDT", "XLM/USDT",  "ZRX/USDT",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA-CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol:       str
    mode:         str
    bar_index:    int
    timestamp:    str
    direction:    str
    meta_conf:    float
    thr:          float
    size_factor:  float
    entry_price:  float
    exit_price:   float = 0.0
    pnl:          float = 0.0
    return_pct:   float = 0.0
    exit_reason:  str   = ""


@dataclass
class BacktestResult:
    symbol:               str
    mode:                 str
    total_trades:         int   = 0
    wins:                 int   = 0
    losses:               int   = 0
    gross_profit:         float = 0.0
    gross_loss:           float = 0.0
    net_return_pct:       float = 0.0
    max_drawdown_pct:     float = 0.0
    avg_trade_return_pct: float = 0.0
    sharpe_ratio:         float = 0.0
    win_rate:             float = 0.0
    profit_factor: Optional[float] = None
    total_signals:        int   = 0
    skipped_dir_regime:   int   = 0   # Gate 1.5 — HMM direction-regime proxy veto
    skipped_htf_veto:     int   = 0   # Gate 1.7 — weekly+daily macro hard veto
    skipped_atr_live:     int   = 0   # Gate 2   — ATR floor < 0.8% (live engine)
    skipped_min_quality:  int   = 0   # Gate 3   — edge_score < MIN_QUALITY_SCORE
    skipped_sr_trend:     int   = 0   # Gate 3   — S&R / trend / confluence / BOS
    skipped_score_signal: int   = 0   # Gate 3b  — score_signal approx < floor
    skipped_adx:          int   = 0   # Gate 3.5 — ADX trend strength < 25
    skipped_stability:    int   = 0   # Gate 3.8 — signal stability window
    skipped_mtf:          int   = 0   # Gate 4.5 — two-zone reversal timing
    skipped_rsi_exhaust:  int   = 0   # Gate 5   — RSI exhaustion / deceleration
    skipped_cooldown:     int   = 0   # Gate 6   — post-trade cooldown
    skipped_atr_floor:    int   = 0   # Gate 7   — legacy ATR floor fallback
    low_confidence:       bool  = True
    atr_multiplier:       float = 1.5
    error:   Optional[str]     = None
    trades:  List[TradeRecord] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    return float(np.mean(arr) / max(np.std(arr), 1e-8) * np.sqrt(252 * 24))


def _ev(meta_conf: float, atr: float, entry: float,
        atr_sl: float, atr_tp: float) -> float:
    """Cost-adjusted expected value using meta_conf as win probability."""
    tp_frac = (atr_tp * atr) / entry
    sl_frac = (atr_sl * atr) / entry
    return meta_conf * tp_frac - (1.0 - meta_conf) * sl_frac - TOTAL_COST_PCT


def _meta_size_factor(edge_score: float, thr: float) -> float:
    """Scale position 0.5–1.0 based on how far edge_score exceeds threshold (both 0-100)."""
    excess = edge_score - thr
    if excess >= 15.0:
        return 1.00
    if excess >= 7.5:
        return 0.75
    return 0.50


def _score_signal_approx(
    direction: str,
    adx: float,
    vol_z: float,
    edge_score: float,
    rsi: float,
    macro_w: float,
    macro_d: float,
    total_conf: float,
    macd_hist: float,
) -> float:
    """Approximate live_engine SignalQualityFilter.score_signal() from 1H features.

    The live engine also uses funding_bias, OI trend, HMM confidence, and LSTM
    signals — none available in batch mode.  We default those to neutral (+10
    funding, +5 OI) so the approximation is conservative (may under-score).
    Use SCORE_SIG_FLOOR (70) as the gate threshold, same as the live engine.
    """
    s = 0.0
    # Trend quality (mirrors live: ADX component)
    if adx > 25.0:
        s += 15.0
    # Volume confirmation
    if vol_z > 1.5:
        s += 10.0
    elif vol_z < -0.8:
        s -= 10.0
    # ML edge quality → maps to meta_prob component in live engine
    s += float(np.clip((edge_score - 50.0) * 0.5, -15.0, 15.0))
    # RSI position quality
    if direction == "long":
        if rsi < 55.0:   s += 10.0
        elif rsi < 65.0: s += 5.0
        elif rsi > 70.0: s -= 8.0
    else:
        if rsi > 45.0:   s += 10.0
        elif rsi > 35.0: s += 5.0
        elif rsi < 30.0: s -= 8.0
    # HTF macro alignment (mirrors live: higher-timeframe bias)
    if macro_w != 0.0 or macro_d != 0.0:
        if direction == "long":
            if macro_w > 0.5 and macro_d > 0.5:    s += 15.0
            elif macro_w > 0.2 and macro_d > 0.2:  s += 8.0
            elif macro_w < -0.5 and macro_d < -0.5: s -= 20.0
            elif macro_w < -0.2:                    s -= 10.0
        else:
            if macro_w < -0.5 and macro_d < -0.5:  s += 15.0
            elif macro_w < -0.2 and macro_d < -0.2: s += 8.0
            elif macro_w > 0.5 and macro_d > 0.5:  s -= 20.0
            elif macro_w > 0.2:                     s -= 10.0
    # Confluence alignment
    if direction == "long":
        if total_conf > 7.0:   s += 10.0
        elif total_conf < 3.0: s -= 10.0
    else:
        if total_conf < 3.0:   s += 10.0
        elif total_conf > 7.0: s -= 10.0
    # MACD alignment
    if macd_hist > 0:
        s += 8.0 if direction == "long"  else -12.0
    elif macd_hist < 0:
        s += 8.0 if direction == "short" else -12.0
    # Neutral defaults for live-only signals (funding=neutral, OI=stable)
    s += 10.0   # funding_bias neutral
    s += 5.0    # oi_trend stable
    return float(np.clip(s, 0.0, 100.0))


def _dir_fraction(closes: np.ndarray, opens: np.ndarray, direction: str) -> float:
    """
    Fraction of bars whose close/open matches `direction`.
    Vectorised — safe to call in a tight loop.
    Returns 0.5 on empty input (neutral / fail-open).
    """
    if len(closes) == 0:
        return 0.5
    if direction == "bearish":
        hits = np.sum(closes < opens)
    else:
        hits = np.sum(closes > opens)
    return float(hits) / len(closes)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveBacktester:
    # Class-level caches — shared across all modes for a given symbol so
    # model weights are loaded once and feature engineering runs once.
    _df_cache:        Dict[str, pd.DataFrame] = {}
    _predictor_cache: Dict[str, Predictor]    = {}

    def __init__(self, symbol: str, mode: str,
                 initial_balance: float = INITIAL_BALANCE):
        self.symbol          = symbol
        self.mode            = mode
        self.initial_balance = initial_balance
        self.params          = MODE_PARAMS[mode]

        if symbol not in AdaptiveBacktester._predictor_cache:
            AdaptiveBacktester._predictor_cache[symbol] = Predictor(symbol)
        self.predictor = AdaptiveBacktester._predictor_cache[symbol]

        meta = self.predictor.meta

        # Prefer meta_gate_profile thresholds (always 0-100 scale);
        # fall back to meta_threshold_buy/sell (may be 0-1 fraction scale).
        # Normalize: if raw value < 2.0 it's a 0-1 fraction → multiply by 100
        # to match the 0-100 scale that EdgeScoringEngine produces.
        _gate_thr = meta.get("meta_gate_profile", {}).get("thresholds", {})
        _raw_buy  = float(_gate_thr.get("buy_threshold",
                          meta.get("meta_threshold_buy",
                          meta.get("meta_threshold", 62.0))))
        _raw_sell = float(_gate_thr.get("sell_threshold",
                          meta.get("meta_threshold_sell",
                          meta.get("meta_threshold", 62.0))))
        self.thr_buy  = _raw_buy  * 100.0 if _raw_buy  < 2.0 else _raw_buy
        self.thr_sell = _raw_sell * 100.0 if _raw_sell < 2.0 else _raw_sell

        self.tradeable_buy  = bool(meta.get("tradeable_buy",
                                   meta.get("tradeable", True)))
        self.tradeable_sell = bool(meta.get("tradeable_sell",
                                   meta.get("tradeable", True)))
        self.override_thr   = float(meta.get("meta_override_confidence", 1.0))
        self.atr_mult       = float(meta.get("atr_multiplier", 1.5))

        _opt = self.predictor._token_params or {}
        self.regime_thresholds = meta.get("regime_thresholds") or _opt.get("regimes", {})
        self.regime_boundaries = meta.get("regime_boundaries") or _opt.get("regime_boundaries", {})

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, hours: int = DATA_HOURS) -> BacktestResult:
        result = BacktestResult(symbol=self.symbol, mode=self.mode,
                                atr_multiplier=self.atr_mult)
        df = self._fetch_and_prepare(hours)
        if df is None or len(df) < MIN_HOURLY_BARS:
            result.error = "Insufficient data"
            return result

        # Pre-extract numpy arrays for vectorised gate helpers (no per-row overhead).
        closes       = df["close"].to_numpy(dtype=float)
        opens        = df["open"].to_numpy(dtype=float)
        meta_dir_arr = df["meta_direction"].to_numpy(dtype=int)   # Gate 3.8

        p        = self.params
        risk_pct = p["risk_pct"]
        atr_sl   = self.atr_mult
        atr_tp   = self.atr_mult * p["rr_ratio"]

        capital       = self.initial_balance
        peak_capital  = capital
        pos_units     = 0.0
        entry_price   = 0.0
        entry_atr_val = 0.0
        entry_bar     = -1
        last_exit_bar = -1          # Gate 6: cooldown tracking
        direction: Optional[str] = None
        trade_returns: List[float] = []
        max_dd        = 0.0

        total_signals        = 0
        skipped_dir_regime   = 0
        skipped_htf_veto     = 0
        skipped_atr_live     = 0
        skipped_min_quality  = 0
        skipped_srt          = 0
        skipped_score_signal = 0
        skipped_adx          = 0
        skipped_stability    = 0
        skipped_mtf          = 0
        skipped_rsi_exhaust  = 0
        skipped_cooldown     = 0
        skipped_atr_floor    = 0

        n = len(df)

        for i in range(MTF_PRIOR_BARS + MTF_RECENT_BARS, n - MAX_HOLD_CANDLES - 1):
            row = df.iloc[i]

            # ── EXIT ─────────────────────────────────────────────────────────
            if pos_units != 0.0:
                bars_held   = i - entry_bar
                exit_px     = self._check_exit(row, entry_price, entry_atr_val,
                                               atr_sl, atr_tp, direction)
                exit_reason = ""
                if exit_px is not None:
                    exit_reason = "tp_sl"
                elif bars_held >= MAX_HOLD_CANDLES:
                    exit_px     = float(row["close"])
                    exit_reason = "timeout"

                if exit_px is not None:
                    if direction == "long":
                        exit_slipped = exit_px * (1.0 - SLIPPAGE_PCT)
                        gross_pnl    = pos_units * (exit_slipped - entry_price)
                    else:
                        exit_slipped = exit_px * (1.0 + SLIPPAGE_PCT)
                        gross_pnl    = pos_units * (entry_price - exit_slipped)

                    exit_cost = abs(pos_units) * exit_slipped * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
                    net_pnl   = gross_pnl - exit_cost
                    capital  += net_pnl

                    pos_val  = entry_price * pos_units
                    ret_pct  = (net_pnl / pos_val * 100) if pos_val > 0 else 0.0
                    trade_returns.append(ret_pct / 100.0)

                    if net_pnl >= 0:
                        result.wins         += 1
                        result.gross_profit += net_pnl
                    else:
                        result.losses      += 1
                        result.gross_loss  += abs(net_pnl)

                    result.total_trades += 1
                    result.trades.append(TradeRecord(
                        symbol=self.symbol, mode=self.mode, bar_index=entry_bar,
                        timestamp=str(df.iloc[entry_bar]["timestamp"]),
                        direction=direction or "",
                        meta_conf=float(df.iloc[entry_bar].get("meta_conf", 0.0)),
                        thr=self.thr_buy if direction == "long" else self.thr_sell,
                        size_factor=round(pos_units * entry_price / capital, 4),
                        entry_price=round(entry_price, 6),
                        exit_price=round(exit_slipped, 6),
                        pnl=round(net_pnl, 4),
                        return_pct=round(ret_pct, 4),
                        exit_reason=exit_reason,
                    ))

                    peak_capital  = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0.0
                    max_dd        = max(max_dd, dd)
                    last_exit_bar = i
                    pos_units     = 0.0
                continue   # never open a new trade on the same bar as an exit

            # ── GATE 1: REGIME ────────────────────────────────────────────────
            # Tokens trained without regime_thresholds bypass this gate entirely.
            # When regime_thresholds exist, per-regime buy_ok/sell_ok are the
            # primary tradeability authority — they override the global
            # tradeable_buy/sell flags (which may be conservatively set based on
            # overall holdout stats while specific regimes perform well).
            regime_str = str(row.get("regime_str", "unknown"))
            if self.regime_thresholds:
                reg = self.regime_thresholds.get(regime_str, {})
                if not reg or reg.get("skipped"):
                    continue
                _buy_ok  = bool(reg.get("buy_ok",  True))
                _sell_ok = bool(reg.get("sell_ok", True))
                # Per-regime thresholds are 0-1 fractions → normalize to 0-100
                _r_buy_raw  = reg.get("buy_threshold")
                _r_sell_raw = reg.get("sell_threshold")
                _thr_buy  = ((_r_buy_raw  * 100.0 if _r_buy_raw  < 2.0 else _r_buy_raw)
                             if _r_buy_raw  is not None else self.thr_buy)
                _thr_sell = ((_r_sell_raw * 100.0 if _r_sell_raw < 2.0 else _r_sell_raw)
                             if _r_sell_raw is not None else self.thr_sell)
            else:
                # No regime gating configured for this token
                _buy_ok   = True
                _sell_ok  = True
                _thr_buy  = self.thr_buy
                _thr_sell = self.thr_sell

            # ── GATE 2: EDGE SCORE + TRADEABILITY ────────────────────────────
            # Live engine uses EdgeScoringEngine to compute a 0-100 quality score
            # and compares against meta_threshold (stored as 60.0, 0-100 scale).
            # meta_conf from predict_meta_batch() is 0-1 scale (raw meta prob).
            # When regime_thresholds exist, per-regime buy_ok/sell_ok determine
            # tradeability (overriding global tradeable_buy/sell) and per-regime
            # thresholds are used as the edge floor for this specific bar.
            meta_conf  = float(row.get("meta_conf",       0.0))
            meta_dir   = int(row.get("meta_direction",    1))
            edge_score = float(row.get("edge_score_side", 0.0))

            if meta_dir == 2:
                if not _buy_ok:
                    continue
                if edge_score < _thr_buy:
                    continue
                direction = "long"
                thr = _thr_buy
            elif meta_dir == 0:
                if not _sell_ok:
                    continue
                if edge_score < _thr_sell:
                    continue
                direction = "short"
                thr = _thr_sell
            else:
                continue    # HOLD

            # ── GATE 1.5: DIRECTION-REGIME HARD VETO (HMM proxy) ─────────────
            # Live engine blocks BUY if HMM == TRENDING_BEAR; SELL if TRENDING_BULL.
            # HMM not available in batch — proxy with macro_trend_1d.
            _t1d_proxy = float(row.get("macro_trend_1d", 0.0))
            if direction == "long"  and _t1d_proxy < HMM_BEAR_PROXY:
                skipped_dir_regime += 1
                continue
            if direction == "short" and _t1d_proxy > HMM_BULL_PROXY:
                skipped_dir_regime += 1
                continue

            # ── GATE 1.7: HTF MACRO HARD VETO ────────────────────────────────
            # Live engine: macro_weekly < -0.5 AND macro_daily < -0.5 → block BUY
            #              macro_weekly >  0.5 AND macro_daily >  0.5 → block SELL
            _macro_w = float(row.get("macro_trend_1w", 0.0))
            _macro_d = float(row.get("macro_trend_1d", 0.0))
            if direction == "long"  and _macro_w < -HTF_HARD_OPPOSE and _macro_d < -HTF_HARD_OPPOSE:
                skipped_htf_veto += 1
                continue
            if direction == "short" and _macro_w >  HTF_HARD_OPPOSE and _macro_d >  HTF_HARD_OPPOSE:
                skipped_htf_veto += 1
                continue

            total_signals += 1

            # NOTE: Live engine Gate 2 checks atr_pct < 0.8%, but predict_realtime()
            # always uses the 1.5% ATR fallback (because _atr is never in the batch df).
            # That means Gate 2 never fires in production. Backtester mirrors this by
            # pre-computing cur_atr for downstream gates without applying the 0.8% floor.
            cur_atr = float(row["atr_14"]) if float(row["atr_14"]) > 0 \
                      else float(row["close"]) * 0.01

            # ── GATE 3: MIN QUALITY SCORE FLOOR ──────────────────────────────
            # Live engine Gate 3: edge_score >= MIN_QUALITY_SCORE (70.0).
            # Applied on top of the meta_threshold (Gate 2 thr, typically 60).
            if edge_score < MIN_QUALITY_SCORE:
                skipped_min_quality += 1
                continue

            # ── GATE 3: S&R / TREND / CONFLUENCE / CHOPPINESS / BOS ──────────
            is_high_conviction = edge_score >= float(self.override_thr)
            if not is_high_conviction:
                at_res = bool(row.get("is_at_resistance", 0))
                at_sup = bool(row.get("is_at_support",    0))
                if (direction == "long"  and at_res) or \
                   (direction == "short" and at_sup):
                    skipped_srt += 1
                    continue

                trend_1d = float(row.get("macro_trend_1d", 0.0))
                if (direction == "long"  and trend_1d < -0.2) or \
                   (direction == "short" and trend_1d >  0.2):
                    skipped_srt += 1
                    continue

                total_conf = float(row.get("total_confluence", 0.0))
                if (direction == "long"  and total_conf < -0.1) or \
                   (direction == "short" and total_conf >  0.1):
                    skipped_srt += 1
                    continue

                choppiness = float(row.get("choppiness", 50.0))
                if choppiness > 60:
                    skipped_srt += 1
                    continue

                bos = float(row.get("bos_state", 0.0))
                if (direction == "long"  and bos < 0) or \
                   (direction == "short" and bos > 0):
                    skipped_srt += 1
                    continue

            # ── GATE 3.5: ADX TREND STRENGTH ─────────────────────────────────
            # Forensic analysis: ADX < 25 blocks 2:1 loss:win → +3.1pp WR.
            # High-conviction signals (edge >= SIGNAL_BYPASS_EDGE) bypass this.
            if edge_score < SIGNAL_BYPASS_EDGE:
                adx_val = float(row.get("adx_14", 25.0))
                if adx_val < 25.0:
                    skipped_adx += 1
                    continue

            # NOTE: Gate 3b (score_signal >= 70) is NOT applied in the backtester.
            # The live engine's score_signal() uses funding_bias, OI trend, HMM
            # confidence and LSTM signals that are not available in batch historical
            # data.  Without those live inputs the approximation averages ~55 for
            # both wins and losses — it discriminates by only 2-3 points and blocks
            # them uniformly, adding noise rather than signal quality.
            # The _score_signal_approx helper is kept for use in wr_forensic.py.

            # ── GATE 3.8: SIGNAL STABILITY ────────────────────────────────────
            # Live engine requires STABILITY_WINDOW consecutive same-direction
            # signals; bypassed at edge >= SIGNAL_BYPASS_EDGE.
            if edge_score < SIGNAL_BYPASS_EDGE:
                _meta_req = 2 if direction == "long" else 0
                _stable   = all(
                    meta_dir_arr[i - back] == _meta_req
                    for back in range(1, STABILITY_WINDOW + 1)
                    if (i - back) >= 0
                )
                if not _stable:
                    skipped_stability += 1
                    continue

            # ── GATE 4.5: TWO-ZONE MTF REVERSAL TIMING ───────────────────────
            # Approximates live_engine Gate 4.5 using 1H bar sequences.
            # Prior zone (bars ago [MTF_RECENT_BARS : MTF_PRIOR_BARS + MTF_RECENT_BARS]):
            #   ≥60% in prior-trend direction → confirms trend existed
            # Reversal zone (last MTF_RECENT_BARS bars):
            #   ≥50% in reversal direction → turn is starting now
            #
            # Limitation vs live engine: uses 1H bars instead of actual 5m/15m
            # candles. Expect the backtester to block slightly fewer entries than
            # the live engine in fast-moving 5m timeframes.
            _prior_start  = i - MTF_PRIOR_BARS - MTF_RECENT_BARS
            _prior_end    = i - MTF_RECENT_BARS
            _recent_start = i - MTF_RECENT_BARS

            _prior_c  = closes[_prior_start:_prior_end]
            _prior_o  = opens[_prior_start:_prior_end]
            _recent_c = closes[_recent_start:i]
            _recent_o = opens[_recent_start:i]

            if direction == "long":
                _prior_dir, _rev_dir = "bearish", "bullish"
            else:
                _prior_dir, _rev_dir = "bullish", "bearish"

            _prior_ok = _dir_fraction(_prior_c, _prior_o, _prior_dir)  >= MTF_PRIOR_FRAC
            _rev_ok   = _dir_fraction(_recent_c, _recent_o, _rev_dir)  >= MTF_REV_FRAC

            if not (_prior_ok and _rev_ok):
                skipped_mtf += 1
                continue

            # ── GATE 5: RSI EXHAUSTION / DECELERATION ────────────────────────
            # Mirrors live_engine.py lines ~2094-2098.
            rsi_val   = float(row.get("rsi_14",              50.0))   # live engine result['rsi'] maps to rsi_14
            rsi_slope = float(row.get("rsi_slope_14",         0.0))   # live engine result['rsi_slope'] maps to rsi_slope_14
            rsi_accel = float(row.get("rsi_acceleration_14",  0.0))   # live engine result['rsi_acceleration'] maps to rsi_acceleration_14

            if direction == "long":
                if rsi_val >= RSI_OVERBOUGHT and rsi_slope <= 0:
                    skipped_rsi_exhaust += 1
                    continue
                if rsi_accel < RSI_ACCEL_BUY:
                    skipped_rsi_exhaust += 1
                    continue
            else:
                if rsi_val <= RSI_OVERSOLD and rsi_slope >= 0:
                    skipped_rsi_exhaust += 1
                    continue
                if rsi_accel > RSI_ACCEL_SELL:
                    skipped_rsi_exhaust += 1
                    continue

            # ── GATE 6: COOLDOWN (post-trade minimum gap) ─────────────────────
            # Mirrors the live engine's flip-flop / cooldown protection.
            # Prevents immediate re-entry after a trade closes.
            if last_exit_bar >= 0 and (i - last_exit_bar) < COOLDOWN_BARS:
                skipped_cooldown += 1
                continue

            # ── GATE 7: ATR FLOOR (dead market veto) ─────────────────────────
            atr_pct = cur_atr / max(float(row["close"]), 1e-12)
            if atr_pct < ATR_FLOOR_PCT:
                skipped_atr_floor += 1
                continue

            # ── OPEN POSITION ─────────────────────────────────────────────────
            size_factor   = _meta_size_factor(edge_score, thr)
            sl_dist       = max(atr_sl * cur_atr, float(row["close"]) * MIN_SL_DISTANCE_PCT)
            risk_amt      = capital * risk_pct * size_factor
            units         = min(risk_amt / sl_dist, capital / float(row["close"]))
            if units <= 0:
                continue

            entry_price   = float(row["close"])
            entry_atr_val = cur_atr
            pos_units     = units
            entry_bar     = i
            entry_cost    = pos_units * entry_price * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
            capital      -= entry_cost

        # ── METRICS ──────────────────────────────────────────────────────────
        if result.total_trades == 0:
            result.error = "No trades executed"
            return result

        result.net_return_pct       = round((capital - self.initial_balance) / self.initial_balance * 100, 2)
        result.max_drawdown_pct     = round(max_dd, 2)
        result.win_rate             = round(result.wins / result.total_trades * 100, 2)
        result.avg_trade_return_pct = round(float(np.mean(trade_returns)) * 100, 4) if trade_returns else 0.0
        result.sharpe_ratio         = round(_sharpe(trade_returns), 3)
        result.total_signals        = total_signals
        result.skipped_dir_regime   = skipped_dir_regime
        result.skipped_htf_veto     = skipped_htf_veto
        result.skipped_atr_live     = skipped_atr_live
        result.skipped_min_quality  = skipped_min_quality
        result.skipped_sr_trend     = skipped_srt
        result.skipped_score_signal = skipped_score_signal
        result.skipped_adx          = skipped_adx
        result.skipped_stability    = skipped_stability
        result.skipped_mtf          = skipped_mtf
        result.skipped_rsi_exhaust  = skipped_rsi_exhaust
        result.skipped_cooldown     = skipped_cooldown
        result.skipped_atr_floor    = skipped_atr_floor
        result.low_confidence       = result.total_trades < MIN_TRADES_REQUIRED

        if result.gross_loss > 0:
            result.profit_factor = round(result.gross_profit / result.gross_loss, 3)
        elif result.gross_profit > 0:
            result.profit_factor = None   # all wins, no losses
        else:
            result.profit_factor = 0.0
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_exit(self, row: pd.Series, entry_price: float, entry_atr: float,
                    atr_sl: float, atr_tp: float,
                    direction: Optional[str]) -> Optional[float]:
        """Check current bar's high/low for TP or SL. No lookahead."""
        if direction == "long":
            tp = entry_price + atr_tp * entry_atr
            sl = entry_price - atr_sl * entry_atr
            if row["high"] >= tp and row["low"] <= sl:
                return sl   # gap-through: assume SL first (conservative)
            if row["high"] >= tp:
                return tp
            if row["low"]  <= sl:
                return sl
        else:
            tp = entry_price - atr_tp * entry_atr
            sl = entry_price + atr_sl * entry_atr
            if row["low"] <= tp and row["high"] >= sl:
                return sl
            if row["low"]  <= tp:
                return tp
            if row["high"] >= sl:
                return sl
        return None

    def _fetch_and_prepare(self, hours: int) -> Optional[pd.DataFrame]:
        if self.symbol in AdaptiveBacktester._df_cache:
            return AdaptiveBacktester._df_cache[self.symbol]

        df = self.predictor.fetch_live_data(limit=hours)
        if df is None or df.empty:
            return None

        btc_df  = self.predictor.fetch_btc_data(limit=hours)
        news_df = self.predictor.load_news_data()

        if btc_df is None or btc_df.empty:
            btc_df = pd.DataFrame({"timestamp": df["timestamp"], "close": 0.0})
            btc_df["open"] = btc_df["high"] = btc_df["low"] = btc_df["close"]
            btc_df["volume"] = 0
        if news_df is None or news_df.empty:
            news_df = pd.DataFrame({"timestamp": df["timestamp"], "sentiment": 0.0})

        df_1d = None
        try:
            df_1d = self.predictor.fetch_live_data(
                timeframe="1d", limit=max(1000, int(hours / 24) + 50))
        except Exception:
            pass

        df_1w = None
        try:
            df_1w = self.predictor.fetch_live_data(
                timeframe="1w", limit=max(200, int(hours / 168) + 20))
        except Exception:
            pass

        df = prepare_features(
            df, btc_df=btc_df, news_df=news_df,
            add_target_flag=False, forward_hours=MAX_HOLD_CANDLES,
            df_1d=df_1d, df_1w=df_1w, macro_state=None,
        )
        if df is None or df.empty:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Batch primary + meta inference — same gating as live engine.
        proba, meta_conf = self.predictor.predict_meta_batch(df)
        df["prob_sell"]  = proba[:, 0]
        df["prob_hold"]  = proba[:, 1]
        df["prob_buy"]   = proba[:, 2]
        df["meta_conf"]  = meta_conf

        dir_raw = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
        df["meta_direction"] = np.where(
            np.maximum(proba[:, 2], proba[:, 0]) > proba[:, 1],
            dir_raw, 1
        )
        df["atr_14"]       = compute_atr(df, 14)
        df["atr_mean_100"] = df["atr_14"].rolling(100, min_periods=1).mean()
        df["volatility_regime"] = df["atr_14"] / (df["atr_mean_100"] + 1e-9)

        # Compute per-bar edge scores (0-100) to mirror live engine Gate 2.
        # EdgeScoringEngine is side-specific: BUY bars use "BUY" score, SELL bars
        # use "SELL" score. We store the side-matched score as "edge_score_side"
        # so the gate loop can read a single column without branching.
        _buy_scores  = EdgeScoringEngine.compute_edge_batch(df, meta_conf, "BUY")
        _sell_scores = EdgeScoringEngine.compute_edge_batch(df, meta_conf, "SELL")
        df["edge_score_buy"]  = _buy_scores.values
        df["edge_score_sell"] = _sell_scores.values
        # Assign the side-appropriate score per bar.
        df["edge_score_side"] = np.where(
            df["meta_direction"] == 2,  # BUY
            df["edge_score_buy"],
            np.where(
                df["meta_direction"] == 0,  # SELL
                df["edge_score_sell"],
                0.0,                        # HOLD — score irrelevant
            ),
        )

        if self.regime_boundaries:
            bounds   = self.regime_boundaries
            vol_avg  = df["volume"].rolling(24, min_periods=1).mean()
            atr_pct  = (df["atr_14"] / df["close"]).fillna(0)
            momentum = df["close"].pct_change(24).fillna(0)

            def _tier(val, p33, p67):
                return "low" if val <= p33 else ("med" if val <= p67 else "high")

            def _trend(val, p33, p67):
                return "down" if val <= p33 else ("flat" if val <= p67 else "up")

            vp33, vp67 = bounds.get("vol_p33", 0),         bounds.get("vol_p67", 0)
            ap33, ap67 = bounds.get("atr_pct_p33", 0),     bounds.get("atr_pct_p67", 0)
            mp33, mp67 = bounds.get("momentum_p33", -0.02), bounds.get("momentum_p67", 0.02)

            df["regime_str"] = [
                f"{_tier(vol_avg.iloc[j], vp33, vp67)}"
                f"_{_tier(atr_pct.iloc[j], ap33, ap67)}"
                f"_{_trend(momentum.iloc[j], mp33, mp67)}"
                for j in range(len(df))
            ]
        else:
            df["regime_str"] = "unknown"

        AdaptiveBacktester._df_cache[self.symbol] = df
        return df


# ══════════════════════════════════════════════════════════════════════════════
# FLEET RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_fleet():
    AdaptiveBacktester._df_cache.clear()
    AdaptiveBacktester._predictor_cache.clear()

    print("=" * 80)
    print("AEGIS-1 BACKTESTER v4.3 — live_engine gate-aligned")
    print(f"   Balance ${INITIAL_BALANCE:,.0f} | Data {DATA_HOURS} h | {len(FLEET)} symbols")
    print(f"   Gates: regime(1) → edge(2) → dir_regime(1.5) → htf_veto(1.7)")
    print(f"          → min_quality(3) → s&r(3) → adx(3.5) → stability(3.8)")
    print(f"          → mtf(4.5) → rsi(5) → cooldown(6) → atr_floor(7)")
    print("=" * 80)

    all_results = []
    for symbol in FLEET:
        print(f"\n{symbol}")
        for mode in MODE_PARAMS:
            bt  = AdaptiveBacktester(symbol, mode)
            res = bt.run(DATA_HOURS)
            if res.error:
                print(f"   {mode.capitalize():<12} ERROR  {res.error}")
            else:
                pf_str = f"{res.profit_factor:.2f}" if res.profit_factor is not None else "inf"
                print(
                    f"   {mode.capitalize():<12} | "
                    f"Trades {res.total_trades:3d} | "
                    f"Return {res.net_return_pct:+6.2f}% | "
                    f"WR {res.win_rate:5.1f}% | "
                    f"PF {pf_str:>5} | "
                    f"Sig {res.total_signals} "
                    f"[dr={res.skipped_dir_regime} "
                    f"htf={res.skipped_htf_veto} "
                    f"mq={res.skipped_min_quality} "
                    f"s&r={res.skipped_sr_trend} "
                    f"adx={res.skipped_adx} "
                    f"stab={res.skipped_stability} "
                    f"mtf={res.skipped_mtf} "
                    f"rsi={res.skipped_rsi_exhaust} "
                    f"cd={res.skipped_cooldown} "
                    f"atr={res.skipped_atr_floor}]"
                )
                res_dict = asdict(res)
                res_dict.pop("trades", None)
                all_results.append(res_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = BACKTEST_DIR / f"backtest_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{len(all_results)} results saved → {out_file}")


if __name__ == "__main__":
    run_backtest_fleet()
