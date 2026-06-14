"""
signal_manager.py — Signal Cooldown Tracker + Beginner Guidance Generator

Manages:
- Per-token per-mode cooldown (prevents back-to-back signals)
- Confidence gate enforcement
- Beginner-friendly trade setup generation (entry zone, SL, TP1/2/3, rationale)
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trader_model.trader_config import (
    MODES, RISK_PROFILES, STRATEGY_NAMES, TRADER_STATE_PATH, TRADER_RECORD_PATH,
)
from src.ml.feature_engine import compute_atr

log = logging.getLogger(__name__)


# ── Cooldown State ────────────────────────────────────────────────────────────

def _load_state() -> Dict[str, Any]:
    if TRADER_STATE_PATH.exists():
        try:
            with open(TRADER_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    TRADER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADER_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def is_on_cooldown(symbol: str, mode: str) -> bool:
    """Return True if the token is still in cooldown for the given mode."""
    state = _load_state()
    key   = f"{symbol}_{mode}"
    last  = state.get(key)
    if last is None:
        return False
    last_time     = datetime.fromisoformat(last)
    cooldown_mins = MODES[mode]['cooldown_minutes']
    elapsed_mins  = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
    return elapsed_mins < cooldown_mins


def record_signal(symbol: str, mode: str) -> None:
    """Mark a signal as fired for this symbol/mode to start the cooldown timer."""
    state = _load_state()
    state[f"{symbol}_{mode}"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)


def cooldown_remaining_minutes(symbol: str, mode: str) -> int:
    """How many minutes remain on the cooldown (0 if not cooling down)."""
    state = _load_state()
    key   = f"{symbol}_{mode}"
    last  = state.get(key)
    if last is None:
        return 0
    last_time     = datetime.fromisoformat(last)
    cooldown_mins = MODES[mode]['cooldown_minutes']
    elapsed_mins  = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
    remaining     = max(0, cooldown_mins - elapsed_mins)
    return int(remaining)


# ── Confidence Gate ────────────────────────────────────────────────────────────

def passes_confidence_gate(confidence: float, risk_profile: str) -> bool:
    """Return True if confidence meets the risk profile's minimum threshold."""
    min_conf = RISK_PROFILES.get(risk_profile, RISK_PROFILES['balanced'])['min_confidence']
    return confidence >= min_conf


# ── Beginner Trade Setup Guide ─────────────────────────────────────────────────

_STRATEGY_EXPLANATIONS = {
    'vwap_bounce':                  'Price is near the Volume Weighted Average Price with above-average volume, suggesting a bounce.',
    'bb_squeeze_breakout':          'Bollinger Bands are tightly squeezed and price is breaking out — a volatility expansion is beginning.',
    'ema_cross_fast':               'Short-term EMA (3) has crossed the EMA (8) in the signal direction, confirmed by momentum.',
    'macd_zero_cross':              'MACD line has crossed the zero line, signaling a shift in momentum direction.',
    'volume_spike_entry':           'Volume spiked 2× above average with a strong directional candle body.',
    'rsi_momentum_cross':           'RSI has crossed the 50-line, indicating a momentum shift.',
    'order_flow_imbalance':         'Buy/sell order flow is imbalanced, suggesting institutional accumulation or distribution.',
    'support_resistance_retest':    'Price is retesting a key support or resistance level with a reversal candle.',
    'opening_range_breakout':       'Price has broken out of the recent 3-candle opening range with volume confirmation.',
    'momentum_burst_candles':       '3 consecutive momentum candles with increasing volume signal a strong directional move.',
    'trend_continuation_pullback':  'Strong trend detected (ADX > 25) — price is pulling back to EMA21, a classic entry point.',
    'ema21_bounce':                 'Price is bouncing off the EMA21 in a trending market, offering a low-risk entry.',
    'vwap_deviation_reversion':     'Price is more than 1.5 standard deviations from VWAP — mean reversion likely.',
    'rsi_divergence':               'RSI divergence detected: price is making a new high/low but RSI is not confirming it.',
    'macd_histogram_reversal':      'MACD histogram has reversed direction after building momentum, signaling a turn.',
    'bb_squeeze_expansion':         'Bollinger Bands were squeezed and are now expanding in the signal direction.',
    'oi_price_divergence':          'Open interest is diverging from price, suggesting a potential squeeze or flush.',
    'news_momentum_catalyst':       'Strong sentiment or news catalyst detected — momentum may continue in this direction.',
    'session_high_low_break':       'Price has broken out of the previous session (Asian/London) range with volume.',
    'ema_golden_death_cross':       'EMA50 has crossed EMA200 (golden or death cross), indicating a major trend change.',
    'supply_demand_zone_retest':    'Price is revisiting a high-volume supply or demand zone — strong reaction expected.',
    'fibonacci_retracement_entry':  'Price is at a key Fibonacci retracement level (38.2%, 50%, or 61.8%) of the recent swing.',
    'higher_timeframe_alignment':   '4h and daily trends are aligned in the same direction — high-probability setup.',
    'market_structure_break':       'Price has broken above the previous swing high (or below swing low) with volume — structure shift.',
    'liquidation_cascade_fade':     'A sharp liquidation move occurred with extreme volume — fade the cascade as price likely recovers.',
}

_MODE_CONTEXT = {
    'scalping': {
        'hold_time':    '15–60 minutes',
        'timeframe':    '5-minute chart',
        'tip':          'Use tight stop losses. Scale out 50% at TP1, let rest run to TP2.',
    },
    'scalping_15m': {
        'hold_time':    '30–120 minutes',
        'timeframe':    '15-minute chart',
        'tip':          'Use tight stop losses. Scale out 50% at TP1, let rest run to TP2.',
    },
    'intraday': {
        'hold_time':    '2–8 hours',
        'timeframe':    '1-hour chart',
        'tip':          'Check the 4h trend before entering. Move SL to breakeven after TP1.',
    },
    'swing': {
        'hold_time':    '2–5 days',
        'timeframe':    '4-hour chart',
        'tip':          'Check weekly trend alignment. Use 1–2% position size. Be patient for the setup.',
    },
}

_RISK_CONTEXT = {
    'conservative': 'Use 1% of your portfolio. Only take A+ setups with confluence.',
    'balanced':     'Use 2% of your portfolio. Look for 2+ confirming signals.',
    'aggressive':   'Use up to 3% of your portfolio. Ensure you have a clear invalidation level.',
}


def generate_beginner_guidance(
    symbol: str,
    direction: str,        # 'BUY' or 'SELL'
    mode: str,
    risk_profile: str,
    confidence: float,
    current_price: float,
    top_strategies: List[str],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate a beginner-friendly trade setup guide.

    Returns a dict with entry zone, SL, TP levels, rationale, and tips.
    """
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES['balanced'])
    sl_mult = profile['sl_atr_mult']
    tp1_mult = profile['tp1_atr_mult']
    tp2_mult = profile['tp2_atr_mult']
    tp3_mult = profile['tp3_atr_mult']

    # ATR-based levels
    atr_value = 0.0
    if df is not None and len(df) >= 15:
        try:
            atr_series = compute_atr(df, 14)
            atr_value  = float(atr_series.iloc[-1])
        except Exception:
            atr_value = current_price * 0.005  # fallback: 0.5% of price

    if atr_value == 0:
        atr_value = current_price * 0.005

    if direction == 'BUY':
        entry_low  = round(current_price * 0.9990, 8)
        entry_high = round(current_price * 1.0010, 8)
        sl_price   = round(current_price - sl_mult * atr_value, 8)
        tp1_price  = round(current_price + tp1_mult * atr_value, 8)
        tp2_price  = round(current_price + tp2_mult * atr_value, 8)
        tp3_price  = round(current_price + tp3_mult * atr_value, 8)
    else:  # SELL
        entry_low  = round(current_price * 0.9990, 8)
        entry_high = round(current_price * 1.0010, 8)
        sl_price   = round(current_price + sl_mult * atr_value, 8)
        tp1_price  = round(current_price - tp1_mult * atr_value, 8)
        tp2_price  = round(current_price - tp2_mult * atr_value, 8)
        tp3_price  = round(current_price - tp3_mult * atr_value, 8)

    sl_pct  = round(abs(current_price - sl_price) / current_price * 100, 2)
    tp1_pct = round(abs(current_price - tp1_price) / current_price * 100, 2)
    rr      = round(tp1_pct / sl_pct, 2) if sl_pct > 0 else 0

    # Strategy explanation
    top_strat = top_strategies[0] if top_strategies else ''
    explanation = _STRATEGY_EXPLANATIONS.get(top_strat, 'Multiple confluence signals detected.')
    supporting  = [_STRATEGY_EXPLANATIONS.get(s, '') for s in top_strategies[1:3] if s]

    mode_ctx   = _MODE_CONTEXT.get(mode, _MODE_CONTEXT['intraday'])
    risk_ctx   = _RISK_CONTEXT.get(risk_profile, _RISK_CONTEXT['balanced'])

    return {
        'direction':    direction,
        'entry_zone':   {'low': entry_low, 'high': entry_high},
        'stop_loss':    sl_price,
        'take_profit':  {'tp1': tp1_price, 'tp2': tp2_price, 'tp3': tp3_price},
        'sl_pct':       sl_pct,
        'tp1_pct':      tp1_pct,
        'risk_reward':  rr,
        'confidence':   round(confidence * 100, 1),
        'primary_signal':   top_strat,
        'rationale':        explanation,
        'supporting_signals': supporting,
        'hold_time':    mode_ctx['hold_time'],
        'chart':        mode_ctx['timeframe'],
        'execution_tip': mode_ctx['tip'],
        'sizing_advice': risk_ctx,
        'beginner_steps': [
            f"1. Open the {mode_ctx['timeframe']} chart for {symbol}",
            f"2. Wait for price to reach entry zone: {entry_low}–{entry_high}",
            f"3. Set stop loss at {sl_price} ({sl_pct}% risk)",
            f"4. Set TP1 at {tp1_price} (take 50% profits here)",
            f"5. Move SL to breakeven after TP1 hits",
            f"6. Let remaining position run to TP2 ({tp2_price}) or TP3 ({tp3_price})",
            f"7. {mode_ctx['tip']}",
        ],
    }


# ── Track Record ──────────────────────────────────────────────────────────────

def append_trader_signal(signal: Dict[str, Any]) -> None:
    """Append a new signal to the trader track record JSON."""
    TRADER_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {'signals': [], 'stats': {}}
    if TRADER_RECORD_PATH.exists():
        try:
            with open(TRADER_RECORD_PATH) as f:
                record = json.load(f)
        except Exception:
            pass

    record.setdefault('signals', []).append(signal)
    # Keep last 1000 signals
    record['signals'] = record['signals'][-1000:]

    with open(TRADER_RECORD_PATH, 'w') as f:
        json.dump(record, f, indent=2)


def compute_trader_stats() -> Dict[str, Any]:
    """Compute performance statistics from trader track record."""
    if not TRADER_RECORD_PATH.exists():
        return {}

    with open(TRADER_RECORD_PATH) as f:
        record = json.load(f)

    signals = record.get('signals', [])
    if not signals:
        return {}

    stats: Dict[str, Any] = {}
    for mode in ['scalping', 'scalping_15m', 'intraday', 'swing']:
        mode_sigs = [s for s in signals if s.get('mode') == mode]
        closed    = [s for s in mode_sigs if s.get('outcome') in ('WIN', 'LOSS')]
        wins      = [s for s in closed if s.get('outcome') == 'WIN']
        stats[mode] = {
            'total':    len(mode_sigs),
            'closed':   len(closed),
            'wins':     len(wins),
            'win_rate': round(len(wins) / len(closed), 3) if closed else 0.0,
        }

    return stats
