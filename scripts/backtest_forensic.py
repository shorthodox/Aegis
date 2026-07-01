#!/usr/bin/env python3
"""
backtest_forensic.py — AEGIS Gate Forensic Auditor

Runs the full backtest gate sequence bar-by-bar and records exactly
which gate killed each signal, the values that caused the block, and
the full signal funnel from raw model output to opened trade.

Usage
-----
  python scripts/backtest_forensic.py                     # first 5 FLEET symbols
  python scripts/backtest_forensic.py --symbol ETH/USDT   # single symbol
  python scripts/backtest_forensic.py --all               # full 63-token fleet

Outputs (in logs/forensic/<timestamp>/)
-----------------------------------------
  <SYMBOL>_audit.csv      — one row per bar where model wanted to trade
  <SYMBOL>_funnel.txt     — gate-by-gate kill count and throughput %
  summary.json            — fleet-wide gate funnel aggregated

Reading the funnel
------------------
  Each row in the audit CSV has:
    blocked_at   — gate name that killed the signal (or "TRADE" if it passed)
    block_reason — the specific sub-condition that fired
    Values for every gate: regime_str, reg_buy_ok, edge_score, rsi_val, etc.

  When blocked_at is "TRADE" the row represents an opened position;
  outcome (win/loss/timeout) and pnl are filled after exit.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings

# Force UTF-8 on Windows consoles that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cost_aware_backtester import (
    AdaptiveBacktester,
    FLEET, DATA_HOURS, MAX_HOLD_CANDLES,
    MTF_PRIOR_BARS, MTF_RECENT_BARS, MTF_PRIOR_FRAC, MTF_REV_FRAC,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_ACCEL_BUY, RSI_ACCEL_SELL,
    COOLDOWN_BARS, ATR_FLOOR_PCT,
    HMM_BEAR_PROXY, HMM_BULL_PROXY, HTF_HARD_OPPOSE,
    MIN_QUALITY_SCORE, SIGNAL_BYPASS_EDGE, STABILITY_WINDOW,
    _dir_fraction,
)

FORENSIC_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\forensic")
FORENSIC_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

GATE_NAMES = [
    "HOLD",              # model output was HOLD — no signal to evaluate
    "G1_REGIME",         # regime unknown / skipped
    "G2_EDGE",           # edge_score < threshold OR regime buy_ok/sell_ok = False
    "G1_5_DIR_REGIME",   # HMM proxy: macro_trend_1d opposing direction
    "G1_7_HTF",          # HTF macro hard veto: weekly+daily both oppose
    "G3_MIN_QUALITY",    # edge_score < MIN_QUALITY_SCORE (70)
    "G3_SR",             # at resistance (BUY) or at support (SELL)
    "G3_TREND",          # HTF daily trend opposing direction
    "G3_CONFLUENCE",     # total_confluence opposing direction
    "G3_CHOPPINESS",     # choppiness > 60
    "G3_BOS",            # break-of-structure opposing direction
    "G3_5_ADX",          # ADX < 25 (weak trend)
    "G3_8_STABILITY",    # signal stability: last N bars not same direction
    "G4_5_MTF",          # two-zone reversal gate blocked
    "G5_RSI_LEVEL",      # RSI overbought/oversold + slope
    "G5_RSI_ACCEL",      # RSI acceleration
    "G6_COOLDOWN",       # post-trade cooldown active
    "G7_ATR_FLOOR",      # ATR too low (dead market)
    "TRADE",             # passed all gates — position opened
]


@dataclass
class SignalTrace:
    symbol:       str
    bar_index:    int
    timestamp:    str
    direction:    str       # "long" / "short"
    meta_conf:    float
    edge_score:   float
    thr:          float
    price:        float
    atr:          float
    regime_str:   str
    reg_buy_ok:   Optional[bool]
    reg_sell_ok:  Optional[bool]
    reg_skipped:  Optional[bool]
    # Gate 3 values
    at_res:       bool = False
    at_sup:       bool = False
    trend_1d:     float = 0.0
    total_conf:   float = 0.0
    choppiness:   float = 50.0
    bos:          float = 0.0
    # Gate 4 values
    ev:           float = 0.0
    ev_floor:     float = 0.0
    # Gate 4.5 values
    prior_frac:   float = 0.0
    rev_frac:     float = 0.0
    # Gate 5 values
    rsi_val:      float = 50.0
    rsi_slope:    float = 0.0
    rsi_accel:    float = 0.0
    # Gate 6 values
    bars_since_exit: int = -1
    # Gate 7 values
    atr_pct:      float = 0.0
    # Verdict
    blocked_at:   str   = ""
    block_reason: str   = ""
    # Trade outcome (filled only when blocked_at == "TRADE")
    trade_exit_bar:   int   = -1
    trade_exit_price: float = 0.0
    trade_pnl:        float = 0.0
    trade_outcome:    str   = ""   # "WIN" / "LOSS" / "TIMEOUT"


@dataclass
class GateFunnel:
    symbol:          str
    total_bars:      int = 0
    model_buy:       int = 0
    model_sell:      int = 0
    hold:            int = 0
    g1_regime:       int = 0
    g2_edge:         int = 0
    g1_5_dir_regime: int = 0
    g1_7_htf:        int = 0
    g3_min_quality:  int = 0
    g3_sr:           int = 0
    g3_trend:        int = 0
    g3_confluence:   int = 0
    g3_choppiness:   int = 0
    g3_bos:          int = 0
    g3_5_adx:        int = 0
    g3_8_stability:  int = 0
    g4_5_mtf:        int = 0
    g5_rsi_level:    int = 0
    g5_rsi_accel:    int = 0
    g6_cooldown:     int = 0
    g7_atr_floor:    int = 0
    trades:          int = 0
    wins:            int = 0
    losses:          int = 0
    timeouts:        int = 0


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ForensicEngine:
    """
    Replays the backtest gate sequence bar-by-bar with full audit logging.
    Uses the same AdaptiveBacktester._fetch_and_prepare() so data is identical.
    """

    def __init__(self, symbol: str, mode: str = "balanced"):
        self.bt     = AdaptiveBacktester(symbol, mode)
        self.symbol = symbol
        self.mode   = mode

    def run(self, hours: int = DATA_HOURS) -> tuple[GateFunnel, List[SignalTrace]]:
        df = self.bt._fetch_and_prepare(hours)
        if df is None or df.empty:
            return GateFunnel(symbol=self.symbol), []

        closes       = df["close"].to_numpy(dtype=float)
        opens        = df["open"].to_numpy(dtype=float)
        meta_dir_arr = df["meta_direction"].to_numpy(dtype=int)   # Gate 3.8

        atr_sl = self.bt.atr_mult
        atr_tp = self.bt.atr_mult * self.bt.params["rr_ratio"]

        funnel = GateFunnel(symbol=self.symbol, total_bars=len(df))
        traces: List[SignalTrace] = []

        # Live trade tracking (to fill trade outcomes later)
        pos_units     = 0.0
        entry_price   = 0.0
        entry_atr     = 0.0
        entry_bar     = -1
        last_exit_bar = -1
        entry_trace_idx = -1
        direction_open: Optional[str] = None

        n = len(df)

        for i in range(MTF_PRIOR_BARS + MTF_RECENT_BARS, n - MAX_HOLD_CANDLES - 1):
            row = df.iloc[i]

            # ── EXIT tracking ─────────────────────────────────────────────────
            if pos_units != 0.0:
                bars_held = i - entry_bar
                exit_px   = self.bt._check_exit(row, entry_price, entry_atr,
                                                atr_sl, atr_tp, direction_open)
                outcome   = ""
                if exit_px is not None:
                    outcome = "TP_SL"
                elif bars_held >= MAX_HOLD_CANDLES:
                    exit_px = float(row["close"])
                    outcome = "TIMEOUT"

                if exit_px is not None:
                    if direction_open == "long":
                        pnl = pos_units * (exit_px - entry_price)
                    else:
                        pnl = pos_units * (entry_price - exit_px)

                    if entry_trace_idx >= 0 and entry_trace_idx < len(traces):
                        t = traces[entry_trace_idx]
                        t.trade_exit_bar   = i
                        t.trade_exit_price = round(exit_px, 6)
                        t.trade_pnl        = round(pnl, 4)
                        if pnl >= 0:
                            t.trade_outcome = "WIN"
                            funnel.wins    += 1
                        else:
                            t.trade_outcome = "LOSS"
                            funnel.losses  += 1
                        if outcome == "TIMEOUT":
                            t.trade_outcome = "TIMEOUT"
                            funnel.timeouts += 1

                    last_exit_bar = i
                    pos_units     = 0.0
                    entry_trace_idx = -1
                continue

            # ── Determine model direction ──────────────────────────────────────
            meta_dir  = int(row.get("meta_direction", 1))
            meta_conf = float(row.get("meta_conf", 0.0))

            if meta_dir not in (0, 2):
                funnel.hold += 1
                funnel.model_buy  += (1 if meta_dir == 2 else 0)
                funnel.model_sell += (1 if meta_dir == 0 else 0)
                continue

            if meta_dir == 2:
                funnel.model_buy += 1
                direction = "long"
            else:
                funnel.model_sell += 1
                direction = "short"

            edge_score = float(row.get("edge_score_side", 0.0))
            price      = float(row["close"])
            cur_atr    = float(row["atr_14"]) if float(row["atr_14"]) > 0 else price * 0.01

            # ── GATE 1: REGIME ────────────────────────────────────────────────
            regime_str = str(row.get("regime_str", "unknown"))
            if self.bt.regime_thresholds:
                reg = self.bt.regime_thresholds.get(regime_str, {})
                if not reg or reg.get("skipped"):
                    # Emit a trace so the funnel audit captures the skip reason
                    t = SignalTrace(
                        symbol=self.symbol, bar_index=i,
                        timestamp=str(row.get("timestamp", "")),
                        direction=direction, meta_conf=round(meta_conf, 5),
                        edge_score=round(edge_score, 2), thr=0.0,
                        price=round(price, 6), atr=round(cur_atr, 6),
                        regime_str=regime_str,
                        reg_buy_ok=reg.get("buy_ok") if reg else None,
                        reg_sell_ok=reg.get("sell_ok") if reg else None,
                        reg_skipped=reg.get("skipped") if reg else None,
                        blocked_at="G1_REGIME",
                        block_reason=f"regime={regime_str!r} skipped={reg.get('skipped') if reg else None} reg_empty={not reg}",
                    )
                    funnel.g1_regime += 1
                    traces.append(t)
                    continue
                _buy_ok  = bool(reg.get("buy_ok",  True))
                _sell_ok = bool(reg.get("sell_ok", True))
                # Per-regime thresholds (0-1 → normalize to 0-100)
                _r_buy_raw  = reg.get("buy_threshold")
                _r_sell_raw = reg.get("sell_threshold")
                _thr_buy  = ((_r_buy_raw  * 100.0 if _r_buy_raw  < 2.0 else _r_buy_raw)
                             if _r_buy_raw  is not None else self.bt.thr_buy)
                _thr_sell = ((_r_sell_raw * 100.0 if _r_sell_raw < 2.0 else _r_sell_raw)
                             if _r_sell_raw is not None else self.bt.thr_sell)
            else:
                reg       = {}
                _buy_ok   = True
                _sell_ok  = True
                _thr_buy  = self.bt.thr_buy
                _thr_sell = self.bt.thr_sell

            thr = _thr_buy if direction == "long" else _thr_sell

            t = SignalTrace(
                symbol=self.symbol, bar_index=i,
                timestamp=str(row.get("timestamp", "")),
                direction=direction,
                meta_conf=round(meta_conf, 5),
                edge_score=round(edge_score, 2),
                thr=round(thr, 2),
                price=round(price, 6),
                atr=round(cur_atr, 6),
                regime_str=regime_str,
                reg_buy_ok=reg.get("buy_ok"),
                reg_sell_ok=reg.get("sell_ok"),
                reg_skipped=reg.get("skipped"),
            )

            # ── GATE 2: EDGE SCORE + REGIME TRADEABILITY ─────────────────────
            # Per-regime buy_ok/sell_ok are the primary tradeability authority.
            # Global tradeable_buy/sell is only used when no regime_thresholds.
            if direction == "long":
                _blocked = not _buy_ok or edge_score < _thr_buy
                _reason  = (f"reg.buy_ok=False regime={regime_str!r}" if not _buy_ok
                            else f"edge={edge_score:.1f} < thr={_thr_buy:.1f}")
            else:
                _blocked = not _sell_ok or edge_score < _thr_sell
                _reason  = (f"reg.sell_ok=False regime={regime_str!r}" if not _sell_ok
                            else f"edge={edge_score:.1f} < thr={_thr_sell:.1f}")
            if _blocked:
                t.blocked_at   = "G2_EDGE"
                t.block_reason = _reason
                funnel.g2_edge += 1
                traces.append(t)
                continue

            # ── GATE 1.5: DIRECTION-REGIME HARD VETO (HMM proxy) ─────────────
            _t1d_proxy = float(row.get("macro_trend_1d", 0.0))
            if direction == "long"  and _t1d_proxy < HMM_BEAR_PROXY:
                t.blocked_at   = "G1_5_DIR_REGIME"
                t.block_reason = f"macro_trend_1d={_t1d_proxy:.3f} < {HMM_BEAR_PROXY} (bear proxy)"
                funnel.g1_5_dir_regime += 1; traces.append(t); continue
            if direction == "short" and _t1d_proxy > HMM_BULL_PROXY:
                t.blocked_at   = "G1_5_DIR_REGIME"
                t.block_reason = f"macro_trend_1d={_t1d_proxy:.3f} > {HMM_BULL_PROXY} (bull proxy)"
                funnel.g1_5_dir_regime += 1; traces.append(t); continue

            # ── GATE 1.7: HTF MACRO HARD VETO ────────────────────────────────
            _macro_w = float(row.get("macro_trend_1w", 0.0))
            _macro_d = float(row.get("macro_trend_1d", 0.0))
            if direction == "long"  and _macro_w < -HTF_HARD_OPPOSE and _macro_d < -HTF_HARD_OPPOSE:
                t.blocked_at   = "G1_7_HTF"
                t.block_reason = f"macro_w={_macro_w:.3f} macro_d={_macro_d:.3f} both < -{HTF_HARD_OPPOSE}"
                funnel.g1_7_htf += 1; traces.append(t); continue
            if direction == "short" and _macro_w >  HTF_HARD_OPPOSE and _macro_d >  HTF_HARD_OPPOSE:
                t.blocked_at   = "G1_7_HTF"
                t.block_reason = f"macro_w={_macro_w:.3f} macro_d={_macro_d:.3f} both > {HTF_HARD_OPPOSE}"
                funnel.g1_7_htf += 1; traces.append(t); continue

            # ── GATE 3: MIN QUALITY SCORE FLOOR ──────────────────────────────
            if edge_score < MIN_QUALITY_SCORE:
                t.blocked_at   = "G3_MIN_QUALITY"
                t.block_reason = f"edge={edge_score:.1f} < MIN_QUALITY_SCORE={MIN_QUALITY_SCORE}"
                funnel.g3_min_quality += 1; traces.append(t); continue

            # ── GATE 3: S&R / TREND / CONFLUENCE / CHOPPINESS / BOS ──────────
            is_high_conviction = edge_score >= float(self.bt.override_thr)
            at_res  = bool(row.get("is_at_resistance", 0))
            at_sup  = bool(row.get("is_at_support",    0))
            t1d     = float(row.get("macro_trend_1d",    0.0))
            tconf   = float(row.get("total_confluence",  0.0))
            chop    = float(row.get("choppiness",       50.0))
            bos     = float(row.get("bos_state",         0.0))

            t.at_res = at_res; t.at_sup = at_sup; t.trend_1d = t1d
            t.total_conf = tconf; t.choppiness = chop; t.bos = bos

            if not is_high_conviction:
                if (direction == "long" and at_res) or (direction == "short" and at_sup):
                    t.blocked_at   = "G3_SR"
                    t.block_reason = f"at_resistance={at_res} at_support={at_sup}"
                    funnel.g3_sr += 1; traces.append(t); continue
                if (direction == "long" and t1d < -0.2) or (direction == "short" and t1d > 0.2):
                    t.blocked_at   = "G3_TREND"
                    t.block_reason = f"trend_1d={t1d:.3f} opposing {direction}"
                    funnel.g3_trend += 1; traces.append(t); continue
                if (direction == "long" and tconf < -0.1) or (direction == "short" and tconf > 0.1):
                    t.blocked_at   = "G3_CONFLUENCE"
                    t.block_reason = f"total_conf={tconf:.3f} opposing {direction}"
                    funnel.g3_confluence += 1; traces.append(t); continue
                if chop > 60:
                    t.blocked_at   = "G3_CHOPPINESS"
                    t.block_reason = f"choppiness={chop:.1f} > 60"
                    funnel.g3_choppiness += 1; traces.append(t); continue
                if (direction == "long" and bos < 0) or (direction == "short" and bos > 0):
                    t.blocked_at   = "G3_BOS"
                    t.block_reason = f"bos_state={bos:.2f} opposing {direction}"
                    funnel.g3_bos += 1; traces.append(t); continue

            # ── GATE 3.5: ADX TREND STRENGTH ─────────────────────────────────
            if edge_score < SIGNAL_BYPASS_EDGE:
                adx_val = float(row.get("adx_14", 25.0))
                if adx_val < 25.0:
                    t.blocked_at   = "G3_5_ADX"
                    t.block_reason = f"adx={adx_val:.1f} < 25 (weak trend)"
                    funnel.g3_5_adx += 1; traces.append(t); continue

            # ── GATE 3.8: SIGNAL STABILITY ────────────────────────────────────
            if edge_score < SIGNAL_BYPASS_EDGE:
                _meta_req = 2 if direction == "long" else 0
                _stable   = all(
                    meta_dir_arr[i - back] == _meta_req
                    for back in range(1, STABILITY_WINDOW + 1)
                    if (i - back) >= 0
                )
                if not _stable:
                    t.blocked_at   = "G3_8_STABILITY"
                    t.block_reason = (f"last {STABILITY_WINDOW} bars not all "
                                      f"{'BUY' if direction == 'long' else 'SELL'}")
                    funnel.g3_8_stability += 1; traces.append(t); continue

            # ── GATE 4.5: TWO-ZONE MTF ────────────────────────────────────────
            _ps = i - MTF_PRIOR_BARS - MTF_RECENT_BARS
            _pe = i - MTF_RECENT_BARS
            _rs = i - MTF_RECENT_BARS

            _pc  = closes[_ps:_pe]; _po = opens[_ps:_pe]
            _rc  = closes[_rs:i];   _ro = opens[_rs:i]

            _prior_dir = "bearish" if direction == "long" else "bullish"
            _rev_dir   = "bullish" if direction == "long" else "bearish"

            prior_frac = _dir_fraction(_pc, _po, _prior_dir)
            rev_frac   = _dir_fraction(_rc, _ro, _rev_dir)
            t.prior_frac = round(prior_frac, 3); t.rev_frac = round(rev_frac, 3)

            if not (prior_frac >= MTF_PRIOR_FRAC and rev_frac >= MTF_REV_FRAC):
                t.blocked_at   = "G4_5_MTF"
                t.block_reason = (f"prior({_prior_dir})={prior_frac:.2f}<{MTF_PRIOR_FRAC} OR "
                                  f"rev({_rev_dir})={rev_frac:.2f}<{MTF_REV_FRAC}")
                funnel.g4_5_mtf += 1
                traces.append(t)
                continue

            # ── GATE 5: RSI EXHAUSTION ────────────────────────────────────────
            rsi_val   = float(row.get("rsi_14", 50.0))
            rsi_slope = float(row.get("rsi_slope_14", 0.0))
            rsi_accel = float(row.get("rsi_acceleration_14", 0.0))
            t.rsi_val = rsi_val; t.rsi_slope = rsi_slope; t.rsi_accel = rsi_accel

            if direction == "long":
                if rsi_val >= RSI_OVERBOUGHT and rsi_slope <= 0:
                    t.blocked_at   = "G5_RSI_LEVEL"
                    t.block_reason = f"rsi={rsi_val:.1f}>={RSI_OVERBOUGHT} slope={rsi_slope:.4f}<=0 (top)"
                    funnel.g5_rsi_level += 1; traces.append(t); continue
                if rsi_accel < RSI_ACCEL_BUY:
                    t.blocked_at   = "G5_RSI_ACCEL"
                    t.block_reason = f"rsi_accel={rsi_accel:.4f}<{RSI_ACCEL_BUY} (RSI free-fall)"
                    funnel.g5_rsi_accel += 1; traces.append(t); continue
            else:
                if rsi_val <= RSI_OVERSOLD and rsi_slope >= 0:
                    t.blocked_at   = "G5_RSI_LEVEL"
                    t.block_reason = f"rsi={rsi_val:.1f}<={RSI_OVERSOLD} slope={rsi_slope:.4f}>=0 (bottom)"
                    funnel.g5_rsi_level += 1; traces.append(t); continue
                if rsi_accel > RSI_ACCEL_SELL:
                    t.blocked_at   = "G5_RSI_ACCEL"
                    t.block_reason = f"rsi_accel={rsi_accel:.4f}>{RSI_ACCEL_SELL} (RSI recovering)"
                    funnel.g5_rsi_accel += 1; traces.append(t); continue

            # ── GATE 6: COOLDOWN ──────────────────────────────────────────────
            bse = i - last_exit_bar if last_exit_bar >= 0 else -1
            t.bars_since_exit = bse
            if last_exit_bar >= 0 and (i - last_exit_bar) < COOLDOWN_BARS:
                t.blocked_at   = "G6_COOLDOWN"
                t.block_reason = f"bars_since_exit={bse} < cooldown={COOLDOWN_BARS}"
                funnel.g6_cooldown += 1; traces.append(t); continue

            # ── GATE 7: ATR FLOOR ─────────────────────────────────────────────
            atr_pct = cur_atr / max(price, 1e-12)
            t.atr_pct = round(atr_pct, 6)
            if atr_pct < ATR_FLOOR_PCT:
                t.blocked_at   = "G7_ATR_FLOOR"
                t.block_reason = f"atr_pct={atr_pct:.5f} < floor={ATR_FLOOR_PCT}"
                funnel.g7_atr_floor += 1; traces.append(t); continue

            # ── PASSED ALL GATES ──────────────────────────────────────────────
            t.blocked_at  = "TRADE"
            t.block_reason = f"edge={edge_score:.1f} meta_conf={meta_conf:.4f}"
            funnel.trades += 1
            entry_price   = price
            entry_atr     = cur_atr
            pos_units     = 1.0    # normalised — only tracking direction here
            entry_bar     = i
            entry_trace_idx = len(traces)
            direction_open  = direction
            traces.append(t)

        return funnel, traces


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _funnel_text(funnel: GateFunnel) -> str:
    total_signals = funnel.model_buy + funnel.model_sell
    sep = "+" + "-" * 57 + "+"
    lines = [
        sep,
        f"|  GATE FUNNEL -- {funnel.symbol:<40}|",
        sep,
        f"|  Total 1H bars:         {funnel.total_bars:>6}                        |",
        f"|  Model BUY signals:     {funnel.model_buy:>6}  ({funnel.model_buy/max(funnel.total_bars,1)*100:5.1f}% of bars)        |",
        f"|  Model SELL signals:    {funnel.model_sell:>6}  ({funnel.model_sell/max(funnel.total_bars,1)*100:5.1f}% of bars)        |",
        sep,
    ]
    gates = [
        ("HOLD (no trade signal)",   funnel.hold),
        ("G1    Regime veto",        funnel.g1_regime),
        ("G2    Edge/regime block",  funnel.g2_edge),
        ("G1.5  Dir-regime veto",    funnel.g1_5_dir_regime),
        ("G1.7  HTF macro veto",     funnel.g1_7_htf),
        ("G3    Min quality <70",    funnel.g3_min_quality),
        ("G3    S&R opposition",     funnel.g3_sr),
        ("G3    HTF trend opposing", funnel.g3_trend),
        ("G3    Confluence oppose",  funnel.g3_confluence),
        ("G3    Choppiness > 60",    funnel.g3_choppiness),
        ("G3    BOS opposing",       funnel.g3_bos),
        ("G3.5  ADX < 25",           funnel.g3_5_adx),
        ("G3.8  Signal stability",   funnel.g3_8_stability),
        ("G4.5  MTF reversal gate",  funnel.g4_5_mtf),
        ("G5    RSI level exhaust",  funnel.g5_rsi_level),
        ("G5    RSI accel exhaust",  funnel.g5_rsi_accel),
        ("G6    Post-trade cdwn",    funnel.g6_cooldown),
        ("G7    ATR floor (dead)",   funnel.g7_atr_floor),
    ]
    for label, count in gates:
        pct = count / max(total_signals, 1) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"|  {label:<24}  {count:>5}  {pct:5.1f}%  {bar:<26}|")

    pass_pct = funnel.trades / max(total_signals, 1) * 100
    lines += [
        sep,
        f"|  TRADES OPENED:         {funnel.trades:>6}  ({pass_pct:5.1f}% pass rate)         |",
        f"|    Wins:                {funnel.wins:>6}                        |",
        f"|    Losses:              {funnel.losses:>6}                        |",
        f"|    Timeouts:            {funnel.timeouts:>6}                        |",
        sep,
    ]
    return "\n".join(lines)


def _regime_breakdown(traces: List[SignalTrace]) -> str:
    from collections import Counter
    regime_gate: Dict[str, Counter] = {}
    for t in traces:
        if t.direction not in ("long", "short"):
            continue
        r = t.regime_str
        if r not in regime_gate:
            regime_gate[r] = Counter()
        regime_gate[r][t.blocked_at] += 1

    lines = ["\nREGIME x GATE BREAKDOWN (top 15 regimes by signal count)"]
    sorted_regimes = sorted(regime_gate.items(), key=lambda x: sum(x[1].values()), reverse=True)[:15]
    for regime, gate_counts in sorted_regimes:
        total = sum(gate_counts.values())
        trade_n = gate_counts.get("TRADE", 0)
        dominant = gate_counts.most_common(1)[0][0]
        lines.append(
            f"  {regime:<22}  signals={total:>4}  trades={trade_n:>3}  "
            f"dominant_block={dominant}"
        )
    return "\n".join(lines)


def _top_blocked_samples(traces: List[SignalTrace], gate: str, n: int = 5) -> str:
    blocked = [t for t in traces if t.blocked_at == gate]
    if not blocked:
        return ""
    lines = [f"\n  Sample blocked signals -- {gate} (last {min(n,len(blocked))})"]
    for t in blocked[-n:]:
        lines.append(
            f"    {t.timestamp[:16]}  {t.direction:<5}  "
            f"edge={t.edge_score:5.1f}  meta_conf={t.meta_conf:.4f}  "
            f"regime={t.regime_str!r}  reason={t.block_reason}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_forensic(symbols: List[str], out_dir: Path):
    AdaptiveBacktester._df_cache.clear()
    AdaptiveBacktester._predictor_cache.clear()

    summary: List[dict] = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  FORENSIC: {symbol}")
        print(f"{'='*60}")

        engine = ForensicEngine(symbol, mode="balanced")
        funnel, traces = engine.run(DATA_HOURS)

        # ── Gate funnel report ────────────────────────────────────────────────
        funnel_text = _funnel_text(funnel)
        print(funnel_text)
        print(_regime_breakdown(traces))

        # Show samples from the 3 biggest blockers
        gate_counts = {g: sum(1 for t in traces if t.blocked_at == g) for g in GATE_NAMES}
        top_blockers = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for gate, _ in top_blockers:
            if gate not in ("HOLD", "TRADE"):
                print(_top_blocked_samples(traces, gate))

        # ── Save audit CSV ─────────────────────────────────────────────────────
        sym_slug = symbol.replace("/", "_")
        csv_path = out_dir / f"{sym_slug}_audit.csv"
        if traces:
            pd.DataFrame([asdict(t) for t in traces]).to_csv(csv_path, index=False)
            print(f"\n  Audit CSV → {csv_path}  ({len(traces)} signal evaluations)")

        funnel_path = out_dir / f"{sym_slug}_funnel.txt"
        funnel_path.write_text(funnel_text + "\n" + _regime_breakdown(traces), encoding="utf-8")

        summary.append(asdict(funnel))

    # ── Fleet summary JSON ─────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Fleet forensic complete. {len(symbols)} symbols.")
    print(f"Summary → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS Backtest Forensic Auditor")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol (e.g. ETH/USDT). Overrides --all.")
    parser.add_argument("--all", action="store_true",
                        help="Run all 63 FLEET symbols (slow).")
    parser.add_argument("--top", type=int, default=5,
                        help="Run first N FLEET symbols (default 5).")
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = FLEET
    else:
        symbols = FLEET[:args.top]

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FORENSIC_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    run_forensic(symbols, out_dir)


if __name__ == "__main__":
    main()
