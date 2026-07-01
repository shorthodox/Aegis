"""
wr_forensic.py — Win-Rate Diagnostic Forensic for AEGIS-1
==========================================================
Answers: WHY is WR low, and WHAT must be fixed immediately?

For each symbol it:
  1. Replays all trades through the backtester gate sequence
  2. Retroactively tests every live-engine gate that is missing or different
  3. Prints a gate impact matrix: prevented_losses, blocked_wins, new_WR, delta
  4. Profiles wins vs losses on entry conditions
  5. Outputs a per-trade CSV and a summary JSON

Usage
-----
  python scripts/wr_forensic.py                         # SOL/USDT, BTC/USDT, ETH/USDT
  python scripts/wr_forensic.py --symbol SOL/USDT       # single symbol
  python scripts/wr_forensic.py --all                   # full 63-token fleet
  python scripts/wr_forensic.py --hours 4000            # extend data window
"""

import sys
import warnings
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]

import numpy as np

# Import shared gate constants and helpers from the backtester
from scripts.cost_aware_backtester import (
    AdaptiveBacktester,
    _dir_fraction, _score_signal_approx,
    MTF_PRIOR_BARS, MTF_RECENT_BARS, MTF_PRIOR_FRAC, MTF_REV_FRAC,
    MAX_HOLD_CANDLES, ATR_FLOOR_PCT, ATR_FLOOR_PCT_LIVE,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_ACCEL_BUY, RSI_ACCEL_SELL,
    COOLDOWN_BARS, MIN_QUALITY_SCORE, SCORE_SIG_FLOOR,
    SIGNAL_BYPASS_EDGE, STABILITY_WINDOW,
    HMM_BEAR_PROXY, HMM_BULL_PROXY, HTF_HARD_OPPOSE,
    FLEET,
)

OUT_DIR = Path("logs/forensic/wr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOLS = ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
DEFAULT_HOURS   = 3_000
DEFAULT_MODE    = "balanced"


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WRTrade:
    symbol:       str
    bar:          int
    timestamp:    str
    direction:    str
    # Entry condition values
    edge_score:   float
    meta_conf:    float
    regime_str:   str
    adx:          float
    vol_z:        float
    rsi:          float
    rsi_slope:    float
    rsi_accel:    float
    macro_w:      float
    macro_d:      float
    total_conf:   float
    macd_hist:    float
    choppiness:   float
    bos:          float
    atr:          float
    atr_pct:      float
    price:        float
    # Retroactive gate flags (True = would have blocked)
    blk_dir_regime:   bool = False
    blk_htf_veto:     bool = False
    blk_atr_live:     bool = False
    blk_min_quality:  bool = False
    blk_score_signal: bool = False
    blk_stability:    bool = False
    blk_adx25:        bool = False
    blk_macd_align:   bool = False
    # Computed
    score_signal: float = 0.0
    # Trade outcome
    exit_bar:     int   = -1
    exit_reason:  str   = ""
    pnl:          float = 0.0
    won:          Optional[bool] = None
    bars_held:    int   = 0


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class WRForensic:
    def __init__(self, symbol: str, mode: str = DEFAULT_MODE, hours: int = DEFAULT_HOURS):
        self.symbol = symbol
        self.mode   = mode
        self.hours  = hours

    def run(self) -> List[WRTrade]:
        bt = AdaptiveBacktester(self.symbol, self.mode)
        df = bt._fetch_and_prepare(self.hours)
        if df is None or len(df) < 200:
            print(f"  [{self.symbol}] Insufficient data — skipping")
            return []

        closes       = df["close"].to_numpy(dtype=float)
        opens        = df["open"].to_numpy(dtype=float)
        meta_dir_arr = df["meta_direction"].to_numpy(dtype=int)
        n            = len(df)

        atr_sl = bt.atr_mult
        atr_tp = bt.atr_mult * bt.params["rr_ratio"]

        trades: List[WRTrade] = []
        pos_units     = 0.0
        entry_price   = 0.0
        entry_atr     = 0.0
        entry_bar_idx = -1
        last_exit_bar = -1
        direction_open: Optional[str] = None

        for i in range(MTF_PRIOR_BARS + MTF_RECENT_BARS, n - MAX_HOLD_CANDLES - 1):
            row = df.iloc[i]

            # ── EXIT ─────────────────────────────────────────────────────────
            if pos_units != 0.0:
                ep   = entry_price
                ea   = entry_atr
                tp   = (ep + atr_tp * ea) if direction_open == "long" else (ep - atr_tp * ea)
                sl   = (ep - atr_sl * ea) if direction_open == "long" else (ep + atr_sl * ea)
                h, l = float(row["high"]), float(row["low"])
                exit_px = None; reason = ""
                if direction_open == "long":
                    if h >= tp and l <= sl: exit_px = sl; reason = "SL"
                    elif h >= tp:           exit_px = tp; reason = "TP"
                    elif l <= sl:           exit_px = sl; reason = "SL"
                else:
                    if l <= tp and h >= sl: exit_px = sl; reason = "SL"
                    elif l <= tp:           exit_px = tp; reason = "TP"
                    elif h >= sl:           exit_px = sl; reason = "SL"
                if exit_px is None and (i - entry_bar_idx) >= MAX_HOLD_CANDLES:
                    exit_px = float(row["close"]); reason = "TIMEOUT"
                if exit_px is not None:
                    pnl = (exit_px - ep) if direction_open == "long" else (ep - exit_px)
                    trades[-1].exit_bar   = i
                    trades[-1].exit_reason = reason
                    trades[-1].pnl        = pnl
                    trades[-1].won        = pnl > 0
                    trades[-1].bars_held  = i - entry_bar_idx
                    last_exit_bar = i
                    pos_units     = 0.0
                continue

            # ── GATE 1: REGIME ────────────────────────────────────────────────
            regime_str = str(row.get("regime_str", "unknown"))
            reg = bt.regime_thresholds.get(regime_str, {})
            if not reg or reg.get("skipped"):
                continue

            # ── GATE 2: EDGE SCORE + TRADEABILITY ────────────────────────────
            meta_conf  = float(row.get("meta_conf",       0.0))
            meta_dir   = int(row.get("meta_direction",    1))
            edge_score = float(row.get("edge_score_side", 0.0))
            if meta_dir == 2:
                if not reg.get("buy_ok") or not bt.tradeable_buy or edge_score < bt.thr_buy:
                    continue
                direction = "long"
            elif meta_dir == 0:
                if not reg.get("sell_ok") or not bt.tradeable_sell or edge_score < bt.thr_sell:
                    continue
                direction = "short"
            else:
                continue

            # ── RETROACTIVE GATE FLAGS ────────────────────────────────────────
            macro_w     = float(row.get("macro_trend_1w", 0.0))
            macro_d     = float(row.get("macro_trend_1d", 0.0))
            adx         = float(row.get("adx_14",          20.0))
            vol_z       = float(row.get("volume_zscore",    0.0))
            rsi         = float(row.get("rsi_14",           50.0))
            rsi_slope   = float(row.get("rsi_slope_14",     0.0))
            rsi_accel   = float(row.get("rsi_acceleration_14", 0.0))
            total_conf  = float(row.get("total_confluence", 5.0))
            macd_hist   = float(row.get("macd_hist",        0.0))
            choppiness  = float(row.get("choppiness",       50.0))
            bos         = float(row.get("bos_state",        0.0))
            cur_atr     = float(row["atr_14"]) if float(row["atr_14"]) > 0 \
                          else float(row["close"]) * 0.01
            atr_pct_val = cur_atr / max(float(row["close"]), 1e-12)

            # Gate 1.5: HMM direction-regime proxy
            blk_dir_regime = (
                (direction == "long"  and macro_d < HMM_BEAR_PROXY) or
                (direction == "short" and macro_d > HMM_BULL_PROXY)
            )
            # Gate 1.7: HTF hard veto
            blk_htf_veto = (
                (direction == "long"  and macro_w < -HTF_HARD_OPPOSE and macro_d < -HTF_HARD_OPPOSE) or
                (direction == "short" and macro_w >  HTF_HARD_OPPOSE and macro_d >  HTF_HARD_OPPOSE)
            )
            # Gate 2 live: ATR floor 0.8%
            blk_atr_live = atr_pct_val < ATR_FLOOR_PCT_LIVE
            # Gate 3: min quality score
            blk_min_quality = edge_score < MIN_QUALITY_SCORE
            # Gate 3b: score_signal approximation
            sig_qual = _score_signal_approx(
                direction, adx, vol_z, edge_score, rsi,
                macro_w, macro_d, total_conf, macd_hist,
            )
            blk_score_signal = edge_score < SIGNAL_BYPASS_EDGE and sig_qual < SCORE_SIG_FLOOR
            # Gate 3.8: signal stability
            if edge_score < SIGNAL_BYPASS_EDGE:
                _meta_req    = 2 if direction == "long" else 0
                blk_stability = not all(
                    meta_dir_arr[i - back] == _meta_req
                    for back in range(1, STABILITY_WINDOW + 1)
                    if (i - back) >= 0
                )
            else:
                blk_stability = False
            # Simple factor gates
            blk_adx25 = adx < 25.0
            blk_macd  = (
                (direction == "long"  and macd_hist < 0) or
                (direction == "short" and macd_hist > 0)
            )

            # ── PASS BASELINE GATES ONLY (v4.0 gate set — before new gates) ──
            # We deliberately do NOT apply the new gates here (1.5, 1.7, ATR_live,
            # min_quality, score_signal, stability) so that the retroactive gate
            # impact matrix can show what each new gate would have contributed.
            # Gate 3 S&R / trend / confluence (existing Gate 3)
            is_hc = edge_score >= float(bt.override_thr)
            if not is_hc:
                at_res = bool(row.get("is_at_resistance", 0))
                at_sup = bool(row.get("is_at_support",    0))
                if (direction == "long" and at_res) or (direction == "short" and at_sup): continue
                if (direction == "long" and macro_d < -0.2) or (direction == "short" and macro_d > 0.2): continue
                if (direction == "long" and total_conf < -0.1) or (direction == "short" and total_conf > 0.1): continue
                if choppiness > 60: continue
                if (direction == "long" and bos < 0) or (direction == "short" and bos > 0): continue
            # Gate 4.5 MTF
            _prior_c  = closes[i - MTF_PRIOR_BARS - MTF_RECENT_BARS : i - MTF_RECENT_BARS]
            _prior_o  = opens[i - MTF_PRIOR_BARS - MTF_RECENT_BARS  : i - MTF_RECENT_BARS]
            _recent_c = closes[i - MTF_RECENT_BARS : i]
            _recent_o = opens[i - MTF_RECENT_BARS  : i]
            _pd, _rd  = ("bearish", "bullish") if direction == "long" else ("bullish", "bearish")
            if not (_dir_fraction(_prior_c, _prior_o, _pd) >= MTF_PRIOR_FRAC and
                    _dir_fraction(_recent_c, _recent_o, _rd) >= MTF_REV_FRAC):
                continue
            # Gate 5 RSI
            if direction == "long":
                if rsi >= RSI_OVERBOUGHT and rsi_slope <= 0: continue
                if rsi_accel < RSI_ACCEL_BUY: continue
            else:
                if rsi <= RSI_OVERSOLD and rsi_slope >= 0: continue
                if rsi_accel > RSI_ACCEL_SELL: continue
            # Gate 6 cooldown
            if last_exit_bar >= 0 and (i - last_exit_bar) < COOLDOWN_BARS:
                continue
            # Gate 7 ATR floor
            if atr_pct_val < ATR_FLOOR_PCT:
                continue

            # ── RECORD TRADE ─────────────────────────────────────────────────
            t = WRTrade(
                symbol=self.symbol, bar=i,
                timestamp=str(row.get("timestamp", "")),
                direction=direction,
                edge_score=edge_score, meta_conf=meta_conf,
                regime_str=regime_str,
                adx=adx, vol_z=vol_z,
                rsi=rsi, rsi_slope=rsi_slope, rsi_accel=rsi_accel,
                macro_w=macro_w, macro_d=macro_d,
                total_conf=total_conf, macd_hist=macd_hist,
                choppiness=choppiness, bos=bos,
                atr=cur_atr, atr_pct=atr_pct_val,
                price=float(row["close"]),
                blk_dir_regime=blk_dir_regime,
                blk_htf_veto=blk_htf_veto,
                blk_atr_live=blk_atr_live,
                blk_min_quality=blk_min_quality,
                blk_score_signal=blk_score_signal,
                blk_stability=blk_stability,
                blk_adx25=blk_adx25,
                blk_macd_align=blk_macd,
                score_signal=sig_qual,
            )
            trades.append(t)
            entry_price   = float(row["close"])
            entry_atr     = cur_atr
            pos_units     = 1.0
            entry_bar_idx = i
            direction_open = direction

        return [t for t in trades if t.won is not None]


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _pct(n, d):
    return 0.0 if d == 0 else n / d * 100.0

def _avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def gate_impact_row(name: str, wins, losses, block_fn):
    bl_w  = sum(1 for t in wins   if block_fn(t))
    bl_l  = sum(1 for t in losses if block_fn(t))
    rem_w = len(wins)   - bl_w
    rem_l = len(losses) - bl_l
    rem   = rem_w + rem_l
    base_wr = _pct(len(wins), len(wins) + len(losses))
    new_wr  = _pct(rem_w, rem) if rem > 0 else 0.0
    delta   = new_wr - base_wr
    sign    = "+" if delta >= 0 else ""
    return {
        "gate": name,
        "bl_loss": bl_l, "bl_win": bl_w,
        "rem_trades": rem, "new_wr": new_wr, "delta": delta,
        "row": f"  {name:<40} | -{bl_l:>2}L -{bl_w:>2}W | trades {rem:>3} | WR {new_wr:5.1f}% | {sign}{delta:.1f}pp",
    }


def print_report(symbol: str, trades: List[WRTrade], mode: str):
    closed = [t for t in trades if t.won is not None]
    if not closed:
        print(f"  [{symbol}] No closed trades to analyze")
        return

    wins   = [t for t in closed if t.won]
    losses = [t for t in closed if not t.won]
    n      = len(closed)
    base_wr = _pct(len(wins), n)

    W = len(wins); L = len(losses)

    print()
    print(f"  {'+'*70}")
    print(f"  {symbol}  mode={mode}  trades={n}  W={W}  L={L}  WR={base_wr:.1f}%")
    print(f"  {'+'*70}")

    # Gate impact matrix (retroactive — gates NOT yet in current backtester
    # are shown with * to highlight them as "what to add")
    print()
    print("  RETROACTIVE GATE IMPACT  (on v4.0 baseline trades — before new gates)")
    print("  [v4.2] = added in backtester v4.2   * = idea only, not added yet")
    print(f"  {'Gate':<44} | {'Blocked':^11} | {'Remain':>6} | {'New WR':>6} | {'Delta':>6}")
    print(f"  {'-'*84}")

    gate_tests = [
        ("HMM dir-regime proxy (Gate 1.5) [v4.2]",
         lambda t: t.blk_dir_regime),
        ("HTF hard veto (Gate 1.7)        [v4.2]",
         lambda t: t.blk_htf_veto),
        ("ATR floor 0.8% (Gate 2 live)    [v4.2]",
         lambda t: t.blk_atr_live),
        ("edge_score >= 70 (Gate 3 floor)  [v4.2]",
         lambda t: t.blk_min_quality),
        ("score_signal >= 70 (Gate 3b)    [v4.2]",
         lambda t: t.blk_score_signal),
        ("stability window=2 (Gate 3.8)   [v4.2]",
         lambda t: t.blk_stability),
        ("adx > 25 (trend filter)           [*]",
         lambda t: t.blk_adx25),
        ("MACD alignment                    [*]",
         lambda t: t.blk_macd_align),
        ("ALL v4.2 gates combined",
         lambda t: t.blk_dir_regime or t.blk_htf_veto or t.blk_atr_live
                   or t.blk_min_quality or t.blk_score_signal or t.blk_stability),
        ("ALL v4.2 + adx25 + MACD",
         lambda t: t.blk_dir_regime or t.blk_htf_veto or t.blk_atr_live
                   or t.blk_min_quality or t.blk_score_signal or t.blk_stability
                   or t.blk_adx25 or t.blk_macd_align),
    ]

    best_delta = 0.0
    best_gate  = ""
    for name, fn in gate_tests:
        r = gate_impact_row(name, wins, losses, fn)
        print(r["row"])
        if r["delta"] > best_delta:
            best_delta = r["delta"]
            best_gate  = name

    # Entry condition profile
    print()
    print("  ENTRY CONDITION PROFILE  (wins vs losses)")
    print(f"  {'Metric':<28} | {'Wins avg':>9} | {'Losses avg':>10} | {'Direction':>10}")
    print(f"  {'-'*65}")

    def _diff(wv, lv, higher_better=True):
        return "good" if (higher_better and wv > lv) or (not higher_better and wv < lv) else "BAD"

    metrics = [
        ("edge_score",   [t.edge_score  for t in wins], [t.edge_score  for t in losses], True),
        ("score_signal", [t.score_signal for t in wins], [t.score_signal for t in losses], True),
        ("adx",          [t.adx         for t in wins], [t.adx         for t in losses], True),
        ("vol_z",        [t.vol_z       for t in wins], [t.vol_z       for t in losses], True),
        ("rsi",          [t.rsi         for t in wins], [t.rsi         for t in losses], None),
        ("rsi_accel",    [t.rsi_accel   for t in wins], [t.rsi_accel   for t in losses], True),
        ("macro_w",      [t.macro_w     for t in wins], [t.macro_w     for t in losses], True),
        ("macro_d",      [t.macro_d     for t in wins], [t.macro_d     for t in losses], True),
        ("total_conf",   [t.total_conf  for t in wins], [t.total_conf  for t in losses], None),
        ("choppiness",   [t.choppiness  for t in wins], [t.choppiness  for t in losses], False),
        ("atr_pct",      [t.atr_pct     for t in wins], [t.atr_pct     for t in losses], None),
        ("bars_held",    [t.bars_held   for t in wins], [t.bars_held   for t in losses], True),
    ]
    for name, wv, lv, hb in metrics:
        wa = _avg(wv); la = _avg(lv)
        tag = _diff(wa, la, hb) if hb is not None else "----"
        print(f"  {name:<28} | {wa:>9.3f} | {la:>10.3f} | {tag:>10}")

    # Exit breakdown
    tp_w = sum(1 for t in wins   if t.exit_reason == "TP")
    sl_w = sum(1 for t in wins   if t.exit_reason == "SL")
    to_w = sum(1 for t in wins   if t.exit_reason == "TIMEOUT")
    tp_l = sum(1 for t in losses if t.exit_reason == "TP")
    sl_l = sum(1 for t in losses if t.exit_reason == "SL")
    to_l = sum(1 for t in losses if t.exit_reason == "TIMEOUT")
    print()
    print(f"  EXIT BREAKDOWN")
    print(f"  wins:   TP={tp_w}  SL={sl_w}  TIMEOUT={to_w}")
    print(f"  losses: TP={tp_l}  SL={sl_l}  TIMEOUT={to_l}")

    # Direction split
    long_t  = [t for t in closed if t.direction == "long"]
    short_t = [t for t in closed if t.direction == "short"]
    lw = sum(1 for t in long_t  if t.won); ll = len(long_t)  - lw
    sw = sum(1 for t in short_t if t.won); sl = len(short_t) - sw
    print()
    print(f"  DIRECTION SPLIT")
    print(f"  long:  W={lw} L={ll}  WR={_pct(lw, ll+lw):.1f}%    short: W={sw} L={sl}  WR={_pct(sw, sl+sw):.1f}%")

    # What to fix
    print()
    print("  WHAT TO FIX IMMEDIATELY")
    print(f"  {'-'*50}")
    if best_gate:
        print(f"  #1 gate with highest WR impact: {best_gate} (+{best_delta:.1f}pp)")
    # Check if score_signal is calibrated wrong
    avg_sig_win  = _avg([t.score_signal for t in wins])
    avg_sig_loss = _avg([t.score_signal for t in losses])
    if avg_sig_win < SCORE_SIG_FLOOR and avg_sig_loss < SCORE_SIG_FLOOR:
        print(f"  WARN: score_signal avg={avg_sig_win:.1f}W/{avg_sig_loss:.1f}L — both below gate floor {SCORE_SIG_FLOOR:.0f}.")
        print(f"        The approximation may under-score due to missing live signals.")
        print(f"        Consider lowering SCORE_SIG_FLOOR to ~{int((avg_sig_win + avg_sig_loss)/2)} for backtesting.")
    elif avg_sig_win - avg_sig_loss < 3.0:
        print(f"  WARN: score_signal barely discriminates (diff={avg_sig_win - avg_sig_loss:.1f}). Check approximation.")
    else:
        print(f"  OK:   score_signal discriminates wins ({avg_sig_win:.1f}) vs losses ({avg_sig_loss:.1f}).")
    # Check edge_score discrimination
    avg_edge_win  = _avg([t.edge_score for t in wins])
    avg_edge_loss = _avg([t.edge_score for t in losses])
    if abs(avg_edge_win - avg_edge_loss) < 2.0:
        print(f"  WARN: edge_score does NOT discriminate W vs L (diff={avg_edge_win - avg_edge_loss:.1f}).")
        print(f"        Entry timing / direction filters are the leverage point, not edge threshold.")
    # Check SL dominance
    if sl_l > 0 and tp_l == 0:
        print(f"  WARN: All {sl_l} losses are SL hits — check if ATR-based SL is too tight for {symbol}.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CSV + JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(trades: List[WRTrade], path: Path):
    if not trades:
        return
    fieldnames = list(asdict(trades[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))


def export_summary(all_summaries: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AEGIS WR Forensic")
    parser.add_argument("--symbol", default=None, help="Single symbol, e.g. SOL/USDT")
    parser.add_argument("--all",    action="store_true", help="Run full 63-token fleet")
    parser.add_argument("--hours",  type=int, default=DEFAULT_HOURS)
    parser.add_argument("--mode",   default=DEFAULT_MODE, choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = FLEET
    else:
        symbols = DEFAULT_SYMBOLS

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AEGIS-1  WIN RATE FORENSIC")
    print(f"   mode={args.mode}  hours={args.hours}  symbols={len(symbols)}")
    print(f"   output -> {run_dir}")
    print("=" * 72)

    all_summaries = []
    # Clear backtester cache so each symbol loads fresh
    AdaptiveBacktester._df_cache.clear()
    AdaptiveBacktester._predictor_cache.clear()

    for symbol in symbols:
        print(f"\n[{symbol}] loading...")
        forensic = WRForensic(symbol, mode=args.mode, hours=args.hours)
        trades   = forensic.run()
        closed   = [t for t in trades if t.won is not None]
        if not closed:
            print(f"  [{symbol}] No trades — skipping report")
            continue

        wins   = [t for t in closed if t.won]
        losses = [t for t in closed if not t.won]
        print_report(symbol, closed, args.mode)

        # Export per-symbol CSV
        sym_safe = symbol.replace("/", "_")
        csv_path = run_dir / f"{sym_safe}_trades.csv"
        export_csv(closed, csv_path)
        print(f"  CSV -> {csv_path}")

        all_summaries.append({
            "symbol":        symbol,
            "mode":          args.mode,
            "hours":         args.hours,
            "total_trades":  len(closed),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(_pct(len(wins), len(closed)), 2),
            "avg_edge_win":  round(_avg([t.edge_score   for t in wins]),   2),
            "avg_edge_loss": round(_avg([t.edge_score   for t in losses]), 2),
            "avg_sig_win":   round(_avg([t.score_signal for t in wins]),   2),
            "avg_sig_loss":  round(_avg([t.score_signal for t in losses]), 2),
        })

    summary_path = run_dir / "summary.json"
    export_summary(all_summaries, summary_path)
    print(f"\nSummary -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
