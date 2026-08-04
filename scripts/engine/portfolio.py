"""Book-level risk: correlation caps and the paper-trading wallet.

PortfolioGuard stops the engine expressing one thesis eight times. VirtualWallet
books the results, with execution costs charged so the published track record is
reachable on a real venue.

The cost constants here are deliberately in step with the training pipeline's
FEE_ROUNDTRIP. If you change one, change the other — a wallet that charges less
than the labels assumed reports an edge that does not exist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.engine.config import TRACK_RECORD_PATH
from scripts.engine.models import Position, TradeRecord

__all__ = ["PortfolioGuard", "VirtualWallet"]


class PortfolioGuard:
    """
    Prevents the engine from opening multiple correlated positions simultaneously.

    Problem it solves:
    ==================
    BTC, ETH, SOL, AVAX, BNB almost always move together.  When the market turns,
    the engine can fire BUY on all five within the same scan cycle.  This creates
    5× correlated exposure — not 5 independent bets.  A single BTC flush wipes
    all five positions at once.

    Solution:
    =========
    Tokens are grouped into correlation clusters.  Within each cluster, only
    MAX_PER_CLUSTER positions can be open at the same time.  The highest-quality
    signal in the cluster wins.

    Portfolio-wide limits:
    - MAX_OPEN_TOTAL:       hard limit on simultaneous open positions
    - MAX_CAPITAL_DEPLOYED: max fraction of wallet balance in open positions

    Clusters are defined statically (good enough — crypto correlations are
    structurally stable within tiers).  Dynamic clustering via PCA on live
    price returns is a Phase 2 improvement.
    """

    # v74 — the caps are BINDING again (user decision 2026-07-19). The old
    # rationale ("independent recommendations, caps only mute opportunities")
    # died on contact with a real session: 16 simultaneous positions opened
    # into one tape — 8 alt shorts that were ONE bet expressed eight times —
    # and bled together on a BTC bounce. Crypto alts run ~0.8 correlation to
    # BTC intraday, so an uncapped book is levered beta, not breadth. The
    # public track record IS the product; it must be built from the best few
    # setups, not everything the model will sign. Capital cap stays open
    # (position values are notional per-signal).
    MAX_PER_CLUSTER    = 2     # one thesis, max two expressions per cluster
    MAX_OPEN_TOTAL     = 5     # the book holds the 5 best setups, not all 58
    MAX_CAPITAL_PCT    = 100.0 # no capital cap — position values are notional per-signal

    # Static correlation clusters (tightest first)
    _CLUSTERS: Dict[str, List[str]] = {
        'MAJORS':    ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'L1_FAST':   ['SOL/USDT', 'AVAX/USDT', 'APT/USDT', 'SUI/USDT', 'NEAR/USDT'],
        'L2':        ['ARB/USDT', 'OP/USDT', 'STRK/USDT', 'IMX/USDT', 'ZK/USDT'],
        'DEFI_BLUE': ['AAVE/USDT', 'UNI/USDT', 'CRV/USDT', 'COMP/USDT', 'LDO/USDT'],
        'AI_INFRA':  ['FET/USDT', 'TAO/USDT', 'ARKM/USDT', 'GRT/USDT'],
        'MEME':      ['PEPE/USDT', 'WIF/USDT', 'DOGE/USDT', 'SHIB/USDT', 'BONK/USDT',
                      'FLOKI/USDT', 'BOME/USDT', 'BRETT/USDT'],
        'XRP_ALTS':  ['XRP/USDT', 'XLM/USDT', 'ADA/USDT', 'TRX/USDT'],
        'LAYER1_MID': ['DOT/USDT', 'ATOM/USDT', 'ALGO/USDT', 'EGLD/USDT',
                       'ICP/USDT', 'FIL/USDT', 'HBAR/USDT'],
    }

    def __init__(self) -> None:
        # Build reverse map: symbol → cluster name
        self._sym_to_cluster: Dict[str, str] = {}
        for cluster, syms in self._CLUSTERS.items():
            for s in syms:
                self._sym_to_cluster[s] = cluster
        # Runtime state (populated from wallet)
        self._open_positions: Dict[str, str] = {}   # symbol → direction ('LONG'/'SHORT')
        self._deployed_usdt:  float          = 0.0  # sum of all open position_values

    def sync_from_wallet(self, open_positions: Dict[str, Any]) -> None:
        """Sync open positions from the VirtualWallet on every scan cycle."""
        self._open_positions = {
            sym: pos.direction
            for sym, pos in open_positions.items()
        }
        self._deployed_usdt = sum(pos.position_value for pos in open_positions.values())

    def _cluster_open_count(self, cluster: str) -> int:
        cluster_syms = set(self._CLUSTERS.get(cluster, []))
        return sum(1 for sym in self._open_positions if sym in cluster_syms)

    def _total_open(self) -> int:
        return len(self._open_positions)

    def can_open(self, symbol: str, wallet_balance: float,
                 position_value: float) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Reason is a human-readable string for the log — critical for debugging.
        """
        # Hard cap on total open positions
        if self._total_open() >= self.MAX_OPEN_TOTAL:
            return False, f'MAX_OPEN_TOTAL={self.MAX_OPEN_TOTAL} reached'

        # Cluster correlation cap
        cluster = self._sym_to_cluster.get(symbol)
        if cluster:
            n_in_cluster = self._cluster_open_count(cluster)
            if n_in_cluster >= self.MAX_PER_CLUSTER:
                open_in_cluster = [s for s in self._open_positions
                                   if self._sym_to_cluster.get(s) == cluster]
                return (False,
                        f'CLUSTER_CAP: {cluster} already has {n_in_cluster} '
                        f'open ({", ".join(open_in_cluster)})')

        # Capital deployment cap — sum actual deployed + new position vs balance.
        if wallet_balance > 0:
            deployed_pct = (self._deployed_usdt + position_value) / wallet_balance
            if deployed_pct >= self.MAX_CAPITAL_PCT:
                return (False,
                        f'CAPITAL_CAP: {deployed_pct:.0%} would be deployed '
                        f'(max {self.MAX_CAPITAL_PCT:.0%})')

        return True, 'OK'

    def get_summary(self) -> Dict[str, Any]:
        cluster_counts: Dict[str, int] = {}
        for sym in self._open_positions:
            c = self._sym_to_cluster.get(sym, 'OTHER')
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        return {
            'open_total':    self._total_open(),
            'max_allowed':   self.MAX_OPEN_TOTAL,
            'cluster_counts': cluster_counts,
        }


class VirtualWallet:
    """Risk 10 % of balance per trade, capped at max_position_usdt."""

    # ── Execution costs (v82) ────────────────────────────────────────────────
    # Until v82 the wallet charged NOTHING and every exit booked at its exact
    # theoretical level (`exit_px=pos.take_profit_1`, `exit_px=trail_stop`, the
    # whole *_via_peak family), so the reported track record was strictly better
    # than anything reachable on a real venue.  With a gross edge of ~0.02R that
    # omission was the difference between "flat" and "losing".
    #
    # 0.04 % taker + 0.01 % slippage per side = 0.10 % round trip, which is the
    # same FEE_ROUNDTRIP the training pipeline already assumes
    # (retrain_model.py:204) — keep the two in step.
    TAKER_FEE_PCT = 0.04   # per side, % of notional
    SLIPPAGE_PCT  = 0.01   # per side, % — adverse fill vs the trigger level

    @classmethod
    def round_trip_cost_pct(cls) -> float:
        """Total cost charged against a slice's gross PnL %, entry + exit."""
        return 2.0 * (cls.TAKER_FEE_PCT + cls.SLIPPAGE_PCT)

    def __init__(self, initial_capital: float, max_position_usdt: float = 1_000.0,
                 track_record_path: Optional[Path] = None):
        self.initial_capital    = initial_capital
        self.balance            = initial_capital
        self.max_position_usdt  = max_position_usdt
        self._track_record_path = track_record_path or TRACK_RECORD_PATH
        self.open_positions:    Dict[str, Position]  = {}
        self.trade_history:     List[TradeRecord]     = []
        self._load_history()

    def _load_history(self) -> None:
        """Restore closed trade history, balance, and open positions from disk on restart."""
        if not self._track_record_path.exists():
            return
        try:
            with open(self._track_record_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            signals   = data.get('signals', [])
            closed    = [s for s in signals if s.get('outcome') in ('WIN', 'LOSS')]
            open_sigs = [s for s in signals if s.get('outcome') == 'OPEN']

            # ── Closed trades → trade_history + adjust balance ─────────────────
            restored = 0
            seen_ids: set = set()
            for s in closed:
                sid = s.get('signal_id', '')
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                try:
                    rec = TradeRecord(
                        signal_id       = sid,
                        symbol          = s['symbol'],
                        direction       = s.get('direction', 'LONG'),
                        side            = s.get('side', 'BUY'),
                        entry_price     = float(s.get('entry_price', 0)),
                        exit_price      = float(s['exit_price']) if s.get('exit_price') else None,
                        entry_time      = s.get('entry_time', ''),
                        close_time      = s.get('close_time'),
                        pnl_pct         = float(s.get('pnl_pct', 0)),
                        pnl_usdt        = float(s.get('pnl_usdt', 0)),
                        outcome         = s['outcome'],
                        exit_reason     = s.get('exit_reason'),
                        meta_confidence = float(s.get('meta_confidence', 0)),
                        position_value  = float(s.get('position_value', 0)),
                        signal_strength = s.get('signal_strength', ''),
                        stop_loss       = float(s.get('stop_loss', 0)),
                        take_profit_1   = float(s.get('take_profit_1', 0)),
                        take_profit_2   = float(s.get('take_profit_2', 0)),
                        take_profit_3   = float(s.get('take_profit_3', 0)),
                        take_profit_4   = float(s.get('take_profit_4', 0)),
                        take_profit_5   = float(s.get('take_profit_5', 0)),
                        atr             = float(s.get('atr', 0)),
                    )
                    self.trade_history.append(rec)
                    self.balance += float(s.get('pnl_usdt', 0))
                    restored += 1
                except Exception:
                    continue
            if restored:
                print(f'[VirtualWallet] Restored {restored} closed trades from disk. '
                      f'Balance: ${self.balance:,.2f}')

            # ── Open positions → restore into wallet so portfolio guard is correct ─
            open_restored = 0
            for s in open_sigs:
                sym = s.get('symbol', '')
                if not sym or sym in self.open_positions:
                    continue
                try:
                    side = s.get('side', '') or ('BUY' if s.get('direction') == 'LONG' else 'SELL')
                    pos  = Position(
                        symbol          = sym,
                        direction       = s.get('direction', 'LONG' if side == 'BUY' else 'SHORT'),
                        side            = side,
                        entry_price     = float(s.get('entry_price', 0)),
                        position_value  = float(s.get('position_value', 0)),
                        # v82: pre-v82 records have no initial_value — fall back to
                        # the surviving size so a restored position still sizes its
                        # remaining TP rungs off a fixed base rather than drifting.
                        initial_value   = float(s.get('initial_value', 0)
                                                or s.get('position_value', 0)),
                        stop_loss       = float(s.get('stop_loss', 0)),
                        signal_id       = s.get('signal_id', ''),
                        entry_time      = s.get('entry_time', ''),
                        meta_confidence = float(s.get('meta_confidence', 0)),
                        atr_multiplier  = float(s.get('atr_multiplier', 1.2)),
                        atr             = float(s.get('atr', 0)),
                        take_profit_1   = float(s.get('take_profit_1', 0)),
                        take_profit_2   = float(s.get('take_profit_2', 0)),
                        take_profit_3   = float(s.get('take_profit_3', 0)),
                        take_profit_4   = float(s.get('take_profit_4', 0)),
                        take_profit_5   = float(s.get('take_profit_5', 0)),
                        signal_strength = s.get('signal_strength', ''),
                        entry_mode      = s.get('entry_mode', ''),
                    )
                    self.open_positions[sym] = pos
                    open_restored += 1
                except Exception:
                    continue
            if open_restored:
                print(f'[VirtualWallet] Restored {open_restored} open positions from disk.')
        except Exception as e:
            print(f'[VirtualWallet] History load error (starting fresh): {e}')

    def position_size(self) -> float:
        return min(self.balance * 0.10, self.max_position_usdt)

    def open_trade(self, pos: Position) -> None:
        self.open_positions[pos.symbol] = pos

    def close_trade(self, symbol: str, exit_price: float,
                    exit_reason: str) -> Optional[TradeRecord]:
        """Full close — removes the position and books PnL on remaining size."""
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        # v82: charge round-trip execution cost on this slice's notional.
        pnl_pct -= self.round_trip_cost_pct()

        pnl_usdt = round(pos.position_value * pnl_pct / 100, 2)
        self.balance += pnl_usdt

        # Whole-trade outcome: include profits already banked by TP partial
        # closes of the same position (same signal_id).  A trade that banked
        # TP1–TP4 and then closed its remainder at break-even is a WIN — the
        # final slice alone would mislabel it a LOSS.
        _banked_usdt = sum(
            t.pnl_usdt for t in self.trade_history
            if t.signal_id == pos.signal_id and 'PARTIAL' in (t.exit_reason or '')
        )
        _trade_outcome = 'WIN' if (pnl_usdt + _banked_usdt) > 0 else 'LOSS'

        rec = TradeRecord(
            signal_id       = pos.signal_id,
            symbol          = symbol,
            direction       = pos.direction,
            side            = pos.side,
            entry_price     = pos.entry_price,
            exit_price      = round(exit_price, 8),
            entry_time      = pos.entry_time,
            close_time      = datetime.now(timezone.utc).isoformat(),
            pnl_pct         = round(pnl_pct, 3),
            pnl_usdt        = pnl_usdt,
            outcome         = _trade_outcome,
            exit_reason     = exit_reason,
            meta_confidence = pos.meta_confidence,
            position_value  = pos.position_value,
            signal_strength = pos.signal_strength,
            stop_loss       = pos.stop_loss,
            take_profit_1   = pos.take_profit_1,
            take_profit_2   = pos.take_profit_2,
            take_profit_3   = pos.take_profit_3,
            take_profit_4   = pos.take_profit_4,
            take_profit_5   = pos.take_profit_5,
            atr             = pos.atr,
        )
        self.trade_history.append(rec)
        return rec

    def partial_close_trade(
        self,
        symbol:       str,
        exit_price:   float,
        exit_reason:  str,
        close_pct:    float,   # fraction of current position_value to close, e.g. 0.20
    ) -> Optional[TradeRecord]:
        """Partial close — reduces position_value, books PnL on the closed slice.

        The position stays open in open_positions so subsequent TP levels can
        still fire.

        v82: close_pct is a fraction of the ORIGINAL allocation
        (`pos.initial_value`), not of the shrinking remainder.  The old code
        applied it to the current position_value while its docstring claimed
        otherwise, so a nominal 15/25/25/15 ladder actually banked
        15.0/21.3/16.0/7.2 % — every rung after the first quietly under-banked
        and the tail rode far more size than the design intended.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return None

        base = pos.initial_value if pos.initial_value > 0 else pos.position_value
        close_value = round(base * close_pct, 2)
        # Never close more than is still open (guards a re-entrant TP cascade).
        close_value = min(close_value, pos.position_value)
        if close_value <= 0:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        # v82: same round-trip cost as a full close, on the slice notional.
        pnl_pct -= self.round_trip_cost_pct()

        pnl_usdt = round(close_value * pnl_pct / 100, 2)
        self.balance  += pnl_usdt
        pos.position_value = round(pos.position_value - close_value, 2)

        rec = TradeRecord(
            signal_id       = pos.signal_id,
            symbol          = symbol,
            direction       = pos.direction,
            side            = pos.side,
            entry_price     = pos.entry_price,
            exit_price      = round(exit_price, 8),
            entry_time      = pos.entry_time,
            close_time      = datetime.now(timezone.utc).isoformat(),
            pnl_pct         = round(pnl_pct, 3),
            pnl_usdt        = pnl_usdt,
            outcome         = 'WIN' if pnl_pct > 0 else 'LOSS',
            exit_reason     = exit_reason,
            meta_confidence = pos.meta_confidence,
            position_value  = close_value,
            signal_strength = pos.signal_strength,
            stop_loss       = pos.stop_loss,
            take_profit_1   = pos.take_profit_1,
            take_profit_2   = pos.take_profit_2,
            take_profit_3   = pos.take_profit_3,
            take_profit_4   = pos.take_profit_4,
            take_profit_5   = pos.take_profit_5,
            atr             = pos.atr,
        )
        self.trade_history.append(rec)
        return rec

    @property
    def summary(self) -> Dict[str, Any]:
        # Exclude TP partial-close records (exit_reason contains 'PARTIAL') from
        # the signal count — they represent slices of an open position, not closed
        # trades, and inflate wins/losses/total when the position is still running.
        _full = [t for t in self.trade_history if 'PARTIAL' not in (t.exit_reason or '')]
        won   = sum(1 for t in _full if t.outcome == 'WIN')
        lost  = sum(1 for t in _full if t.outcome == 'LOSS')
        total = won + lost
        pnl_u = round(self.balance - self.initial_capital, 2)
        return {
            'initial_capital': self.initial_capital,
            'balance':         round(self.balance, 2),
            'total_pnl_usdt':  pnl_u,
            'total_pnl_pct':   round(pnl_u / self.initial_capital * 100, 3),
            'total_trades':    total,
            'won':             won,
            'lost':            lost,
            'win_rate':        round(won / total, 3) if total else 0.0,
            'open_positions':  len(self.open_positions),
        }
