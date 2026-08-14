"""Plain data carriers shared across the engine.

No behaviour lives here beyond field defaults — these are the shapes that move
between the gate, the wallet, the exit manager and the track record. Kept in
their own module so every other engine module can import them without pulling in
the orchestration.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# NOTE: deliberately no `from __future__ import annotations` here. It would turn
# every field's .type into a string, so dataclasses.fields() and
# typing.get_type_hints() report something different from the original
# definitions these were extracted from. The classes are round-tripped through
# asdict() into the track record; keep them introspectable.

__all__ = ["TokenConfig", "Position", "TradeRecord", "RegimeState"]


@dataclass
class TokenConfig:
    symbol: str
    mode:   str = 'balanced'


@dataclass
class Position:
    symbol:          str
    direction:       str    # LONG | SHORT
    side:            str    # BUY  | SELL
    entry_price:     float
    position_value:  float  # USDT allocated
    stop_loss:       float  # LIVE stop. Mutated by the break-even ratchet and
                            # the trailing logic. NEVER use this to compute R.
    signal_id:       str
    entry_time:      str
    meta_confidence: float
    atr_multiplier:  float
    # The risk the trade was ACTUALLY taken with. Set once at open, never
    # mutated — unlike stop_loss, which the break-even ratchet rewrites to
    # entry_price the moment TP1 is tagged (exits.py:346).
    #
    # Every R-multiple, R:R and expectancy figure must derive from THIS. Using
    # stop_loss silently produced infinite-R for any trade that reached TP1:
    # CRV/USDT on 2026-08-14 closed a real +1.40% win and published
    # entry 0.2513 / stop 0.2513, a zero-risk trade. The win rate survives that
    # error; expectancy does not, and expectancy is the metric the published
    # record is moving toward.
    #
    # Defaults to 0.0 for records written before this field existed. Consumers
    # must treat 0.0 as "unknown" and decline to compute R rather than dividing
    # by it.
    entry_stop:      float = 0.0
    atr:             float = 0.0   # ATR at entry (used for trailing stop distance)
    initial_value:   float = 0.0   # v82: allocation AT ENTRY.  position_value shrinks
                                   # as TP rungs bank, so partial sizing must be a
                                   # fraction of THIS, not of the remainder.
    take_profit_1:   float = 0.0   # TP1: 1.0R from entry — 15% partial close
    take_profit_2:   float = 0.0   # TP2: 2.0R from entry — 25% partial + break-even + trail on
    take_profit_3:   float = 0.0   # TP3: structural target — 25% partial close
    take_profit_4:   float = 0.0   # TP4: 1.618 fib extension — 15% partial close
    take_profit_5:   float = 0.0   # TP5: 2.618 fib extension — close remainder (RR anchor)
    signal_strength: str   = ''    # risk tier at entry: STRONG | NORMAL | RISKY
    entry_mode:      str   = ''    # structure-gate verdict detail at entry
                                   # (support_reversal / breakout_* / GATE_SKIPPED: …)
    quality_score:   float = 0.0   # SignalQualityFilter score AT ENTRY — displayed
                                   # for open positions instead of a live re-score
    gate_warnings:   list  = field(default_factory=list)  # advisory-gate ledger AT
                                   # ENTRY — keeps the chart gate breakdown complete
                                   # for open positions (rebuilt away otherwise)
    entry_support:    float = 0.0  # S/R the STRUCTURE GATE judged AT ENTRY — shown
    entry_resistance: float = 0.0  # for open positions so the chart's S/R lines
                                   # reflect the entry structure, not a live re-score
                                   # that drifts after entry (a breakdown short can
                                   # look like a naive "sell at support" otherwise)


@dataclass
class TradeRecord:
    signal_id:       str
    symbol:          str
    direction:       str
    side:            str
    entry_price:     float
    exit_price:      Optional[float]
    entry_time:      str
    close_time:      Optional[str]
    pnl_pct:         float
    pnl_usdt:        float
    outcome:         str              # OPEN | WIN | LOSS
    exit_reason:     Optional[str]
    meta_confidence: float
    position_value:  float
    signal_strength: str   = ''
    stop_loss:       float = 0.0   # stop AT EXIT — ratcheted, not the risk taken
    entry_stop:      float = 0.0   # stop AT ENTRY — the only valid R denominator
    take_profit_1:   float = 0.0
    take_profit_2:   float = 0.0
    take_profit_3:   float = 0.0
    take_profit_4:   float = 0.0
    take_profit_5:   float = 0.0
    atr:             float = 0.0


@dataclass
class RegimeState:
    """Snapshot of the current market micro-structure for one symbol."""
    regime:              str         # one of the canonical labels in config.py
    confidence:          float       # 0.0 – 1.0
    trade_allowed:       bool
    preferred_strategies: List[str]
    max_position_pct:    float       # fraction of balance, e.g. 0.10
