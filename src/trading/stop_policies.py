"""Stop placement and stop movement, behind one interface.

UNIT CONVENTION — READ BEFORE ADDING A CONSTANT
================================================
Every ``*_PCT`` name in this module and in ``trader_gate`` is **PERCENT**:
``MIN_STOP_PCT = 0.50`` means 0.50% of price, NOT 0.005 and NOT 50%.
Prices and distances are in price units. ATR is in price units.

This is not pedantry. A reference implementation circulated with the same
constant names in FRACTIONS (``MIN_STOP_PCT = 0.010``) — identical names, 100x
apart, both living in the stop path. A fraction/percent mix-up here does not
raise; it silently places a stop 100x too tight or too wide. Hence the loud
guards below, which fail at import rather than in production.

WHAT THIS MODULE IS
===================
The engine currently derives a stop in two places that disagree by design:
``trader_gate`` places it against structure, then a percent band clamps it. Each
policy below isolates one of those rules so they can be measured against each
other on the same trades instead of argued about.

**Nothing here is wired into the live engine.** It is the substrate the sweep
runs on. ``PercentBandStop`` reproduces the deployed behaviour exactly and
exists as the control; adopting anything else is a separate, measured decision.

A NOTE ON THE INTERFACE
=======================
The specified Protocol had two methods, ``initial_stop`` and ``update_stop``.
That is not sufficient: candle-close invalidation differs from the deployed rule
not in WHERE the stop sits but in WHEN it is allowed to trigger. Folding that
into ``update_stop`` would mean a policy silently returning a level it does not
intend to be honoured intrabar. ``triggered()`` is therefore a third member,
defaulting to the intrabar behaviour every current policy uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, runtime_checkable

Side = Literal["LONG", "SHORT"]

# ── the deployed band (mirrors trader_gate; see below for why it is re-read) ──
from src.trading import trader_gate as _tg

# Structural placement, mirrored from trader_gate stage 2. These are ATR
# MULTIPLES, not percents — the suffix distinguishes them deliberately.
STOP_BUFFER_ATR = 0.55      # stop sits this far beyond the level
MIN_STOP_ATR    = 1.50      # noise floor for the invalidation
MAX_STOP_ATR    = 3.00      # beyond this the setup is refused

# ── unit guards: fail at import, not in production ───────────────────────────
def _guard_pct(name: str, value: float) -> None:
    """A percent-unit constant that arrived as a fraction is the bug this catches."""
    if value == 0.0:
        return                                  # 0 is the documented "disabled"
    assert 0.05 < value < 10.0, (
        f"{name}={value!r} is out of range for a PERCENT-unit stop constant. "
        f"0.50 means 0.50%. A value near 0.005 is a fraction and is 100x too "
        f"tight; a value above 10 is 100x too wide."
    )


def _guard_atr(name: str, value: float) -> None:
    assert 0.0 < value < 20.0, f"{name}={value!r} is not a plausible ATR multiple"


_guard_pct("trader_gate.MIN_STOP_PCT", float(getattr(_tg, "MIN_STOP_PCT", 0.0) or 0.0))
_guard_pct("trader_gate.MAX_STOP_PCT", float(getattr(_tg, "MAX_STOP_PCT", 0.0) or 0.0))
for _n, _v in (("STOP_BUFFER_ATR", STOP_BUFFER_ATR),
               ("MIN_STOP_ATR", MIN_STOP_ATR),
               ("MAX_STOP_ATR", MAX_STOP_ATR)):
    _guard_atr(_n, _v)


# ── contexts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalContext:
    """Everything a policy may see when placing the FIRST stop."""
    side:         Side
    entry:        float
    atr:          float               # ATR(14), price units
    level:        float = 0.0         # structural level the setup leans on, 0 if none
    atr_pct_rank: float = 0.5         # 0..1 percentile of this bar's ATR% vs its own history
    vwap:         float = 0.0
    ema20:        float = 0.0
    ema50:        float = 0.0

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1


@dataclass(frozen=True)
class BarContext:
    """One closed candle, plus the running state a trail needs."""
    side:    Side
    entry:   float
    high:    float
    low:     float
    close:   float
    atr:     float
    extreme: float                    # best price seen since fill
    tp1_hit: bool = False
    tp2_hit: bool = False
    vwap:    float = 0.0
    ema20:   float = 0.0
    ema50:   float = 0.0

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1


@runtime_checkable
class StopPolicy(Protocol):
    name: str

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        """Price level, or None if this policy declines the signal."""

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        """Proposed level. Always passed through apply_stop() by the caller."""

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        """Has this stop been hit on this bar?"""


# ── the one place a stop is allowed to move ──────────────────────────────────

def apply_stop(side: Side, entry: float, active_stop: float,
               proposed: float, *, log: Optional[list] = None) -> float:
    """Ratchet-only. A stop may tighten toward entry and past it, never widen.

    Every policy writes through here so "the stop only ever moves one way" is a
    property of the module rather than a habit each policy has to remember. A
    widening proposal is discarded and recorded — silently clamping would hide a
    policy bug behind correct-looking behaviour.
    """
    if proposed is None or proposed <= 0:
        return active_stop
    sign = 1 if side == "LONG" else -1
    tighter = proposed > active_stop if sign == 1 else proposed < active_stop
    if tighter:
        return float(proposed)
    if log is not None and abs(proposed - active_stop) > 1e-12:
        log.append(("WIDEN_REJECTED", active_stop, proposed))
    return active_stop


def _intrabar_hit(ctx: BarContext, active_stop: float) -> bool:
    return (ctx.low <= active_stop) if ctx.sign == 1 else (ctx.high >= active_stop)


def _structural_stop(ctx: SignalContext) -> Optional[float]:
    """trader_gate stage 2, before the percent band. Returns None on refusal."""
    s = ctx.sign
    if ctx.atr <= 0 or ctx.entry <= 0:
        return None
    if ctx.level > 0:
        stop = ctx.level - s * STOP_BUFFER_ATR * ctx.atr
    else:
        stop = ctx.entry - s * MIN_STOP_ATR * ctx.atr
    risk = abs(ctx.entry - stop)
    if risk < MIN_STOP_ATR * ctx.atr:                 # widen to the noise floor
        stop = ctx.entry - s * MIN_STOP_ATR * ctx.atr
        risk = MIN_STOP_ATR * ctx.atr
    if risk > MAX_STOP_ATR * ctx.atr:                 # too far to defend
        return None
    return stop


# ── policies ─────────────────────────────────────────────────────────────────

@dataclass
class PercentBandStop:
    """THE DEPLOYED CONTROL. Structure, then clamped into the percent band.

    Reproduces production exactly, including the part under scrutiny: on a 1h alt
    the 1.50 ATR structural floor is typically well above MAX_STOP_PCT, so the
    clamp discards the structural placement on most signals and the stop lands
    inside the noise the floor exists to clear.
    """
    name: str = "PercentBand(0.50-0.70%)"
    lo_pct: float = field(default_factory=lambda: float(getattr(_tg, "MIN_STOP_PCT", 0.0) or 0.0))
    hi_pct: float = field(default_factory=lambda: float(getattr(_tg, "MAX_STOP_PCT", 0.0) or 0.0))

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        base = _structural_stop(ctx)
        if base is None:
            return None
        risk = abs(ctx.entry - base)
        if self.lo_pct > 0 and self.hi_pct > 0:
            lo, hi = ctx.entry * self.lo_pct / 100.0, ctx.entry * self.hi_pct / 100.0
            risk = min(max(risk, lo), hi)
        return ctx.entry - ctx.sign * risk

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        return active_stop

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return _intrabar_hit(ctx, active_stop)


@dataclass
class StructureStop:
    """The main candidate: stage 2 with the clamp removed."""
    name: str = "Structure(unclamped)"

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        return _structural_stop(ctx)

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        return active_stop

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return _intrabar_hit(ctx, active_stop)


@dataclass
class AtrStop:
    """Simple reference: k x ATR from entry, ignoring structure."""
    k: float = 1.6
    name: str = ""

    def __post_init__(self):
        _guard_atr("AtrStop.k", self.k)
        if not self.name:
            self.name = f"ATR(k={self.k:g})"

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        if ctx.atr <= 0:
            return None
        return ctx.entry - ctx.sign * self.k * ctx.atr

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        return active_stop

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return _intrabar_hit(ctx, active_stop)


@dataclass
class CandleInvalidationStop:
    """Structure placement, but only a CLOSE beyond it invalidates.

    A wick through the level is a liquidity sweep, not a broken thesis. The cost
    is real and must not be hidden: the position is carried to the close of the
    breaching bar, so the realised loss is whatever the close is, which can be
    materially worse than the stop. Phase 2's diagnostic is what decides whether
    the wicks saved outweigh the closes paid.
    """
    name: str = "CandleInvalidation"

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        return _structural_stop(ctx)

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        return active_stop

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return (ctx.close <= active_stop) if ctx.sign == 1 else (ctx.close >= active_stop)


@dataclass
class VolCompressionStop:
    """ATR multiple that scales with where this bar's ATR sits in its own range.

    Tighter in compression, wider in expansion. `k` interpolates linearly between
    k_lo at the 0th ATR percentile and k_hi at the 100th.
    """
    k_lo: float = 1.2
    k_hi: float = 2.2
    name: str = ""

    def __post_init__(self):
        _guard_atr("VolCompressionStop.k_lo", self.k_lo)
        _guard_atr("VolCompressionStop.k_hi", self.k_hi)
        assert self.k_lo <= self.k_hi, "k_lo must not exceed k_hi"
        if not self.name:
            self.name = f"VolComp({self.k_lo:g}-{self.k_hi:g})"

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        if ctx.atr <= 0:
            return None
        r = min(max(ctx.atr_pct_rank, 0.0), 1.0)
        k = self.k_lo + (self.k_hi - self.k_lo) * r
        return ctx.entry - ctx.sign * k * ctx.atr

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        return active_stop

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return _intrabar_hit(ctx, active_stop)


@dataclass
class VwapEmaTrailStop:
    """Structure entry stop, then trail to VWAP / EMA once the trade is working.

    Trails only after TP1 so it cannot tighten a trade that has not yet proved
    anything — the same reasoning that keeps the breakeven ratchet behind TP1.
    """
    anchor: Literal["vwap", "ema20", "ema50"] = "ema20"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"VwapEmaTrail({self.anchor})"

    def initial_stop(self, ctx: SignalContext) -> Optional[float]:
        return _structural_stop(ctx)

    def update_stop(self, ctx: BarContext, active_stop: float) -> float:
        if not ctx.tp1_hit:
            return active_stop
        anchor = {"vwap": ctx.vwap, "ema20": ctx.ema20, "ema50": ctx.ema50}[self.anchor]
        if anchor <= 0:
            return active_stop
        return anchor          # apply_stop() enforces ratchet-only

    def triggered(self, ctx: BarContext, active_stop: float) -> bool:
        return _intrabar_hit(ctx, active_stop)


ALL_POLICIES = (
    PercentBandStop, StructureStop, AtrStop, CandleInvalidationStop,
    VolCompressionStop, VwapEmaTrailStop,
)
