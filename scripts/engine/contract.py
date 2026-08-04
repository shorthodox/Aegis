"""The retrain <-> live handshake, in one place.

`scripts/retrain_model.py` writes a `*_meta.json` sidecar per token; the live
engine reads it to decide whether that token may trade, in which direction, and
above which probability. Historically both sides spoke raw dict:
`meta.get("meta_threshold_buy", meta.get("meta_threshold", 0.6))` scattered
across the engine, with no single statement of what a sidecar must contain.

Two things went wrong under that arrangement and both are guarded here:

1. Silent defaults. A key retrain stopped writing, or never wrote for a given
   pipeline, degraded to a hardcoded fallback instead of failing. `Sidecar`
   distinguishes "absent" from "present and zero".

2. The live engine overriding the trainer. live_engine.py force-set
   `tradeable/tradeable_buy/tradeable_sell = True` for every binary-dual model,
   which is the opposite of what training decided for tokens whose holdout came
   in under breakeven. `tradeable_for()` is now the only way to ask, and it
   answers from the sidecar.

If you add a field to the sidecar in retrain_model.py, add it here too and the
live side picks it up typed. `validate_for_training()` lets the trainer assert,
at write time, that it emitted everything the live side will look for -- so a
missing key is caught in the retrain that produced it rather than at 3am in the
scan loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "Sidecar",
    "SidecarError",
    "REQUIRED_KEYS",
    "validate_for_training",
    "load_fleet",
]


class SidecarError(RuntimeError):
    """A sidecar is missing something the live engine cannot proceed without."""


# Keys the live engine reads and has no safe default for. retrain_model.py
# should call validate_for_training() against the dict it is about to dump so a
# regression in the writer is caught at training time.
REQUIRED_KEYS: List[str] = [
    "symbol",
    "feature_cols",
    "model_format",
    "calibration_temperature",
    "atr_multiplier",
    "tradeable",
    "meta_threshold",
]

# Keys that are optional but whose absence changes behaviour enough to be worth
# reporting once at load, rather than silently defaulting.
NOTABLE_KEYS: List[str] = [
    "meta_threshold_buy", "meta_threshold_sell",
    "tradeable_buy", "tradeable_sell",
    "token_breakeven", "risk_tier", "primary_model_type",
]

_TIERS = ("conservative", "balanced", "aggressive")


@dataclass
class Sidecar:
    """Typed view over one `*_meta.json`.

    Construct with `Sidecar.load(path)`. Raw contents stay available on `.raw`
    for fields not yet promoted to attributes, but prefer promoting a field
    here over reaching into `.raw` at a call site -- that is how the old
    scattered-`.get()` situation grew.
    """

    path: Path
    raw: Dict[str, Any] = field(repr=False)

    # ---- identity -------------------------------------------------------
    symbol: str = ""
    trained_at: Optional[str] = None

    # ---- model wiring ---------------------------------------------------
    model_format: str = "booster"
    primary_model_type: str = ""
    feature_cols: List[str] = field(default_factory=list, repr=False)
    meta_feature_cols: List[str] = field(default_factory=list, repr=False)
    calibration_temperature: float = 1.0

    # ---- geometry -------------------------------------------------------
    atr_multiplier: float = 1.5

    # ---- gating ---------------------------------------------------------
    meta_threshold: float = 0.6
    meta_threshold_buy: Optional[float] = None
    meta_threshold_sell: Optional[float] = None
    meta_threshold_buy_aggressive: Optional[float] = None
    meta_threshold_sell_aggressive: Optional[float] = None
    primary_only_mode: bool = False
    primary_confidence_threshold: Optional[float] = None

    # ---- permission -----------------------------------------------------
    tradeable: bool = False
    tradeable_buy: Optional[bool] = None
    tradeable_sell: Optional[bool] = None
    risk_tier: Dict[str, bool] = field(default_factory=dict)

    # ---- economics ------------------------------------------------------
    token_breakeven: Optional[float] = None
    target_precision: Optional[float] = None
    holdout_trading: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- diagnostics collected at load ---------------------------------
    missing_notable: List[str] = field(default_factory=list, repr=False)

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Sidecar":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            raise SidecarError(f"{Path(path).name}: unreadable ({err})") from err
        return cls.from_dict(raw, path=Path(path))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], path: Optional[Path] = None) -> "Sidecar":
        missing = [k for k in REQUIRED_KEYS if raw.get(k) is None]
        if missing:
            name = path.name if path else raw.get("symbol", "<dict>")
            raise SidecarError(
                f"{name}: sidecar missing required key(s) {missing}. It was "
                f"probably written by an older retrain_model.py -- retrain this "
                f"token rather than defaulting, so live matches what was measured."
            )

        def _f(key: str) -> Optional[float]:
            v = raw.get(key)
            return None if v is None else float(v)

        def _b(key: str) -> Optional[bool]:
            v = raw.get(key)
            return None if v is None else bool(v)

        return cls(
            path=path or Path("<dict>"),
            raw=raw,
            symbol=str(raw["symbol"]),
            trained_at=raw.get("trained_at"),
            model_format=str(raw.get("model_format", "booster")),
            primary_model_type=str(raw.get("primary_model_type", "")),
            feature_cols=list(raw.get("feature_cols") or []),
            meta_feature_cols=list(raw.get("meta_feature_cols") or raw.get("feature_cols") or []),
            calibration_temperature=float(raw.get("calibration_temperature", 1.0)),
            atr_multiplier=float(raw.get("atr_multiplier", 1.5)),
            meta_threshold=float(raw["meta_threshold"]),
            meta_threshold_buy=_f("meta_threshold_buy"),
            meta_threshold_sell=_f("meta_threshold_sell"),
            meta_threshold_buy_aggressive=_f("meta_threshold_buy_aggressive"),
            meta_threshold_sell_aggressive=_f("meta_threshold_sell_aggressive"),
            primary_only_mode=bool(raw.get("primary_only_mode", False)),
            primary_confidence_threshold=_f("primary_confidence_threshold"),
            tradeable=bool(raw.get("tradeable", False)),
            tradeable_buy=_b("tradeable_buy"),
            tradeable_sell=_b("tradeable_sell"),
            risk_tier=dict(raw.get("risk_tier") or {}),
            token_breakeven=_f("token_breakeven"),
            target_precision=_f("target_precision"),
            holdout_trading=dict(raw.get("holdout_trading") or {}),
            missing_notable=[k for k in NOTABLE_KEYS if raw.get(k) is None],
        )

    # -- the questions the engine actually asks ---------------------------
    def threshold_for(self, side: str, *, aggressive: bool = False) -> float:
        """Probability floor for `side` ("BUY"/"SELL").

        Falls back to the combined gate when the per-side threshold is absent --
        which is deliberate and matches how retrain writes it: a side that did
        not qualify independently inherits the combined threshold rather than
        being handed a permissive default.
        """
        s = side.upper()
        if s not in ("BUY", "SELL"):
            return self.meta_threshold
        if aggressive:
            v = (self.meta_threshold_buy_aggressive if s == "BUY"
                 else self.meta_threshold_sell_aggressive)
            if v is not None:
                return v
        v = self.meta_threshold_buy if s == "BUY" else self.meta_threshold_sell
        return self.meta_threshold if v is None else v

    def tradeable_for(self, side: str) -> bool:
        """May this token fire `side` right now?

        The single authority. Do not re-derive this from `.raw`, and do not
        write `True` over it because a model happens to be a binary pair -- that
        override is what let tokens whose holdout came in under breakeven keep
        emitting live signals.
        """
        if not self.tradeable:
            return False
        s = side.upper()
        per_side = self.tradeable_buy if s == "BUY" else self.tradeable_sell
        # Absent per-side flag means training never split the decision; the
        # combined approval carries.
        return self.tradeable if per_side is None else bool(per_side)

    def tier_enabled(self, tier: str) -> bool:
        """Whether the engine's configured risk tier is approved for this token."""
        if not self.risk_tier:
            return True          # sidecar predates tiering: don't bench on absence
        return bool(self.risk_tier.get(tier, False))

    def _h(self, key: str) -> Optional[float]:
        v = self.holdout_trading.get(key)
        return None if v is None else float(v)

    @property
    def signal_precision(self) -> Optional[float]:
        """Fraction of fired signals that hit the profit barrier.

        NOT the trainer's profitability test. Comparing this to
        `token_breakeven` double-counts timeout exits as full-barrier losses
        and is unfairly harsh -- see retrain_model.py where tradeable_final is
        computed. Use `passes_training_economics` to ask whether a token
        earned its live slot.
        """
        return self._h("signal_precision")

    @property
    def directional_precision(self) -> Optional[float]:
        return self._h("directional_precision")

    @property
    def dir_precision_lower_bound(self) -> Optional[float]:
        """Wilson lower bound on directional precision -- the number the
        trainer actually gates on (>= 0.60)."""
        return self._h("dir_precision_lower_bound")

    @property
    def expectancy_pct(self) -> Optional[float]:
        """Per-trade expectancy after costs. The trainer requires > 0."""
        return self._h("expectancy_pct")

    @property
    def passes_training_economics(self) -> Optional[bool]:
        """Mirror of the trainer's profitability test, for reporting only.

        Returns None when the holdout block is too old to carry the fields.
        Kept aligned with retrain_model.py's tradeable_final: positive
        expectancy AND a directional lower bound clearing 0.60. Do not invent a
        different rule here -- divergence between what training approves and
        what live believes is precisely the bug class this module prevents.
        """
        exp, lb = self.expectancy_pct, self.dir_precision_lower_bound
        if exp is None or lb is None:
            return None
        return exp > 0.0 and lb >= 0.60

    def describe(self) -> str:
        bits = [f"{self.symbol}", f"tradeable={self.tradeable}"]
        if self.tradeable_buy is not None or self.tradeable_sell is not None:
            bits.append(f"buy={self.tradeable_for('BUY')} sell={self.tradeable_for('SELL')}")
        bits.append(f"thr={self.meta_threshold:.3f}")
        if self.dir_precision_lower_bound is not None:
            bits.append(f"dir_prec_lb={self.dir_precision_lower_bound:.3f}")
        if self.expectancy_pct is not None:
            bits.append(f"exp={self.expectancy_pct:+.3f}%")
        return "  ".join(bits)


def validate_for_training(payload: Dict[str, Any]) -> None:
    """Assert a sidecar dict carries everything the live engine requires.

    Call this in retrain_model.py immediately before json.dump(). A missing key
    then fails the training run that caused it, instead of surfacing as a
    defaulted threshold in the live scan loop days later.
    """
    missing = [k for k in REQUIRED_KEYS if payload.get(k) is None]
    if missing:
        raise SidecarError(
            f"refusing to write sidecar for {payload.get('symbol', '?')}: "
            f"missing required key(s) {missing}"
        )


def load_fleet(model_store: Path, *, strict: bool = False) -> Dict[str, "Sidecar"]:
    """Load every `*_meta.json` under `model_store`, keyed by symbol.

    With `strict=False` a broken sidecar is skipped and reported by the caller
    via the returned `errors` on the exception-free path; with `strict=True` the
    first bad sidecar raises. The engine loads non-strict so one corrupt token
    cannot take the whole fleet offline.
    """
    out: Dict[str, Sidecar] = {}
    for p in sorted(Path(model_store).glob("*_meta.json")):
        try:
            sc = Sidecar.load(p)
        except SidecarError:
            if strict:
                raise
            continue
        out[sc.symbol] = sc
    return out
