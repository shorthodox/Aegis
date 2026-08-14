# Why TAO and BCH lost while moving the right way — 2026-08-14

Both signals called the direction correctly. Both were stopped out first, and price then went
where the signal said it would. This is an investigation into why, based on the two closed
records and the code paths that produced them.

**Two separate defects, one symptom.** Both trades ended with a stop sitting *between entry and
the level the trade leaned on*, so the move that tested the level collected the stop before the
thesis was tested. They got there by different routes.

---

## The evidence

From `/data/track_record.json`:

| | Entry | Stop | Exit | Risk | In ATR | Exit reason |
|---|---|---|---|---|---|---|
| **TAO/USDT** | 198.80 | 196.2156 | 194.80 | **1.3000%** | 1.765 | `STOP_HIT` |
| **BCH/USDT** | 203.20 | 201.5714 | 200.20 | **0.8015%** | **1.500** | `STOP_HIT` |

Two suspiciously exact numbers. TAO is at `MAX_STOP_PCT` (1.30%) to four decimals. BCH is at
`MIN_STOP_ATR` (1.50 ATR) to four decimals. Neither is a structural level — both are clamps.

Support at the time (TAO 193.70 from the chart card, reconstructed below and confirmed exact):

```
TAO:  entry 198.80 → support 193.70 is 2.565% away.  Stop sat at 1.300%.
BCH:  entry 203.20 → support 199.70 is 1.722% away.  Stop sat at 0.802%.
```

**In both cases the stop was closer to entry than the level was.** Price fell toward the level,
took the stop, and reversed. TAO's low of 194.80 never even reached 193.70.

---

## Defect 1 — `_budget_band` undoes the support-clearing (TAO)

`risk.py:calculate_stops`, the path taken when the gate supplies no `sl_override`:

```python
risk = ((price - support) + buf) if (0 < support < price) else cap
risk = max(floor, min(risk, cap))
sl   = price - risk
if 0 < support < price:
    sl = min(sl, support - buf)            # ← clears the level. 192.968
risk = self._budget_band(price, price - sl)  # ← re-bands. 2.5844
sl   = price - risk                          # ← 196.2156, back ABOVE support
```

The comment immediately above that block states the intent plainly:

> *"the support-clearing exception is deliberately allowed to widen past the band. A stop sitting
> ON the level is the one place the market reliably collects it, which is worse than a wide stop."*

**The code does the opposite of its own comment.** `_budget_band` runs *after* the
support-clearing `min()`, so it pulls the stop straight back over the level the previous line
just cleared. The exception is written, and then cancelled two lines later.

Reconstructed against TAO's real inputs (`price 198.8, support 193.70, atr 1.46428571`):

```
structural stop below support : 192.9679   (2.934% risk)   ← correct, clears the level
after _budget_band            : 196.2156   (1.300% risk)   ← what actually shipped
RECORDED stop                 : 196.2156   ✅ exact match
```

The stop that would have survived the test — 192.97, below support — was replaced by one 3.25
points higher, inside the noise the level generates. Price bottomed at 194.80. **The correct
stop would not have been hit.**

## Defect 2 — the override path's support-clearing never fired (BCH)

BCH's stop is exactly `1.50 × ATR`, which is `MIN_STOP_ATR` — a **trader_gate** constant, not a
`risk.py` one. So BCH came through the `sl_override` branch, where the gate's own stop is taken
verbatim. That branch has its own support-clearing guard:

```python
sl = sl_override
if 0 < support < price and sl >= support - buf:
    sl = min(sl, support - buf)
```

With `support = 199.70`, `sl = 201.5714 ≥ 199.157`, this should have widened the stop to 199.157.
It did not — the recorded stop is the unmodified gate value. **Therefore `result['support']` was
0, absent, or not below price at 12:48:46.**

So the gate leaned on a level within `1.50 ATR` of entry (its floor bound, meaning the structural
distance was *tighter* than the floor), while the rolling support 1.72% away never entered the
calculation.

Unlike TAO, this one cannot be reconstructed further, for the reason in §4.

## Defect 3 — a working order can never become a trade

This is the direct answer to *"the entry should be a bit later."*

`trader_gate` already models the behaviour you want. `AT_LEVEL_ATR = 0.35`:

```python
if dist_atr <= AT_LEVEL_ATR and ok:
    action = ACTION_ENTER      # at market, price is already at the level
elif dist_atr <= REACH_ATR:    # 2.50
    action = ACTION_WORK       # resting order AT the level, expires in 8 bars
else:
    reject('level is N ATR away — not a trade yet')
```

and `entry_px = price if action == ACTION_ENTER else level`. A far-from-level setup is supposed
to rest an order *at the level* and wait.

**The engine never fills it.** `_working_orders` is declared as
`Dict[str, float]` — a map of key to first-seen timestamp — and every use of it is
`setdefault`, `pop`, or an age comparison. On `ACTION_WORK` the engine sets `fire = False`,
`signal = 'HOLD'`, publishes a card with `pending_target = level`, and returns. Nothing anywhere
converts a working order into a position when price reaches the level. After 8 bars it expires.

Consequently **every trade the system takes is a market entry**, and
`entry_price = round(price, 8)` in `positions.py` confirms it — the open path never reads
`plan.level`.

The mechanism for entering later, at the level, exists in the gate and is not implemented
downstream. Working orders are a display feature.

## 4 — What cannot be audited, and why

`Position` carries `entry_support` and `entry_resistance` — explicitly, per its own comment, so
the chart can show the S/R the trade was judged against. **`TradeRecord` does not carry them.**

So once a trade closes, the level it leaned on is gone. The question *"was the stop below the
level?"* — the exact question this investigation exists to answer — is unanswerable from the
published record. TAO was only reconstructible because the chart card still displays 193.70 and
the arithmetic happened to match to four decimals. BCH is not recoverable at all.

This is the same defect class as `entry_stop`, fixed earlier today: the record keeps the numbers
that are convenient to write, not the ones needed to audit the decision.

---

## What this does and does not say about performance

**Structural, and needs no statistics:** the stops in both trades were set by clamps, not by
structure, and both sat between entry and the level. That is determined from two trades, and one
would have sufficed.

**Statistical, and unsupported:** nothing here says the strategy is profitable or unprofitable.
Two trades is two trades. The `n=8` analysis in `SYSTEMS_REVIEW.md` applies unchanged — no
strategy conclusion may rest on this sample.

What can be said is narrower and still useful: **in both trades the recorded stop was closer to
entry than the level the setup was built on, and in TAO's case the correct stop demonstrably
would not have been hit.**

---

## Recommendations, in order of value

1. **Move `_budget_band` before the support-clearing**, or exempt a support-cleared stop from the
   band entirely — which is what the existing comment already claims happens. One-line ordering
   change; makes the code do what it says. Directly fixes TAO's failure mode.

2. **Persist `entry_support` / `entry_resistance` on `TradeRecord`.** Without them this
   investigation cannot be repeated on the next pair of trades. Additive, no behaviour change.

3. **Decide what a working order is for.** Either implement the fill — when price reaches
   `plan.level` inside the expiry window, open at the level — or stop publishing a
   `pending_target` the system will never act on. Today it is a card that promises an entry the
   engine cannot take. This is the largest change of the three and needs a decision, not a patch.

4. **Investigate why `result['support']` was absent for BCH.** The override path's
   support-clearing depends on it, and it silently does nothing when the field is missing. A stop
   guard that no-ops on missing input is the same silent-failure pattern catalogued in
   `SYSTEMS_REVIEW.md` §2.

None of these are implemented — this pass is analysis only.
