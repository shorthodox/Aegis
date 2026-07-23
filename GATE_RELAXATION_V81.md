# Gate Relaxation v81 - Fire at Support/Resistance Zones

## Core Problem
Engine was too strict with the "wait for exact level" logic - signals stayed PENDING indefinitely because price rarely reaches within 1.2% of calculated support/resistance. This caused ZERO fires in 12+ hours despite many tradable setups.

## Changes Made

### 1. **Guard M: Relaxed Pending Logic (Lines 3542-3580)**
**Before:** Fire ONLY within 1.2% (`PENDING_NEAR_PCT`) of the level, otherwise PENDING forever.

**After:** Fire if ANY of these conditions are true:
- Price is **at the level** (within 1.2%)
- Price **came from the level** (tag-and-reject reversal)
- **5m reversal confirmed** (3+ of last 4 candles in signal direction OR reversal pattern)
- Price is **in the correct S/R zone** (rp ≤ 0.35 for BUY, rp ≥ 0.65 for SELL)

```python
_price_in_zone = ((new_side == 'BUY' and _rp_m <= self.STRUCT_SUPPORT_ZONE) or
                 (new_side == 'SELL' and _rp_m >= self.STRUCT_RESISTANCE_ZONE))

_should_wait = (_target_m is None or 
               (not _at_level_m and not _came_from_m and not _has_5m_reversal and not _price_in_zone))
```

### 2. **Guard M: Zone Awareness (Lines 3507-3530)**
**Before:** Hard-blocked signals in wrong location (BUY below 0.5, SELL above 0.5).

**After:** Only hard-block if in **OPPOSITE zone WITHOUT a break**:
- BUY in RESISTANCE_ZONE (rp ≥ 0.65) → blocked ONLY if resistance not broken
- SELL in SUPPORT_ZONE (rp ≤ 0.35) → blocked ONLY if support not broken
- BUY in SUPPORT_ZONE or SELL in RESISTANCE_ZONE → **ALLOWED**

```python
_in_correct_zone = ((new_side == 'BUY' and _rp_m <= self.STRUCT_SUPPORT_ZONE) or
                   (new_side == 'SELL' and _rp_m >= self.STRUCT_RESISTANCE_ZONE))

# Only hard-block if NOT at level AND NOT coming from level AND in opposite zone
if (_rp_m is not None and not _at_level_m and not _came_from_m and not _in_correct_zone):
    # Block only wrong-zone entries
```

### 3. **Guard K: Zone-Based Location Gate (Lines 3668-3684)**
**Before:** Strict "wrong location" veto based on midpoint (0.5).

**After:** Allow fires at S/R zones, block only dangerous wrong-zone entries:
- Allow BUY at SUPPORT_ZONE (rp ≤ 0.35) ✓
- Allow SELL at RESISTANCE_ZONE (rp ≥ 0.65) ✓
- Block BUY at RESISTANCE_ZONE (unless already broken) ✗
- Block SELL at SUPPORT_ZONE (unless already broken) ✗

```python
_wrong_zone = (
    (new_side == 'BUY'  and _at_resist_zone and not bool(result.get('resistance_broken_recent'))) or
    (new_side == 'SELL' and _at_support_zone and not bool(result.get('support_broken_recent')))
)
```

## Firing Flow (Revised)

1. **Check if in correct zone** → if yes, proceed
2. **Check if at level or came from level** → if yes, proceed  
3. **Check 5m reversal (3x5m candles)** → if yes, proceed
4. **Check if in opposite zone without break** → if yes, BLOCK
5. **Guard J: 5m momentum confirmation** → requires 3x5m candles in direction or reversal pattern
6. **Guard K: Important level check** → block wrong-side fires at known resisted/supported levels (unless broken)
7. **Fire at zone with 5m confirmation** ✓

## What This Means

- ✅ **SELL at resistance in BULL regime** → fires if 5m shows bearish turn + 3x5m candles
- ✅ **BUY at support in BEAR regime** → fires if 5m shows bullish turn + 3x5m candles  
- ✅ **Fire when coming from level** (tag-reject reversal) → immediate fire with zone confirmation
- ✅ **Fire when 5m reversal confirmed at zone** → no need to wait for exact level touch
- ✅ **Fire when model agrees and technicals align** → volume, MACD, BOS gates all agree
- ❌ **Block dangerous reverse entries** → BUY at unbroken resistance, SELL at unbroken support

## Expected Result

**Before:** ~0 signals in 12 hours (too strict pending)  
**After:** Signals fire when market reaches S/R zones with 5m reversal confirmation + volume/momentum alignment

---

**Version:** v81  
**Date:** 2026-07-23  
**Doctrine:** Fire at Support/Resistance Zones with 5m Confirmation
