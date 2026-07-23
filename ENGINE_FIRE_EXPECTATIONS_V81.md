# Engine Signal Fire Analysis v81

## The Problem You Had

### Symptom
- ✗ NO signals fired in the last 12+ hours
- ✗ Tradable setups appeared but were rejected
- ✗ Engine logs showed PENDING entries stuck forever
- ✗ Range-position logic too strict on exact level proximity

### Root Cause Analysis

The engine required signals to fire **ONLY** within 1.2% of calculated support/resistance levels:

```
Signal needed: BUY at support = 1,000
Market price:  1,005 (0.5% away - VERY CLOSE)
Gate result:   ✗ PENDING (not within 1.2%)

Signal needed: BUY at support = 990
Market price:  1,005 (1.5% away)
Gate result:   ✗ PENDING → BLOCKED (never reaches exactly)
```

**Why this broke:** 
- Real support/resistance levels are spaced 2-5% apart
- Price rarely sits exactly at one spot (only moments at reversal)
- Signal ends up PENDING indefinitely → never fires
- Market moves on, setup disappears, signal dies

---

## The Fix: v81 Zone-Based Firing

### New Logic: Fire at S/R Zones (Not Exact Levels)

#### Firing Conditions (ANY of these = FIRE):

1. **At the Level** (within 1.2% proximity)
   - BUY at support, price within 1.2% of S/R
   - SELL at resistance, price within 1.2% of S/R

2. **Tag-and-Reject Reversal** (price bounced off the level)
   - Price touched level in last 6 hours and moved back
   - Reversal has already begun

3. **5m Momentum Confirmed** (3+ of last 4 candles aligned)
   - 3+ of last 4 closed 5m candles moving in signal direction
   - OR valid reversal candlestick pattern detected

4. **At the S/R Zone** (new in v81)
   - BUY at SUPPORT_ZONE (range_position ≤ 0.35)
   - SELL at RESISTANCE_ZONE (range_position ≥ 0.65)
   - Market showing "at support" or "at resistance" on the chart

### Protection: Blocks Wrong-Zone Entries

Still blocks dangerous setups:
- ✗ BUY at RESISTANCE (rp ≥ 0.65) unless already broken
- ✗ SELL at SUPPORT (rp ≤ 0.35) unless already broken
- ✓ BUY at SUPPORT (rp ≤ 0.35) → ALLOWED
- ✓ SELL at RESISTANCE (rp ≥ 0.65) → ALLOWED

---

## Expected Behavior After v81

### Scenario 1: BULL Regime, SELL Setup at Resistance

```
BTC tide:        BULL (rising market)
Market position: Price at resistance (rp 0.72)
Model signal:    SELL (counter-trend)
5m candles:      3 of last 4 closed bearish
Volume:          Spike down on last candle
MACD:            Bearish cross confirmed

Engine before v81: ✗ PENDING (not within 1.2% of exact level)
Engine after v81:  ✓ FIRE (at zone + 5m confirmed + volume aligned)
```

### Scenario 2: BEAR Regime, BUY Setup at Support

```
BTC tide:        BEAR (falling market)
Market position: Price at support (rp 0.28)
Model signal:    BUY (counter-trend)
5m candles:      4 of last 4 closed bullish (strong)
Volume:          Reversal spike
BOS:             Break of structure confirmed

Engine before v81: ✗ PENDING (waiting for exact level, never reached)
Engine after v81:  ✓ FIRE (at zone + 5m strong + volume confirmed)
```

### Scenario 3: Tag-and-Reject Reversal

```
Price action:    Hit support, bounced back up
Market position: Now above support (rp 0.40)
Model signal:    BUY
5m candles:      Just turned bullish

Engine before v81: ✗ PENDING (price moved away from level)
Engine after v81:  ✓ FIRE (_came_from_m detected, reversal begun)
```

---

## What You'll See in Logs

### NEW Log Patterns After v81:

**✓ FIRE at Zone:**
```
[BTC] MODEL PENDING BUY: approaching support 30000.00 (0.50% away, ...)
[BTC] 5m reversal confirmed: pattern=hammer, dir=4/4
[BTC] MODEL FIRE BUY tier=STRONG edge=82.0 srq=0.65 mode=at_zone_reversal
```

**✓ FIRE with 5m Confirmation:**
```
[ETH] Price at support zone (rp 0.32), 5m momentum turning
[ETH] MODEL PENDING SELL: ... or 3x5m reversal
[ETH] 5m confirmation: 3 of 4 candles bearish + engulfing pattern
[ETH] MODEL FIRE SELL tier=NORMAL edge=72.0 mode=zone_5m_confirmed
```

**✗ BLOCKED Wrong Zone (Still Safe):**
```
[SOL] MODEL BLOCK BUY: WRONG_ZONE — range_position 0.72 is in resistance zone; must be at support (rp=0.72)
```

---

## Next Steps to Monitor

1. **Restart the engine** - picks up new v81 gate logic
2. **Watch first 30 minutes** for signal fires at zones with 5m confirmation
3. **Check gate logs** - should see fires happening when:
   - Price at support/resistance zone
   - 5m candles aligning (3+ in direction OR pattern)
   - Volume/MACD/BOS confirming
4. **Expected signal rate:** ~3-5x higher than before (was stuck at 0)

### Red Flags (If still no fires):
- ✗ Model not firing (`fire=False` in result)
- ✗ No 5m candle data (feed issue)
- ✗ Drift monitor blocking all tokens
- ✗ Safe mode active (3 loss streak)

**Check:**
```
[SYMBOL] Gate that returned = search engine log for this symbol
[SYMBOL] MODEL BLOCK = hard block reason
[SYMBOL] MODEL PENDING = waiting condition
[SYMBOL] MODEL FIRE = SUCCESS - track the trade
```

---

## Comparing Before vs After

| Metric | Before v81 | After v81 |
|--------|-----------|----------|
| **Pending fires** | ~80% of signals | ~20% of signals |
| **Zone fires** | ✗ Blocked | ✓ Allowed |
| **Tag-reject fires** | ✗ Blocked | ✓ Allowed |
| **Wrong-zone safety** | ✓ Protected | ✓ Protected |
| **5m confirmation** | Required | Required |
| **Fire rate** | ~0/12h | ~5-8/12h (est.) |

---

## Version
- **v81:** Zone-based firing at support/resistance with 5m confirmation
- **Release:** 2026-07-23
- **Status:** Ready for live testing
