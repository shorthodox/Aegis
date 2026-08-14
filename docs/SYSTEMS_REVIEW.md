# Systems Review — 2026-08-14

Analysis only. No code changed in this pass.

Covers: what the 8-trade sample can and cannot support, the CRV stop anomaly, the systemic
failure mode behind four separate incidents, an audit of how conclusions were reached, and a
draft public note for the track-record restart.

---

## 0. What the 8 trades can and cannot support

**Read this before any section below that mentions the win rate.**

Observed: 8 closed trades, 1 win, 7 losses. Win rate 12.5%. The measured expectation for the
0.70% stop these trades ran under is a **38.3%** win rate.

Exact binomial, n=8, p=0.383:

```
P(X = 0)  = 0.02100
P(X = 1)  = 0.10430
P(X ≤ 1)  = 0.12530      →  12.53%
expected wins = 3.06
```

**A run this bad or worse happens roughly one time in eight when the model is performing exactly
as measured.** That is unremarkable. The sample is fully consistent with the geometry being the
only problem and the run being unlucky.

**No strategy change may be justified by this win rate.** For scale, the 95% confidence interval
on win rate is ±21.3 points at n=20 and still ±13.5 points at n=50. Distinguishing 12.5% from
38.3% needs a sample an order of magnitude larger than we have.

### Two categories, kept separate throughout

| | Needs | Status |
|---|---|---|
| **Statistical claims** — win rate, expectancy, whether the edge is real | Large n | **Nothing is supported.** n=8 |
| **Structural claims** — is the code doing what it was written to do | One observation | **Fully supported.** Deterministic, repeated |

Everything actionable in this review is structural. The performance question remains open and
will stay open until the restarted sample is large enough.

### The structural finding, stated deterministically

10 of 12 trades carried a stop at **exactly 0.700%** of entry — to four decimal places, across
ten different tokens and price scales spanning 0.039 to 607.85:

```
TRB 12.78→12.69054   PYTH 0.03945→0.03917385   TIA 0.3063→0.3041559
DOGE 0.06985→0.06936105   HBAR 0.06536→0.06490248   NEAR 1.612→1.600716
INJ 4.554→4.522122   COMP 15.92→15.80856   SUI 0.6778→0.6730554
```

Ten identical values across ten independent structures is not a coincidence and does not need
statistics. It is `MAX_STOP_PCT` at `trader_gate.py:848-860` overwriting the structural stop
computed upstream, on essentially every trade. **n=1 would have proven it. We have ten.**

The two exceptions confirm the same mechanism:

- **BNB at 0.6630%** — a structural stop that happened to land *inside* the band, so the clamp
  left it alone. Proof the clamp only bites when structure is wider, which is nearly always.
- **ADA at 1.3000%** — placed at 10:47:42, after the fix deployed. The new cap, and the first
  signal in the book with a structural stop.

---

## 1. The CRV anomaly — resolved, and not a bug

**CRV/USDT: entry 0.2513, stop 0.2513, distance 0.0000%, outcome WIN, pnl +1.40%.**

### It cannot have come from the stop placement

`MIN_STOP_PCT = 0.50`, and the clamp is `_banded = min(max(_structural, _lo), _hi)` with
`_lo = entry × 0.50%`. For CRV that floor is 0.00126. The clamp is mathematically incapable of
returning a zero-distance stop, and the guard rails upstream (`1.50×ATR` floor at `:704-710`,
`MAX_STOP_ATR` reject at stage 2) all widen rather than narrow. **No stop-placement path
produces `stop == entry`.**

### What actually produced it

`scripts/engine/exits.py:346`, on TP1 being tagged:

```python
_partial('TP1_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[0], pos.take_profit_1)
self._tp1_hit[symbol] = True
pos.stop_loss = pos.entry_price          # break-even
print(f'[{symbol}] TP1_HIT ... SL to break-even ({pos.entry_price:.6g})')
```

The **break-even ratchet**. Once TP1 is banked the stop moves to entry to protect the rung. The
same move appears at `:497` and `:499` after TP2.

The arithmetic closes it. The v87 TP1 is **1.5%** (`trader_gate.py:152`). CRV booked **+1.40%**.
1.5% gross − 0.10% round-trip cost = **1.40% net**, exact to the reported precision.

**CRV hit TP1, banked a partial, and its stop ratcheted to break-even. The stop in the record is
the stop at exit, not the stop at entry.**

### Can it have been booked as a win incorrectly?

**No.** It is a real win, correctly classified. The trade reached its first profit objective and
banked it. It is the smallest win the ladder can produce, but it is a win, and removing it would
misstate the record in the other direction.

**No other trade in the book has a degenerate stop.** The scan across all 12 covers every
outcome, and CRV is the only one — consistent with it being the only trade to reach TP1.

### The real finding here is a reporting defect, not a trading one

The track record's `stop_loss` field holds the **current** stop, which for any ratcheted trade
is not the stop the trade was taken with. Consequences:

1. **Published R:R is wrong for any trade that reached TP1.** CRV reads as an infinite-R trade.
2. **"Did it hit its stop?" is unanswerable from the record** — the field has been overwritten.
3. It cost real time in this investigation: the anomaly read as a data-integrity failure serious
   enough to call the loss column unreliable, and it was a display artefact.

The fix is additive and small — record `entry_stop` alongside the live `stop_loss` — but it is a
change and this pass is analysis. Logged, not implemented.

---

## 2. The systemic finding — one failure mode, four instances

This is the priority section.

### The four incidents

| # | Incident | The repo said | Production did |
|---|---|---|---|
| 1 | `firestore.rules` | A reviewed ruleset with field-level protection | Ran `40d739b2` from 2026-05-22 — `allow read, write`, no protection. The file had **never been deployed** |
| 2 | `STATE_DIR` / volume | State persists on a mounted volume | **No volume existed.** `/app/data` on the container overlay, wiped every deploy |
| 3 | Engine Firestore writes | Mirrors the track record to `engine_state` | Wrote to database `(default)`, which **has never existed**. Every write failed |
| 4 | `stop_policies.py` | Stop placement behind one measurable interface | Never imported. A downstream clamp in `trader_gate.py` decided every stop |

### The common mechanism

All four are the same error: **an artefact existing in the repository was treated as evidence
that it was in effect in production.** Writing is not deploying; importing is not wiring;
configuring is not mounting; a constant is not a behaviour.

That assumption is normal and usually harmless. What made these four survive is that **every one
had a silent failure path**. Nothing anywhere reported the divergence:

| # | Why it stayed invisible |
|---|---|
| 1 | Rules are inert until deployed, and nothing compares repo to live. A rules file that is never deployed produces **no error at all** — not at boot, not at build, not at runtime. The only signal was an absence of denials nobody was looking for. |
| 2 | `mkdir(parents=True, exist_ok=True)` **creates the missing directory**, so an unmounted path is indistinguishable from a mounted one by every signal the process had: it exists, it is writable, writes succeed, reads come back. The `except` clause then silently fell back to the in-repo dir. |
| 3 | `try/except` that sets `_FS_DOWN = True` and returns. A circuit breaker that trips on the **first** call, never retries, and never reports. The comment above it even anticipated the failure — *"the volume only helps if it's actually mounted, which kept not being the case"* — and the code recorded the risk instead of asserting against it. |
| 4 | No failure at all — the clamp is doing exactly what it was written to do. The module's own docstring says *"NOT imported by the engine"*, so the divergence was documented and then forgotten. Nothing distinguishes "measured policy" from "policy in force". |

Incidents 1–3 share a sharper sub-pattern: **the operation that should have failed loudly
succeeded quietly instead** — `mkdir` creating, `except` swallowing, a non-deploy being a no-op.
Incident 4 is the pure form: no error was possible, because nothing was wrong except the belief.

### What would have caught the most, per unit of effort

| Mechanism | Catches | Misses | Effort |
|---|---|---|---|
| **Runtime self-report endpoint** — reports *observed* state | **1, 2, 3, 4** | — | ~1 file |
| Config reconciliation check | 1, 2, 3 | 4 (no config is wrong) | medium |
| Startup assertions that fail loudly | 2, 3 | 1 (outside the app), 4 (nothing to assert) | small, per-site |
| Post-deploy smoke suite | 1, 2, partially 3 | 4 | medium–large |

**The self-report endpoint wins, and not narrowly.** Every diagnosis in this incident required
going around the application to ask the live system directly: the Rules API for the ruleset,
`railway ssh` for the mount, the Firestore Admin API for the database list, production logs for
the clamp. **All four were invisible from inside the app and trivially visible from outside it.**
That is the gap.

Startup assertions are still worth having for the fatal subset — the mount guard shipped in
`da69311a` is one, and it converts incident 2 from silent corruption into a refused boot. But
they only cover conditions where the right response is to not start. Incidents 1 and 4 are not
of that kind.

### The smallest thing that works

**One authenticated endpoint that reports what the running process actually observes.** Not what
it was configured with — what it can verify, right now:

- Firestore: which database id, and a live probe confirming it exists and is writable
- State: `STATE_DIR`, `is_mount()`, writable, free space
- Rules: the live ruleset id fetched from the Rules API, beside the hash of the repo file
- Exits: the effective stop band, and whether it is currently binding on live signals
- Build: the deployed commit SHA

**The distinction that makes it work: every line must be *derived at runtime*, never printed
from a constant.** An endpoint that echoes `MAX_STOP_PCT` tells you nothing — the constant was
never in doubt. An endpoint that reports "the last 20 signals all stopped at exactly the cap"
would have shown incident 4 on day one. `Path(STATE_DIR).is_mount()` catches incident 2;
`Path(STATE_DIR)` printed as a string does not.

Four instances of one bug do not need four fixes. They need **one way to ask the system what it
is actually doing**, and the discipline to ask it after every deploy.

---

## 3. Decision-making audit

The hypothesis was: *inference where execution was available and cheap.* **Confirmed** — six
cases, and the cheap check existed in every one.

| # | Conclusion | Evidence actually available | What was assumed | Cheap check that would have prevented it |
|---|---|---|---|---|
| 1 | "`app.js:155`'s client trade-close fails in production" | The live ruleset, one API call away | That `firestore.rules` in the repo was what production ran | Query the live ruleset first — **the same call that later disproved it** |
| 2 | "The Railway service is gone" — from a 404 on `qg0oi1ii` | `railway status`, `railway domain list` | That the hostname in DNS was the service's current domain | `railway domain list` — showed the real host answering 200 |
| 3 | "`scalp_signals` has no reader" | The frontend source | That a Python call-site inventory covered all consumers | `grep -r scalp_signals web/` — one command, found the reader immediately |
| 4 | "Two databases, both holding data" | The Firestore Admin API | That a named DB in code implied a named DB in existence | `GET /v1/projects/{p}/databases` — showed only one |
| 5 | "Two rules are broken" (first C2 run) | The seed file itself | That uids copied from a truncated listing were complete | Assert seeded ids exist before testing — **now in the harness** |
| 6 | "The event-loop work is finished" (after `60ddeebe`) | The same grep, re-run | That fixing the loops I had found meant finding all of them | Re-run the search after fixing, not before |

Three observations that go beyond the pattern:

**The correction always came from execution, never from more careful reading.** In every case the
error was fixed by running something — an API call, a grep, a probe. Re-reading produced
confident wrong answers repeatedly. This is why C2's insistence on the emulator was correct, and
why it immediately paid: it caught #5, which was my own error one level deeper than the rules.

**Cases 1 and 4 are the same error as the systemic finding in §2.** A conclusion drawn from the
repository was assumed to hold in production. The audit finding and the systems finding are one
thing seen from two directions — which is itself evidence that §2's mechanism is the right one.

**Where inference was correct and necessary.** Not all of it was wrong. The `MAX_STOP_PCT`
diagnosis was reached by reading code and confirmed by data; the B1′-vs-B1‴ analysis correctly
predicted that an OR would resolve to the stale stub, before any test ran. The distinction is
that these were *predictions about behaviour under stated conditions*, verified afterwards. The
failures were all *assumptions about the state of the world* that could have been checked and
weren't.

---

## 4. Track-record restart note (draft)

For publication alongside the reset. Factual, no spin.

> ### Track record reset — 14 August 2026
>
> The published track record has been reset to zero. This note explains why, and what the
> previous numbers were.
>
> **What happened.** Until 13 August, the server had no persistent disk attached. Runtime state
> was written to the container's temporary filesystem, which is erased every time the service is
> redeployed. The track record was therefore wiped on each deploy without anyone noticing. A
> Firestore backup that was supposed to protect against exactly this had been misconfigured since
> it was written and had never saved a single record. Both safeguards were broken at the same
> time, and neither reported a failure.
>
> A persistent volume was attached on 13 August and verified by confirming a test file survives a
> redeploy. State now persists.
>
> **What the previous sample was.** After the volume was attached, the engine recorded **8 closed
> trades: 1 win, 7 losses.** We are not presenting that as a result, in either direction. Eight
> trades cannot establish a win rate — at the strategy's measured 38.3%, a run of 1 win or fewer
> in 8 happens about one time in eight by chance alone. It is too small to be evidence of
> anything, and we would say the same if it had come out the other way.
>
> **What changed.** Those 8 trades exposed a real defect. The engine calculates a stop from market
> structure — below the support level a trade leans on — and a separate risk cap was then pulling
> every stop in to 0.70% of the entry price. Ten of the twelve signals in that sample had stops at
> exactly 0.70%, on ten different tokens. The stop was no longer marking the point where the trade
> idea fails; it was sitting in front of that point, and normal price movement toward the level
> was closing trades before the idea had been tested.
>
> The cap has been widened to 1.30% so stops sit at the structural level again. Testing over
> 19,140 historical trades measured this as better on expectancy, win rate, and return per unit of
> risk. **Expect fewer signals** — a wider stop means a trade must offer more to be worth taking,
> so more setups are now rejected.
>
> **Why the record restarts rather than continues.** Those 8 trades were taken with geometry we
> have since established was wrong. Carrying them forward would mix two different strategies in
> one number. The sample restarts from the first signal placed with the corrected stop.
>
> We publish losses as readily as wins. The reset is not an attempt to bury a bad run — the run is
> stated above in full. It restarts because the numbers would not mean anything otherwise.

---

## Logged, not implemented

Analysis-only pass. Deliberately not built:

1. **The runtime self-report endpoint** (§2) — the single highest-value item in this review
2. **`entry_stop` recorded alongside `stop_loss`** (§1) — published R:R is wrong for ratcheted trades
3. **`trial_end_ts`** — the fix that closes the paywall, already queued
