# Task 0 — Authorisation Translation Report

**Status:** awaiting approval. No code written.
**Scope:** the `users` collection and its subcollections only, per the Phase 1 scope decision.

Four findings below contradict or extend the Phase 0 inventory, because Phase 0 scanned Python
and the authorisation surface is in the rules file and the JS. Two are live security holes.
One blocks Task 3 as currently specified.

---

## 1. `firestore.rules` reproduced verbatim

Source: `firestore.rules` at repo root, `rules_version = '2'`. Reproduced in full; the
`users` block is §1 lines 60–91.

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    function isAuthed() {
      return request.auth != null;
    }

    // True if the doc ID matches the caller's uid OR their email.
    // Needed because the backend keys user docs by email while the
    // client SDK (ensureUserDocumentV2) keys them by uid.
    function ownsDoc(userId) {
      return request.auth.uid == userId
          || request.auth.token.email == userId;
    }

    // Fetch user data — try uid-keyed doc first (client-created),
    // fall back to email-keyed doc (backend-created via Admin SDK).
    function getUserData() {
      let uidPath   = /databases/$(database)/documents/users/$(request.auth.uid);
      let emailPath = /databases/$(database)/documents/users/$(request.auth.token.email);
      return exists(uidPath)
        ? get(uidPath).data
        : (request.auth.token.email != null && exists(emailPath)
            ? get(emailPath).data
            : null);
    }

    function isSubscriptionActive() {
      let d = getUserData();
      let plan = d != null ? d.get('plan', d.get('tier', '')) : '';
      return plan == 'trial'
          || plan == 'pro'
          || plan == 'premium'
          || plan == 'intermediate'
          || plan == 'basic';
    }

    function planFieldsUnchanged() {
      let protected = ['plan', 'tier', 'trial_end', 'trialEnd',
                       'subscription_active', 'subscription'];
      return !request.resource.data.diff(resource.data)
                .affectedKeys().hasAny(protected);
    }

    match /users/{userId} {
      allow read: if isAuthed() && ownsDoc(userId);

      allow create: if isAuthed()
        && ownsDoc(userId)
        && (!('plan' in request.resource.data)
            || request.resource.data.plan == 'trial')
        && !('subscription_active' in request.resource.data);

      allow update: if isAuthed()
        && ownsDoc(userId)
        && planFieldsUnchanged();

      allow delete: if false;

      match /preferences/{doc=**} {
        allow read, write: if isAuthed() && ownsDoc(userId);
      }

      match /trades/{doc=**} {
        allow read:  if isAuthed() && ownsDoc(userId);
        allow write: if false;
      }
    }

    match /signals/{signalId} {
      allow read:  if isAuthed() && isSubscriptionActive();
      allow write: if false;
    }

    match /analytics/{docId} {
      allow read:  if isAuthed();
      allow write: if false;
    }

    match /reviews/{reviewId} {
      allow read:  if true;
      allow write: if false;
    }

    match /trades/{tradeId} {
      allow read:  if isAuthed()
        && resource.data.get('userId', '') == request.auth.uid;
      allow write: if false;
    }

    match /subscriptions/{userId} {
      allow read:  if isAuthed() && request.auth.uid == userId;
      allow write: if false;
    }

    match /transactions/{transactionId} {
      allow read: if isAuthed()
        && resource.data.get('userId', '') == request.auth.uid;
      allow create: if isAuthed()
        && request.resource.data.get('userId', '') == request.auth.uid;
      allow update, delete: if false;
    }

    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

---

## 2. Rule → FastAPI dependency mapping

**Default posture, applied to every endpoint below:** the target row is derived from the
verified Firebase token. No endpoint accepts a user identifier from the client. This is
strictly tighter than `ownsDoc()`, which accepted a client-supplied path segment and then
compared it — the comparison is what made it safe, and removing the parameter removes the
class of bug entirely.

`ownsDoc(userId)` matched **either** `request.auth.uid` **or** `request.auth.token.email`.
Its faithful equivalent is a resolver that looks up by both, because the dual key space means
a caller's row may be filed under either:

```python
async def current_user_row(token = Depends(verify_firebase_token)) -> UserRow:
    """Resolve the caller's row. Mirrors ownsDoc(): uid-keyed first, then email-keyed.
    Never takes an identifier from the request."""
    row = repo.users.by_doc_id(token.uid)
    if row is None and token.email:
        row = repo.users.by_doc_id(token.email)
    if row is None:
        raise HTTPException(404, "User not found")
    return row
```

| Rule | Condition | FastAPI equivalent | On failure |
|---|---|---|---|
| `users` **read** | `isAuthed() && ownsDoc(userId)` | `Depends(current_user_row)` — no ID parameter exists to tamper with | 401 unauthenticated / 404 no row |
| `users` **create** | `isAuthed() && ownsDoc() && plan in (absent,'trial') && no subscription_active` | `POST /api/me/bootstrap`; server sets `plan='trial'`; `plan`, `subscription_*`, `trial_*` are **not accepted from the body** | 409 if row exists |
| `users` **update** | `isAuthed() && ownsDoc() && planFieldsUnchanged()` | `PATCH /api/me` with a strict Pydantic allow-list model; unlisted fields rejected, not ignored | 422 on unknown field |
| `users` **delete** | `if false` | no endpoint exists | 405 |
| `preferences/**` read+write | `isAuthed() && ownsDoc()` | `GET/PATCH /api/me/preferences` via `current_user_row` | 401 / 404 |
| `trades/**` read | `isAuthed() && ownsDoc()` | `GET /api/me/trades` — `WHERE user_doc_id = :caller`, never a supplied id | 401 / 404 |
| `trades/**` write | `if false` | no client write path; existing `/api/trades/*` keeps its Admin-SDK-equivalent server path | 405 |

**Allow-list for `PATCH /api/me`** — derived by subtracting everything the rules protect, and
everything §3.1 says they *should* have protected, from the observed field set:

`full_name`, `phone_number`, `location`, `loginMethods`, `lastLogin`, `preferences` (map),
`usage` (map).

Everything else — `plan`, `plan_type`, `tier`, `trial_*`, `subscription*`, `suspended`,
`suspension_reason`, `status`, `expires_at`, `activated_at`, `api_key_hash`, `dev_code_used`,
`capital`, `risk_pct`, `order_id`, `payment_id` — is server-written only.

---

## 3. Flagged: rules that are more permissive than they should be

You asked for inherited over-permission to be called out rather than faithfully reproduced.
There are three, and the first two are exploitable today.

### 3.1 `planFieldsUnchanged()` protects six fields; entitlement depends on more ⚠️ **live hole**

```javascript
let protected = ['plan', 'tier', 'trial_end', 'trialEnd',
                 'subscription_active', 'subscription'];
```

The Phase 0 field inventory found 33 fields on `users`. Several are **not** in the protected
list, so an authenticated user can write them to their own document via the client SDK. I
checked each against its server-side read sites rather than assuming impact:

| Unprotected field | Read at | Real consequence |
|---|---|---|
| `trial_used` | `main.py:4282` — `if user_doc.get("trial_used")` | **Exploitable.** Set `false` → redeem the trial again |
| `trial_active` | `main.py:4372` | Exploitable in that flow |
| `subscription_end` | `main.py:5644` | Reminder-email scheduling only — not an entitlement gate |
| `trial_start` | — | Display arithmetic only |
| `suspended`, `suspension_reason` | **no read sites** | None. Dead field — nothing enforces suspension |
| `plan_type`, `status`, `expires_at`, `activated_at` | **no read sites** | None today; latent if ever wired up |
| `api_key_hash` | auth path | Self-inflicted only — overwrites the caller's own key |

**Correction to an earlier draft of this report:** `is_trial_expired()` (`main.py:3129`) reads
only `plan`, `subscription.status`, and `trial_end` — all three *are* protected. So the update
path to trial escalation is closed, and the severity here is lower than it first appeared. The
one clearly exploitable field on the update path is `trial_used`, which gates trial redemption
at `main.py:4282` and is not consulted by `is_trial_expired()` at all.

`plan` is protected and `/api/signals` gates on `plan == 'pro'` (`main.py:1963`), so the
direct "make myself pro" path is closed.

**Recommendation: do not reproduce.** The `PATCH /api/me` allow-list in §2 excludes all of
them. This *tightens* behaviour, so it needs your explicit sign-off — a client currently
writing any of these will start getting 422.

### 3.2 `allow create` omits the trial fields ⚠️ **live hole**

`planFieldsUnchanged()` is only applied to `update`. The `create` rule checks `plan` and
`subscription_active` and nothing else — so `trial_end`, `trial_start`, and `subscription_end`
can be set to arbitrary values **at document creation**.

**This is the serious one**, because `trial_end` is precisely what `is_trial_expired()` reads
(`main.py:3129`). Setting it at create time yields a trial that never expires — the escalation
that §3.1's update path does *not* give you.

This is reachable. `trial-countdown.js:585` writes to the **email-keyed** doc:

```js
const docKey = auth.currentUser?.email || userId;
setDoc(doc(db, 'users', docKey), {
  trial_start: ..., trial_end: ..., trial: {...}
}, { merge: true }).catch(e => console.log('Silent update of trial data failed', e));
```

For a user whose row exists only under their **uid**, this `setDoc` on the email key is a
*create*, not an update — so `planFieldsUnchanged()` never runs and an arbitrary `trial_end`
is accepted. The dual key space is what makes it reachable. The `.catch()` swallowing the
failure is why nobody has noticed the update-path denials.

**Recommendation: do not reproduce.** `POST /api/me/bootstrap` sets trial dates server-side.

### 3.3 Rules and backend disagree on what "subscribed" means

`isSubscriptionActive()` admits `basic`, `intermediate`, `trial`, `pro`, `premium` for reading
`signals`. The backend requires `plan == 'pro'` (`main.py:1963`). The rules are the more
permissive of the two, so a `basic` user reads signals directly from Firestore that the API
would refuse them.

Out of Phase 1 scope (`signals` stays on Firestore), but it is a live entitlement bypass and
should be tracked.

---

## 4. Blocked: `onSnapshot` cannot be replaced by a `fetch`

Task 3 asks for "the exact `fetch` that replaces it". For four call sites there is no such
thing — they are **real-time listeners**, not reads. Phase 0 recorded "no listeners" because
it scanned Python; the listeners are all in the JS.

| Site | Target | In Phase 1 scope? |
|---|---|---|
| `app.js:162` | `users/{uid}/trades` | **yes** |
| `gatekeeper.js:2283` | `users/{uid}/trades` | **yes** |
| `gatekeeper.js:2891` | `users/{uid}/preferences/settings` | **yes** |
| `gatekeeper.js:2142` | `signals` | no — stays on Firestore |

Three options for the two in-scope targets, needs your decision (**Q1** below):

- **(a) Poll.** `GET` on an interval, closest to current behaviour from the UI's perspective,
  simplest to implement, costs a request per interval per user.
- **(b) Push over the existing WebSocket.** There is already a WS channel
  (`/ws/track-record`); trades and preferences could ride it. No new polling load, but it
  widens the WS contract.
- **(c) Leave both subcollections on Firestore this phase.** Port only the parent `users`
  document. Smallest Phase 1, defers the problem.

I recommend **(c)** for this phase and **(b)** later: it keeps Phase 1 to exactly the sensitive
thing (subscription state on the parent document) and avoids designing a realtime replacement
under migration pressure.

---

## 5. Other findings affecting the schema

**5.1 Two parallel field vocabularies.** The client SDK writes camelCase, the Admin SDK writes
snake_case, into the same document. Both are live data:

| Client (JS) | Server (Python) |
|---|---|
| `lastLogin`, `joinDate`, `loginMethods` | `last_login` |
| `trial.startDate`, `trial.endDate`, `trial.expiryNotified` | `trial_start`, `trial_end` |
| `preferences{}`, `usage{}` maps | — |

The Phase 0 schema captured only the snake_case set. **The `users` table must carry both** or
the port silently drops client-written data. Proposed: explicit columns for the snake_case
canon, plus `client_fields_json` preserving the camelCase set verbatim, so nothing is lost and
reconciliation stays a later data task — consistent with the Q1 decision on `doc_id`.

**5.2 A second nested map.** Phase 0 found `subscription.*`. There is also `trial.*`
(`trial-countdown.js:696` writes `'trial.expiryNotified'`). Flattening per Task 1 therefore
covers `subscription_*` **and** `trial_*` dotted paths.

**5.3 `serverTimestamp()` is used** by the client (`auth.js:158`, `gatekeeper.js:2880`) for
`lastLogin`/`joinDate`. Server-side equivalent is `datetime.now(timezone.utc).isoformat()`,
consistent with the ISO-8601 convention.

**5.4 A frontend write that the rules already deny.** `app.js:155` does
`updateDoc(tradeRef, {status:'closed', closeTime: new Date()})` against
`users/{uid}/trades/{id}`, where the rule is `allow write: if false`. This fails in production
today. The server path `POST /api/trades/{id}/close` already exists and does the same thing.
Port the *server* behaviour; do not build a client write path for it.

---

## 5.5 camelCase field set checked against real read sites

Applying the §3.1 method to the client vocabulary, as requested. Most of it is indeed
write-only bookkeeping — but not all, and the exception is the one that was predicted.

| Field | Read at | Verdict |
|---|---|---|
| `lastLogin` | nowhere (3 write sites, 0 reads) | Dead bookkeeping |
| `joinDate` | `trial-countdown.js:346` | Consumed — last-resort trial-start fallback |
| `loginMethods` | `auth.js:131` | Consumed, benign (read to append) |
| `displayName`, `photoURL`, `phone` | nowhere server-side | Client display only |
| `usage.*` | nowhere | Dead bookkeeping |
| `trial.allowedTokens`, `trial.allowedTimeframes` | nowhere (`gatekeeper.js`'s `allowedTokens` is an unrelated local) | Dead |
| `trial.active`, `trial.expiryNotified` | `trial-countdown.js:688` | Consumed |
| `trial.startDate` | `trial-countdown.js:346` | Consumed — **first** in a 4-way fallback |
| `trial.endDate` | `authManager.js:106` | **Consumed, and takes precedence over `trial_end`** ⚠️ |

### The residual hole: `trial.*` is unprotected and wins the precedence contest

`authManager.js:100-111` resolves the trial deadline in this order:

```js
if (userData.trial?.endDate != null)  endDate = userData.trial.endDate;   // ← FIRST
else if (userData.trial_end != null)  endDate = userData.trial_end;       // ← fallback
else if (userData.trialEnd  != null)  endDate = userData.trialEnd;
```

`trial-countdown.js:346` does the same thing for the start date:
`data?.trial?.startDate || data?.trial_start || data?.trialStart || data?.joinDate`.

The rules fix protects the snake_case scalars `trial_end` / `trial_start` / `trial_used`. It
does **not** protect the `trial` map, so a client can still write `trial: {endDate: '2099-…'}`
to its own document and the client-side gate will honour it — because it is consulted *before*
the protected field.

**Severity: client-side only.** Server entitlement (`is_trial_expired()`, `main.py:3129`) reads
`trial_end`, which is now protected on both paths, so API access is not bypassable this way.
The `_signTrialEnd()` tamper-detection in `authManager.js` does not help, because it signs
whatever value came out of Firestore — including a forged one.

**Why this was not fixed in the same pass:** adding `trial` to `adminOnlyFields()` would break
two legitimate flows — signup writes the whole `trial` map (`auth.js:89`), and
`trial-countdown.js:699` legitimately updates `trial.expiryNotified`. Firestore rules cannot
distinguish subfields of a map in `affectedKeys()`, so a wholesale lock is too blunt, and the
value itself cannot be range-checked (the rules file already notes at its head that an ISO
string cannot be compared with `request.time`).

**Recommended fix — one line, no flow breaks:** invert the precedence in `authManager.js` so
the server-authoritative `trial_end` is consulted first and `trial.endDate` is demoted to a
legacy fallback. Same for `trial-countdown.js:346`. This removes the unprotected field's
authority without locking the map. Not applied — awaiting your call, per Q5.

## 5.6 Near-miss worth recording

My first version of `createFieldsSafe()` put `subscription` in `adminOnlyFields()`. Legitimate
signup (`auth.js:82`) writes `subscription: {status:'none', …}` in the initial document, so
that rule would have **rejected every new account**. Caught before shipping by walking the
signup payload against the new rule rather than assuming it only carried benign fields.

`subscription` is now permitted on create only when `status != 'active'`, and stays fully
locked on update. The general lesson: every tightening here has to be validated against the
actual create payload, because the create path had no field guard at all before today.

## 6. Questions before Task 1

**Q1.** `onSnapshot` replacement strategy — (a) poll, (b) WebSocket, or (c) defer both
subcollections to a later phase? I recommend (c) for Phase 1.

**Q2.** Sign-off to *tighten* §3.1 and §3.2 rather than reproduce them. This is a deliberate
behaviour change: clients currently able to write `trial_used`, `suspended`, etc. will start
receiving 422. I believe that is the point, but it is your call and it is the one place where
Phase 1 is not behaviour-preserving.

**Q3.** Does "port `users`" include the `preferences` subcollection? It is owner-read/write
with no server-side equivalent today, so it needs a table and two endpoints. Folded into Q1
option (c) if you take that route.

**Q4.** The rules define `subscriptions`, `transactions`, and a root-level `trades` collection.
None appear in the Phase 0 backend inventory or the frontend scan. Confirm they are vestigial
and I will not port them — but they are worth checking for data before the rules are ever
tightened, since anything in them is currently readable by its owner.
