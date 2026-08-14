# B1″ — Collapse the `users` dual key space

**Status:** plan only. Nothing here has been executed. Approved as a follow-up *after* the
rules deploy, deliberately not in the same change — it modifies user data, and only one
variable should move at a time.

**Prerequisite:** B1‴ (email-first precedence in `getUserData()`) is live and stable.

---

## The defect

`users` documents exist under two key spaces:

| Key space | Written by | Content |
|---|---|---|
| **email** — `users/{email}` | backend, `create_user_doc()` (`main.py:3106`) | the real record: plan, subscription, trial dates, payment ids |
| **uid** — `users/{firebase_uid}` | browser, `ensureUserDocumentV2()` (`web/src/scripts/auth.js:61`) | a stub: `plan:'trial'`, `subscription.status:'none'`, written once and never updated |

Measured 2026-08-14 across all 15 documents: **7 humans hold both**, and in 7 of 7 pairs the
email doc carries truth while the uid twin is the stub. One human (`animeshkukreti05@gmail.com`)
holds only a uid doc. Eight humans, fifteen documents.

B1‴ routes around this by reading the email doc first. It does not remove the second document,
and it does not stop new ones appearing.

## Why reconciliation alone is not enough

Deleting the 7 stubs today regenerates them on the next signup, because the browser still
creates one. **The cause has to go first, then the data.** Doing it in the other order produces
a clean database that quietly re-dirties itself.

---

## Change 1 — stop the browser creating a user document

### The redundancy

`create_user_doc(email, ...)` at `main.py:3106` already writes the authoritative document on
**every** signup path:

| Caller | Path |
|---|---|
| `main.py:4127` | email/password signup |
| `main.py:4165` | Google signup |
| `main.py:3152` | social login |
| `main.py:3313` | OTP verification |

So `ensureUserDocumentV2()` is not filling a gap. It writes a second, worse copy of a document
the server already owns — hardcoding `plan:'trial'` client-side, with no server validation, and
under a different key.

### The change

`web/src/scripts/auth.js:61-123` — `ensureUserDocumentV2()` currently:

```js
const userDocRef = doc(db, 'users', user.uid);   // :64
const docSnap    = await getDoc(userDocRef);     // :65
if (!docSnap.exists()) {
  const userData = { /* uid, email, plan:'trial', subscription:{...}, trial:{...} */ };
  await setDoc(userDocRef, userData);            // :123   ← the stub
}
```

It should **read the server's document instead of creating its own**:

- Point the read at the email key, matching `create_user_doc()`
- Delete the `setDoc` branch entirely — the browser stops being a writer of record
- Keep the existing-user `updateDoc` at `:135` for `loginMethods` / `lastLogin`; those are
  display bookkeeping the rules still permit, and they are not entitlement

**Ordering hazard, and the reason this needs care.** The browser may reach
`ensureUserDocumentV2()` before the server's signup call has committed. Today the client papers
over that by creating a document. Once it stops, the read can miss. It needs a bounded retry
(a few hundred ms, a handful of attempts) and an explicit failure surfaced to the user — **not
a silent `.catch()`**, which is the pattern that hid the rule denials for weeks.

`web/src/scripts/gatekeeper.js:2868-2881` — `ensureUserDocument()` is a second, smaller writer
of the same uid-keyed document (`{uid, email, joinDate, lastLogin}`, no `plan`). It did not
create the `plan:'trial'` stubs, but it does create uid-keyed documents and should be folded
into the same change.

---

## Change 2 — reconcile the 7 existing pairs

Only after Change 1 is deployed and a signup has been observed to produce exactly one document.

**Merge rule: the email document wins.** It is the server-written record and holds the real
plan, subscription, trial dates and payment ids in all 7 pairs. The uid twin contributes only
the camelCase client bookkeeping the server never writes — `lastLogin`, `joinDate`,
`loginMethods`, `preferences`, `usage`, `trial.*`.

Per pair:

1. Read both documents; snapshot both to a local NDJSON dump **before** any write
2. Copy any client-only field present on the uid doc and absent on the email doc onto the email doc
3. Verify the merged email doc against the snapshot field by field
4. Delete the uid doc
5. Re-verify: the human resolves to exactly one document, and `isSubscriptionActive()` is unchanged

Do **not** batch all 7. One human at a time, verified, with the pro account
(`animeshkukreti60@gmail.com`) done **last** — it is the only entitled record and the only one
whose loss is not recoverable from a re-signup.

The 8th human (`animeshkukreti05@gmail.com`, uid-only) needs nothing. After Change 1 they remain
uid-keyed and resolve via B1‴'s fallback, which is exactly the path that fallback exists for.

### Verification query

```sql
-- after reconciliation this must return 0
SELECT COUNT(*) FROM users a JOIN users b
  ON a.email = b.email AND a.doc_id <> b.doc_id;
```

(SQL form retained from the shelved migration plan; against Firestore, the equivalent is
"no uid-keyed document whose `email` field matches an existing email-keyed document id".)

---

## Sequencing

1. Rules deploy (B1‴ + the create-path guard) — **separate, first**
2. Observe stable. No user data touched.
3. Change 1 — browser stops writing. Deploy. Watch one real signup produce one document.
4. Change 2 — reconcile 7 pairs, one at a time, pro account last.
5. Verification query returns 0.

## Risks

| Risk | Mitigation |
|---|---|
| Browser reads before server commits, signup appears broken | Bounded retry + visible error; never a silent catch |
| A merge drops a client-only field | Snapshot both docs first; field-by-field verify before delete |
| The pro account's entitlement is lost mid-merge | Done last, verified individually, dump retained |
| Stubs reappear after reconciliation | Change 1 ships first and is confirmed by observing a real signup |
