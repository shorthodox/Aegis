# Deployed Firestore rulesets

One line per `firebase deploy --only firestore:rules`. The point of this file is that a future
audit can check repo-vs-production divergence in ten seconds instead of querying the live API.

**Divergence is not hypothetical.** Between 2026-05-22 and 2026-08-14 the repo's
`firestore.rules` was rewritten several times and never deployed once. Every analysis of "the
rules" during that period described a file that production had never seen — including a
security audit that concluded a client write was being denied when it was in fact allowed.

## How to check current state in ten seconds

```bash
# which ruleset is live, and when it was released
curl -sS -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://firebaserules.googleapis.com/v1/projects/aegis-d78e1/releases
```

Compare the `rulesetName` against the newest row below. If the newest row's ruleset id is not
the live one, the repo and production have diverged.

## Rollback

All rulesets are retained by Firebase — 9 exist as of 2026-08-14, back to 2026-04-29. To roll
back:

**Console:** Firebase Console → `aegis-d78e1` → Firestore Database → **Rules** tab → **History**
(top-right, beside Publish) → select the timestamp → **Restore**.

That republishes the old ruleset as a new release. The bad ruleset stays in history; nothing is
destroyed by a rollback.

## Log

| Ruleset ID | Released | Commit | Notes |
|---|---|---|---|
| `40d739b2-379f-4f97-9b73-f8ed001e6ec0` | 2026-05-22 | *(pre-repo — no matching commit)* | **Rollback target.** `allow read, write` on `users/{userId}` with no field protection: `plan` client-writable, so any authenticated user could self-grant `pro`. `signals` readable by any authenticated user with no subscription check. This is what production ran for ~3 months. |
| *(pending)* | *(pending)* | *(pending)* | First deploy of the repo rules. Adds `createFieldsSafe()` create-path guard, `adminOnlyFields` (incl. `trial_start`, `trial_used`, `comped*`), email-first `getUserData()` precedence, `isSubscriptionActive()` on `signals`, and removes the vestigial `/trades`, `/subscriptions`, `/transactions` blocks. Validated by `tests/rules` — 12 pass, 1 known-open (see below). |

## Known open after the pending deploy

`isSubscriptionActive()` is a membership test — `plan == 'trial'` passes regardless of
`trial_end` or `subscription.status`. Signup mints a `trial` document, so **any account still
reads every signal**. The deploy closes the *write* escalation; it does not close the paywall.

The rules language cannot compare the ISO-8601 string in `trial_end` against `request.time`.
Closing it needs `trial_end_ts` — a native Firestore Timestamp written server-side alongside
the existing string — so the rule can evaluate `request.time < trial_end_ts`. Queued.

`tests/rules/rules.test.mjs` carries this as a deliberately failing test named
`EXPECTED FAIL — stub says trial, email doc expired/inactive, must fail closed`. **When that
test starts passing, the paywall is closed** — do not delete it to make the suite green.
