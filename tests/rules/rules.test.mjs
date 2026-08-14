/**
 * Executes firestore.rules against the emulator. Does not reason about them.
 *
 * The reason this exists: on 2026-08-14 a careful read of the rules concluded
 * that app.js's client trade-close "fails in production". It does not — the
 * DEPLOYED ruleset (40d739b2, 2026-05-22) permits that write, and the repo file
 * that was being read had never been deployed at all. Reading rules produces
 * confident wrong answers. Running them does not.
 *
 * The seed is not fixtures. seed_users.json is the 15 real production documents
 * exported verbatim, with every email consistently pseudonymised so a committed
 * fixture carries no real address while rule evaluation stays byte-identical.
 * Hand-written fixtures encode what you think the data looks like; these carry
 * the mixed camelCase/snake_case vocabulary, the trial.* map beside
 * subscription.*, the missing fields on stubs, and both key spaces.
 *
 * Identities that matter, from the real data:
 *   user3@example.test  — the only entitled account. plan=pro, EMAIL-keyed.
 *                         Its uid twin AJXC4jLowFexRzXgd577lSAcKAj1 is a plan=trial stub.
 *                         This is the account B1‴ exists for.
 *   user2@example.test  — the one human with NO email-keyed doc. Resolves
 *                         through the uid fallback.
 *   user1@example.test  — email doc says trial/inactive and EXPIRED; uid stub
 *                         wKRMnYLuREbFLiwYcjoTMvXbgU02 says trial. The fail-closed case.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { test, before, after, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  initializeTestEnvironment,
  assertSucceeds,
  assertFails,
} from '@firebase/rules-unit-testing';
import { doc, getDoc, setDoc, updateDoc } from 'firebase/firestore';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..');

const PRO_EMAIL   = 'user3@example.test';       // entitled, email-keyed
const PRO_UID     = 'AJXC4jLowFexRzXgd577lSAcKAj1';         // its plan=trial stub
const UIDONLY_EM  = 'user2@example.test';       // no email-keyed doc
const UIDONLY_UID = 'qxKii1jQn0ebOs9pm1qqr2TfSxd2';
const EXPIRED_EM  = 'user1@example.test';       // trial/inactive, expired
const EXPIRED_UID = 'wKRMnYLuREbFLiwYcjoTMvXbgU02';         // its plan=trial stub

let env;

/** Firestore rejects our {__ts__} marker; timestamps become ISO strings, which
 *  is what the snake_case fields hold in production anyway. */
function deTs(v) {
  if (Array.isArray(v)) return v.map(deTs);
  if (v && typeof v === 'object') {
    if ('__ts__' in v) return v.__ts__;
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, deTs(x)]));
  }
  return v;
}

before(async () => {
  env = await initializeTestEnvironment({
    projectId: 'aegis-rules-test',
    firestore: {
      rules: readFileSync(join(REPO, 'firestore.rules'), 'utf8'),
      host: '127.0.0.1',
      port: 8080,
    },
  });

  const seed = JSON.parse(readFileSync(join(HERE, 'seed_users.json'), 'utf8'));

  // Guard: every id these tests reference must actually be in the seed. The
  // first run of this suite "failed" on two rules that were in fact fine — the
  // constants below had been copied from a listing that truncated uids to 16
  // chars, so the documents were never written and every lookup saw an absent
  // doc. A harness that seeds nothing and reports rule failures is worse than
  // no harness.
  for (const id of [PRO_EMAIL, PRO_UID, UIDONLY_UID, EXPIRED_EM, EXPIRED_UID]) {
    assert.ok(id in seed, `seed is missing ${id} — test constants are out of sync with seed_users.json`);
  }
  await env.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    for (const [id, data] of Object.entries(seed)) {
      await setDoc(doc(db, 'users', id), deTs(data));
    }
    await setDoc(doc(db, 'signals', 'BTC_USDT'), { symbol: 'BTC/USDT', fire: true });
    await setDoc(doc(db, 'analytics', 'global_performance'), { win_rate: 0.5 });
  });
});

after(async () => { await env?.cleanup(); });

/** Authenticated context whose token carries BOTH uid and email, which is what
 *  a real Firebase ID token carries and what ownsDoc()/getUserData() read. */
const as = (uid, email) => env.authenticatedContext(uid, { email, email_verified: true });

describe('users — write protection', () => {
  test('signup writing subscription:{status:none} is ALLOWED', async () => {
    const db = as('brandNewUid', 'newbie@example.test').firestore();
    await assertSucceeds(setDoc(doc(db, 'users', 'brandNewUid'), {
      uid: 'brandNewUid', email: 'newbie@example.test', plan: 'trial',
      subscription: { status: 'none', startDate: null, endDate: null },
      trial: { active: true, endDate: '2026-09-01T00:00:00Z', expiryNotified: false },
      loginMethods: ['password'], preferences: { capital: 10000, riskPct: 2 },
      usage: { signalCount: 0 },
    }));
  });

  test('benign client bookkeeping is ALLOWED', async () => {
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    const ref = doc(db, 'users', EXPIRED_UID);
    await assertSucceeds(updateDoc(ref, { lastLogin: '2026-08-14T00:00:00Z' }));
    await assertSucceeds(updateDoc(ref, { loginMethods: ['password', 'google'] }));
    await assertSucceeds(updateDoc(ref, { preferences: { capital: 5000 } }));
    await assertSucceeds(updateDoc(ref, { 'trial.expiryNotified': true }));
  });

  test('client writing plan:pro is DENIED', async () => {
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    await assertFails(updateDoc(doc(db, 'users', EXPIRED_UID), { plan: 'pro' }));
  });

  test('client writing trial_end / trial_start / trial_used is DENIED', async () => {
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    const ref = doc(db, 'users', EXPIRED_UID);
    await assertFails(updateDoc(ref, { trial_end: '2099-01-01T00:00:00Z' }));
    await assertFails(updateDoc(ref, { trial_start: '2026-01-01T00:00:00Z' }));
    await assertFails(updateDoc(ref, { trial_used: false }));
  });

  test('client writing an arbitrary trial_end at CREATE is DENIED', async () => {
    // The escalation found on 2026-08-13: setDoc(merge) on the email key is a
    // CREATE when only the uid doc exists, so the update guard never ran.
    const db = as('freshUid', 'fresh@example.test').firestore();
    await assertFails(setDoc(doc(db, 'users', 'fresh@example.test'), {
      trial_end: '2099-01-01T00:00:00Z',
    }, { merge: true }));
  });

  test('client writing comped markers is DENIED', async () => {
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    const ref = doc(db, 'users', EXPIRED_UID);
    await assertFails(updateDoc(ref, { comped: true }));
    await assertFails(updateDoc(ref, { comped_reason: 'self-granted' }));
    await assertFails(updateDoc(ref, { comped_by: 'me' }));
  });

  test('user A cannot read or write user B document', async () => {
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    await assertFails(getDoc(doc(db, 'users', PRO_EMAIL)));
    await assertFails(updateDoc(doc(db, 'users', PRO_EMAIL), { lastLogin: 'x' }));
    await assertFails(getDoc(doc(db, 'users', PRO_UID)));
  });

  test('owner reads own doc under BOTH key spaces', async () => {
    const db = as(PRO_UID, PRO_EMAIL).firestore();
    await assertSucceeds(getDoc(doc(db, 'users', PRO_EMAIL)));
    await assertSucceeds(getDoc(doc(db, 'users', PRO_UID)));
  });
});

describe('signals — entitlement (B1‴ precedence under test)', () => {
  test('email-keyed entitled user CAN read signals', async () => {
    // Resolves the email doc (plan=pro), not the uid stub. The whole point.
    const db = as(PRO_UID, PRO_EMAIL).firestore();
    await assertSucceeds(getDoc(doc(db, 'signals', 'BTC_USDT')));
  });

  test('uid-only human CAN read signals via the fallback', async () => {
    const db = as(UIDONLY_UID, UIDONLY_EM).firestore();
    await assertSucceeds(getDoc(doc(db, 'signals', 'BTC_USDT')));
  });

  test('user with NO document CANNOT read signals', async () => {
    const db = as('ghostUid', 'ghost@example.test').firestore();
    await assertFails(getDoc(doc(db, 'signals', 'BTC_USDT')));
  });

  test('unauthenticated CANNOT read signals', async () => {
    const db = env.unauthenticatedContext().firestore();
    await assertFails(getDoc(doc(db, 'signals', 'BTC_USDT')));
  });

  test('EXPECTED FAIL — stub says trial, email doc expired/inactive, must fail closed', async () => {
    // This asserts the behaviour we WANT, and the current rules do not provide
    // it. isSubscriptionActive() is a membership test: plan == 'trial' passes
    // regardless of trial_end or subscription.status. B1‴ makes the rule read
    // the RIGHT document; it does not make that document's expiry mean
    // anything. Closing this needs trial_end_ts (a native Timestamp the rule
    // can compare against request.time) — queued, not in this change.
    const db = as(EXPIRED_UID, EXPIRED_EM).firestore();
    await assertFails(getDoc(doc(db, 'signals', 'BTC_USDT')));
  });
});
