# Firestore → SQLite Migration Plan (Phase 0 output)

**Status:** awaiting approval. No code written yet.
**Target:** SQLite on a Railway volume (confirmed — the Supabase project provisioned on
2026-08-13 will go unused).
**Scope:** storage swap only. Firebase Auth, the signal engine, exit logic, and model code
are not touched.

---

## 1. Headline discovery findings

Five things the handoff did not anticipate. Two of them need your decision before Phase 1.

> ### ⚠️ CORRECTION (2026-08-13, after Task 0) — §1.1 and §1.3 below are WRONG
>
> The Firestore Admin API was queried directly over REST once the TLS workaround was in
> place. The project has **exactly one database**:
>
> ```
> name='default'   type=FIRESTORE_NATIVE   location=asia-south2
> ```
>
> **`(default)` does not exist.** Querying it returns
> `NOT_FOUND: The database (default) does not exist for project aegis-d78e1`.
>
> Every engine module calling `_fs.client()` — `state.py:50`, `engine.py:713`,
> `scalp.py:85` — has therefore been writing to a database that is not there. Consequences:
>
> - **The track record has never had a Firestore backup.** `_fs_save_track_record()` fails on
>   the first call and trips `_FS_DOWN`, permanently for that process.
>   `_hydrate_track_record_from_firestore()` has never restored anything. §1.3 called it "a
>   slimmed DR mirror"; there is no mirror. `STATE_DIR/track_record.json` on the Railway
>   volume is the **only** copy in existence. This makes §6.1 urgent, not optional.
> - `scalp_signals` root cause confirmed: the writes fail, which is why `bot-record.html`
>   reads an empty collection and falls back to `/data/scalp_trades.json`.
> - `engine.py:675` does **not** double the write volume — it fails immediately. The quota
>   exhaustion is attributable to `main.py`'s producer alone.
> - **Phase 2 exports one database, not two.** The `--database` flag and the namespacing of
>   dump files are no longer needed.
>
> The sections below are retained as written, for audit trail. Read them through this
> correction.

### 1.1 There are TWO Firestore databases, not one ⚠️

Confirmed from the installed SDK: `google/cloud/firestore_v1/base_client.py` defines
`DEFAULT_DATABASE = "(default)"`. So these are different databases:

| Client construction | Database | Used by |
|---|---|---|
| `firestore.client(database_id="default")` | `default` *(a named DB)* | `main.py:555`, and the frontend via `getFirestore(app, "default")` |
| `firestore.client()` | `(default)` | `state.py:50`, `engine.py:713`, `scalp.py:85` |

**Consequence:** `signals` exists in *both* databases with different content and different
writers. Any export that assumes one database loses half the data. Per your decision, both
are migrated, so `signals` becomes two tables.

### 1.2 Two independent signal writers, one writing into the void

- `main.py` producer → `signals` in `default` — **the frontend reads this one**
- `engine.py:675 _push_signals_to_firestore()` → `signals` in `(default)` — **nothing reads this**

Both fire on roughly the same cadence, so actual write volume is about double the ~17.9k/day
estimated from `main.py` alone. This is the most likely cause of the free-tier write-quota
exhaustion that triggered the outage. Migrating as-is (your choice) carries it forward; worth
a separate cleanup commit afterwards.

### 1.3 The track record's primary store is a file, not Firestore

The handoff calls the track record the thing that must survive above all. Its primary store is
`STATE_DIR/track_record.json` on the Railway volume (`scripts/engine/config.py:104`).
Firestore holds only a slimmed disaster-recovery mirror at `engine_state/track_record` in the
`(default)` database, read by `_hydrate_track_record_from_firestore()` when the volume comes
up empty.

**The migration's real risk is therefore the volume, not Firestore.** The Firestore side of
the track record is a backup of a backup.

### 1.4 `users` is keyed inconsistently — email in some paths, Firebase UID in others ⚠️

| Dependency | Returns | Document path |
|---|---|---|
| `get_current_user` | **email** | `users/{email}` — used by `get_user_doc` and most endpoints |
| `get_firebase_uid` | **Firebase UID** | `users/{uid}` — used by `/api/trades/execute`, `/api/trades/{id}/close` |

The `users` collection contains documents under **both** key spaces.

**Resolved:** surrogate integer PK; `doc_id TEXT UNIQUE NOT NULL` holds the original verbatim;
`firebase_uid` and `email` are separate indexed columns populated from whatever each row
carries. Faithful and auditable, and reconciliation becomes a later data task rather than a
second migration. Normalising to UID during import was rejected: it would silently merge or
drop rows with no way to audit what happened.

**Why this matters beyond schema aesthetics:** if one human has rows under both key spaces, a
lookup via the wrong path returns nothing — which presents to a paying customer as "no
subscription". **Before Phase 2, obtain one number: how many distinct humans hold rows under
both key spaces.** Zero means this is latent and ignorable. Non-zero means subscription state
is already split in production, and that bug outranks this migration.

### 1.5 Mixed timestamp representations

Almost everything stores ISO-8601 UTC strings (`datetime.now(timezone.utc).isoformat()`).
The exception is `phone_verifications`, which stores **native datetimes** — `main.py:940`
notes "Firestore returns DatetimeWithNanoseconds … no conversion needed", and `main.py:1696`
queries it with `where("expires_at", "<", now)` passing a `datetime` object.

**Decision:** store all timestamps as ISO-8601 UTC TEXT (one convention, per handoff). The
`phone_verifications` repository functions convert back to tz-aware `datetime` on read, so
callers see exactly what they see today. Documented exception, no behaviour change.

---

## 2. Firestore-specific behaviour in use

Searched for all of it. What is **not** used, and therefore needs no equivalent:
`SERVER_TIMESTAMP`, `Increment`, `ArrayUnion`/`ArrayRemove`, `array_contains`, `on_snapshot`
listeners, and Firestore transactions. (The "transaction" matches in `main.py` are all Paddle
and Dodo payment transactions.)

What **is** used:

| Behaviour | Sites | Relational equivalent |
|---|---|---|
| Batched writes | `main.py:1332,1348,1448,1455`; `engine.py:714` | One SQL transaction per batch |
| `set(..., merge=True)` | `main.py:1451,1464,7197` | `INSERT … ON CONFLICT(pk) DO UPDATE SET …` |
| Auto-generated doc IDs | `main.py:6918` (`.document()`), `main.py:6853` (`.add()`) | `uuid4().hex` generated in the repository layer |
| Nested subcollection | `users/{uid}/trades/{trade_id}` | `trades` table with `user_doc_id` FK |
| Dotted field-path updates | `subscription.status`, `subscription.canceled_at` | `json_set(subscription_json, '$.status', ?)` via SQLite JSON1 |
| `where(field, "in", [...])` | `main.py:1071` | `WHERE status IN (?, ?)` |
| Range query on timestamp | `main.py:1696` | `WHERE expires_at < ?` (ISO strings sort chronologically) |
| Dynamic/variable payload keys | `signals` documents | Indexed scalar columns + `payload_json TEXT` for the remainder |

---

## 3. Collection inventory

Operation counts from an automated scan of `main.py`, `state.py`, `engine.py`, `scalp.py`,
`live_engine.py`.

| Collection | DB | Reads | Writes | Doc ID | Notes |
|---|---|---|---|---|---|
| `users` | `default` | 5 | 21 | email **or** uid | 33 distinct fields; busiest table |
| `dev_codes` | `default` | 7 | 5 | the code; plus `current_token` sentinel | |
| `signals` | `default` | 2 | 1 (batched, 60 docs) | `SYMBOL_USDT` | frontend reads this |
| `dev_keys` | `default` | 0 | 2 | key id | |
| `analytics` | `default` | 0 | 2 | fixed `global_performance` | single-row table |
| `phone_verifications` | `default` | 3 | 4 | email | native datetimes |
| `reviews` | `default` | 0 | 1 | auto-generated | |
| `users/{uid}/trades` | `default` | 1 | 2 | auto-generated | subcollection |
| `engine_state` | `(default)` | 1 | 2 | fixed `track_record` | the DR mirror |
| `signals` (engine) | `(default)` | 0 | 1 (batched) | `SYMBOL_USDT` | dead writes — no reader anywhere |
| `scalp_signals` | `(default)` | 0 | 1 | `SYMBOL_mode` | **read by the frontend, from the WRONG database** |

### 3.1 Frontend reads — the half a Python inventory cannot see

The JS modular API is `collection(db, 'name')`, which a Python call-site scan misses entirely.
Full frontend inventory, all bound to `getFirestore(app, 'default')`:

| Collection | Reads | Backend writer targets | Agrees? |
|---|---|---|---|
| `users` | 18 | `default` | ✅ |
| `signals` | 1 | `default` | ✅ |
| `analytics` | 1 | `default` | ✅ |
| `scalp_signals` | 1 (`bot-record.html:448`) | **`(default)`** | ❌ **broken** |

**`scalp_signals` is a live production bug.** `scalp.py:85` writes to `(default)`;
`bot-record.html:448` queries `collection(db,'scalp_signals')` with
`orderBy('entry_time','desc'), limit(200)` against `default`. The ScalpBot Record page is
reading an empty collection and silently falling back to `/data/scalp_trades.json`
(`bot-record.html:470`). It therefore **does** need a table, and needs an `entry_time` index.
Fixing the writer's database target is a separate commit, not this migration.

### 3.2 Which `signals` is authoritative — decided on evidence

`default` is authoritative for everything the frontend reads. Nothing — backend or frontend —
reads `signals` from `(default)`, so `engine.py`'s push is confirmed dead. `(default)` has
exactly one live consumer: `engine_state/track_record`, whose sole writer and sole reader both
target it, so it is self-consistent.

### 3.3 Production row counts — no longer a blocker

`verify_migration.py` asserts **export-dump count == SQLite row count**, not live-Firestore ==
SQLite. The live database keeps moving during the migration, so the latter is unstable by
construction; the former is the invariant that actually matters (that the import was
lossless), and `wc -l` on the NDJSON provides it for free.

**Connectivity note:** local Firestore access is currently broken by TLS interception, not by
account state. The certificate presented for `oauth2.googleapis.com` is issued by
`CN=Avast Web/Mail Shield Root` — Avast's root is in the Windows store (so `curl` and browsers
work) but not in certifi (so every Python call fails `CERTIFICATE_VERIFY_FAILED`). Quota
exhaustion would surface as `RESOURCE_EXHAUSTED` *after* a successful handshake, so this is
unrelated to billing. Fix before running the export: `pip install pip-system-certs`, or
exclude `*.googleapis.com` from Avast's HTTPS scanning.

---

## 4. Proposed schema (`migrations/001_initial.sql`)

Conventions: all timestamps ISO-8601 UTC TEXT; all money/prices REAL; booleans INTEGER 0/1;
original Firestore document ID preserved verbatim in every table.

```sql
CREATE TABLE schema_migrations (
  version     INTEGER PRIMARY KEY,
  applied_at  TEXT NOT NULL
);

-- Surrogate PK. doc_id holds the Firestore document ID verbatim — email for most rows,
-- Firebase UID for rows created via /api/trades/*. firebase_uid and email are populated
-- from whatever each row actually carries, so the dual key space stays visible and
-- auditable instead of being silently collapsed at import time.
CREATE TABLE users (
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id                 TEXT UNIQUE NOT NULL,
  email                  TEXT,
  firebase_uid           TEXT,
  full_name              TEXT,
  phone_number           TEXT,
  plan                   TEXT,
  plan_type              TEXT,
  status                 TEXT,
  provider               TEXT,
  location               TEXT,
  otp_verified           INTEGER DEFAULT 0,
  trial_active           INTEGER DEFAULT 0,
  trial_used             INTEGER DEFAULT 0,
  trial_start            TEXT,
  trial_end              TEXT,
  subscription_end       TEXT,
  subscription_id        TEXT,
  subscription_json      TEXT,          -- nested map; dotted updates via json_set()
  activated_at           TEXT,
  last_login             TEXT,
  expires_at             TEXT,
  suspended              INTEGER DEFAULT 0,
  suspension_reason      TEXT,
  api_key_hash           TEXT,
  api_key_last_generated TEXT,
  dev_code_used          TEXT,
  capital                REAL,
  risk_pct               REAL,
  order_id               TEXT,
  payment_id             TEXT,
  extra_json             TEXT           -- forward-compat for unmapped keys
);
CREATE INDEX idx_users_email        ON users(email);
CREATE INDEX idx_users_uid          ON users(firebase_uid);
CREATE INDEX idx_users_phone        ON users(phone_number);
CREATE INDEX idx_users_plan         ON users(plan);
CREATE INDEX idx_users_sub_status   ON users(json_extract(subscription_json, '$.status'));

-- Reconciliation query for the pre-Phase-2 check in §1.4: rows whose email and
-- firebase_uid both appear, under different doc_ids, are the same human split in two.
--   SELECT COUNT(*) FROM users a JOIN users b
--     ON a.email = b.email AND a.doc_id <> b.doc_id;

CREATE TABLE trades (
  id             TEXT PRIMARY KEY,      -- was an auto-generated Firestore ID
  user_doc_id    TEXT NOT NULL REFERENCES users(doc_id) ON DELETE CASCADE,
  symbol         TEXT NOT NULL,
  side           TEXT NOT NULL,
  entry_price    REAL, stop_loss REAL, take_profit REAL,
  risk_percent   REAL, leverage REAL,
  position_units REAL, notional_value REAL,
  status         TEXT NOT NULL DEFAULT 'open',
  signal_id      TEXT,
  demat_status   TEXT,
  open_time      TEXT NOT NULL,
  close_time     TEXT
);
CREATE INDEX idx_trades_user   ON trades(user_doc_id);
CREATE INDEX idx_trades_status ON trades(status);

-- Two source databases, two tables. `signals` is what the frontend reads;
-- `engine_signals` is engine.py's parallel write to the (default) database.
CREATE TABLE signals (
  doc_id       TEXT PRIMARY KEY,        -- e.g. BTC_USDT
  symbol       TEXT NOT NULL,
  fire         INTEGER DEFAULT 0,
  status       TEXT,
  timestamp    TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX idx_signals_fire   ON signals(fire);
CREATE INDEX idx_signals_status ON signals(status);

CREATE TABLE engine_signals (
  doc_id       TEXT PRIMARY KEY,
  symbol       TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at   TEXT
);

-- Read by bot-record.html:448 with orderBy('entry_time','desc'), limit(200).
CREATE TABLE scalp_signals (
  doc_id       TEXT PRIMARY KEY,        -- SYMBOL_mode
  symbol       TEXT NOT NULL,
  mode         TEXT NOT NULL,
  entry_time   TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX idx_scalp_entry_time ON scalp_signals(entry_time DESC);

CREATE TABLE dev_codes (
  code        TEXT PRIMARY KEY,         -- also holds the 'current_token' sentinel row
  source      TEXT, plan TEXT, label TEXT,
  active_code TEXT,                     -- sentinel row only
  features_json TEXT,
  created_at  TEXT, expires_at TEXT,
  created_by  TEXT, used_by TEXT, used_at TEXT
);
CREATE INDEX idx_dev_codes_expires ON dev_codes(expires_at);

CREATE TABLE dev_keys (
  key_id      TEXT PRIMARY KEY,
  key         TEXT, created_by TEXT,
  features_json TEXT,
  created_at  TEXT, expires_at TEXT,
  last_used   TEXT, usage_count INTEGER DEFAULT 0
);

CREATE TABLE phone_verifications (
  email         TEXT PRIMARY KEY,
  phone_number  TEXT,
  signup_token  TEXT,
  otp_code      TEXT,
  expires_at    TEXT NOT NULL,          -- ISO TEXT; repo converts to datetime on read
  extra_json    TEXT
);
CREATE INDEX idx_phone_ver_token   ON phone_verifications(signup_token);
CREATE INDEX idx_phone_ver_phone   ON phone_verifications(phone_number);
CREATE INDEX idx_phone_ver_expires ON phone_verifications(expires_at);

CREATE TABLE analytics (
  doc_id        TEXT PRIMARY KEY,       -- always 'global_performance'
  win_rate      REAL, expectancy REAL, profit_factor REAL,
  max_drawdown  REAL, total_trades INTEGER,
  updated_at    TEXT
);

CREATE TABLE reviews (
  id          TEXT PRIMARY KEY,         -- was auto-generated
  payload_json TEXT NOT NULL,
  created_at  TEXT
);

CREATE TABLE engine_state (
  doc_id       TEXT PRIMARY KEY,        -- always 'track_record'
  signals_json TEXT, summary_json TEXT,
  gate_version TEXT, generated_at TEXT,
  generation   INTEGER
);
```

Every column above appears in a `WHERE` or `ORDER BY` at some call site, or is indexed
because the Phase 0 scan shows a query against it.

---

## 5. Call-site → replacement mapping

Repository modules, one per entity, plain typed functions. No route handler or engine module
touches SQL directly.

| Call site | Current | Replacement |
|---|---|---|
| `main.py:936-951` | `_otp_get/_otp_set/_otp_update/_otp_delete` | `repo.phone_verifications.{get,upsert,update,delete}` |
| `main.py:955,962` | `where(signup_token/phone_number ==).limit(1)` | `repo.phone_verifications.{by_token,by_phone}` |
| `main.py:1696` | `where(expires_at <).stream()` + per-doc delete | `repo.phone_verifications.delete_expired(now)` — one `DELETE` |
| `main.py:1071` | `where(status in [...]).stream()` | `repo.signals.closed_trades()` |
| `main.py:1079,1148` | `analytics/global_performance.set()` | `repo.analytics.upsert(...)` |
| `main.py:1334` | `signals.stream()` + batch | `repo.signals.all()` + one transaction |
| `main.py:1448-1464` | batched `set(merge=True)` × 60 | `repo.signals.upsert_many(rows)` — one transaction |
| `main.py:3054,3069` | `users.document(email).get()`, `where(phone_number ==)` | `repo.users.{get,by_phone}` |
| `main.py:3115-7325` | 21 × `users.update({...})` | `repo.users.update(doc_id, **fields)`; dotted paths via `json_set` |
| `main.py:5603,5641` | `where(plan ==)`, `where(subscription.status ==)` | `repo.users.{by_plan,by_subscription_status}` |
| `main.py:6266-6694` | `dev_codes` get/set/delete/update | `repo.dev_codes.*` |
| `main.py:5810,6794` | `dev_keys` set/update | `repo.dev_keys.*` |
| `main.py:6853` | `reviews.add()` (auto-ID) | `repo.reviews.create()` — `uuid4().hex` |
| `main.py:6918-6935` | `users/{uid}/trades` subcollection | `repo.trades.{create,get,close}` |
| `main.py:7051,7183` | `signals.document(symbol)` get/set merge | `repo.signals.{get,upsert}` |
| `state.py:70,83,101` | `engine_state/track_record` set/delete/get | `repo.engine_state.{save,clear,load}` |
| `engine.py:714-734` | batched `signals` push to `(default)` | `repo.engine_signals.upsert_many()` |
| `scalp.py:96` | `scalp_signals.set()` | `repo.scalp_signals.upsert()` |

**Library choice:** stdlib `sqlite3`, not SQLAlchemy. The query surface is entirely
single-table lookups by primary key, four indexed equality filters, one range filter, and one
`IN` — there are no joins beyond `trades → users`, no dynamic query building, and no second
dialect to target. An ORM would add a dependency and a mapping layer for no benefit here.

---

## 6. Resolved decisions

**Q1 — `users` key.** Surrogate integer PK; `doc_id TEXT UNIQUE NOT NULL` verbatim; indexed
`firebase_uid` and `email`. See §1.4, including the pre-Phase-2 reconciliation count.

**Q2 — `scalp_signals`.** Investigated as directed, and the conclusion reversed: it **is**
read, by `bot-record.html:448`, from the wrong database. It gets a table and an `entry_time`
index. Fixing the writer's DB target is a separate commit. See §3.1.

**Q3 — verification invariant.** `verify_migration.py` asserts export-dump count == SQLite row
count. Live counts are not an assertion target. See §3.3.

**Q4 — `main.py:7278`.** Ported faithfully, `# type: ignore` left alone, `TODO` added, kept
out of this diff so a bisect stays meaningful. It raises `TypeError` for every caller today,
so there is no working behaviour to preserve and zero users can depend on it. Fixed in the
commit immediately after cutover.

## 6.1 The track record has no backup — decide before Phase 4 ⚠️

Established in §1.3: the track record lives in `STATE_DIR/track_record.json` on the Railway
volume, not in Firestore. Two consequences the original handoff got wrong:

1. Phase 2's "verify every closed trade" was aimed at the wrong artefact. The Firestore
   `engine_state` mirror is a slimmed copy, not the record itself.
2. **Litestream will not protect it.** Litestream replicates SQLite, not arbitrary JSON. So
   the product's core credibility claim currently has no backup at all, and would still have
   none after a by-the-book Phase 4.

Two ways to close it:

- **(a) `rclone` sync of `STATE_DIR/*.json` to the same R2 bucket — recommended.** Keeps this
  migration a pure storage swap. Folding the track record into SQLite means rewriting
  `VirtualWallet._load_history` and `_save_track_record`, which is engine code and therefore
  barred by hard constraint #4.
- **(b) Fold it into SQLite** so Litestream covers it with one mechanism. Cleaner long-term,
  but it is a behaviour change to the engine and belongs in its own change.

Recommendation: (a) in Phase 4, (b) later if desired. **This work is not finished without one
of them.**

---

## 7. Sequencing

Phases 1–4 proceed as specified in the handoff once this plan is approved. Two adjustments
follow from discovery:

- **Phase 2 exports two databases**, not one. `export_firestore.py` takes a `--database` flag
  and is run twice; dumps are namespaced so the `signals` collision cannot silently merge.
  Requires the TLS fix in §3.3 first.
- **Phase 4 adds an `rclone` sync for `STATE_DIR/*.json`** alongside Litestream, per §6.1.
- **Before Phase 2**, run the §1.4 reconciliation count against the export dumps.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Volume not mounted → SQLite written to container FS, lost on redeploy | Startup assertion that `DATABASE_PATH`'s parent is a mount point; fail loudly, do not silently create |
| Two processes open the DB | Single replica enforced and documented; WAL + `busy_timeout=5000` per connection |
| `signals` collision merges two databases' data | Separate tables, namespaced dump files, verified by row count per source |
| Import run twice duplicates rows | All imports `INSERT … ON CONFLICT(pk) DO UPDATE`, keyed on the original document ID |
| Track record corrupted | Verify script checks every closed trade field-by-field, not a sample |
