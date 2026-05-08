# Firestore Setup Guide for AEGIS v1.0

## Overview
This guide walks you through setting up Firestore for the AEGIS trading platform with proper security, collections, and indexing.

---

## Step 1: Firebase Project Setup

### 1.1 Create/Access Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create a new project: **aegis-d78e1** (or select existing)
3. Enable **Firestore Database** (if not already enabled)
4. Select region: **asia-south2** (as per firebase.json)

### 1.2 Enable Authentication Methods
In Firebase Console → Authentication → Sign-in method:
- ✅ **Google** - Enable
- ✅ **Email/Password** - Enable  
- ✅ **Phone** - Enable (requires Recaptcha verification)
- ✅ **Anonymous** - Optional

---

## Step 2: Create Firestore Collections

### 2.1 Users Collection
Create collection: **`users`**

**Document ID:** `{uid}` (Firebase Auth UID)

**Fields:**
```javascript
{
  uid: "user_firebase_uid",
  email: "user@example.com",
  phone: "+91XXXXXXXXXX",
  displayName: "User Name",
  photoURL: "https://...",
  
  // Plan information
  plan: "trial", // enum: "trial" | "basic" | "premium" | "pro"
  
  // Subscription details
  subscription: {
    status: "none", // enum: "active" | "expired" | "none"
    startDate: Timestamp,
    endDate: Timestamp,
    renewalDate: Timestamp
  },
  
  // Trial tracking (important for free users)
  trial: {
    active: true,
    startDate: Timestamp,
    endDate: Timestamp, // startDate + 3 days
    expiryNotified: false,
    allowedTokens: ["BTC", "ETH", "SOL", "ARB", "AAVE"],
    allowedTimeframes: ["30m", "1h"]
  },
  
  // Auth methods used
  loginMethods: ["google", "email", "phone"],
  
  // Timestamps
  joinDate: Timestamp,
  lastLogin: Timestamp,
  
  // User preferences
  preferences: {
    capital: 10000,
    riskPct: 2,
    theme: "dark",
    notifications: true
  },
  
  // Usage tracking
  usage: {
    totalSignalsToday: 0,
    lastSignalTime: Timestamp,
    signalCount: 0
  }
}
```

### 2.2 Signals Collection
Create collection: **`signals`**

**Document ID:** `{symbol}` (e.g., "BTC", "ETH", "SOL")

**Fields:**
```javascript
{
  symbol: "BTC",
  price: 45000.50,
  signal: "BUY", // enum: "BUY" | "SELL" | "HOLD" | "AVOID"
  entry: 45000,
  sl: 44500,
  tp: 46500,
  timeframe: "1h",
  confidence: 0.85,
  riskRewardRatio: 2.5,
  timestamp: Timestamp,
  updatedAt: Timestamp,
  source: "ml_model", // source of signal
  active: true
}
```

### 2.3 Subscriptions Collection
Create collection: **`subscriptions`**

**Document ID:** Auto-generated

**Fields:**
```javascript
{
  userId: "user_firebase_uid",
  plan: "premium", // enum: "basic" | "premium" | "pro"
  status: "active", // enum: "active" | "pending" | "cancelled"
  startDate: Timestamp,
  endDate: Timestamp,
  renewalDate: Timestamp,
  paymentId: "payment_id_from_cashfree",
  paymentGateway: "cashfree",
  amount: 999,
  currency: "INR",
  billingCycle: "monthly", // enum: "monthly" | "quarterly" | "annual"
  autoRenew: true
}
```

### 2.4 Trades Collection (per user)
Create subcollection: **`users/{uid}/trades`**

**Document ID:** Auto-generated

**Fields:**
```javascript
{
  symbol: "BTC/USDT",
  type: "long", // enum: "long" | "short"
  entry: 45000,
  exit: null,
  sl: 44500,
  tp: 46500,
  size: 0.1,
  leverage: 1,
  status: "open", // enum: "open" | "closed" | "partial"
  pnl: 150,
  pnlPercent: 0.33,
  openedAt: Timestamp,
  closedAt: null,
  notes: "Signal from ML model",
  source: "dashboard", // enum: "dashboard" | "api" | "alert"
  timeframe: "1h"
}
```

---

## Step 3: Set Security Rules

Update **firestore.rules** with the provided rules (already updated in your project).

**Key Security Features:**
- Users can only read/write their own documents
- Signals are readable by all authenticated users
- Only admins can update signals
- Subscriptions readable by owner only
- No public access

---

## Step 4: Create Indexes (if needed)

Go to Firebase Console → Firestore → Indexes

**Auto-created indexes:**
- `users` collection (usually auto-indexed on creation)
- `signals` collection

**Manual indexes to create (if needed):**
1. **Collection:** `subscriptions`
   - Fields: `userId` (Asc), `status` (Asc)
   - For query: `subscriptions where userId=X and status=active`

2. **Collection:** `users/{uid}/trades`
   - Fields: `status` (Asc), `openedAt` (Desc)
   - For query: `trades where status=open order by openedAt desc`

---

## Step 5: Update Firestore Configuration

### 5.1 Verify firebase.json
```json
{
  "firestore": {
    "database": "default",
    "location": "asia-south2",
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  }
}
```

### 5.2 Update firestore.indexes.json
```json
{
  "indexes": [
    {
      "collectionGroup": "subscriptions",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "userId", "order": "ASCENDING" },
        { "fieldPath": "status", "order": "ASCENDING" }
      ]
    },
    {
      "collectionGroup": "trades",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "status", "order": "ASCENDING" },
        { "fieldPath": "openedAt", "order": "DESCENDING" }
      ]
    }
  ]
}
```

---

## Step 6: Deploy Rules and Indexes

### Using Firebase CLI:

```bash
# Install Firebase CLI (if not already installed)
npm install -g firebase-tools

# Login to Firebase
firebase login

# Set active project
firebase use aegis-d78e1

# Deploy Firestore rules
firebase deploy --only firestore:rules

# Deploy indexes
firebase deploy --only firestore:indexes
```

---

## Step 7: Set Up Firestore Emulator (Local Development)

### 7.1 Install Emulator
```bash
firebase init emulators
# Select Firestore and Authentication
```

### 7.2 Start Emulator
```bash
firebase emulators:start
```

### 7.3 Use in Code
```javascript
if (location.hostname === 'localhost') {
  connectFirestoreEmulator(db, 'localhost', 8080);
  connectAuthEmulator(auth, 'http://localhost:9099');
}
```

---

## Step 8: Initialize Trial for New Users

When a user signs up, the trial is automatically created in **auth.js**:

```javascript
trial: {
  active: true,
  startDate: serverTimestamp(),
  endDate: serverTimestamp(), // Backend will add 3 days
  expiryNotified: false,
  allowedTokens: ['BTC', 'ETH', 'SOL', 'ARB', 'AAVE'],
  allowedTimeframes: ['30m', '1h']
}
```

**Firestore Functions** (backend) should update `endDate` server-side:
```javascript
// In Cloud Functions
const trialStart = new Date();
const trialEnd = new Date(trialStart.getTime() + 3 * 24 * 60 * 60 * 1000);
await updateDoc(userRef, { 'trial.endDate': Timestamp.fromDate(trialEnd) });
```

---

## Step 9: Test Firestore Setup

### 9.1 Create Test User
1. Go to Firebase Console → Authentication
2. Create test user: `test@aegis.com` / `Test@1234`

### 9.2 Verify Data
1. Go to Firestore → Collections
2. Should see `users` collection with test user document
3. Check `users/{test-uid}` has all required fields

### 9.3 Test Security Rules
Run this in browser console:
```javascript
import { db } from './gatekeeper.js';
import { doc, getDoc } from "firebase/firestore";

// Should succeed (own document)
const myDoc = await getDoc(doc(db, 'users', currentUser.uid));
console.log('✅ Can read own document:', myDoc.data());

// Should fail (other user's document)
try {
  const otherDoc = await getDoc(doc(db, 'users', 'other-uid'));
  console.log('❌ SECURITY ISSUE: Can read other user data!');
} catch (e) {
  console.log('✅ Cannot read other user data (correct):', e.message);
}
```

---

## Step 10: Backup & Monitoring

### 10.1 Enable Backup
Firebase Console → Firestore → Backups → Create backup

### 10.2 Monitor Usage
Firebase Console → Firestore → Usage tab
- Track read/write operations
- Monitor storage usage
- Set up budget alerts

---

## Common Issues & Solutions

### Issue: "Permission denied" errors
**Solution:** Check security rules - ensure user UID matches document path

### Issue: Trial not showing countdown
**Solution:** Ensure `endDate` is a Firestore Timestamp, not string

### Issue: Slow queries
**Solution:** Create composite indexes for queries with multiple conditions

### Issue: Data not syncing
**Solution:** Check internet connection, verify Firestore rules, restart app

---

## Next Steps

1. ✅ Deploy Firestore rules
2. ✅ Create collections in Firebase Console
3. ✅ Test authentication flows
4. ✅ Deploy Cloud Functions (if using backend triggers)
5. ✅ Set up monitoring and alerts
6. ✅ Create scheduled jobs for trial expiry checks
7. ✅ Implement payment webhook handlers

---

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/auth/signup` | POST | Create new user |
| `/auth/login` | POST | User login |
| `/auth/logout` | POST | User logout |
| `/signals/list` | GET | Get all signals |
| `/subscriptions/create` | POST | Create subscription |
| `/trades/open` | POST | Open trading position |
| `/trades/close` | POST | Close trading position |
| `/user/profile` | GET | Get user profile |
| `/user/update` | PUT | Update user settings |

---

## Support & Documentation

- Firebase Docs: https://firebase.google.com/docs/firestore
- Security Rules: https://firebase.google.com/docs/firestore/security/get-started
- Firestore Best Practices: https://firebase.google.com/docs/firestore/best-practices

