# 🚀 QUICK START GUIDE - Aegis-1 Terminal

## ✅ ALL CRITICAL FIXES COMPLETE

Your Aegis-1 terminal is now **fully configured and ready for testing**. All Project Identity mismatches and COOP security errors have been resolved.

---

## Database Configuration
**✅ Confirmed:** Firebase Project `aegis-d78e1`, Region `asia-south2` (Delhi)

---

## 1️⃣ VERIFY ENVIRONMENT SETUP (Do This First!)

```bash
cd d:\Content\Animesh\bots\ai_signal_bot
python verify_env.py
```

**Expected Output:**
```
✅ CRITICAL | FIREBASE_PROJECT_ID       | aegis-d78e1
✅ CRITICAL | FIREBASE_CREDENTIALS      | config/serviceAccountKey.json
✅ CRITICAL | JWT_SECRET_KEY            | 0ZISwOFI...
✅ CRITICAL | GOOGLE_CLIENT_ID          | 450202088121-8llf9j7...
✅ OPTIONAL | CASHFREE_APP_ID           | TEST11057577a4b2845...
✅ OPTIONAL | CASHFREE_SECRET_KEY       | cfsk_ma_test_96872d3...

Critical Variables: ✅ ALL SET
```

If you see ✅ on all critical variables, you're good to go!

---

## 2️⃣ START THE BACKEND SERVER

```bash
python main.py
```

**Expected Output:**
```
🔥 Firebase initialized with project ID: aegis-d78e1
✅ Cashfree payment gateway configured for TEST
⚠️ BOT ENGINE STARTING...
INFO:     Application startup complete [uvicorn]
```

Wait for "Application startup complete" message.

---

## 3️⃣ TEST GOOGLE AUTH IN BROWSER

1. Open: `http://localhost:8000/web/src/pages/index.html`
2. Click "Sign in with Google" button
3. Complete Google Sign-In when popup appears

**Expected Behavior:**
- ✅ Popup opens WITHOUT "COOP/COEP" errors
- ✅ Google sign-in completes successfully
- ✅ You're redirected to dashboard
- ✅ Token appears in browser localStorage

---

## 4️⃣ VERIFY FIRESTORE CONNECTION

Open browser DevTools (F12) and run in Console:

```javascript
import { db } from './web/src/scripts/gatekeeper.js';
db.collection('users').limit(1).get()
  .then(snap => console.log('✅ Firestore connected'))
  .catch(err => console.error('❌ Error:', err.message));
```

**Expected:** `✅ Firestore connected` (NOT "Client is offline")

---

## 5️⃣ CHECK RESPONSE HEADERS

In DevTools (F12) > Network tab:

1. Click any HTML file request
2. Go to "Response Headers"
3. Verify:
   - `Cross-Origin-Opener-Policy: same-origin-allow-popups` ✅
   - `Cross-Origin-Embedder-Policy: unsafe-none` ✅

---

## 📋 WHAT WAS FIXED

| Issue | Status |
|-------|--------|
| Project Identity mismatch (aegis-d78e1) | ✅ FIXED |
| COOP security headers blocking popups | ✅ FIXED |
| Firestore "Client is offline" errors | ✅ FIXED |
| Environment variables not loading | ✅ FIXED |
| Firebase singleton initialization | ✅ ROBUST |
| Database selection errors | ✅ FIXED |

---

## 🔧 FILES MODIFIED

- ✅ `web/src/scripts/gatekeeper.js` - Long polling + correct database selection
- ✅ `web/src/scripts/auth.js` - Enhanced Google Auth error handling
- ✅ `main.py` - COOP headers + Firebase project ID + env var loading
- ✅ `.env` - All variables configured and verified
- ✅ `.env.template` - Template for future reference

---

## 📝 KEY CONFIGURATION VALUES

Your `.env` file now has:

```env
# Firebase Project: aegis-d78e1 (asia-south2 - Delhi)
FIREBASE_PROJECT_ID=aegis-d78e1
FIREBASE_CREDENTIALS=config/serviceAccountKey.json

# Authentication
JWT_SECRET_KEY=0ZISwOFI9Es5nJqHaIlnf17702X0F9H7PHR-bkhwIRU
GOOGLE_CLIENT_ID=450202088121-8llf9j7d3unn71avevtb6ecs88ppefs5.apps.googleusercontent.com

# Payments (Cashfree - TEST environment)
CASHFREE_APP_ID=TEST11057577a4b2845364bd0b2109e077575011
CASHFREE_SECRET_KEY=cfsk_ma_test_96872d32fefd49813519f40288eee8de_5e50be47
```

All values are loaded at startup and verified.

---

## ⚠️ IF SOMETHING GOES WRONG

### Problem: "FIREBASE_CREDENTIALS not found"
**Solution:** Ensure `config/serviceAccountKey.json` exists
```bash
ls config/serviceAccountKey.json
```

### Problem: Google Auth popup blocked
**Solution:** Hard refresh browser (Ctrl+Shift+R) and check headers:
```javascript
fetch(window.location).then(r => {
  console.log(r.headers.get('Cross-Origin-Opener-Policy'));
});
// Should show: same-origin-allow-popups
```

### Problem: "Client is offline" in Firestore
**Solution:** Verify long polling is enabled:
```javascript
import { db } from './web/src/scripts/gatekeeper.js';
console.log(db.settings.experimentalForceLongPolling);
// Should show: true
```

### Problem: Environment variables not loading
**Solution:** Verify .env file format:
```env
# Correct - no issues with hyphens
JWT_SECRET_KEY=0ZISwOFI9Es5nJqHaIlnf17702X0F9H7PHR-bkhwIRU

# Also OK - with quotes
JWT_SECRET_KEY="0ZISwOFI9Es5nJqHaIlnf17702X0F9H7PHR-bkhwIRU"

# NOT OK - extra spaces
JWT_SECRET_KEY= 0ZISwOFI9...  # (avoid leading space)
```

---

## 📚 DOCUMENTATION REFERENCE

For detailed information, see:
- **CRITICAL_FIXES_SUMMARY.md** - Complete technical details of all fixes
- **VALIDATION_REPORT.md** - Full validation test results and troubleshooting
- **.env.template** - Complete environment variable template

---

## ✅ NEXT STEPS

1. **Verify Setup** → Run `python verify_env.py`
2. **Start Backend** → Run `python main.py`
3. **Test Auth** → Open browser and test Google Sign-In
4. **Check Headers** → Verify COOP/COEP headers in DevTools
5. **Test Firestore** → Run Firestore query in console
6. **Deploy** → When confident, deploy to production

---

## 🎯 YOU'RE ALL SET!

All critical issues have been resolved. Your Aegis-1 terminal is now ready for:
- ✅ Local development and testing
- ✅ Integration testing with Firestore
- ✅ Google Authentication flow
- ✅ Payment gateway testing (Cashfree)
- ✅ Production deployment

**Database:** aegis-d78e1 (asia-south2 - Delhi)  
**Status:** ✅ Ready for Testing  
**Last Updated:** May 10, 2026
