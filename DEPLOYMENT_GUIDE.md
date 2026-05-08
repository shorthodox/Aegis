# AEGIS v1.0 – Complete Deployment & Testing Guide

## 📋 Implementation Checklist

### ✅ Completed Components
- [x] Firestore schema and security rules
- [x] Mobile-responsive authentication UI (Login/Signup)
- [x] Phone OTP authentication with Recaptcha
- [x] Free trial 3-day countdown system
- [x] Dashboard component with mobile navigation
- [x] main.py updated with Firestore signals endpoints
- [x] CSS optimized for Android and iPhones
- [x] Signal restriction logic for trial users

---

## 🚀 Deployment Steps

### Step 1: Update Firebase Console

#### 1.1 Enable Authentication Methods
Go to [Firebase Console](https://console.firebase.google.com) → Authentication → Sign-in method:

1. **Google Sign-In**
   - Enable it
   - Add your web domain to authorized JavaScript origins

2. **Email/Password**
   - Enable it
   - Set password requirements (min 8 chars recommended)

3. **Phone Authentication**
   - Enable it
   - Add Recaptcha secret key (create if needed)
   - Add authorized domains

4. **Update Firebase Config in gatekeeper.js**
   - Project ID
   - API Key
   - Auth Domain
   - Storage Bucket

#### 1.2 Create Firestore Collections
Follow the [FIRESTORE_SETUP_GUIDE.md](../FIRESTORE_SETUP_GUIDE.md) for:
- Creating `users` collection
- Creating `signals` collection
- Creating `subscriptions` collection
- Setting up proper indexes

#### 1.3 Deploy Security Rules
```bash
cd path/to/project
firebase login
firebase use aegis-d78e1
firebase deploy --only firestore:rules
```

### Step 2: Update Environment Variables

#### 2.1 Backend (.env file)
```bash
# Firebase
FIREBASE_CREDENTIALS=/path/to/serviceAccountKey.json

# JWT
JWT_SECRET_KEY=your_super_secret_jwt_key_here
ALGORITHM=HS256

# Cashfree Payment
CASHFREE_APP_ID=your_cashfree_app_id
CASHFREE_SECRET_KEY=your_cashfree_secret_key
CASHFREE_ENV=TEST  # Change to PROD for live payments

# Email Service
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password

# Base URL
BASE_URL=https://your-domain.com  # For production
```

#### 2.2 Frontend Configuration
Update [web/src/scripts/gatekeeper.js](../web/src/scripts/gatekeeper.js):
```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "aegis-d78e1.firebaseapp.com",
  projectId: "aegis-d78e1",
  storageBucket: "aegis-d78e1.firebasestorage.app",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

### Step 3: Install & Deploy

#### 3.1 Backend Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally first
python main.py

# Deploy to production (e.g., Railway, Heroku, Google Cloud Run)
# For Railway:
railway up
```

#### 3.2 Frontend Deployment
```bash
# Deploy to Firebase Hosting
npm install -g firebase-tools
firebase login
firebase deploy --only hosting
```

### Step 4: Test Authentication Flow

#### 4.1 Test Email/Password Signup
1. Go to homepage
2. Click "Client Login"
3. Choose "Continue with Email"
4. Enter email and password
5. Verify account created in Firebase Console

#### 4.2 Test Google Login
1. Click "Client Login"
2. Choose "Continue with Google"
3. Select Google account
4. Verify redirected and logged in

#### 4.3 Test Phone OTP
1. Click "Client Login"
2. Choose "Continue with Phone"
3. Enter phone number (+91XXXXXXXXXX)
4. Enter OTP received
5. Verify account created

#### 4.4 Test Trial Countdown
1. Signup with new account
2. Check Firestore - user doc should have trial data
3. Verify countdown displays on pages
4. Test trial signal restrictions

### Step 5: Test Signal Retrieval

#### 5.1 Add Test Signals to Firestore
In Firebase Console → Firestore → Collections → signals → Add document:

**Document ID:** BTC
```json
{
  "symbol": "BTC",
  "signal": "BUY",
  "entry": 45000,
  "sl": 44500,
  "tp": 46500,
  "timeframe": "1h",
  "confidence": 0.85,
  "timestamp": "2025-05-06T00:00:00Z",
  "active": true
}
```

#### 5.2 Test API Endpoints
```bash
# Get all signals
curl http://localhost:8000/api/signals

# Get specific signal
curl http://localhost:8000/api/signals/BTC

# Get public signals (no auth)
curl http://localhost:8000/api/public/signals
```

#### 5.3 Test in Dashboard
1. Login with test account
2. Go to /pages/dashboard.html
3. Verify signals display
4. For trial user, verify only 5 tokens show
5. For trial user, verify only 30m/1h timeframes show

### Step 6: Mobile Testing

#### 6.1 Test on iOS
1. Open web app in Safari on iPhone
2. Add to home screen
3. Test:
   - Navigation tabs in mobile nav
   - Auth modal on mobile view
   - Signal cards responsive
   - Trial countdown visible

#### 6.2 Test on Android
1. Open web app in Chrome on Android
2. Test:
   - Hamburger menu working
   - Buttons accessible on small screens
   - Keyboard doesn't overlap inputs
   - Trial countdown updates

#### 6.3 Responsive Testing Tools
```bash
# Chrome DevTools
- Ctrl+Shift+I → Toggle device toolbar (Ctrl+Shift+M)
- Test: iPhone 12, iPhone 14, Pixel 5, etc.

# Firefox Responsive Design Mode
- Ctrl+Shift+M
```

### Step 7: Performance Optimization

#### 7.1 Firestore Optimization
- Enable automatic scaling
- Create composite indexes for queries
- Monitor read/write operations

#### 7.2 Frontend Optimization
```bash
# Minify CSS and JS
npm install -g cssnano uglify-js

# Check bundle size
npm install -g bundle-analyzer

# Optimize images
# Use WebP format where possible
```

#### 7.3 Caching Strategy
- Cache signals for 5 minutes client-side
- Use service worker for offline support
- Cache auth tokens in localStorage

---

## 🧪 Test Scenarios

### Scenario 1: New User Signup Flow
**Steps:**
1. User visits homepage
2. Clicks "Client Login" → Opens auth modal
3. Selects "Continue with Email"
4. Enters new email, password, name
5. Account created in Firestore
6. Trial activated (3 days)
7. Redirected to dashboard

**Expected:**
- ✅ User doc created in Firestore
- ✅ Trial fields populated
- ✅ Countdown displays
- ✅ 5 tokens access only

---

### Scenario 2: Trial Expiry Notification
**Steps:**
1. Create user with trial ending today
2. Check every 5 minutes (simulate)
3. At 24h before expiry, send notification
4. At expiry, update trial.active to false

**Expected:**
- ✅ Notification sent
- ✅ UI shows "Trial Expired"
- ✅ User redirected to pricing page

---

### Scenario 3: Signal Access Control
**Steps:**
1. Login as trial user
2. View signals in dashboard
3. Try to access SOL (not in allowed list)
4. System blocks access

**Expected:**
- ✅ Only [BTC, ETH, SOL, ARB, AAVE] visible
- ✅ Only 30m/1h timeframes shown
- ✅ Toast message "Trial restriction"

---

### Scenario 4: Mobile Navigation
**Steps:**
1. Resize browser to 375px width
2. Click hamburger menu
3. Navigate to different sections
4. Verify all pages load

**Expected:**
- ✅ Menu slides in from left
- ✅ Navigation items accessible
- ✅ Login/Logout button visible
- ✅ Trial countdown displays

---

## 🔍 Testing Checklist

- [ ] Email/Password authentication works
- [ ] Google authentication works
- [ ] Phone OTP authentication works
- [ ] Trial countdown displays correctly
- [ ] Trial expires after 3 days
- [ ] Trial user sees restrictions
- [ ] Signals load from Firestore
- [ ] Mobile layout responsive on all devices
- [ ] iOS app works in Safari
- [ ] Android app works in Chrome
- [ ] Payment gateway configured (if needed)
- [ ] Error messages are clear
- [ ] Loading states show properly
- [ ] Network errors handled gracefully
- [ ] Logout works properly

---

## 🐛 Troubleshooting

### Issue: "Cannot find Firebase config"
**Solution:**
- Check gatekeeper.js firebaseConfig
- Verify Firebase project ID matches console

### Issue: "OTP not sending"
**Solution:**
- Enable Phone auth in Firebase Console
- Add Recaptcha key
- Check phone number format (+91...)

### Issue: "Trial countdown not showing"
**Solution:**
- Check Firestore user doc has trial fields
- Verify trial-countdown.js loaded
- Check browser console for errors

### Issue: "Signals not displaying"
**Solution:**
- Verify signals exist in Firestore collection
- Check API endpoint returning data
- Inspect network tab for 404 errors

### Issue: "Mobile layout broken"
**Solution:**
- Clear browser cache
- Check CSS media queries
- Test in different browsers
- Verify viewport meta tag in HTML

---

## 📊 Monitoring & Analytics

### Firebase Console Monitoring
- Firestore → Monitoring tab
- Authentication → Analytics tab
- Performance metrics

### Error Tracking
- Use Firebase Crashlytics
- Monitor API error rates
- Track failed authentications

### User Analytics
- Track signup funnel
- Monitor trial conversions
- Measure engagement metrics

---

## 📝 Next Steps

1. **Deploy to production**
   - Use environment-specific configs
   - Set up CI/CD pipeline
   - Configure domain SSL certificate

2. **Enable notifications**
   - Send trial expiry emails
   - Push notifications for signals
   - SMS for OTP (using Twilio or similar)

3. **Add payment integration**
   - Complete Cashfree setup
   - Test payment webhook
   - Implement subscription management

4. **Scale Firestore**
   - Set up automatic backups
   - Monitor database growth
   - Optimize queries and indexes

5. **Security hardening**
   - Enable HTTPS everywhere
   - Add rate limiting
   - Implement CSRF protection
   - Regular security audits

---

## 📚 Useful Resources

- [Firebase Documentation](https://firebase.google.com/docs)
- [Firestore Best Practices](https://firebase.google.com/docs/firestore/best-practices)
- [Firebase CLI Reference](https://firebase.google.com/docs/cli)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Mobile Responsive Design](https://www.w3schools.com/css/css_rwd_intro.asp)

---

## ✨ Final Checklist

- [ ] All environment variables configured
- [ ] Firebase rules deployed
- [ ] Firestore collections created with indexes
- [ ] Frontend built and tested
- [ ] Backend running without errors
- [ ] Authentication flows tested on all methods
- [ ] Mobile layout tested on multiple devices
- [ ] Trial system working correctly
- [ ] Signals endpoint responding with data
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backup plan ready
- [ ] Go-live date set

---

**Created:** 2025-05-06  
**Version:** AEGIS v1.0  
**Status:** Ready for Deployment ✅

