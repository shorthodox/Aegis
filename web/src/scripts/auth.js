// ============================================================
// AEGIS Authentication Module – Mobile-First Design
// Separate Login/Signup flows with multiple auth methods
// ============================================================

import {
  auth,
  db
} from './gatekeeper.js';

import { AuthManager } from '../auth/authManager.js';

import {
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  signInWithPhoneNumber,
  RecaptchaVerifier,
  PhoneAuthProvider,
  linkWithCredential,
  updateProfile,
  onAuthStateChanged,
  GoogleAuthProvider,
  sendPasswordResetEmail,
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";

import {
  doc,
  setDoc,
  getDoc,
  updateDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";

const googleProvider = new GoogleAuthProvider();

// ============================================================
// STATE & DOM ELEMENTS
// ============================================================
let currentAuthMode = 'login'; // 'login' | 'signup'
let recaptchaVerifier = null;
let confirmationResult = null;

// ============================================================
// HELPER: Create/Update User Document in Firestore
// ============================================================
export async function ensureUserDocumentV2(user, authMethod = 'email', phone = '') {
  if (!user) return null;

  const userDocRef = doc(db, 'users', user.uid);
  const docSnap = await getDoc(userDocRef);

  const now = new Date();
  const trialEndDate = new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000); // +3 days

  if (!docSnap.exists()) {
    // NEW USER - CREATE WITH TRIAL
    const userData = {
      uid: user.uid,
      email: user.email || '',
      phone: phone || user.phoneNumber || '',
      displayName: user.displayName || 'User',
      photoURL: user.photoURL || '',
      
      // Plan info
      plan: 'trial',
      
      // Subscription
      subscription: {
        status: 'none',
        startDate: null,
        endDate: null,
        renewalDate: null
      },
      
      // Trial tracking
      trial: {
        active: true,
        startDate: serverTimestamp(),
        endDate: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
        expiryNotified: false,
        allowedTokens: ['BTC', 'ETH', 'SOL', 'ARB', 'AAVE'], // 5 tokens max
        allowedTimeframes: ['30m', '1h']
      },
      
      // Auth methods used
      loginMethods: [authMethod],
      
      // Timestamps
      joinDate: serverTimestamp(),
      lastLogin: serverTimestamp(),
      
      // User preferences
      preferences: {
        capital: 10000,
        riskPct: 2,
        theme: 'dark',
        notifications: true
      },
      
      // Usage tracking
      usage: {
        totalSignalsToday: 0,
        lastSignalTime: null,
        signalCount: 0
      }
    };
    
    await setDoc(userDocRef, userData);
    console.log('✅ New user document created');
    return { ...userData, isNewUser: true };
  } else {
    // EXISTING USER - UPDATE LOGIN METHOD & LAST LOGIN
    const existingData = docSnap.data();
    
    // Add auth method if not already present
    const loginMethods = new Set(existingData.loginMethods || []);
    loginMethods.add(authMethod);
    
    await updateDoc(userDocRef, {
      loginMethods: Array.from(loginMethods),
      lastLogin: serverTimestamp()
    });
    
    console.log('✅ Existing user login updated');
    return { ...existingData, isNewUser: false };
  }
}

// ============================================================
// GOOGLE LOGIN/SIGNUP
// ============================================================
export async function handleGoogleAuth() {
  try {
    googleProvider.addScope('email');
    googleProvider.addScope('profile');

    const result = await signInWithPopup(auth, googleProvider);
    const user = result.user;

    // Check if this is an existing user (has a Firestore doc already)
    const userDocRef = doc(db, 'users', user.uid);
    const docSnap = await getDoc(userDocRef);

    if (docSnap.exists()) {
      // Returning user — update last login and complete auth immediately
      const existingData = docSnap.data();
      const loginMethods = new Set(existingData.loginMethods || []);
      loginMethods.add('google');
      await updateDoc(userDocRef, { loginMethods: Array.from(loginMethods), lastLogin: serverTimestamp() });
      const idToken = await user.getIdToken();
      AuthManager.setToken(idToken);
      AuthManager.setUser(existingData);
      localStorage.setItem('authenticated', 'true');
      return { success: true, user, isNewUser: false, message: 'Logged in successfully!' };
    }

    // New user — phone verification required before Firestore doc is created
    return { success: true, user, isNewUser: true, message: 'Phone verification required' };
  } catch (error) {
    console.error('Google Auth error:', error.code, error.message);
    return { success: false, message: error.message || 'Google authentication failed' };
  }
}

// Called after phone OTP is verified for a new Google user.
export async function completeGoogleSignupWithPhone(user, phone) {
  try {
    const userData = await ensureUserDocumentV2(user, 'google', phone);
    const idToken = await user.getIdToken();
    AuthManager.setToken(idToken);
    AuthManager.setUser(userData);
    localStorage.setItem('authenticated', 'true');
    return { success: true };
  } catch (error) {
    console.error('completeGoogleSignupWithPhone error:', error);
    return { success: false, message: error.message || 'Failed to complete signup' };
  }
}

// ============================================================
// EMAIL/PASSWORD SIGNUP
// ============================================================
// ============================================================
// OTP helpers — call backend before creating Firebase account
// ============================================================
export async function sendOTPForSignup(email, phone = null) {
  try {
    const res = await fetch('/auth/send-otp-for-registration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, phone })
    });
    const data = await res.json();
    if (!res.ok) return { success: false, message: data.detail || 'Failed to send OTP' };
    return { success: true, message: data.message };
  } catch {
    return { success: false, message: 'Network error. Please try again.' };
  }
}

// Helper: obtain MSG91 widget token from server
export async function fetchMsg91WidgetToken() {
  try {
    const res = await fetch('/auth/msg91-token');
    if (!res.ok) return null;
    const j = await res.json();
    return j.token || null;
  } catch (err) {
    console.warn('Could not fetch MSG91 token', err);
    return null;
  }
}

// Server-side verify proxy for MSG91 access token
export async function verifyMsg91AccessToken(accessToken) {
  try {
    const res = await fetch('/auth/msg91-verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 'access_token': accessToken })
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || 'Verification failed');
    }
    return await res.json();
  } catch (err) {
    console.warn('MSG91 verify proxy error', err);
    throw err;
  }
}

// ============================================================
// MSG91 OTPWidget Integration Helpers
// These wrappers try to use a globally-loaded `OTPWidget` (from provider
// script) or dynamically import the `@msg91comm/sendotp-sdk` package when
// available (bundler environment). They expose simple send/retry/verify
// methods used by the signup UI.
// ============================================================
export async function initMsg91Widget(widgetId, authToken) {
  try {
    if (window.OTPWidget && typeof window.OTPWidget.initializeWidget === 'function') {
      window.OTPWidget.initializeWidget(widgetId, authToken);
      return true;
    }

    // Try dynamic import when running in a bundler (npm-installed package)
    try {
      const mod = await import('@msg91comm/sendotp-sdk');
      const { OTPWidget } = mod;
      if (OTPWidget && typeof OTPWidget.initializeWidget === 'function') {
        OTPWidget.initializeWidget(widgetId, authToken);
        // keep a reference for other helper functions
        window.OTPWidget = OTPWidget;
        return true;
      }
    } catch (e) {
      // dynamic import may fail in plain browser environment; ignore
    }

    console.warn('MSG91 OTPWidget not available to initialize');
    return false;
  } catch (err) {
    console.error('initMsg91Widget error', err);
    return false;
  }
}

export async function sendOTPWithWidget(identifier) {
  if (!identifier) throw new Error('identifier required (phone or email)');
  const widget = window.OTPWidget;
  if (!widget || typeof widget.sendOTP !== 'function') {
    throw new Error('OTPWidget not initialized or sendOTP not available');
  }
  return await widget.sendOTP({ identifier });
}

export async function retryOTPWithWidget(reqId, retryChannel = undefined) {
  const widget = window.OTPWidget;
  if (!widget || typeof widget.retryOTP !== 'function') {
    throw new Error('OTPWidget not initialized or retryOTP not available');
  }
  const body = { reqId };
  if (typeof retryChannel !== 'undefined') body.retryChannel = retryChannel;
  return await widget.retryOTP(body);
}

export async function verifyOTPWithWidget(reqId, otp) {
  if (!reqId || !otp) throw new Error('reqId and otp required');
  const widget = window.OTPWidget;
  if (!widget || typeof widget.verifyOTP !== 'function') {
    throw new Error('OTPWidget not initialized or verifyOTP not available');
  }
  const body = { reqId, otp };
  return await widget.verifyOTP(body);
}

export async function verifyOTPForSignup(email, otp, phone = null) {
  try {
    const res = await fetch('/auth/verify-otp-for-registration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, otp, phone })
    });
    const data = await res.json();
    if (!res.ok) return { success: false, message: data.detail || 'Invalid OTP' };
    return { success: true, signup_token: data.signup_token };
  } catch {
    return { success: false, message: 'Network error. Please try again.' };
  }
}

// ============================================================
// EMAIL/PASSWORD SIGNUP (called only after OTP is verified)
// ============================================================
export async function handleEmailSignup(email, password, displayName, signupToken = null, mobile = null) {
  try {
    if (!email || !password || !displayName) {
      throw new Error('Please fill in all fields');
    }

    if (password.length < 8) {
      throw new Error('Password must be at least 8 characters');
    }

    // signupToken is only present in the legacy email-OTP flow.
    // Firebase Phone Auth flow sets it to null; that's fine — phone was verified client-side.
    if (signupToken) {
      sessionStorage.setItem('otp_signup_token', signupToken);
    }
    if (mobile) sessionStorage.setItem('pending_phone', mobile);

    // OTP already proved email is reachable — create account and sign in directly.
    // fetchSignInMethodsForEmail is deprecated and broken when Firebase's email
    // enumeration protection is enabled; let createUserWithEmailAndPassword handle
    // duplicates via auth/email-already-in-use instead.
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    await updateProfile(user, {
      displayName: displayName,
      photoURL: `https://ui-avatars.com/api/?name=${encodeURIComponent(displayName)}&background=00f2ff&color=000`
    });

    // Create Firestore document immediately (email was verified via OTP)
    const userData = await ensureUserDocumentV2(user, 'email');

    const idToken = await user.getIdToken();

    // Provision the backend user record (users/{email} keyed doc with otp_verified:true).
    // Without this, /auth/me looks up by email key and returns 404/403, kicking the
    // user back from dashboard even after a successful trial start.
    try {
      const provisionRes = await fetch('/api/users/provision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`
        },
        body: JSON.stringify({
          uid: user.uid,
          email: user.email,
          display_name: displayName,
          provider: 'password',
          signup_token: signupToken,
          phone_number: mobile || null
        })
      });
      if (provisionRes.ok) {
        sessionStorage.removeItem('otp_signup_token');
      }
    } catch (provisionErr) {
      console.warn('[signup] Backend provision failed (non-fatal):', provisionErr.message);
    }

    AuthManager.setToken(idToken);
    AuthManager.setUser(userData);
    localStorage.setItem('authenticated', 'true');

    console.log('✅ Email signup successful');
    return { success: true, user, message: 'Account created successfully!', userData };
  } catch (error) {
    console.error('❌ Email signup error:', error.code, error.message);
    let message = 'Signup failed. Please try again.';
    if (error.code === 'auth/email-already-in-use') {
      message = 'Email already in use. Please sign in instead.';
    } else if (error.code === 'auth/invalid-email') {
      message = 'Invalid email address. Please check and try again.';
    } else if (error.code === 'auth/weak-password') {
      message = 'Password must be at least 8 characters.';
    } else if (error.message) {
      message = error.message;
    }
    return { success: false, message };
  }
}

// ============================================================
// EMAIL/PASSWORD LOGIN
// ============================================================
export async function handleEmailLogin(email, password) {
  try {
    if (!email || !password) {
      throw new Error('Please enter email and password');
    }

    console.log('🔐 Logging in with email...');
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // Gate: check Firestore — only users who completed OTP-verified signup exist here
    const userDocRef = doc(db, 'users', user.uid);
    const docSnap = await getDoc(userDocRef);
    if (!docSnap.exists()) {
      await signOut(auth);
      return {
        success: false,
        needsSignup: true,
        message: 'No account found for this email. Please create an account first.'
      };
    }

    // Firestore doc confirmed — update last login
    const userData = await ensureUserDocumentV2(user, 'email');

    // Store token
    const idToken = await user.getIdToken();
    AuthManager.setToken(idToken);
    AuthManager.setUser(userData);
    localStorage.setItem('authenticated', 'true');

    console.log('✅ Email login successful');
    return { success: true, user, message: 'Logged in successfully!', userData };
  } catch (error) {
    console.error('❌ Email login error:', error.code, error.message);

    let userMessage = 'Login failed';
    if (error.code === 'auth/user-not-found') {
      userMessage = 'No account found with this email. Please sign up first.';
    } else if (error.code === 'auth/wrong-password' || error.code === 'auth/invalid-credential') {
      userMessage = 'Incorrect password. Please try again.';
    } else if (error.code === 'auth/invalid-email') {
      userMessage = 'Invalid email address.';
    } else if (error.code === 'auth/too-many-requests') {
      userMessage = 'Too many failed attempts. Please try again later.';
    }

    return {
      success: false,
      message: userMessage
    };
  }
}

// ============================================================
// PHONE/OTP SIGNUP
// ============================================================
export async function setupPhoneRecaptcha() {
  try {
    if (window.recaptchaVerifier) {
      return window.recaptchaVerifier;
    }
    
    const verifier = new RecaptchaVerifier(auth, 'recaptcha-container', {
      'size': 'invisible',
      'callback': (response) => {
        console.log('✅ reCAPTCHA verified:', response);
      },
      'expired-callback': () => {
        console.warn('⚠️ reCAPTCHA expired');
      }
    });
    
    window.recaptchaVerifier = verifier;
    return verifier;
  } catch (error) {
    console.error('❌ reCAPTCHA setup error:', error);
    throw error;
  }
}

export async function sendPhoneOTP(phoneNumber, displayName) {
  try {
    if (!phoneNumber || !displayName) {
      throw new Error('Phone number and name required');
    }

    // Always clear stale verifier — previous errors leave it in a broken state
    if (window.recaptchaVerifier) {
      try { window.recaptchaVerifier.clear(); } catch (_) {}
      window.recaptchaVerifier = null;
    }

    const formattedPhone = phoneNumber.startsWith('+') ? phoneNumber : '+91' + phoneNumber;

    const recaptcha = await setupPhoneRecaptcha();
    confirmationResult = await signInWithPhoneNumber(auth, formattedPhone, recaptcha);
    
    window.currentPhoneData = { phoneNumber: formattedPhone, displayName };
    
    console.log('✅ OTP sent successfully');
    return { success: true, message: 'OTP sent! Check your phone.' };
  } catch (error) {
    console.error('❌ OTP send error:', error.code, error.message);
    return { 
      success: false, 
      message: error.message || 'Failed to send OTP'
    };
  }
}

export async function verifyPhoneOTP(otpCode) {
  try {
    if (!otpCode || !confirmationResult) {
      throw new Error('OTP code required');
    }
    
    console.log('✔️ Verifying OTP...');
    const userCredential = await confirmationResult.confirm(otpCode);
    const user = userCredential.user;
    
    // Update profile if we have display name
    if (window.currentPhoneData?.displayName) {
      await updateProfile(user, {
        displayName: window.currentPhoneData.displayName,
        photoURL: `https://ui-avatars.com/api/?name=${encodeURIComponent(window.currentPhoneData.displayName)}&background=00f2ff&color=000`
      });
    }
    
    // Create user document
    await ensureUserDocumentV2(user, 'phone');
    
    // Store token
    const idToken = await user.getIdToken();
    AuthManager.setToken(idToken);
    localStorage.setItem('authenticated', 'true');
    
    confirmationResult = null;
    window.currentPhoneData = null;
    
    console.log('✅ Phone OTP verified successfully');
    return { success: true, user, message: 'Account created via phone!' };
  } catch (error) {
    console.error('❌ OTP verification error:', error.code, error.message);
    return {
      success: false,
      message: error.message || 'Invalid OTP'
    };
  }
}

// Lightweight variant used during email/password signup.
// Confirms the Firebase phone OTP, signs out the temp phone user,
// then lets handleEmailSignup create the real email/password account.
export async function verifyPhoneOTPForRegistration(otpCode) {
  try {
    if (!otpCode || !confirmationResult) {
      throw new Error('OTP code required');
    }
    await confirmationResult.confirm(otpCode);
    confirmationResult = null;
    // Do NOT sign out here — callers manage auth state:
    // • Google flow: current user stays signed in to write Firestore doc
    // • Email flow: createUserWithEmailAndPassword switches the session automatically
    return { success: true };
  } catch (error) {
    const code = error.code || '';
    console.error('Phone OTP registration verify error:', code, error.message);
    let message = `Verification failed (${code || 'unknown'}). Please try again.`;
    if (code === 'auth/invalid-verification-code' || code === 'auth/invalid-credential') {
      message = 'Incorrect code. Please check and retry.';
    } else if (code === 'auth/code-expired') {
      message = 'Code expired. Please request a new one.';
    } else if (code === 'auth/too-many-requests') {
      message = 'Too many attempts. Please wait a moment and try again.';
    }
    return { success: false, message };
  }
}

// For Google signup: links the phone credential to the Google user WITHOUT
// replacing auth.currentUser. confirmationResult.confirm() would sign in a
// new phone user, kicking out the Google user and breaking the Firestore write.
export async function verifyAndLinkPhoneToGoogle(otpCode, googleUser) {
  try {
    if (!otpCode || !confirmationResult) throw new Error('OTP code required');
    const verificationId = confirmationResult.verificationId;
    const phoneCredential = PhoneAuthProvider.credential(verificationId, otpCode);
    try {
      await linkWithCredential(googleUser, phoneCredential);
    } catch (linkErr) {
      const lc = linkErr.code || '';
      // Already linked to this account or to an orphaned test account — OTP was valid either way.
      // Firebase validates the code BEFORE checking for conflicts, so credential-already-in-use
      // means the code was correct. Our Firestore uniqueness check already passed, so proceed.
      if (lc === 'auth/provider-already-linked' || lc === 'auth/credential-already-in-use') {
        confirmationResult = null;
        return { success: true };
      }
      throw linkErr;
    }
    confirmationResult = null;
    return { success: true };
  } catch (error) {
    const code = error.code || '';
    console.error('Phone link to Google error:', code, error.message);
    let message = `Verification failed (${code || 'unknown'}). Please try again.`;
    if (code === 'auth/invalid-verification-code' || code === 'auth/invalid-credential') {
      message = 'Incorrect code. Please check and retry.';
    } else if (code === 'auth/code-expired') {
      message = 'Code expired. Please request a new one.';
    } else if (code === 'auth/too-many-requests') {
      message = 'Too many attempts. Please wait a moment and try again.';
    }
    return { success: false, message };
  }
}

// ============================================================
// LOGOUT
// ============================================================
export async function handleLogout() {
  try {
    await signOut(auth);
    AuthManager.logout();
    localStorage.removeItem('authenticated');
    localStorage.removeItem('userSession');
    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    localStorage.removeItem('trial_end_timestamp');
    localStorage.removeItem('trial_end_sig');
    localStorage.removeItem('cached_uid');
    Object.keys(localStorage).forEach(k => {
      if (k.startsWith('trialStart_')) localStorage.removeItem(k);
    });
    window.location.href = '/';
    return { success: true };
  } catch (error) {
    console.error('❌ Logout error:', error);
    return { success: false, message: error.message };
  }
}

// ============================================================
// PASSWORD RESET
// ============================================================
// All reset emails are delivered exclusively via our Neo SMTP domain
// so they land in the inbox branded as gatekeeper.sbs — never via Firebase.
export async function handlePasswordReset(email) {
  if (!email) return { success: false, message: 'Email required.' };
  try {
    const res = await fetch('/auth/send-password-reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      return { success: true, message: data.message || 'Reset link sent — check your inbox (and spam folder).' };
    }
    if (res.status === 429) {
      return { success: false, message: 'Too many attempts. Please wait a few minutes and try again.' };
    }
    // SMTP failure — silently fall back to Firebase's own reset email
    if (data.detail === 'smtp_failure') {
      try {
        await sendPasswordResetEmail(auth, email);
        return { success: true, message: 'Reset link sent — check your inbox (and spam folder).' };
      } catch (fbErr) {
        if (fbErr.code === 'auth/user-not-found') {
          return { success: true, message: 'If an account with this email exists, a reset link has been sent.' };
        }
        return { success: false, message: 'Failed to send reset email. Please contact support@gatekeeper.sbs.' };
      }
    }
    return {
      success: false,
      message: data.detail || 'Failed to send reset email. Please try again.',
    };
  } catch (error) {
    console.error('❌ Password reset error:', error);
    return { success: false, message: 'Network error. Please check your connection and try again.' };
  }
}

// ============================================================
// AUTH STATE LISTENER
// ============================================================
export function subscribeToAuthState(callback) {
  return onAuthStateChanged(auth, async (user) => {
    if (user) {
      const userData = await getDoc(doc(db, 'users', user.uid));
      callback({
        authenticated: true,
        user,
        userData: userData.data()
      });
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
    } else {
      callback({
        authenticated: false,
        user: null,
        userData: null
      });
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: false } }));
    }
  });
}

// ============================================================
// HELPER: Check if User is Authenticated
// ============================================================
export function isUserAuthenticated() {
  return localStorage.getItem('authenticated') === 'true' || localStorage.getItem('userSession') || !!AuthManager.getToken();
}

// ============================================================
// HELPER: Get Current User ID
// ============================================================
export function getCurrentUserId() {
  return auth.currentUser?.uid || null;
}

// ============================================================
// HELPER: Get User Data
// ============================================================
export async function getUserData(userId) {
  if (!userId) return null;
  try {
    const userDoc = await getDoc(doc(db, 'users', userId));
    return userDoc.exists() ? userDoc.data() : null;
  } catch (error) {
    console.error('❌ Error fetching user data:', error);
    return null;
  }
}
