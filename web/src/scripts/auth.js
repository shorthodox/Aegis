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
  updateProfile,
  fetchSignInMethodsForEmail,
  onAuthStateChanged,
  GoogleAuthProvider
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
export async function ensureUserDocumentV2(user, authMethod = 'email') {
  if (!user) return null;

  // Using the db instance imported from gatekeeper.js
  // The firebaseConfig should be properly initialized in gatekeeper.js
  // Make sure gatekeeper.js has the correct Firestore configuration
  
  const userDocRef = doc(db, 'users', user.uid);
  const docSnap = await getDoc(userDocRef);
  
  const now = new Date();
  const trialEndDate = new Date(now.getTime() + 3 * 24 * 60 * 60 * 1000); // +3 days
  
  if (!docSnap.exists()) {
    // NEW USER - CREATE WITH TRIAL
    const userData = {
      uid: user.uid,
      email: user.email || '',
      phone: user.phoneNumber || '',
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
    console.log('✅ New user document created:', user.uid);
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
    
    console.log('✅ Existing user login updated:', user.uid);
    return { ...existingData, isNewUser: false };
  }
}

// ============================================================
// GOOGLE LOGIN/SIGNUP
// ============================================================
export async function handleGoogleAuth() {
  try {
    console.log('🔐 Starting Google Auth...');
    // Set scopes to help with popup/window.closed operations
    googleProvider.addScope('email');
    googleProvider.addScope('profile');
    
    const result = await signInWithPopup(auth, googleProvider);
    const user = result.user;
    
    // Create/update user document
    const userData = await ensureUserDocumentV2(user, 'google');
    
    // Store token
    const idToken = await user.getIdToken();
    AuthManager.setToken(idToken);
    AuthManager.setUser(userData);
    localStorage.setItem('authenticated', 'true');
    
    console.log('✅ Google Auth successful:', user.email);
    return { success: true, user, message: 'Logged in successfully!', userData };
  } catch (error) {
    console.error('❌ Google Auth error:', error.code, error.message);
    // COOP error handling: if the error is about window.closed being blocked
    if (error.code === 'auth/popup-blocked' || error.message?.includes('window.closed')) {
      console.error('⚠️ COOP/COEP Error detected. Make sure your server sends proper headers.');
      console.error('Expected headers: Cross-Origin-Opener-Policy: same-origin-allow-popups, Cross-Origin-Embedder-Policy: unsafe-none');
    }
    return { 
      success: false, 
      message: error.message || 'Google authentication failed'
    };
  }
}

// ============================================================
// EMAIL/PASSWORD SIGNUP
// ============================================================
// ============================================================
// OTP helpers — call backend before creating Firebase account
// ============================================================
export async function sendOTPForSignup(email) {
  try {
    const res = await fetch('/auth/send-otp-for-registration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const data = await res.json();
    if (!res.ok) return { success: false, message: data.detail || 'Failed to send OTP' };
    return { success: true, message: data.message };
  } catch {
    return { success: false, message: 'Network error. Please try again.' };
  }
}

export async function verifyOTPForSignup(email, otp) {
  try {
    const res = await fetch('/auth/verify-otp-for-registration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, otp })
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

    if (!signupToken) {
      throw new Error('Email verification required before creating an account.');
    }

    // Store token so provisionUserFromFirebase can present it to the backend
    sessionStorage.setItem('otp_signup_token', signupToken);
    if (mobile) sessionStorage.setItem('pending_phone', mobile);

    // Check if Firebase account already exists for this email
    const methods = await fetchSignInMethodsForEmail(auth, email);
    if (methods.length > 0) {
      throw new Error('Email already in use. Please sign in instead.');
    }

    // OTP already proved email is reachable — create account and sign in directly
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    await updateProfile(user, {
      displayName: displayName,
      photoURL: `https://ui-avatars.com/api/?name=${encodeURIComponent(displayName)}&background=00f2ff&color=000`
    });

    // Create Firestore document immediately (email was verified via OTP)
    const userData = await ensureUserDocumentV2(user, 'email');

    const idToken = await user.getIdToken();
    AuthManager.setToken(idToken);
    AuthManager.setUser(userData);
    localStorage.setItem('authenticated', 'true');

    console.log('✅ Email signup successful:', email);
    return { success: true, user, message: 'Account created successfully!', userData };
  } catch (error) {
    console.error('❌ Email signup error:', error.code, error.message);
    return { success: false, message: error.message || 'Signup failed' };
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

    console.log('✅ Email login successful:', email);
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
    
    // Format phone number (ensure +country code)
    const formattedPhone = phoneNumber.startsWith('+') ? phoneNumber : '+91' + phoneNumber;
    
    console.log('📞 Sending OTP to:', formattedPhone);
    
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
    window.location.href = '/web/src/pages/index.html';
    return { success: true };
  } catch (error) {
    console.error('❌ Logout error:', error);
    return { success: false, message: error.message };
  }
}

// ============================================================
// PASSWORD RESET
// ============================================================
// Routed through backend so the email arrives from our trusted Neo domain
// (animeshkukreti@gatekeeper.sbs) instead of Firebase's noreply address.
export async function handlePasswordReset(email) {
  try {
    if (!email) throw new Error('Email required');
    const res = await fetch('/auth/send-password-reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const data = await res.json();
    if (!res.ok) return { success: false, message: data.detail || 'Failed to send reset email' };
    return { success: true, message: data.message };
  } catch (error) {
    console.error('❌ Password reset error:', error);
    return { success: false, message: 'Network error. Please try again.' };
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
