// ============================================================
// AEGIS Authentication Module – Mobile-First Design
// Separate Login/Signup flows with multiple auth methods
// ============================================================

import { 
  auth, 
  db, 
  googleProvider 
} from './gatekeeper.js';

import {
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  signInWithPhoneNumber,
  RecaptchaVerifier,
  updateProfile,
  fetchSignInMethodsForEmail,
  sendPasswordResetEmail,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";

import {
  doc,
  setDoc,
  getDoc,
  updateDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";

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
        endDate: serverTimestamp(), // Will be set by backend
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
    return userData;
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
    return existingData;
  }
}

// ============================================================
// GOOGLE LOGIN/SIGNUP
// ============================================================
export async function handleGoogleAuth() {
  try {
    console.log('🔐 Starting Google Auth...');
    const result = await signInWithPopup(auth, googleProvider);
    const user = result.user;
    
    // Create/update user document
    await ensureUserDocumentV2(user, 'google');
    
    // Store token
    const idToken = await user.getIdToken();
    localStorage.setItem('access_token', idToken);
    localStorage.setItem('authenticated', 'true');
    
    console.log('✅ Google Auth successful:', user.email);
    return { success: true, user, message: 'Logged in successfully!' };
  } catch (error) {
    console.error('❌ Google Auth error:', error.code, error.message);
    return { 
      success: false, 
      message: error.message || 'Google authentication failed'
    };
  }
}

// ============================================================
// EMAIL/PASSWORD SIGNUP
// ============================================================
export async function handleEmailSignup(email, password, displayName) {
  try {
    if (!email || !password || !displayName) {
      throw new Error('Please fill in all fields');
    }
    
    if (password.length < 8) {
      throw new Error('Password must be at least 8 characters');
    }
    
    console.log('📝 Creating email account...');
    
    // Check if email already exists
    const methods = await fetchSignInMethodsForEmail(auth, email);
    if (methods.length > 0) {
      throw new Error('Email already in use. Please try logging in instead.');
    }
    
    // Create user
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;
    
    // Update profile
    await updateProfile(user, {
      displayName: displayName,
      photoURL: `https://ui-avatars.com/api/?name=${encodeURIComponent(displayName)}&background=00f2ff&color=000`
    });
    
    // Create user document in Firestore
    await ensureUserDocumentV2(user, 'email');
    
    // Store token
    const idToken = await user.getIdToken();
    localStorage.setItem('access_token', idToken);
    localStorage.setItem('authenticated', 'true');
    
    console.log('✅ Email signup successful:', email);
    return { success: true, user, message: 'Account created successfully!' };
  } catch (error) {
    console.error('❌ Email signup error:', error.code, error.message);
    return { 
      success: false, 
      message: error.message || 'Signup failed'
    };
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
    
    // Update user document
    await ensureUserDocumentV2(user, 'email');
    
    // Store token
    const idToken = await user.getIdToken();
    localStorage.setItem('access_token', idToken);
    localStorage.setItem('authenticated', 'true');
    
    console.log('✅ Email login successful:', email);
    return { success: true, user, message: 'Logged in successfully!' };
  } catch (error) {
    console.error('❌ Email login error:', error.code, error.message);
    
    let userMessage = 'Login failed';
    if (error.code === 'auth/user-not-found') {
      userMessage = 'Email not found. Please sign up instead.';
    } else if (error.code === 'auth/wrong-password') {
      userMessage = 'Incorrect password. Please try again.';
    } else if (error.code === 'auth/invalid-email') {
      userMessage = 'Invalid email address.';
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
    localStorage.setItem('access_token', idToken);
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
    console.log('👋 Logging out...');
    await signOut(auth);
    localStorage.removeItem('access_token');
    localStorage.removeItem('authenticated');
    console.log('✅ Logout successful');
    return { success: true };
  } catch (error) {
    console.error('❌ Logout error:', error);
    return { success: false, message: error.message };
  }
}

// ============================================================
// PASSWORD RESET
// ============================================================
export async function handlePasswordReset(email) {
  try {
    if (!email) throw new Error('Email required');
    
    console.log('📧 Sending password reset email...');
    await sendPasswordResetEmail(auth, email);
    console.log('✅ Password reset email sent');
    return { 
      success: true, 
      message: 'Check your email for password reset link'
    };
  } catch (error) {
    console.error('❌ Password reset error:', error);
    return { 
      success: false, 
      message: error.message || 'Failed to send reset email'
    };
  }
}

// ============================================================
// AUTH STATE LISTENER
// ============================================================
export function subscribeToAuthState(callback) {
  return onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log('✅ User authenticated:', user.email);
      const userData = await getDoc(doc(db, 'users', user.uid));
      callback({
        authenticated: true,
        user,
        userData: userData.data()
      });
    } else {
      console.log('❌ User not authenticated');
      callback({
        authenticated: false,
        user: null,
        userData: null
      });
    }
  });
}

// ============================================================
// HELPER: Check if User is Authenticated
// ============================================================
export function isUserAuthenticated() {
  return localStorage.getItem('authenticated') === 'true' && auth.currentUser;
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
