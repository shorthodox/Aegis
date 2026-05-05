// ============================================================
// Aegis‑1 Gatekeeper – Sovereign Onboarding (Triple‑Step)
// ============================================================
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-app.js";
import {
  getFirestore, doc, getDoc, setDoc, updateDoc,
  serverTimestamp, onSnapshot
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";
import {
  getAuth, onAuthStateChanged, signInWithPopup, signOut,
  GoogleAuthProvider, OAuthProvider, signInWithEmailAndPassword,
  createUserWithEmailAndPassword, sendSignInLinkToEmail, isSignInWithEmailLink,
  signInWithEmailLink, RecaptchaVerifier, signInWithPhoneNumber,
  fetchSignInMethodsForEmail, updateProfile
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";

// -------------------------------------------------------------------
// Firebase Configuration (with Realtime Database URL)
// -------------------------------------------------------------------
const firebaseConfig = {
  apiKey: "AIzaSyDtudUL2sE1_fKbzIro5d2IP0-M2dYI6x4",
  authDomain: "aegis-d78e1.firebaseapp.com",
  projectId: "aegis-d78e1",
  storageBucket: "aegis-d78e1.firebasestorage.app",
  messagingSenderId: "623998601232",
  appId: "1:623998601232:web:288a89514d84ac3573a295",
  measurementId: "G-V6RWEEWT7L",
  databaseURL: "https://aegis-d78e1-default-rtdb.asia-southeast1.firebasedatabase.app"
};

// -------------------------------------------------------------------
// API Base URL – dynamic, falls back to current origin
// -------------------------------------------------------------------
const API_BASE_URL = window.location.origin; // e.g., https://gatekeeper.sbs or http://localhost:8000

let firebaseApp;
if (!globalThis._firebaseApp) {
  firebaseApp = initializeApp(firebaseConfig);
  globalThis._firebaseApp = firebaseApp;
} else {
  firebaseApp = globalThis._firebaseApp;
}

export const auth = getAuth(firebaseApp);
export const db = getFirestore(firebaseApp);
export { onAuthStateChanged, signInWithPopup };

// -------------------------------------------------------------------
// OAuth Providers (Google only exported)
// -------------------------------------------------------------------
export const googleProvider = new GoogleAuthProvider();

// Microsoft and Apple providers kept for internal use (not exported)
const microsoftProvider = new OAuthProvider('microsoft.com');
const appleProvider = new OAuthProvider('apple.com');

// -------------------------------------------------------------------
// Helper: extract token from URL (for custom backend)
// -------------------------------------------------------------------
export function extractTokenFromHash() {
  try {
    const hash = window.location.hash.substring(1);
    const params = new URLSearchParams(hash);
    const token = params.get('token');
    if (token) {
      localStorage.setItem('access_token', token);
      window.history.replaceState({}, document.title, window.location.pathname);
      console.log("✅ JWT extracted from URL and stored");
      return token;
    }
    return null;
  } catch (error) {
    console.error("❌ extractTokenFromHash error:", error.name, error.message);
    return null;
  }
}
extractTokenFromHash();

export function getJWTToken() {
  return localStorage.getItem('access_token');
}

// Check JWT expiration (returns true if token is expired)
function isJWTExpired(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const exp = payload.exp * 1000; // convert to milliseconds
    return Date.now() >= exp;
  } catch (e) {
    console.warn("JWT parse error, assuming expired:", e);
    return true;
  }
}

export async function getFirebaseIdToken() {
  const user = auth.currentUser;
  if (!user) return null;
  try {
    return await user.getIdToken();
  } catch (error) {
    console.error("Error getting Firebase token:", error);
    return null;
  }
}

export async function getCurrentUserToken() {
  const jwt = getJWTToken();
  if (jwt && !isJWTExpired(jwt)) {
    return jwt;
  } else if (jwt) {
    console.log("Stored JWT expired, clearing and falling back to Firebase");
    localStorage.removeItem('access_token');
  }
  return await getFirebaseIdToken();
}

export async function fetchPaymentConfig() {
  try {
    const response = await fetch(`${API_BASE_URL}/payment/config`);
    if (!response.ok) return { cashfree: { enabled: false, environment: 'TEST' } };
    return await response.json();
  } catch (error) {
    console.error('fetchPaymentConfig error:', error);
    return { cashfree: { enabled: false, environment: 'TEST' } };
  }
}

// -------------------------------------------------------------------
// User document management (with setupComplete flag and type enforcement)
// -------------------------------------------------------------------
export async function ensureUserDocument(user) {
  if (!user) return null;
  const userDocRef = doc(db, "users", user.uid);
  try {
    const docSnap = await getDoc(userDocRef);
    if (!docSnap.exists()) {
      // Enforce correct types for numeric fields
      const userData = {
        email: user.email,
        displayName: user.displayName || "",
        photoURL: user.photoURL || "",
        plan: "trial",
        capital: 10000,          // number
        risk_pct: 2,             // number
        setupComplete: false,
        trial_end: serverTimestamp(),
        trial_expired: false,
        expiry_notice_sent: false,
        join_date: serverTimestamp(),
        lastLogin: serverTimestamp()
      };
      await setDoc(userDocRef, userData);
      console.log("✅ New user document created for:", user.uid);
      return userData;
    } else {
      // Update lastLogin
      await updateDoc(userDocRef, { lastLogin: serverTimestamp() });
      const data = docSnap.data();
      // Ensure numeric fields are numbers (defensive)
      if (typeof data.capital !== 'number') {
        await updateDoc(userDocRef, { capital: Number(data.capital) || 10000 });
      }
      if (typeof data.risk_pct !== 'number') {
        await updateDoc(userDocRef, { risk_pct: Number(data.risk_pct) || 2 });
      }
      return data;
    }
  } catch (error) {
    console.error("ensureUserDocument error:", error);
    return null;
  }
}

export function subscribeUserSettings(user, callback) {
  if (!user) return () => {};
  const userDocRef = doc(db, "users", user.uid);
  return onSnapshot(userDocRef,
    (docSnap) => {
      if (docSnap.exists()) callback(docSnap.data());
    },
    (error) => console.error("Settings listener error:", error)
  );
}

export async function updateUserSetting(user, field, value) {
  if (!user) return;
  const userDocRef = doc(db, "users", user.uid);
  try {
    // Type safety for capital and risk_pct
    let finalValue = value;
    if (field === 'capital' || field === 'risk_pct') {
      finalValue = Number(value);
      if (isNaN(finalValue)) finalValue = field === 'capital' ? 10000 : 2;
    }
    await updateDoc(userDocRef, { [field]: finalValue });
    console.log(`✅ Updated ${field} to ${finalValue}`);
  } catch (error) {
    console.error(`updateUserSetting (${field}) error:`, error);
  }
}

// -------------------------------------------------------------------
// Step 3: Onboarding – save profile and mark setupComplete
// -------------------------------------------------------------------
export async function updateUserOnboarding(uid, data) {
  if (!uid) return { success: false, error: "No user ID provided" };
  const userDocRef = doc(db, "users", uid);
  try {
    const updateData = {
      fullName: data.fullName || "",
      location: data.location || "",
      avatarUrl: data.avatarUrl || null,
      setupComplete: true
    };
    await updateDoc(userDocRef, updateData);
    console.log("✅ Onboarding data saved for:", uid);
    return { success: true };
  } catch (error) {
    console.error("updateUserOnboarding error:", error);
    return { success: false, error: error.message };
  }
}

// -------------------------------------------------------------------
// Logout – clears both Firebase session and local JWT
// -------------------------------------------------------------------
export async function logout() {
  try {
    await signOut(auth);
    localStorage.removeItem('access_token');
    console.log("✅ User signed out, JWT cleared");
    return { success: true };
  } catch (error) {
    console.error("Logout error:", error);
    return { success: false, error: error.message };
  }
}

// -------------------------------------------------------------------
// Social Login (only Google exported) with enhanced error handling
// -------------------------------------------------------------------
export async function signInWithGoogle() {
  try {
    const result = await signInWithPopup(auth, googleProvider);
    console.log("Google sign-in successful, user:", result.user.email);
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    let friendlyMsg = "Google login failed. ";
    switch (error.code) {
      case 'auth/popup-closed-by-user':
        friendlyMsg += "Popup closed before completing sign-in.";
        break;
      case 'auth/network-request-failed':
        friendlyMsg += "Network error. Check your connection.";
        break;
      default:
        friendlyMsg += error.message;
    }
    console.error("❌ Google login error:", error.code, error.message);
    return { success: false, error: friendlyMsg };
  }
}

// -------------------------------------------------------------------
// Email OTP (backend endpoints) – using dynamic BASE_URL
// -------------------------------------------------------------------
export async function sendEmailOTP(email) {
  try {
    const response = await fetch(`${API_BASE_URL}/send-otp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const data = await response.json();
    if (response.ok && data.success) {
      return { success: true, message: data.message };
    } else {
      return { success: false, error: data.detail || "Failed to send OTP" };
    }
  } catch (error) {
    console.error("sendEmailOTP error:", error);
    return { success: false, error: error.message };
  }
}

export async function verifyEmailOTP(email, otp) {
  try {
    const response = await fetch(`${API_BASE_URL}/verify-otp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, otp })
    });
    const data = await response.json();
    if (response.ok && data.success) {
      return { success: true, message: data.message };
    } else {
      return { success: false, error: data.detail || "Invalid OTP" };
    }
  } catch (error) {
    console.error("verifyEmailOTP error:", error);
    return { success: false, error: error.message };
  }
}

// -------------------------------------------------------------------
// Legacy methods (kept for backward compatibility, not exported)
// -------------------------------------------------------------------
export async function signInWithEmail(email, password) {
  try {
    console.log("🔑 Attempting email sign-in:", email);
    const result = await signInWithEmailAndPassword(auth, email, password);
    console.log("✅ Firebase auth sign-in successful:", result.user.uid);
    
    // Verify Firestore document exists
    console.log("📝 Verifying Firestore user document...");
    const userData = await ensureUserDocument(result.user);
    console.log("✅ Firestore document verified:", userData);
    
    return { success: true, user: result.user };
  } catch (error) {
    let message = "Login failed. ";
    console.error("❌ Sign-in error:", error.code, error.message);
    if (error.code === 'auth/wrong-password') message += "Incorrect password.";
    else if (error.code === 'auth/user-not-found') message += "No account found. Please register.";
    else message += error.message;
    return { success: false, error: message };
  }
}

export async function registerWithEmail(email, password, displayName = "") {
  try {
    console.log("📝 Attempting to register:", email);
    const result = await createUserWithEmailAndPassword(auth, email, password);
    console.log("✅ Firebase auth account created:", result.user.uid);
    
    if (displayName) {
      await updateProfile(result.user, { displayName: displayName });
      console.log("✅ Display name updated:", displayName);
    }
    
    // Ensure Firestore document is created
    console.log("📝 Creating Firestore user document...");
    const userData = await ensureUserDocument(result.user);
    console.log("✅ Firestore document created/verified:", userData);
    
    return { success: true, user: result.user };
  } catch (error) {
    let message = "Registration failed. ";
    console.error("❌ Registration error:", error.code, error.message);
    if (error.code === 'auth/email-already-in-use') message += "Email already in use.";
    else message += error.message;
    return { success: false, error: message };
  }
}

export async function sendMagicLink(email) {
  const actionCodeSettings = {
    url: window.location.href,
    handleCodeInApp: true
  };
  try {
    await sendSignInLinkToEmail(auth, email, actionCodeSettings);
    window.localStorage.setItem('emailForSignIn', email);
    return { success: true, message: "Magic link sent! Check your inbox." };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

export async function completeMagicLinkSignIn() {
  if (isSignInWithEmailLink(auth, window.location.href)) {
    let email = window.localStorage.getItem('emailForSignIn');
    if (!email) email = window.prompt('Please provide your email for confirmation');
    try {
      const result = await signInWithEmailLink(auth, email, window.location.href);
      window.localStorage.removeItem('emailForSignIn');
      await ensureUserDocument(result.user);
      return { success: true, user: result.user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
  return null;
}

let recaptchaVerifier = null;
let confirmationResult = null;
export function setupRecaptcha(containerId) {
  if (!recaptchaVerifier) {
    recaptchaVerifier = new RecaptchaVerifier(auth, containerId, {
      'size': 'invisible',
      'callback': () => console.log('reCAPTCHA resolved')
    });
  }
  return recaptchaVerifier;
}
export async function sendPhoneOTP(phoneNumber, recaptchaContainerId) {
  try {
    const verifier = setupRecaptcha(recaptchaContainerId);
    confirmationResult = await signInWithPhoneNumber(auth, phoneNumber, verifier);
    return { success: true, message: "OTP sent!" };
  } catch (error) {
    let msg = "Failed to send OTP. ";
    if (error.code === 'auth/invalid-phone-number') msg += "Invalid phone number.";
    else msg += error.message;
    return { success: false, error: msg };
  }
}
export async function verifyPhoneOTP(code) {
  if (!confirmationResult) return { success: false, error: "No OTP request active." };
  try {
    const result = await confirmationResult.confirm(code);
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    return { success: false, error: "Invalid or expired OTP code." };
  }
}

export async function checkAccountExists(email) {
  try {
    const methods = await fetchSignInMethodsForEmail(auth, email);
    return { exists: methods.length > 0, methods };
  } catch (error) {
    return { exists: false, error: error.message };
  }
}

// -------------------------------------------------------------------
// Plan & token visibility (dashboard helpers)
// -------------------------------------------------------------------
export function validateAccess(userPlan, trialEnd) {
  if (!trialEnd) return { valid: userPlan !== 'trial', plan: userPlan };
  const normalizedTrialEnd = trialEnd instanceof Date ? trialEnd : (trialEnd.toDate ? trialEnd.toDate() : new Date(trialEnd));
  const now = new Date();
  if (userPlan === 'trial' && now > normalizedTrialEnd) return { valid: false, plan: 'trial', expired: true, trialEnd: normalizedTrialEnd };
  return { valid: true, plan: userPlan, trialEnd: normalizedTrialEnd };
}

export function startTrialCountdown(trialEnd, displayElement) {
  if (!displayElement) return () => {};
  function updateTimer() {
    const now = new Date();
    const diff = trialEnd - now;
    if (diff <= 0) {
      displayElement.innerText = "Trial expired";
      return;
    }
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (3600000)) / (1000 * 60));
    displayElement.innerText = `Pro Trial: ${hours}h ${minutes}m left`;
  }
  updateTimer();
  const interval = setInterval(updateTimer, 60000);
  return () => clearInterval(interval);
}

const BIG5 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"];
const BASIC_TOKENS = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "DOGE/USDT"];
export function isTokenVisible(symbol, userPlan, trialActive = true) {
  // Pro users see everything
  if (userPlan === 'pro') return true;

  // Basic users have a limited toolkit
  if (userPlan === 'basic') return BASIC_TOKENS.includes(symbol);

  // Trial users within the 72h window see only the BIG5 sample tokens
  if (userPlan === 'trial' && trialActive) return BIG5.includes(symbol);

  // Non-subscribed or expired trial users see no signals
  return false;
}

export function getUpgradeModal() {
  let modal = document.getElementById('upgradeModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'upgradeModal';
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-card">
        <h3>🚀 Upgrade to Pro</h3>
        <p>See all 58 tokens, real‑time signals, trade simulation, and Alpha Mode.</p>
        <a href="index.html#pricing" class="btn-pro">View Plans</a>
        <button id="closeUpgradeModal">Close</button>
      </div>
    `;
    document.body.appendChild(modal);
    document.getElementById('closeUpgradeModal').onclick = () => modal.style.display = 'none';
  }
  return modal;
}
