// ============================================================
// Aegis‑1 Gatekeeper – Auth + Firestore (Streamlined)
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

// Firebase Configuration
const firebaseConfig = {
  apiKey: "AIzaSyDtudUL2sE1_fKbzIro5d2IP0-M2dYI6x4",
  authDomain: "aegis-d78e1.firebaseapp.com",
  projectId: "aegis-d78e1",
  storageBucket: "aegis-d78e1.firebasestorage.app",
  messagingSenderId: "623998601232",
  appId: "1:623998601232:web:288a89514d84ac3573a295",
  measurementId: "G-V6RWEEWT7L"
};

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

// ---- OAuth Providers (Google only exported, others kept for internal use) ----
export const googleProvider = new GoogleAuthProvider();

// Microsoft and Apple providers are kept for backend compatibility (not exported)
const microsoftProvider = new OAuthProvider('microsoft.com');
const appleProvider = new OAuthProvider('apple.com');

// ---- Helper: extract token from URL (for custom backend) ----
export function extractTokenFromHash() {
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const token = params.get('token');
  if (token) {
    localStorage.setItem('access_token', token);
    window.history.replaceState({}, document.title, window.location.pathname);
    return token;
  }
  return null;
}
extractTokenFromHash();

export function getJWTToken() {
  return localStorage.getItem('access_token');
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
  if (jwt) return jwt;
  return await getFirebaseIdToken();
}

// ---- User document management ----
export async function ensureUserDocument(user) {
  if (!user) return null;
  const userDocRef = doc(db, "users", user.uid);
  try {
    const docSnap = await getDoc(userDocRef);
    if (!docSnap.exists()) {
      const userData = {
        email: user.email,
        displayName: user.displayName || "",
        photoURL: user.photoURL || "",
        plan: "trial",
        capital: 10000,
        risk_pct: 2,
        join_date: serverTimestamp(),
        lastLogin: serverTimestamp()
      };
      await setDoc(userDocRef, userData);
      console.log("✅ New user document created for:", user.uid);
      return userData;
    } else {
      await updateDoc(userDocRef, { lastLogin: serverTimestamp() });
      return docSnap.data();
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
    await updateDoc(userDocRef, { [field]: value });
    console.log(`✅ Updated ${field} to ${value}`);
  } catch (error) {
    console.error(`updateUserSetting (${field}) error:`, error);
  }
}

// ---- Logout function ----
export async function logout() {
  try {
    await signOut(auth);
    localStorage.clear();
    console.log("✅ User signed out successfully");
    return { success: true };
  } catch (error) {
    console.error("Logout error:", error);
    return { success: false, error: error.message };
  }
}

// ---- Google Sign‑in (only exported social provider) ----
export async function signInWithGoogle() {
  try {
    const result = await signInWithPopup(auth, googleProvider);
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    console.error("Google login error:", error);
    return { success: false, error: error.message };
  }
}

// ---- Email/Password (Sign In & Register) ----
export async function signInWithEmail(email, password) {
  try {
    const result = await signInWithEmailAndPassword(auth, email, password);
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    let message = "Login failed. ";
    if (error.code === 'auth/wrong-password') message += "Incorrect password.";
    else if (error.code === 'auth/user-not-found') message += "No account found. Please register.";
    else message += error.message;
    return { success: false, error: message };
  }
}

export async function registerWithEmail(email, password, displayName = "") {
  try {
    const result = await createUserWithEmailAndPassword(auth, email, password);
    if (displayName) {
      await updateProfile(result.user, { displayName: displayName });
    }
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    let message = "Registration failed. ";
    if (error.code === 'auth/email-already-in-use') message += "Email already in use.";
    else message += error.message;
    return { success: false, error: message };
  }
}

// ---- Passwordless Email Link (Magic Link) ----
const actionCodeSettings = {
  url: window.location.href,
  handleCodeInApp: true
};

export async function sendMagicLink(email) {
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
    if (!email) {
      email = window.prompt('Please provide your email for confirmation');
    }
    try {
      const result = await signInWithEmailLink(auth, email, window.location.href);
      window.localStorage.removeItem('emailForSignIn');
      await ensureUserDocument(result.user);
      return { success: true, user: result.user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
  return null; // not a magic link sign-in
}

// ---- Phone Number (OTP) ----
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
    console.error("OTP send error:", error);
    let msg = "Failed to send OTP. ";
    if (error.code === 'auth/invalid-phone-number') msg += "Invalid phone number.";
    else msg += error.message;
    return { success: false, error: msg };
  }
}

export async function verifyPhoneOTP(code) {
  if (!confirmationResult) {
    return { success: false, error: "No OTP request active. Please request a new code." };
  }
  try {
    const result = await confirmationResult.confirm(code);
    await ensureUserDocument(result.user);
    return { success: true, user: result.user };
  } catch (error) {
    return { success: false, error: "Invalid or expired OTP code." };
  }
}

// ---- Helper: Check existing account (for linking) ----
export async function checkAccountExists(email) {
  try {
    const methods = await fetchSignInMethodsForEmail(auth, email);
    return { exists: methods.length > 0, methods };
  } catch (error) {
    return { exists: false, error: error.message };
  }
}

// ---- Plan & token visibility ----
export function validateAccess(userPlan, joinDate) {
  if (!joinDate) return { valid: false, plan: 'basic' };
  const now = new Date();
  const trialEnd = new Date(joinDate.getTime() + 72 * 60 * 60 * 1000);
  if (userPlan === 'trial' && now > trialEnd) {
    return { valid: false, plan: 'basic', expired: true };
  }
  return { valid: true, plan: userPlan, trialEnd };
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
export function isTokenVisible(symbol, userPlan, trialActive = true) {
  if (userPlan === 'pro') return true;
  if (userPlan === 'trial' && trialActive) return true;
  return BIG5.includes(symbol);
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