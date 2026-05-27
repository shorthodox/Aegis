// ============================================================
// Aegis‑1 Gatekeeper – Sovereign Dashboard (Complete)
// ============================================================
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-app.js";
import {
  getFirestore, doc, getDoc, setDoc, updateDoc, collection, query, where, orderBy, limit, onSnapshot,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";
import {
  getAuth, onAuthStateChanged, signOut
} from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";

import { AuthManager } from '../auth/authManager.js';
import { loadThirdPartyScript } from './iframeGuard.js';

function _showPaymentLoader() {
  if (document.getElementById('aegis-pay-loader')) return;
  const el = document.createElement('div');
  el.id = 'aegis-pay-loader';
  el.innerHTML = `
    <style>
      #aegis-pay-loader {
        position: fixed; inset: 0; z-index: 99999;
        background: rgba(0,0,0,0.82); backdrop-filter: blur(6px);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 20px;
      }
      #aegis-pay-loader .apl-ring {
        width: 56px; height: 56px;
        border: 3px solid rgba(0,242,255,0.15);
        border-top-color: #00f2ff;
        border-radius: 50%;
        animation: apl-spin 0.75s linear infinite;
      }
      #aegis-pay-loader .apl-text {
        font-family: monospace; font-size: 13px;
        color: rgba(0,242,255,0.85); letter-spacing: 0.12em;
        text-transform: uppercase; font-weight: 700;
      }
      #aegis-pay-loader .apl-sub {
        font-family: monospace; font-size: 10px;
        color: rgba(255,255,255,0.3); letter-spacing: 0.08em;
      }
      @keyframes apl-spin { to { transform: rotate(360deg); } }
    </style>
    <div class="apl-ring"></div>
    <div class="apl-text">Connecting to Gateway</div>
    <div class="apl-sub">Please wait&hellip;</div>
  `;
  document.body.appendChild(el);
}

function _hidePaymentLoader() {
  const el = document.getElementById('aegis-pay-loader');
  if (el) el.remove();
}

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDtudUL2sE1_fKbzIro5d2IP0-M2dYI6x4",
  authDomain: "aegis-d78e1.firebaseapp.com",
  databaseURL: "https://aegis-d78e1-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "aegis-d78e1", // This is crucial for Firestore
  storageBucket: "aegis-d78e1.firebasestorage.app",
  messagingSenderId: "623998601232",
  appId: "1:623998601232:web:288a89514d84ac3573a295",
  measurementId: "G-V6RWEEWT7L"
};

// -------------------------------------------------------------------
// Initialize Firebase (Singleton pattern to prevent double initialization)
// -------------------------------------------------------------------
let firebaseApp;

// Check if Firebase is already initialized
if (!globalThis._firebaseApp) {
  try {
    firebaseApp = initializeApp(firebaseConfig);
    globalThis._firebaseApp = firebaseApp;
    console.log('✅ Firebase initialized successfully');
  } catch (error) {
    console.error('❌ Firebase initialization error:', error);
    throw error;
  }
} else {
  firebaseApp = globalThis._firebaseApp;
}

export const auth = getAuth(firebaseApp);
export const db = getFirestore(firebaseApp, "default");

// Global timeframe state
export let currentTimeframe = '1h';
window.activeTimeframe = currentTimeframe;

// -------------------------------------------------------------------
// API Base URL & Hash Token Extraction
// -------------------------------------------------------------------
const API_BASE_URL = window.location.origin;

// Extract token from OAuth redirect hash
if (window.location.hash.includes('token=')) {
  const tokenFragment = window.location.hash.split('token=')[1].split('&')[0];
  if (tokenFragment && typeof AuthManager !== 'undefined') {
    AuthManager.setToken(tokenFragment);
    window.location.hash = ''; // Clear hash for security
  }
}

// -------------------------------------------------------------------
// Global State
// -------------------------------------------------------------------
let currentUser = null;
let currentUserData = null;
let userPlan = 'trial';
let trialEnd = null;
let trialActive = true;

// Listen for trial expiration from dashboard countdown
document.addEventListener('trialExpired', () => {
  trialActive = false;
  allowedTokens = [];
  localStorage.setItem('cachedAllowedTokens', JSON.stringify([]));
  // Refresh UI to update plan badge immediately when trial expires
  updateUI();
});

// Update UI if the trial status asynchronously finishes loading/refreshing
window.addEventListener('trial-status-updated', () => {
  if (typeof AuthManager !== 'undefined') {
    trialActive = AuthManager.isTrialValid();
    if (!trialActive && typeof showSubscriptionExpiredOverlay === 'function') {
      showSubscriptionExpiredOverlay();
    }
    updateUI();
  }
});

// Re-verify subscription state when the user returns to this tab.
// Debounced to at most once per 60 seconds so it never hammers Firestore.
let _lastFocusVerifyAt = 0;
document.addEventListener('visibilitychange', async () => {
  if (document.hidden || !currentUser?.uid) return;
  const now = Date.now();
  if (now - _lastFocusVerifyAt < 60_000) return;
  _lastFocusVerifyAt = now;

  const token = typeof AuthManager !== 'undefined' ? AuthManager.getToken() : null;
  if (!token) return;

  // Lightweight integrity check: if trial_end_timestamp was tampered with,
  // verifyTrialEndIntegrity() returns false → force Firestore re-read.
  if (typeof AuthManager !== 'undefined' && AuthManager.verifyTrialEndIntegrity) {
    const intact = await AuthManager.verifyTrialEndIntegrity();
    if (intact === false) {
      // Signature mismatch — clear the tampered value and re-derive from Firestore.
      localStorage.removeItem('trial_end_timestamp');
      localStorage.removeItem('trial_end_sig');
    }
  }

  try {
    const userDocRef = doc(db, 'users', currentUser.uid);
    const snap = await getDoc(userDocRef);
    if (!snap.exists()) return;
    const data = snap.data();
    const plan = (data.plan || data.tier || '').toLowerCase();
    if (['pro', 'premium', 'intermediate', 'basic'].includes(plan)) {
      if (typeof window.clearExpiredView === 'function') window.clearExpiredView();
      return;
    }
    const rawEnd = data.trial_end || data.trialEnd || data.trial?.endDate;
    if (rawEnd) {
      const endDate = rawEnd.toDate ? rawEnd.toDate() : new Date(rawEnd);
      if (endDate < new Date() && typeof window.setExpiredView === 'function') {
        window.setExpiredView();
      }
    }
  } catch (_) {
    // Silently ignore — the WebSocket server-authority tick is still enforcing.
  }
});

let allowedTokens = [];
let ws = null;
let signalsUnsubscribe = null;
let tradesUnsubscribe = null;
let countdownInterval = null;
let currentAlphaMode = false;

// DOM Elements
let signalsContainer, balanceDisplay, capitalDisplay, riskDisplay;
let alphaToggleBtn, alphaStatus, upgradeBtn, logoutBtn, trialBanner, planBadge;
let capitalInput, riskInput, saveSettingsBtn;

// -------------------------------------------------------------------
// Token Lists
// -------------------------------------------------------------------
const BIG5_TOKENS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "AAVE/USDT"];
const PRO_TOKENS = []; // Will be populated from backend

let currentRiskProfile = 'balanced';

// -------------------------------------------------------------------
// Signal Status Determination
// -------------------------------------------------------------------
function getSignalStatus(signal) {
  /**
   * Determine if signal is Active, Expired, or Stopped Out based on current price.
   * - ACTIVE: Signal is still valid
   * - EXPIRED: Target price has been hit (success)
   * - STOPPED_OUT: Stop loss has been hit (failure)
   */
  if (!signal || !window.currentTickers) return 'ACTIVE';

  const currentPrice = window.currentTickers[signal.symbol];
  if (currentPrice === undefined) return 'ACTIVE';

  const tp = parseFloat(signal.tp) || 0;
  const sl = parseFloat(signal.sl) || 0;

  if (tp <= 0 || sl <= 0) return 'ACTIVE';

  // For LONG positions: expired if current price >= tp
  if (signal.direction === 'LONG') {
    if (currentPrice >= tp) return 'EXPIRED';
    if (currentPrice <= sl) return 'STOPPED_OUT';
  }
  // For SHORT positions: expired if current price <= tp
  else if (signal.direction === 'SHORT') {
    if (currentPrice <= tp) return 'EXPIRED';
    if (currentPrice >= sl) return 'STOPPED_OUT';
  }

  return 'ACTIVE';
}

// -------------------------------------------------------------------
// Subscription & Trial Locking
// -------------------------------------------------------------------
function showSubscriptionExpiredOverlay() {
  // Delegate to dashboard.js if available
  if (typeof window.setExpiredView === 'function') {
    window.setExpiredView();
    return;
  }

  const overlay = document.getElementById('subscriptionExpiredOverlay');
  const mainContent = document.getElementById('dashboard-main-content');

  if (overlay) overlay.classList.remove('hidden');
  if (mainContent) {
    mainContent.classList.add('hidden');
    mainContent.style.display = 'none';
  }

  // Disable all interactive elements
  document.querySelectorAll('button, input, select, .signal-card').forEach(el => {
    if (!el.closest('#subscriptionExpiredOverlay') && !el.closest('#access-expired-card') && !el.closest('#guardian-drawer')) {
      el.disabled = true;
      el.style.pointerEvents = 'none';
      el.style.opacity = '0.5';
    }
  });
}

// -------------------------------------------------------------------
// Signal Debouncing & Risk-Based Sorting
// -------------------------------------------------------------------

function sortSignalsByRisk(signals) {
  if (!Array.isArray(signals)) return signals;

  const riskWeights = {
    conservative: { ai_prob: 0.75, risk_pct: 0.015 },
    balanced: { ai_prob: 0.65, risk_pct: 0.025 },
    aggressive: { ai_prob: 0.55, risk_pct: 0.035 },
    sniper: { ai_prob: 0.45, risk_pct: 0.045 }
  };

  const weights = riskWeights[currentRiskProfile] || riskWeights.balanced;

  return signals.sort((a, b) => {
    const scoreA = (a.ai_prob || 0) * weights.ai_prob + (1 / (a.risk_pct || 0.02)) * weights.risk_pct;
    const scoreB = (b.ai_prob || 0) * weights.ai_prob + (1 / (b.risk_pct || 0.02)) * weights.risk_pct;
    return scoreB - scoreA; // Higher score first
  });
}

// -------------------------------------------------------------------
// Mobile Optimization
// -------------------------------------------------------------------
function setupMobileOptimizations() {
  // Mobile menu toggle is owned by the inline <script> in dashboard.html.
  // Handlers registered here caused triple-binding + stopImmediatePropagation
  // conflicts — deliberately left empty so only one owner fires.
}

// -------------------------------------------------------------------
// Initialize Dashboard
// -------------------------------------------------------------------

function initGatekeeper() {
  // Try to load saved timeframe preference
  const savedTf = localStorage.getItem('activeTimeframe');
  if (savedTf) {
    // Note: userPlan may not be fully loaded here yet, but we will validate later in applyUserData or updateDashboardData
    currentTimeframe = savedTf;
  }
  window.activeTimeframe = currentTimeframe;

  if (!document.getElementById('dashboard-main-content')) return;
  initializeElements();
  attachEventListeners();

  // Fast path: if a valid cached JWT exists, start loading immediately without
  // waiting for Firebase's cold-start (~500 ms–2 s). onAuthStateChanged still
  // fires in the background to silently refresh the token.
  let _fastPathFired = false;
  const _cachedToken = typeof AuthManager !== 'undefined' ? AuthManager.getToken() : null;
  if (_cachedToken && !isJWTExpired(_cachedToken)) {
    _fastPathFired = true;
    loadUserFromBackend(_cachedToken).catch(() => {});
  }

  onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log("Firebase user detected:", user.uid);
      const token = await user.getIdToken();
      if (typeof AuthManager !== 'undefined') AuthManager.setToken(token);
      if (!_fastPathFired) {
        await loadUserFromBackend(token, user);
      }
      // Fast path already running — just keep the stored token current so
      // WebSocket reconnects and API calls use the fresh Firebase token.
    } else {
      if (!_fastPathFired) {
        checkAuthAndLoad();
      }
    }
  });
  setupFooter();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initGatekeeper);
} else {
  initGatekeeper();
}

function initializeElements() {
  signalsContainer = document.getElementById('signalsContainer');
  balanceDisplay = document.getElementById('balanceDisplay');
  capitalDisplay = document.getElementById('capitalDisplay');
  riskDisplay = document.getElementById('riskDisplay');
  alphaToggleBtn = document.getElementById('alphaToggle');
  alphaStatus = document.getElementById('alphaStatus');
  upgradeBtn = document.getElementById('upgradeBtn');
  logoutBtn = document.getElementById('logoutBtn');
  trialBanner = document.getElementById('trialBanner');
  planBadge = document.getElementById('planBadge');
  capitalInput = document.getElementById('capitalInput');
  riskInput = document.getElementById('riskInput');
  saveSettingsBtn = document.getElementById('saveSettingsBtn');
}

function attachEventListeners() {
  // Alpha toggle is handled via modal — no direct listener on alphaToggleBtn
  if (upgradeBtn) upgradeBtn.addEventListener('click', () => {
    window.location.href = '/web/src/pages/pricing.html';
  });
  if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);
  if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveUserSettings);

  // Alpha Modal bindings
  const alphaToggleContainer = document.getElementById('alphaToggleContainer');
  const alphaModal = document.getElementById('alpha-modal');
  const alphaConfirm = document.getElementById('alpha-confirm');
  const alphaCancel = document.getElementById('alpha-cancel');

  if (alphaToggleContainer && alphaModal) {
    alphaToggleContainer.addEventListener('click', () => {
      if (userPlan !== 'pro') {
        showUpgradeModal();
        return;
      }
      alphaModal.classList.remove('hidden');
    });
  }
  if (alphaCancel && alphaModal) {
    alphaCancel.addEventListener('click', () => {
      alphaModal.classList.add('hidden');
    });
  }
  if (alphaConfirm && alphaModal) {
    alphaConfirm.addEventListener('click', () => {
      alphaModal.classList.add('hidden');
      toggleAlphaMode();
    });
  }

  // Timeframe listeners
  const tfBtns = document.querySelectorAll('.tf-btn');
  tfBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const tf = btn.getAttribute('data-tf');
      const requiresPro = btn.getAttribute('data-pro') === 'true';

      if (requiresPro && !['pro', 'intermediate'].includes(userPlan)) {
        showUpgradeModal();
        return;
      }

      currentTimeframe = tf;
      window.activeTimeframe = currentTimeframe;

      // Update UI active state
      tfBtns.forEach(b => {
        b.classList.remove('bg-cyan/20', 'text-cyan', 'font-bold');
        if (!b.classList.contains('cursor-not-allowed')) {
          b.classList.add('text-gray-400');
          b.classList.remove('text-white');
        }
      });

      btn.classList.add('bg-cyan/20', 'text-cyan', 'font-bold');
      btn.classList.remove('text-gray-400', 'text-gray-500');

      // Re-render signals from memory
      if (typeof window.latestSignals !== 'undefined' && Object.keys(window.latestSignals).length > 0) {
        debouncedFilterAndRenderSignals();
      }
    });
  });

  const strategySelect = document.getElementById('strategy-matchmaker');
  if (strategySelect) {
    strategySelect.addEventListener('change', (e) => {
      currentRiskProfile = e.target.value;
      if (typeof window.latestSignals !== 'undefined' && Object.keys(window.latestSignals).length > 0) {
        debouncedFilterAndRenderSignals();
      }
    });
  }

  // Setup mobile optimizations
  setupMobileOptimizations();
}

// ============================================================
// ── Live price DOM refresh ─────────────────────────────────────────────
// Reads window.currentTickers vs window.previousTickers and updates every
// .live-price element immediately — no setTimeout, no debounce.
// Flash classes (price-flash-up / price-flash-down) fire a 400ms CSS animation
// then are removed so each price change produces a visible blip.
function _refreshPriceElements() {
  const prev = window.previousTickers || {};
  const curr = window.currentTickers;
  if (!curr || typeof curr !== 'object') return;

  Object.entries(curr).forEach(([sym, price]) => {
    const currentPrice = parseFloat(price);
    if (isNaN(currentPrice)) return;

    const idStr = sym.replace('/', '-');
    const previousPrice = parseFloat(prev[sym] ?? currentPrice);
    const priceStr = currentPrice < 0.01 ? currentPrice.toFixed(6) : currentPrice.toFixed(4);

    window.dispatchEvent(new CustomEvent('priceUpdate', { detail: { symbol: sym, price: currentPrice } }));

    document.querySelectorAll(`.live-price[data-symbol="${idStr}"]`).forEach(el => {
      el.textContent = `$${priceStr}`;
      el.classList.remove('price-up', 'price-down', 'price-flash-up', 'price-flash-down');
      if (currentPrice > previousPrice) {
        el.classList.add('price-up', 'price-flash-up');
        setTimeout(() => el.classList.remove('price-flash-up'), 400);
      } else if (currentPrice < previousPrice) {
        el.classList.add('price-down', 'price-flash-down');
        setTimeout(() => el.classList.remove('price-flash-down'), 400);
      }
    });

    // Market overview card % change badge
    const changeSpan = document.getElementById(`market-card-change-${idStr}`);
    if (changeSpan && previousPrice !== currentPrice) {
      const pct = ((currentPrice - previousPrice) / previousPrice) * 100;
      changeSpan.textContent = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
      changeSpan.className = `text-[10px] font-mono font-semibold ${pct >= 0 ? 'text-green-400' : 'text-red-400'}`;
    }
  });
}

// Pulses the WS status dot so users can see data is flowing even when prices
// aren't changing. Clears itself after 200 ms.
let _pulseTimer = null;
function _pulseLiveIndicator() {
  document.querySelectorAll('#ws-status-dot, #ws-status-dot-mobile, #ws-status-dot-inner').forEach(d => {
    d.classList.add('data-received');
  });
  if (_pulseTimer) clearTimeout(_pulseTimer);
  _pulseTimer = setTimeout(() => {
    document.querySelectorAll('#ws-status-dot, #ws-status-dot-mobile, #ws-status-dot-inner').forEach(d => {
      d.classList.remove('data-received');
    });
  }, 200);
}

// Fast-path ticker update — called directly for type:"ticker" messages so they
// never enter the full updateDashboardData pipeline.
function applyTickerUpdates(tickers) {
  if (!tickers || typeof tickers !== 'object') return;
  window.previousTickers = { ...(window.currentTickers || {}) };
  window.currentTickers = { ...(window.currentTickers || {}), ...tickers };
  _refreshPriceElements();
  _pulseLiveIndicator();
}

// DEBOUNCE RENDER TO PREVENT EXCESSIVE DOM UPDATES
// ============================================================
let renderTimeout = null;
let lastRenderTime = 0;
const MIN_RENDER_INTERVAL = 100; // Minimum 100ms between renders

function debouncedFilterAndRenderSignals() {
  if (renderTimeout) clearTimeout(renderTimeout);

  const now = Date.now();
  const timeSinceLastRender = now - lastRenderTime;

  if (timeSinceLastRender < MIN_RENDER_INTERVAL) {
    // Schedule render after debounce period
    renderTimeout = setTimeout(() => {
      lastRenderTime = Date.now();
      filterAndRenderSignals();
    }, MIN_RENDER_INTERVAL - timeSinceLastRender);
  } else {
    // Render immediately
    lastRenderTime = now;
    filterAndRenderSignals();
  }
}

function filterAndRenderSignals() {
  const currentSignals = {};
  
  const strategySelect = document.getElementById('strategy-matchmaker');
  const currentStrategy = strategySelect ? strategySelect.value : '';

  // Validate timeframe for trial users
  if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
      if (!['15m', '30m', '1h'].includes(currentTimeframe)) {
          currentTimeframe = '1h';
          if (typeof window !== 'undefined') window.activeTimeframe = '1h';
      }
  }

  Object.values(window.latestSignals || {}).forEach(sig => {
    if (sig.timeframe === currentTimeframe) {
      // Debouncing should not hide signals from the UI
      // If UI flickering is an issue, we should debounce the render function itself, 
      // not drop the signals from the array.

      let isMatch = true;
      if (currentStrategy) {
        const acc = (sig.trading_accuracy || 0.5); // Provide fallback if no accuracy data
        if (currentStrategy === 'conservative') {
          if (acc <= 0.75) isMatch = false;
        } else if (currentStrategy === 'balanced') {
          if (acc <= 0.60 || acc > 0.75) isMatch = false;
        } else if (currentStrategy === 'aggressive') {
          if (acc < 0.50 || acc > 0.70) isMatch = false;
        } else if (currentStrategy === 'sniper') {
          if (acc <= 0.40) isMatch = false;
        }
      }
      if (isMatch) {
        currentSignals[sig.symbol] = sig;
      }
    }
  });

  // Sort signals by risk profile
  const sortedSignals = sortSignalsByRisk(Object.values(currentSignals));
  const sortedSignalsObj = {};
  sortedSignals.forEach(sig => {
    sortedSignalsObj[sig.symbol] = sig;
  });

  renderSignals(sortedSignalsObj);

  // Sync signal direction badges on market token cards
  if (typeof window.updateMarketCardSignalBadges === 'function') {
    window.updateMarketCardSignalBadges();
  }
}

// -------------------------------------------------------------------
// Authentication & User Data
// -------------------------------------------------------------------
async function checkAuthAndLoad() {
  const token = AuthManager.getToken();
  // bypassAuth is a dev-only escape hatch — disabled in production (non-localhost) environments.
  const _isDev = ['localhost', '127.0.0.1'].includes(window.location.hostname);
  const bypassAuth = _isDev && (
      localStorage.getItem('dashboardAuthBypass') === 'true' ||
      window.location.search.includes('bypassAuth=true')
  );

  console.log('gatekeeper.checkAuthAndLoad:', {
    tokenPresent: !!token,
    tokenMasked: token ? `${token.slice(0, 10)}...` : null,
    bypassAuth
  });

  if (!token) {
    if (bypassAuth) {
      console.warn('gatekeeper: no auth token found, bypassing redirect for local testing');
      return;
    }
    console.warn('gatekeeper: no auth token found, redirecting to login');
    redirectToLogin();
    return;
  }

  // Check if token is expired
  if (isJWTExpired(token)) {
    if (bypassAuth) {
      console.warn('gatekeeper: auth token expired, bypassing redirect for local testing');
      return;
    }
    console.warn('gatekeeper: auth token expired, clearing token and redirecting to login');
    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    redirectToLogin();
    return;
  }

  await loadUserFromBackend(token);

  // Check for trial expiration after loading user data
  if (userPlan === 'trial' && !trialActive) {
    showSubscriptionExpiredOverlay();
  }
}

function isJWTExpired(token) {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(base64));
    const exp = payload.exp * 1000;
    return Date.now() >= exp;
  } catch (e) {
    return true;
  }
}

async function loadUserFromBackend(token, firebaseUser = null) {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const userData = await response.json();
      applyUserData(userData, token);
    } else if (response.status === 404 && firebaseUser) {
      // New Firebase user — auto-provision a backend profile then continue
      await provisionUserFromFirebase(firebaseUser, token);
    } else if (response.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('authToken');
      redirectToLogin();
    } else if (response.status === 403) {
      // Account exists but was not OTP-verified (created before the verification gate).
      // Sign the Firebase user out so they can't re-enter the dashboard loop.
      try { await import("https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js").then(m => m.signOut(auth)); } catch (_) {}
      localStorage.clear();
      sessionStorage.clear();
      redirectToLogin();
    } else {
      console.error('Failed to load user data:', response.status);
      redirectToLogin();
    }
  } catch (error) {
    console.error('Load user error:', error);
    redirectToLogin();
  }
}

function applyUserData(userData, token) {
  currentUser = { email: userData.email, uid: userData.uid || auth.currentUser?.uid || userData.email, token };
  currentUserData = userData;
  userPlan = userData.plan || 'trial';
  trialEnd = userData.trial_end ? new Date(userData.trial_end) : null;

  if (typeof AuthManager !== 'undefined') {
    AuthManager.setUser(userData);
  }

  const isActive = userData.trial_active ?? true;
  trialActive = typeof AuthManager !== 'undefined' ? AuthManager.isTrialValid() : isActive;

  if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
    allowedTokens = BIG5_TOKENS;
    localStorage.setItem('cachedAllowedTokens', JSON.stringify(allowedTokens));
  }

  // Start WebSocket immediately — don't block on the /user/limits round-trip.
  // BIG5_TOKENS is already set above as a safe fallback for trial users.
  updateUI();
  startWebSocket(token);

  // Fetch limits and secondary data in the background; refresh UI when done.
  loadUserLimits().then(() => {
    updateUI();
    setupFirestoreListeners();
    loadGlobalPerformanceData();
    document.dispatchEvent(new CustomEvent('dashboardUserLoaded', { detail: { userData: currentUserData } }));
  });
}

async function provisionUserFromFirebase(firebaseUser, token, attempt = 1) {
  const MAX_ATTEMPTS = 3;

  // Detect provider — email/password accounts must present the OTP signup_token
  const provider = firebaseUser.providerData?.[0]?.providerId || 'firebase';
  const isPasswordUser = provider === 'password';
  const signupToken = sessionStorage.getItem('otp_signup_token');

  if (isPasswordUser && !signupToken) {
    // No OTP-verified token — block provisioning and send back to signup
    console.warn('[provision] Password user missing OTP signup_token — redirecting to login');
    redirectToLogin();
    return;
  }

  if (attempt === 1) showProvisioningState();

  try {
    const response = await fetch(`${API_BASE_URL}/api/users/provision`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        uid: firebaseUser.uid,
        email: firebaseUser.email || null,
        display_name: firebaseUser.displayName || null,
        provider,
        signup_token: isPasswordUser ? signupToken : null,
        phone_number: sessionStorage.getItem('pending_phone') || null
      })
    });

    if (response.ok) {
      sessionStorage.removeItem('otp_signup_token'); // single-use — clear after provisioning
      sessionStorage.removeItem('pending_phone');
      hideProvisioningState();
      const userData = await response.json();
      // Force-refresh the Firebase ID token so it picks up email_verified=true,
      // which the backend just set via Admin SDK. Without this the stale token
      // still carries email_verified=false and every subsequent API call returns 401.
      let freshToken = token;
      try {
        freshToken = await firebaseUser.getIdToken(true);
        if (typeof AuthManager !== 'undefined') AuthManager.setToken(freshToken);
      } catch (e) {
        console.warn('[provision] Token refresh failed, using original token:', e.message);
      }
      applyUserData(userData, freshToken);
    } else if (response.status === 409) {
      // Duplicate phone number — account suspended by backend
      hideProvisioningState();
      let detail = 'This phone number is already registered to another account.';
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      try {
        const { signOut: _signOut } = await import("https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js");
        await _signOut(auth);
      } catch (_) {}
      localStorage.clear();
      sessionStorage.clear();
      alert(detail);
      redirectToLogin();
    } else if (response.status === 403) {
      // OTP token rejected by backend — never retry, send to signup
      hideProvisioningState();
      redirectToLogin();
    } else if (attempt < MAX_ATTEMPTS) {
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return provisionUserFromFirebase(firebaseUser, token, attempt + 1);
    } else {
      hideProvisioningState();
      showProvisionErrorState();
    }
  } catch (error) {
    console.error('provisionUserFromFirebase error:', error);
    if (attempt < MAX_ATTEMPTS) {
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return provisionUserFromFirebase(firebaseUser, token, attempt + 1);
    }
    hideProvisioningState();
    showProvisionErrorState();
  }
}

function showProvisioningState() {
  if (document.getElementById('provision-overlay')) return;
  const overlay = document.createElement('div');
  overlay.id = 'provision-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.93);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;';
  overlay.innerHTML = `
    <i class="fas fa-circle-notch fa-spin" style="font-size:2.5rem;color:#22d3ee;"></i>
    <p style="color:#e2e8f0;font-size:1.1rem;font-weight:600;">Setting up your account&hellip;</p>
    <p style="color:#94a3b8;font-size:0.85rem;">This only happens once.</p>
  `;
  document.body.appendChild(overlay);
}

function hideProvisioningState() {
  document.getElementById('provision-overlay')?.remove();
}

function showProvisionErrorState() {
  const overlay = document.createElement('div');
  overlay.id = 'provision-error-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.95);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;';
  overlay.innerHTML = `
    <i class="fas fa-exclamation-triangle" style="font-size:2.5rem;color:#f97316;"></i>
    <h2 style="color:#f97316;font-size:1.3rem;font-weight:700;margin:0;">Account Setup Failed</h2>
    <p style="color:#94a3b8;font-size:0.9rem;text-align:center;max-width:320px;margin:0;">
      We couldn&apos;t create your account profile after 3 attempts. Please retry or contact support.
    </p>
    <div style="display:flex;gap:1rem;margin-top:0.5rem;">
      <button onclick="location.reload()" style="padding:10px 28px;background:linear-gradient(135deg,#f97316,#ef4444);border:none;color:white;border-radius:8px;font-size:1rem;font-weight:bold;cursor:pointer;">
        Retry
      </button>
      <button onclick="window.location.href='/web/src/pages/index.html'" style="padding:10px 28px;background:transparent;border:2px solid rgba(255,255,255,0.2);color:white;border-radius:8px;font-size:1rem;font-weight:bold;cursor:pointer;">
        Go Home
      </button>
    </div>
  `;
  document.body.appendChild(overlay);
}

async function loadUserLimits() {
  const token = AuthManager.getToken();
  if (!token) return;

  try {
    const response = await fetch(`${API_BASE_URL}/user/limits`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const limits = await response.json();
      // Backend is authoritative for allowed tokens for all plans (including trial)
      if (limits.allowed_tokens && limits.allowed_tokens.length > 0) {
        allowedTokens = limits.allowed_tokens;
        localStorage.setItem('cachedAllowedTokens', JSON.stringify(allowedTokens));
      }
      return limits;
    } else {
      const errorText = await response.text().catch(() => 'No response body');
      console.warn(`Backend failed to provide limits (Status: ${response.status} ${response.statusText}). Response: ${errorText}. Using fallback.`);
      allowedTokens = allowedTokens.length > 0 ? allowedTokens : BIG5_TOKENS;
    }
  } catch (error) {
    console.error('Load limits error:', error);
    allowedTokens = allowedTokens.length > 0 ? allowedTokens : BIG5_TOKENS;
  }
  return null;
}

function updateUI() {
  if (!trialActive && typeof signalsContainer !== 'undefined' && signalsContainer) {
    signalsContainer.innerHTML = '';
  }

  // Update plan badge — reflects server-authoritative plan
  if (planBadge) {
    planBadge.className = 'text-sm font-bold mt-1';
    const isTrialValid = typeof AuthManager !== 'undefined' ? AuthManager.isTrialValid() : trialActive;
    const p = (userPlan || 'trial').toLowerCase();

    if (p === 'pro' || p === 'premium') {
      planBadge.innerHTML = '<i class="fas fa-crown"></i> PRO';
      planBadge.classList.add('text-yellow-500');
    } else if (p === 'intermediate') {
      planBadge.innerHTML = '<i class="fas fa-bolt"></i> INTERMEDIATE';
      planBadge.classList.add('text-purple-400');
    } else if (p === 'basic') {
      planBadge.innerHTML = '<i class="fas fa-shield-alt"></i> BASIC';
      planBadge.classList.add('text-blue-400');
    } else if (isTrialValid && p === 'trial') {
      planBadge.innerHTML = '<i class="fas fa-flask"></i> TRIAL ACTIVE';
      planBadge.classList.add('text-cyan');
    } else {
      planBadge.innerHTML = '<i class="fas fa-clock"></i> TRIAL EXPIRED';
      planBadge.classList.add('text-red-500');
    }
  }

  // Update Aegis logo click handler
  const aegisLogoBtn = document.getElementById('aegis-logo-btn');
  if (aegisLogoBtn) {
    aegisLogoBtn.addEventListener('click', () => {
      window.location.href = '/web/src/pages/index.html';
    });
  }

  // Update return home button
  const returnHomeBtn = document.getElementById('returnHomeBtn');
  if (returnHomeBtn) {
    returnHomeBtn.addEventListener('click', () => {
      window.location.href = '/web/src/pages/index.html';
    });
  }

  // We leave trial banner manipulation to trial-countdown.js
  // Update alpha mode visibility (Pro only - or available for all if requested)
  if (alphaToggleBtn) {
    if (userPlan === 'trial' || userPlan === 'basic') {
      alphaToggleBtn.style.display = 'none';
      alphaToggleBtn.classList.add('feature-locked');
    } else {
      alphaToggleBtn.style.display = 'flex';
      alphaToggleBtn.classList.remove('feature-locked');
    }
  }

  // Unlock timeframe buttons for paid plans
  if (['pro', 'premium', 'intermediate'].includes(userPlan)) {
    document.querySelectorAll('.tf-btn[data-pro="true"]').forEach(btn => {
      const lockIcon = btn.querySelector('.fa-lock');
      if (lockIcon) lockIcon.remove();
      btn.disabled = false;
      btn.classList.remove('opacity-50', 'cursor-not-allowed', 'text-gray-500');
    });
  }
}

// -------------------------------------------------------------------
// WebSocket Connection
// -------------------------------------------------------------------
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const baseReconnectDelay = 1000; // 1 second
let heartbeatInterval = null;
let heartbeatTimeout = null;

function cleanupWebSocket() {
  if (ws) {
    ws.onopen = null;
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
    ws = null;
  }
  stopHeartbeat();
}

function startWebSocket(token) {
  // Check if features are blocked before connecting
  const isTrialValid = typeof AuthManager !== 'undefined' ? AuthManager.isTrialValid() : trialActive;
  const plan = (userPlan || 'trial').toLowerCase();
  const hasPaidPlan = ['pro', 'premium', 'intermediate', 'basic'].includes(plan);

  if (!hasPaidPlan && !isTrialValid) {
    console.warn('[WS] Features are blocked (trial expired). WebSocket connection aborted.');
    updateConnectionStatus('DISCONNECTED', 'red');
    return;
  }

  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error('[WS] Max WebSocket reconnection attempts reached');
    updateConnectionStatus('DISCONNECTED', 'red');
    return;
  }

  // State Management: Clean up existing instance before creating a new one
  cleanupWebSocket();
  
  // Expose disconnect function so dashboard.js can terminate connection on expiry
  window.disconnectGatekeeperWebSocket = cleanupWebSocket;

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/dashboard`;

  console.log(`[WS] Connecting to ${wsUrl}...`);
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] ✅ WebSocket connected');
    reconnectAttempts = 0; // Reset on successful connection
    updateConnectionStatus('CONNECTED', 'green');
    ws.send(JSON.stringify({ token, type: 'auth' }));

    // Start heartbeat
    startHeartbeat();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.type === 'pong') {
        resetHeartbeatTimeout();
        return;
      }

      // Any message resets the timeout because the connection is alive
      resetHeartbeatTimeout();

      if (data.type === 'error') {
        console.error('[WS] WebSocket error message:', data.message);
        if (data.message === 'Invalid token') {
          // Token was rejected by the server — attempt a silent Firebase refresh
          // before the connection closes so the next reconnect has a valid token.
          if (auth.currentUser) {
            auth.currentUser.getIdToken(true)
              .then(refreshed => {
                if (typeof AuthManager !== 'undefined') AuthManager.setToken(refreshed);
              })
              .catch(() => {});
          }
        }
        return;
      }

      // Fast path: ticker-only messages bypass the full updateDashboardData
      // pipeline — they go straight to DOM via applyTickerUpdates (no signal
      // card re-render, no debounce, no defer) for immediate price flashes.
      if (data.type === 'ticker' && data.tickers) {
        applyTickerUpdates(data.tickers);
        return;
      }

      // Full signal update (every ~500 ms) — runs the complete pipeline.
      updateDashboardData(data);
    } catch (e) {
      console.error('[WS] WebSocket parse error:', e);
    }
  };

  ws.onerror = (error) => {
    console.error('[WS] WebSocket error:', error);
    updateConnectionStatus('ERROR', 'red');
  };

  ws.onclose = (event) => {
    console.warn(`[WS] WebSocket disconnected (code: ${event.code}, reason: ${event.reason || 'None'}), reconnecting...`);
    ws = null;
    updateConnectionStatus('DISCONNECTED', 'red');
    
    // Stop heartbeat to avoid ghost pings
    stopHeartbeat();
    
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
    reconnectAttempts++;

    setTimeout(async () => {
      let freshToken = AuthManager.getToken() || token;

      // Firebase ID tokens expire after 1 hour. If the stored token is expired
      // and the user still has an active Firebase session, refresh it silently
      // so the backend doesn't reject the reconnection with "Invalid token".
      if (isJWTExpired(freshToken) && auth.currentUser) {
        try {
          freshToken = await auth.currentUser.getIdToken(true);
          AuthManager.setToken(freshToken);
        } catch (refreshErr) {
          console.warn('[WS] Token refresh failed on reconnect:', refreshErr.message || refreshErr);
          redirectToLogin();
          return;
        }
      } else if (isJWTExpired(freshToken)) {
        // Token expired and no Firebase session — send back to login
        console.warn('[WS] Token expired and no active Firebase session — redirecting to login');
        redirectToLogin();
        return;
      }

      startWebSocket(freshToken);
    }, delay);
  };
}

function startHeartbeat() {
  stopHeartbeat();
  heartbeatInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 2000);
  resetHeartbeatTimeout();
}

function resetHeartbeatTimeout() {
  if (heartbeatTimeout) clearTimeout(heartbeatTimeout);
  // If no message received for 10s, connection is dead — reconnect
  heartbeatTimeout = setTimeout(() => {
    console.warn('[WS] Heartbeat timeout. Reconnecting...');
    cleanupWebSocket();
    if (typeof AuthManager !== 'undefined') {
      const token = AuthManager.getToken();
      if (token) startWebSocket(token);
    }
  }, 10000);
}

function stopHeartbeat() {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
  if (heartbeatTimeout) {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = null;
  }
}

function updateConnectionStatus(status, color) {
  const statusDots = document.querySelectorAll('#ws-status-dot, #ws-status-dot-mobile, #ws-status-dot-inner');
  const statusTexts = document.querySelectorAll('#ws-status-text, #ws-status-text-inner');

  statusDots.forEach(dot => {
    dot.className = `status-dot text-${color}-500 bg-current`;
  });

  statusTexts.forEach(text => {
    text.textContent = status;
  });
}

// -------------------------------------------------------------------
// S&R Proximity Alert Toast
// -------------------------------------------------------------------
const _srAlertCooldowns = {};

function showSRAlertToast(alert) {
  const key = `${alert.symbol}_${alert.alert_state}`;
  const now = Date.now();
  // Suppress if same symbol+state fired within the last 60 s
  if (_srAlertCooldowns[key] && now - _srAlertCooldowns[key] < 60000) return;
  _srAlertCooldowns[key] = now;

  const container = document.getElementById('sr-alert-toast');
  if (!container) return;

  const isSupport = alert.alert_state === 'NEAR_SUPPORT';
  const color    = isSupport ? '#00ff88' : '#ff5252';
  const icon     = isSupport ? 'fa-level-down-alt' : 'fa-level-up-alt';
  const label    = isSupport ? 'NEAR SUPPORT' : 'NEAR RESISTANCE';
  const dist     = isSupport ? alert.dist_to_support_pct : alert.dist_to_resistance_pct;
  const line     = isSupport ? alert.support_line : alert.resistance_line;
  const lineStr  = line != null ? parseFloat(line).toFixed(4) : '—';
  const distStr  = dist != null ? `${parseFloat(dist).toFixed(2)}% away` : '';

  const toast = document.createElement('div');
  toast.className = 'sr-toast-item';
  toast.style.cssText = `border-left:3px solid ${color}`;
  toast.innerHTML = `
    <div class="sr-toast-row">
      <i class="fas ${icon}" style="color:${color}"></i>
      <strong>${alert.symbol}</strong>
      <span style="color:${color};font-weight:700;font-size:0.65rem;letter-spacing:1px">${label}</span>
    </div>
    <div class="sr-toast-detail">@ ${lineStr} &bull; ${distStr}</div>
  `;
  container.appendChild(toast);
  // Auto-dismiss after 8 s with fade-out
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 400); }, 7600);
}

function updateDashboardData(data) {
  // Save previous prices and pre-merge before any rendering so:
  //   1. Signal cards render with the latest prices from this message.
  //   2. _refreshPriceElements() can compare old vs new for direction arrows.
  window.previousTickers = { ...(window.currentTickers || {}) };
  if (data.tickers && Object.keys(data.tickers).length > 0) {
    window.currentTickers = { ...(window.currentTickers || {}), ...data.tickers };
  }

  // Update balance
  if (balanceDisplay && data.balance !== undefined) {
    balanceDisplay.textContent = `$${data.balance.toFixed(2)}`;
  }

  // Update signals with plan filtering FIRST
  if (signalsContainer && data.signals) {
    // Populate window.latestSignals from WebSocket
    Object.entries(data.signals).forEach(([sym, sig]) => {
      // For trial users, create signals for multiple timeframes (15m, 30m, 1h)
      const trialTimeframes = ['15m', '30m', '1h'];

      if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
        // For trial/basic users, create the same signal for all trial timeframes
        trialTimeframes.forEach(tf => {
          const key = `${sym}_${tf}`;
          window.latestSignals = window.latestSignals || {};
          const signalObj = {
            symbol: sym,
            signal: sig.signal || 'WAITING',
            ai_prob: sig.ai_prob || sig.confidence || 0,
            signal_strength: sig.signal_strength || 'NORMAL',
            risk_pct: sig.risk_pct || 2,
            atr: sig.atr || 0,
            timeframe: tf,
            direction: sig.direction || "NEUTRAL",
            entry_price: sig.entry_price || 0,
            sl: sig.sl || 0,
            tp: sig.tp || 0,
            confidence_score: sig.confidence_score || 0,
            signal_id: sig.signal_id || "",
            trading_accuracy: sig.trading_accuracy || 0.5,
            profitability_index: sig.profitability_index || 0,
            sr_telemetry: sig.sr_telemetry || null,
            confluence: sig.confluence || null,
            probabilities: sig.probabilities || {},
            shap_values: sig.shap_values || [],
            expectancy: sig.expectancy ?? null,
            max_dd: sig.max_dd ?? null,
            profit_factor: sig.profit_factor ?? null,
            win_rate: sig.win_rate ?? null,
            total_trades: sig.total_trades ?? 0,
          };
          // Determine and set signal status
          signalObj.status = getSignalStatus(signalObj);
          window.latestSignals[key] = signalObj;
        });
      } else {
        // For pro users, use the actual timeframe from data or default to 1h
        const tf = sig.timeframe || '1h';
        const key = `${sym}_${tf}`;
        window.latestSignals = window.latestSignals || {};
        const signalObj = {
          symbol: sym,
          signal: sig.signal || 'WAITING',
          ai_prob: sig.ai_prob || sig.confidence || 0,
          signal_strength: sig.signal_strength || 'NORMAL',
          risk_pct: sig.risk_pct || 2,
          atr: sig.atr || 0,
          timeframe: tf,
          direction: sig.direction || "NEUTRAL",
          entry_price: sig.entry_price || 0,
          sl: sig.sl || 0,
          tp: sig.tp || 0,
          confidence_score: sig.confidence_score || 0,
          signal_id: sig.signal_id || "",
          trading_accuracy: sig.trading_accuracy || 0.5,
          profitability_index: sig.profitability_index || 0,
          sr_telemetry: sig.sr_telemetry || null,
          macro_regime: sig.macro_regime || null,
          confluence: sig.confluence || null,
          probabilities: sig.probabilities || {},
          shap_values: sig.shap_values || [],
          expectancy: sig.expectancy ?? null,
          max_dd: sig.max_dd ?? null,
          profit_factor: sig.profit_factor ?? null,
          win_rate: sig.win_rate ?? null,
          total_trades: sig.total_trades ?? 0,
        };
        // Determine and set signal status
        signalObj.status = getSignalStatus(signalObj);
        window.latestSignals[key] = signalObj;
      }
      // Auto-save new BUY/SELL signals to history (deduped by signal_id inside addToSignalHistory)
      if (sig.signal_id && !['HOLD', 'WAITING', 'NEUTRAL'].includes(sig.signal || 'WAITING') &&
          typeof window.addToSignalHistory === 'function') {
        window.addToSignalHistory({ ...sig, symbol: sym });
      }
    });

    // Populate ALL timeframes from the full map the backend sends.
    // This is what makes tab-switching work: without it, pro signals are only
    // stored under the engine's native timeframe key and never match the user's tab.
    if (data.timeframes && ['pro', 'premium', 'intermediate'].includes(userPlan)) {
      window.latestSignals = window.latestSignals || {};
      Object.entries(data.timeframes).forEach(([sym, tfMap]) => {
        Object.entries(tfMap).forEach(([tf, sig]) => {
          if (!sig) return;
          const key = `${sym}_${tf}`;
          const tfSignalObj = {
            symbol: sym,
            signal: sig.signal || 'WAITING',
            ai_prob: sig.ai_prob || sig.confidence || 0,
            signal_strength: sig.signal_strength || 'NORMAL',
            risk_pct: sig.risk_pct || 2,
            atr: sig.atr || 0,
            timeframe: tf,
            direction: sig.direction || 'NEUTRAL',
            entry_price: sig.entry_price || 0,
            sl: sig.sl || 0,
            tp: sig.tp || 0,
            confidence_score: sig.confidence_score || 0,
            signal_id: sig.signal_id || '',
            trading_accuracy: sig.trading_accuracy || 0.5,
            profitability_index: sig.profitability_index || 0,
            sr_telemetry: sig.sr_telemetry || null,
            macro_regime: sig.macro_regime || null,
            confluence: sig.confluence || null,
            probabilities: sig.probabilities || {},
            shap_values: sig.shap_values || [],
            expectancy: sig.expectancy ?? null,
            max_dd: sig.max_dd ?? null,
            profit_factor: sig.profit_factor ?? null,
            win_rate: sig.win_rate ?? null,
            total_trades: sig.total_trades ?? 0,
          };
          tfSignalObj.status = getSignalStatus(tfSignalObj);
          window.latestSignals[key] = tfSignalObj;
        });
      });
    }

    debouncedFilterAndRenderSignals();

    // S&R proximity alert toasts
    if (data.sr_alerts && Array.isArray(data.sr_alerts)) {
      data.sr_alerts.forEach(alert => showSRAlertToast(alert));
    }
  }

  // Fallback: Extract prices from signals for any symbol not in data.tickers.
  // Always update (not just when missing) so prices refresh as signals recompute.
  if (window.latestSignals) {
    window.currentTickers = window.currentTickers || {};
    Object.values(window.latestSignals).forEach(sig => {
      if (sig && sig.symbol) {
        const p = sig.price || sig.entry_price;
        if (p && !(data.tickers && data.tickers[sig.symbol])) {
          window.currentTickers[sig.symbol] = p;
        }
      }
    });
  }

  // Apply live price DOM updates immediately — previousTickers was saved at
  // the top of this function so direction arrows and flash animations are correct.
  if (window.currentTickers && typeof window.currentTickers === 'object') {
    _refreshPriceElements();
    _pulseLiveIndicator();
  }

  // Update open trades — accept any of the keys the backend might use.
  // Filter to current user's positions only to prevent ghost trades from other
  // sessions or global backtest data leaking into the active UI.
  const rawTrades = data.open_trades ?? data.positions ?? data.fleet ?? null;
  if (rawTrades !== null) {
    const uid = currentUser?.uid || currentUser?.email || null;
    const incomingTrades = uid
      ? rawTrades.filter(t => !t.user_id || t.user_id === uid)
      : rawTrades;
    if (incomingTrades.length > 0) localStorage.setItem('lastKnownTrades', JSON.stringify(incomingTrades));
    if (typeof window.renderTrades === 'function') window.renderTrades(incomingTrades);
  }

  // Update alpha mode status
  if (data.alpha_mode !== undefined) {
    currentAlphaMode = data.alpha_mode;
    if (alphaStatus) {
      alphaStatus.textContent = currentAlphaMode ? 'ACTIVE' : 'STANDBY';
      alphaStatus.className = currentAlphaMode ? 'alpha-active' : 'alpha-standby';
    }
    if (alphaToggleBtn) {
      alphaToggleBtn.classList.toggle('active', currentAlphaMode);
    }
  }

  // Update warmup progress
  if (data.warmup) {
    const warmupEl = document.getElementById('warmupProgress');
    if (warmupEl) warmupEl.textContent = `Warmup: ${data.warmup}`;
  }

  // ── Server-authority enforcement ──────────────────────────────────
  // The server sends trial_expired on every WS tick (~1 s). If the
  // overlay was dismissed from the console, this re-applies it within
  // the next tick. Conversely, a valid subscription removes the overlay.
  if (typeof data.trial_expired === 'boolean') {
    const expiredCard = document.getElementById('subscriptionExpiredOverlay');
    const isOverlayVisible = expiredCard && !expiredCard.classList.contains('hidden');

    if (data.trial_expired && !isOverlayVisible) {
      if (typeof window.setExpiredView === 'function') window.setExpiredView();
    } 
    // WebSocket should not have the power to unlock features
    // else if (!data.trial_expired && isOverlayVisible) {
    //   if (typeof window.clearExpiredView === 'function') window.clearExpiredView();
    // }
  }
}


// -------------------------------------------------------------------
// Signal Rendering with Plan Filtering
// -------------------------------------------------------------------
function renderSignals(signals) {
  if (!signalsContainer) return;

  const signalEntries = Object.entries(signals);

  // Strict check using AuthManager
  const effectiveTrialActive = typeof AuthManager !== 'undefined' ? AuthManager.isTrialValid() : ((trialActive === null && userPlan === 'trial') ? true : trialActive);

  function getUserTier() {
    if (userPlan === 'pro') return 3;
    if (userPlan === 'intermediate') return 2;
    if (userPlan === 'basic') return 1;
    return 1; // Trial or none
  }


  // Filter signals based on plan tier
  const filteredEntries = signalEntries.filter(([key, signal]) => {
    const symbol = signal.symbol;
    // PRO and INTERMEDIATE subscribers see all incoming tokens — no filter
    if (userPlan === 'pro' || userPlan === 'premium' || userPlan === 'intermediate') return true;
    // Trial/basic: restrict to allowedTokens (BIG5 by default)
    return allowedTokens.includes(symbol);
  });

  if (filteredEntries.length === 0) {
    const isExplicitlyExpired = userPlan === 'expired' || userPlan === 'none' || window.trialExpiredTriggered === true;
    const isTrialPlan = userPlan === 'trial' || userPlan === 'trial-active' || effectiveTrialActive;

    if (!isExplicitlyExpired && isTrialPlan) {
      signalsContainer.innerHTML = `
        <div class="no-signals">
          <i class="fas fa-spinner fa-spin"></i>
          <p>Preparing your trial token cards...</p>
          <div style="font-size:10px; margin-top:20px; text-align:left; color:#888; max-height:150px; overflow-y:auto; word-break:break-all;">
            <p>DEBUG INFO:</p>
            <p>userPlan: ${userPlan}</p>
            <p>allowedTokens: ${JSON.stringify(allowedTokens)}</p>
            <p>currentTimeframe: ${currentTimeframe}</p>
            <p>latestSignals count: ${Object.keys(window.latestSignals || {}).length}</p>
            <p>First 5 keys: ${Object.keys(window.latestSignals || {}).slice(0,5).join(', ')}</p>
          </div>
        </div>
      `;
      return;
    }

    signalsContainer.innerHTML = `
      <div class="no-signals">
        <i class="fas fa-chart-line"></i>
        <p>No signals available for your plan</p>
        ${userPlan !== 'pro' ? '<a href="/web/src/pages/pricing.html" class="upgrade-link">Upgrade to Pro for 58 tokens →</a>' : ''}
      </div>
    `;
    return;
  }

  // Populate simulation select dropdown
  const simSelect = document.getElementById('sim-symbol');
  if (simSelect) {
    const currentSelection = simSelect.value;
    simSelect.innerHTML = '<option value="">Select a signal...</option>' +
      filteredEntries.map(([key, signal]) => `<option value="${signal.symbol}">${signal.symbol}</option>`).join('');
    if (filteredEntries.some(([key, signal]) => signal.symbol === currentSelection)) {
      simSelect.value = currentSelection;
    }
  }

  const strategySelect = document.getElementById('strategy-matchmaker');
  const isStrategyActive = strategySelect && strategySelect.value !== '';

  signalsContainer.innerHTML = filteredEntries.map(([key, signal]) => {
    const symbol = signal.symbol;
    const signalType = signal.signal || 'WAITING';
    const timeframe = signal.timeframe || '1h'; // Default to 1h if not provided
    const signalStatus = signal.status || getSignalStatus(signal);
    const signalClass = getSignalClass(signalType, signalStatus);
    const cardTypeClass = getSignalCardType(signal.direction);
    const confidence = (signal.ai_prob || 0) * 100;

    // Determine status badge
    let statusBadge = '';
    let statusIndicator = '';
    if (signalStatus === 'EXPIRED') {
      statusBadge = '<span class="bg-green-500/20 text-green-400 border border-green-500/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider">✓ TARGET HIT</span>';
      statusIndicator = ' opacity-60';
    } else if (signalStatus === 'STOPPED_OUT') {
      statusBadge = '<span class="bg-red-500/20 text-red-400 border border-red-500/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider">✗ STOPPED OUT</span>';
      statusIndicator = ' opacity-50';
    }

    let directionBadge = '';
    if (signal.direction === 'LONG') {
      directionBadge = '<span class="bg-green-500/20 text-green-400 border border-green-500/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider">LONG</span>';
    } else if (signal.direction === 'SHORT') {
      directionBadge = '<span class="bg-red-500/20 text-red-400 border border-red-500/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider">SHORT</span>';
    }

    const slStr = signal.sl ? signal.sl.toFixed(4) : '-';
    const tpStr = signal.tp ? signal.tp.toFixed(4) : '-';
    const entryStr = signal.entry_price ? signal.entry_price.toFixed(4) : '-';
    const profIndex = (signal.profitability_index || 0).toFixed(2);

    const confluence = signal.confluence || { trend: 50, momentum: 50, volume: 50 };

    let matchClasses = '';
    let matchBadge = '';
    if (isStrategyActive) {
      matchClasses = 'border-cyan shadow-[0_0_15px_rgba(0,242,255,0.4)]';
      matchBadge = '<span class="bg-cyan/20 text-cyan border border-cyan/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider animate-pulse">ALPHA MATCH</span>';
    }

    // S&R proximity inline badge
    const _srT = signal.sr_telemetry;
    const _macro = signal.macro_regime || {};
    const srBadge = _srT && _srT.alert_state && _srT.alert_state !== 'NONE' ? (() => {
      const _sup = _srT.alert_state === 'NEAR_SUPPORT';
      const _badgeStyle = _sup
        ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300 animate-pulse'
        : 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse';
      const _icon = _sup ? 'fa-level-down-alt' : 'fa-level-up-alt';
      const _label = _sup ? 'NEAR SUPPORT' : 'NEAR RESISTANCE';
      const _dist = _sup ? _srT.dist_to_support_pct : _srT.dist_to_resistance_pct;
      const _line = _sup ? _srT.support_line : _srT.resistance_line;
      const _distanceText = _dist != null ? `${parseFloat(_dist).toFixed(2)}% away` : 'distance unknown';
      const _lineText = _line != null ? parseFloat(_line).toFixed(4) : '---';
      return `<div class="flex items-center justify-between rounded-full border px-3 py-1.5 text-[10px] font-mono ${_badgeStyle}">
        <span class="flex items-center gap-2 font-semibold uppercase tracking-[0.08em] text-[10px]"><i class="fas ${_icon}"></i>${_label}</span>
        <span class="text-slate-300">${_lineText} · ${_distanceText}</span>
      </div>`;
    })() : (_srT ? (() => {
      const supportLine = _srT.support_line != null ? parseFloat(_srT.support_line).toFixed(4) : '---';
      const resistanceLine = _srT.resistance_line != null ? parseFloat(_srT.resistance_line).toFixed(4) : '---';
      const supportDist = _srT.dist_to_support_pct != null ? `${parseFloat(_srT.dist_to_support_pct).toFixed(2)}%` : 'N/A';
      const resistanceDist = _srT.dist_to_resistance_pct != null ? `${parseFloat(_srT.dist_to_resistance_pct).toFixed(2)}%` : 'N/A';
      return `<div class="rounded-xl border border-white/10 bg-slate-950/70 px-3 py-1.5 text-[10px] font-mono text-slate-400">
        <div class="flex items-center justify-between gap-3">
          <span class="font-semibold">S: ${supportDist}</span>
          <span class="font-semibold">R: ${resistanceDist}</span>
        </div>
        <div class="mt-0.5 text-[9px] text-slate-500">S ${supportLine} · R ${resistanceLine}</div>
      </div>`;
    })() : '');
    const macroBadge = (_macro && (_macro.confluence_score !== undefined || _macro.trend_1d !== undefined)) ? (() => {
      const trendLabel = _macro.trend_1d === 1 ? 'BULLISH 1D' : _macro.trend_1d === -1 ? 'BEARISH 1D' : 'NEUTRAL 1D';
      const score = typeof _macro.confluence_score === 'number' && _macro.confluence_score > 0 ? _macro.confluence_score.toFixed(0) : '—';
      return `<div class="flex items-center justify-between px-1.5 py-1 rounded text-[10px] font-mono bg-slate-900/80 border border-slate-700 text-slate-200">
        <span style="font-weight:700">${trendLabel}</span>
        <span style="color:#7dd3fc">Confluence ${score}</span>
      </div>`;
    })() : '';

    return `
      <div class="signal-card ${cardTypeClass}${statusIndicator} cursor-pointer hover:shadow-[0_0_15px_rgba(0,242,255,0.2)] transition-all transform hover:-translate-y-1 overflow-hidden ${matchClasses}" onclick="window.openSignalDetails('${symbol}', '${timeframe}')" data-symbol="${symbol}" data-status="${signalStatus}">
        <div class="signal-header flex justify-between items-center">
          <div class="flex items-center">
            <span class="signal-symbol font-bold">${symbol}</span>
            <span class="signal-timeframe text-xs text-gray-500 ml-2">${timeframe}</span>
            ${directionBadge}
            ${statusBadge}
            ${matchBadge}
          </div>
          <div class="flex items-center gap-2">
            <button class="view-logic-btn text-[10px] bg-white/5 border border-white/10 px-2 py-0.5 rounded text-gray-400 hover:text-white transition-colors" onclick="window.toggleScorecard(event, '${symbol}')">View Logic <i class="fas fa-chevron-down"></i></button>
            <div class="price-container text-sm flex font-mono ml-1"><span class="live-price" data-symbol="${symbol.replace('/', '-')}">${window.currentTickers && window.currentTickers[symbol] ? '$' + parseFloat(window.currentTickers[symbol]).toFixed(4) : '-'}</span></div>
            <span class="signal-badge ${signalClass}">${signalType}</span>
          </div>
        </div>
        <div class="signal-details mt-3">
          <div class="signal-confidence flex justify-between items-center mb-2">
            <div class="confidence-bar flex-1 h-1.5 bg-black/50 rounded overflow-hidden mr-3">
              <div class="confidence-fill h-full bg-current" style="width: ${confidence}%"></div>
            </div>
            <span class="text-xs font-mono">AI: ${confidence.toFixed(1)}%</span>
          </div>
          <div class="signal-meta grid grid-cols-2 gap-2 mt-3 text-xs text-gray-400">
            <span class="signal-strength ${signal.signal_strength?.toLowerCase()}">
              <i class="fas fa-bolt"></i> ${signal.signal_strength || 'NORMAL'}
            </span>
            <span class="signal-risk flex justify-between items-center">
              <span><i class="fas fa-shield-alt"></i> Risk: ${signal.risk_pct || 2}%</span>
              <span class="text-orange font-bold px-1 rounded bg-orange/10 border border-orange/20" title="Profitability Index">PI: ${profIndex}</span>
            </span>
            <span class="col-span-2 text-cyan font-mono bg-black/30 p-1.5 rounded flex justify-between">
               <span>Entry: ${entryStr}</span>
               <span>SL: ${slStr} | TP: ${tpStr}</span>
            </span>
          </div>
          ${(srBadge || macroBadge) ? `<div class="mt-2 pt-1.5 border-t border-white/5 space-y-1">${macroBadge}${srBadge}</div>` : ''}
          <div class="slide-down-container mt-2 ${window.openScorecards && window.openScorecards.has(symbol) ? 'open' : ''}">
            <div class="slide-down-content ${ (userPlan === 'trial' || userPlan === 'basic') ? 'feature-locked relative' : '' }" id="scorecard-${symbol.replace('/', '-')}">
              ${ (userPlan === 'trial' || userPlan === 'basic') ? `
              <div class="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md rounded border border-white/10">
                <i class="fas fa-lock text-white/50 text-xl mb-2"></i>
                <div class="text-[10px] text-gray-400 mb-2">Logic Locked</div>
                <button class="upgrade-btn text-[10px] bg-cyan/20 text-cyan px-2 py-1 rounded border border-cyan/30 hover:bg-cyan/30 transition-colors" onclick="window.location.href='/web/src/pages/pricing.html'; event.stopPropagation();">Upgrade to Pro</button>
              </div>
              ` : '' }
              <div class="pt-2 border-t border-white/10 mt-2 ${ (userPlan === 'trial' || userPlan === 'basic') ? 'opacity-30 blur-sm pointer-events-none' : '' }">
                <div class="text-[10px] font-bold text-gray-400 uppercase mb-2">AI Reasoning: XGBoost Confluence</div>
                <div class="mb-2">
                  <div class="flex justify-between text-[10px]"><span class="text-gray-400">Trend Alignment (EMA 50/200)</span><span class="text-cyan font-mono">${confluence.trend}%</span></div>
                  <div class="confluence-bar"><div class="fill" style="width: ${confluence.trend}%;"></div></div>
                </div>
                <div class="mb-2">
                  <div class="flex justify-between text-[10px]"><span class="text-gray-400">Momentum (RSI Regime)</span><span class="text-cyan font-mono">${confluence.momentum}%</span></div>
                  <div class="confluence-bar"><div class="fill" style="width: ${confluence.momentum}%;"></div></div>
                </div>
                <div class="mb-2">
                  <div class="flex justify-between text-[10px]"><span class="text-gray-400">Volume Delta</span><span class="text-cyan font-mono">${confluence.volume}%</span></div>
                  <div class="confluence-bar"><div class="fill" style="width: ${confluence.volume}%;"></div></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

window.toggleScorecard = function (event, symbol) {
  event.stopPropagation(); // prevent card click
  const containerId = `scorecard-${symbol.replace('/', '-')}`;
  const content = document.getElementById(containerId);
  if (!content) return;

  const wrapper = content.parentElement;
  
  if (!window.openScorecards) window.openScorecards = new Set();
  const isOpen = wrapper.classList.contains('open');

  // Calculate tier
  let userTier = 0;
  if (userPlan === 'pro') userTier = 3;
  else if (userPlan === 'intermediate') userTier = 2;
  else if (userPlan === 'basic') userTier = 1;

  if (userTier < 2) {
    content.classList.add('feature-locked', 'locked');

    // Toggle opening with lock
    wrapper.classList.toggle('open');
    if (!isOpen) {
      window.openScorecards.add(symbol);
    } else {
      window.openScorecards.delete(symbol);
    }

    // If they click on the lock overlay (which is placed via pseudo element pointer-events:auto),
    // trigger upgrade modal.
    content.onclick = (e) => {
      e.stopPropagation();
      if (typeof showUpgradeModal === 'function') showUpgradeModal();
    };
  } else {
    content.classList.remove('premium-lock-blur', 'locked');
    wrapper.classList.toggle('open');
    if (!isOpen) {
      window.openScorecards.add(symbol);
    } else {
      window.openScorecards.delete(symbol);
    }
  }
}

window.openSignalDetails = function (symbol, timeframe) {
  const key = `${symbol}_${timeframe}`;
  const sig = window.latestSignals && window.latestSignals[key];
  if (!sig) return;
  
  if (typeof window.showSignalDetailsModal === 'function') {
    window.showSignalDetailsModal(sig);
  }
}

window.selectSignal = function (symbol, timeframe) {
  const key = `${symbol}_${timeframe}`;
  const sig = window.latestSignals && window.latestSignals[key];
  if (!sig) return;

  // Delegate signal history and terminal prefill to trading-rooms.js
  if (typeof window.addToSignalHistory === 'function') window.addToSignalHistory(sig);
  if (typeof window.prefillFromSignal === 'function') window.prefillFromSignal(sig);

  // Switch to the terminal tab
  if (typeof window.switchRoom === 'function') {
    window.switchRoom('terminal');
  }

  // Dispatch event for app.js to catch
  document.dispatchEvent(new CustomEvent('signalRowClicked', {
    detail: {
      pair: symbol,
      entry: sig.entry_price || 0,
      signal_id: sig.signal_id,
      ...sig
    }
  }));
}


function getSignalClass(signal, status) {
  const s = String(signal).toUpperCase();

  // If signal is expired or stopped out, use neutral/expired styling
  if (status === 'EXPIRED' || status === 'STOPPED_OUT') {
    return 'expired';
  }

  if (s.includes('BUY') || s.includes('LONG')) return 'buy';
  if (s.includes('SELL') || s.includes('SHORT')) return 'sell';
  return 'neutral';
}

function getSignalCardType(direction) {
  if (direction === 'LONG') return 'bullish';
  if (direction === 'SHORT') return 'bearish';
  return 'hold';
}


// -------------------------------------------------------------------
// Global Performance Data (analytics/global_performance)
// -------------------------------------------------------------------
async function loadGlobalPerformanceData() {
  try {
    const perfRef = doc(db, 'analytics', 'global_performance');
    const perfSnap = await getDoc(perfRef);

    const FALLBACKS = { expectancy: '0.00%', maxdd: '-0.00%', profitFactor: '1.00', winRate: '0%', totalTrades: '0' };

    const d = perfSnap.exists() ? perfSnap.data() : {};

    const expectancy   = d.mathematical_expectancy ?? d.expectancy   ?? FALLBACKS.expectancy;
    const maxdd        = d.max_drawdown            ?? d.maxDrawdown   ?? FALLBACKS.maxdd;
    const profitFactor = d.profit_factor           ?? d.profitFactor  ?? FALLBACKS.profitFactor;
    const winRate      = d.win_rate != null
      ? `${(parseFloat(d.win_rate) * (d.win_rate <= 1 ? 100 : 1)).toFixed(1)}%`
      : FALLBACKS.winRate;
    const totalTrades  = d.total_trades            ?? d.totalTrades   ?? FALLBACKS.totalTrades;

    const safe = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    safe('analytics-expectancy',    String(expectancy));
    safe('analytics-maxdd',         String(maxdd));
    safe('analytics-profit-factor', String(profitFactor));
    safe('analytics-win-rate',      String(winRate));
    safe('analytics-total-trades',  String(totalTrades));
  } catch (err) {
    console.warn('[loadGlobalPerformanceData] Firestore error — showing fallbacks:', err);
    ['analytics-expectancy', 'analytics-maxdd', 'analytics-profit-factor',
     'analytics-win-rate', 'analytics-total-trades'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '—';
    });
  }
}

// -------------------------------------------------------------------
// Firestore Real-time Listeners
// -------------------------------------------------------------------
function setupFirestoreListeners() {
  if (signalsUnsubscribe) { signalsUnsubscribe(); signalsUnsubscribe = null; }
  if (tradesUnsubscribe) { tradesUnsubscribe(); tradesUnsubscribe = null; }

  const token = AuthManager.getToken();
  if (!token) return;

  // Listen to signals collection
  const signalsQuery = query(collection(db, 'signals'), orderBy('timestamp', 'desc'), limit(150));

  window.latestSignals = {}; // Accumulate signals to prevent flickering
  signalsUnsubscribe = onSnapshot(signalsQuery, (snapshot) => {
    snapshot.docChanges().forEach(change => {
      if (change.type === 'added' || change.type === 'modified') {
        const data = change.doc.data();
        let symbol = data.symbol || change.doc.id;
        if (symbol && symbol.includes('_') && !symbol.includes('/')) {
            symbol = symbol.replace('_', '/');
        }
        const tf = data.timeframe || '1h';
        const key = `${symbol}_${tf}`;

        // Apply plan filtering
        if (!['pro', 'premium', 'intermediate'].includes(userPlan) && !allowedTokens.includes(symbol)) {
          return;
        }

        const _fsConf = data.confluence_scorecards || {};
        const _fsEm = data.expectancy_matrix || {};
        const _existingSignal = window.latestSignals && window.latestSignals[key];
        const _computedConfluence = data.confluence || (_fsConf.trend ? {
          trend: _fsConf.trend === 'Aligned' ? 80 : 35,
          momentum: Math.min(100, Math.max(0, Math.round((_fsConf.efficiency || 0.5) * 100))),
          volume: _fsConf.volume === 'high' ? 78 : (_fsConf.volume === 'normal' ? 58 : 38),
        } : null);
        const signalObj = {
          symbol: symbol,
          signal: data.signal || 'WAITING',
          ai_prob: data.ai_prob || data.confidence || 0,
          signal_strength: data.signal_strength || 'NORMAL',
          risk_pct: data.risk_pct || 2,
          atr: data.atr || 0,
          direction: data.direction || "NEUTRAL",
          entry_price: data.entry_price || 0,
          sl: data.sl || 0,
          tp: data.tp || 0,
          confidence_score: (data.ai_prob || data.confidence || 0) * 100,
          signal_id: data.signal_id || "",
          trading_accuracy: data.trading_accuracy || 0.5,
          profitability_index: data.profitability_index || 0,
          sr_telemetry: data.sr_telemetry || null,
          macro_regime: data.macro_regime || null,
          confluence: _computedConfluence || (_existingSignal ? _existingSignal.confluence : null),
          probabilities: data.probabilities || data.raw_probabilities || {},
          shap_values: data.shap_values || data.shap_contributions || [],
          expectancy: data.expectancy ?? (_fsEm.historical_expectancy ?? null),
          max_dd: data.max_dd ?? (_fsEm.max_dd_pct != null ? -Math.abs(_fsEm.max_dd_pct) : null),
          profit_factor: data.profit_factor ?? (_fsEm.profitability_index != null ? Math.max(0.01, _fsEm.profitability_index) : null),
          win_rate: data.win_rate ?? (data.trading_accuracy != null ? Math.round(data.trading_accuracy * 1000) / 10 : null),
          total_trades: data.total_trades ?? 0,
        };

        // Auto-save new BUY/SELL signals to history (deduped by signal_id inside addToSignalHistory)
        if (signalObj.signal_id && !['HOLD', 'WAITING', 'NEUTRAL'].includes(signalObj.signal) &&
            typeof window.addToSignalHistory === 'function') {
          window.addToSignalHistory(signalObj);
        }

        if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
          const trialTimeframes = ['15m', '30m', '1h'];
          trialTimeframes.forEach(trialTf => {
            const key = `${symbol}_${trialTf}`;
            window.latestSignals[key] = { ...signalObj, timeframe: trialTf };
          });
        } else {
          const allTimeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
          allTimeframes.forEach(proTf => {
            const key = `${symbol}_${proTf}`;
            window.latestSignals[key] = { ...signalObj, timeframe: proTf };
          });
        }
      } else if (change.type === 'removed') {
        const data = change.doc.data();
        let symbol = data.symbol || change.doc.id;
        if (symbol && symbol.includes('_') && !symbol.includes('/')) {
            symbol = symbol.replace('_', '/');
        }

        if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
          const trialTimeframes = ['15m', '30m', '1h'];
          trialTimeframes.forEach(trialTf => {
            const key = `${symbol}_${trialTf}`;
            delete window.latestSignals[key];
          });
        } else {
          const allTimeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];
          allTimeframes.forEach(proTf => {
            const key = `${symbol}_${proTf}`;
            delete window.latestSignals[key];
          });
        }
      }
    });

    if (Object.keys(window.latestSignals).length > 0) {
      debouncedFilterAndRenderSignals();
    }
  }, (error) => {
    console.error('Signals listener error:', error);
    if (error.code === 'permission-denied') {
      if (!['pro', 'premium', 'intermediate'].includes(userPlan)) {
        if (signalsContainer) {
          signalsContainer.innerHTML = '<div class="no-signals"><i class="fas fa-lock text-red-500 mb-2 text-2xl"></i><p>Please upgrade to view signals.</p></div>';
        }
        showUpgradePrompt();
        if (auth && auth.currentUser) {
          auth.currentUser.getIdToken(true).catch(e => console.warn('Failed to force token refresh on permission denied:', e));
        }
        if (typeof setExpiredView === 'function') {
          setExpiredView();
        }
        if (typeof blockAllFeatures === 'function') {
          blockAllFeatures();
        }
      }
    } else {
      console.warn('Signals stream aborted/errored. Reconnecting in 5s...');
      setTimeout(setupFirestoreListeners, 5000);
    }
  });

  // Listen to user's trades in the user's subcollection so updates from
  // the backend API (which writes to users/{uid}/trades) reconcile properly.
  const firebaseUid = auth.currentUser?.uid;
  if (firebaseUid) {
    const userTradesCol = collection(db, 'users', firebaseUid, 'trades');
    const tradesQuery = query(userTradesCol, where('status', '==', 'open'));

    tradesUnsubscribe = onSnapshot(tradesQuery, (snapshot) => {
      const trades = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      // Always sync so closed-trade pruning clears the cache even when 0 open trades remain
      localStorage.setItem('lastKnownTrades', JSON.stringify(trades));
      if (typeof window.renderTrades === 'function') window.renderTrades(trades);
    }, (error) => {
      console.error('Trades listener error:', error);
      if (error.code !== 'permission-denied') {
        console.warn('Trades stream aborted/errored. Reconnecting in 5s...');
        setTimeout(setupFirestoreListeners, 5000);
      }
    });
  }

  // Allow the analytics room to render from localStorage while the snapshot loads
  window.forceTradesRefresh = function () {
    let trades = [];
    try {
      const lk = localStorage.getItem('lastKnownTrades');
      if (lk) trades = JSON.parse(lk);
    } catch (_) {}

    const analyticsEntry = localStorage.getItem('analyticsActiveTrade');
    if (analyticsEntry) {
      try {
        const trade = JSON.parse(analyticsEntry);
        if (trade && trade.status === 'open') {
          const alreadyIn = trades.some(t =>
            (trade.signalId && t.signalId === trade.signalId) ||
            (t.symbol === trade.symbol && String(t.entryPrice) === String(trade.entryPrice))
          );
          if (!alreadyIn) trades.unshift(trade);
        }
      } catch (_) {}
    }

    const uid = (auth && auth.currentUser && auth.currentUser.uid) || (currentUser && currentUser.uid);
    if (uid && trades.length > 0) {
      const containsOtherUsers = trades.some(t => t.userId && t.userId !== uid);
      if (containsOtherUsers) {
        trades = trades.filter(t => (t.userId && t.userId === uid) || String(t.id || '').startsWith('sim-'));
      }
    }

    if (trades.length > 0 && typeof window.renderTrades === 'function') { window.renderTrades(trades); }
  };
}

// -------------------------------------------------------------------
// Alpha Mode Toggle (Pro only)
// -------------------------------------------------------------------
async function toggleAlphaMode() {
  const isSubActive = AuthManager.getSubscriptionStatus() === 'active';

  if (userPlan !== 'pro' && !isSubActive) {
    showUpgradeModal();
    return;
  }

  const token = AuthManager.getToken();
  if (!token) return;

  try {
    const response = await fetch(`${API_BASE_URL}/alpha/toggle`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (response.ok) {
      const data = await response.json();
      currentAlphaMode = data.alpha_mode;

      // Update UI theme/state for Alpha Mode
      updateAlphaTheme(currentAlphaMode);

      if (alphaStatus) {
        alphaStatus.textContent = currentAlphaMode ? 'ACTIVE' : 'STANDBY';
        alphaStatus.className = currentAlphaMode ? 'alpha-active' : 'alpha-standby';
      }
      if (alphaToggleBtn) {
        alphaToggleBtn.classList.toggle('active', currentAlphaMode);
      }
      console.log(`Alpha mode ${currentAlphaMode ? 'enabled' : 'disabled'}`);
    } else if (response.status === 403) {
      showUpgradeModal();
    } else if (response.status === 503) {
      alert('Engine is warming up. Please try again in a few seconds.');
    } else {
      alert('Failed to toggle Alpha Mode. Please try again.');
    }
  } catch (error) {
    console.error('Toggle alpha error:', error);
    alert('Network error. Please check your connection and try again.');
  }
}

function updateAlphaTheme(isActive) {
  const body = document.body;
  if (isActive) {
    body.classList.add('alpha-mode-active');
    // Add alpha-specific styling
    body.style.setProperty('--accent-color', 'var(--orange)');
    body.style.setProperty('--glow-color', 'rgba(255, 140, 0, 0.6)');
  } else {
    body.classList.remove('alpha-mode-active');
    // Reset to default
    body.style.setProperty('--accent-color', 'var(--cyan)');
    body.style.setProperty('--glow-color', 'rgba(0, 242, 255, 0.6)');
  }
}

// -------------------------------------------------------------------
// User Settings
// -------------------------------------------------------------------
async function saveUserSettings() {
  const token = AuthManager.getToken();
  if (!token) return;

  const newCapital = parseFloat(capitalInput?.value);
  const newRisk = parseFloat(riskInput?.value);

  if (isNaN(newCapital) || newCapital < 100) {
    alert('Capital must be at least $100');
    return;
  }
  if (isNaN(newRisk) || newRisk < 0.5 || newRisk > 10) {
    alert('Risk percentage must be between 0.5% and 10%');
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/user/settings`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ capital: newCapital, risk_pct: newRisk })
    });

    if (response.ok) {
      alert('Settings saved successfully');
      if (capitalDisplay) capitalDisplay.textContent = `$${newCapital.toFixed(2)}`;
      if (riskDisplay) riskDisplay.textContent = `${newRisk}%`;
    } else {
      alert('Failed to save settings');
    }
  } catch (error) {
    console.error('Save settings error:', error);
    alert('Network error');
  }
}

// Trial countdown logic removed from here as it is managed in trial-countdown.js

// -------------------------------------------------------------------
// Upgrade Modal & Paddle Integration
// -------------------------------------------------------------------
function showUpgradePrompt() {
  if (userPlan === 'pro') return;

  const modal = getUpgradeModal();
  if (modal) modal.style.display = 'flex';
}

function getUpgradeModal() {
  let modal = document.getElementById('upgradeModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'upgradeModal';
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-card">
        <span class="close-modal">&times;</span>
        <h3>🚀 Unlock Full Power</h3>
        <p>Upgrade to Pro for:</p>
        <ul>
          <li>All 58 trading signals</li>
          <li>Alpha Mode (AI conviction)</li>
          <li>Real-time WebSocket feed</li>
          <li>Priority execution alerts</li>
        </ul>
        <div class="pricing-options" style="display:flex; flex-direction:column; gap:0.5rem; margin-bottom:1rem;">
          <button onclick="window.AegisDashboard?.subscribeToPlan('basic')" class="btn-outline" style="padding: 0.6rem; font-size: 0.85rem;">Basic ($3.60/mo)</button>
          <button onclick="window.AegisDashboard?.subscribeToPlan('intermediate')" class="btn-outline" style="padding: 0.6rem; font-size: 0.85rem; border-color: var(--primary-cyan);">Intermediate ($24/mo)</button>
          <button onclick="window.AegisDashboard?.subscribeToPlan('pro')" class="btn-pro" style="padding: 0.6rem; font-size: 0.85rem;">Pro ($40/mo)</button>
        </div>
        <button id="closeUpgradeModalBtn" class="btn-secondary">Maybe Later</button>
      </div>
    `;
    document.body.appendChild(modal);

    const closeBtn = modal.querySelector('.close-modal');
    const closeModalBtn = document.getElementById('closeUpgradeModalBtn');

    if (closeBtn) closeBtn.onclick = () => modal.style.display = 'none';
    if (closeModalBtn) closeModalBtn.onclick = () => modal.style.display = 'none';

    modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };
  }
  return modal;
}

// Razorpay Standard Checkout Integration
window.AegisDashboard = {
  subscribeToPlan: async (planType) => {
    const allowedPlans = ['basic', 'intermediate', 'pro'];
    const planName = allowedPlans.includes(planType) ? planType : 'basic';
    const token = AuthManager.getToken();

    if (!token) {
      alert('Please log in first');
      window.location.href = '/web/src/pages/index.html';
      return;
    }

    _showPaymentLoader();
    try {
      // 1. Get Razorpay key_id from backend (keeps secret off the frontend)
      const configResp = await fetch(`${API_BASE_URL}/payment/config`);
      const config = await configResp.json().catch(() => ({}));
      const keyId = config?.razorpay?.key_id;
      if (!keyId) {
        _hidePaymentLoader();
        alert('Payment gateway is not configured. Please contact support.');
        return;
      }

      // 2. Detect currency from timezone
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const currency = (tz === 'Asia/Calcutta' || tz === 'Asia/Kolkata') ? 'INR' : 'USD';

      // 3. Create Razorpay order on backend (amount converted from USD at live rate)
      const orderResp = await fetch(`${API_BASE_URL}/api/create-order`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan: planName, currency })
      });
      if (!orderResp.ok) {
        _hidePaymentLoader();
        if (orderResp.status === 401 || orderResp.status === 403) {
          alert('Your session has expired. Please sign in again.');
          window.location.href = '/web/src/pages/index.html';
          return;
        }
        const err = await orderResp.json().catch(() => ({}));
        alert('Could not create payment order. ' + (err.detail || 'Please try again.'));
        return;
      }
      const orderData = await orderResp.json();

      // 4. Load Razorpay checkout.js on-demand
      try {
        await loadThirdPartyScript('https://checkout.razorpay.com/v1/checkout.js');
      } catch (sdkErr) {
        _hidePaymentLoader();
        console.error('[Razorpay] Checkout script failed to load:', sdkErr);
        alert('Payment gateway failed to load. Please disable ad-blockers and refresh.');
        return;
      }

      // 5. Open Razorpay modal; verify signature on success
      _hidePaymentLoader();
      await new Promise((resolve) => {
        const userEmail = currentUser?.email || '';
        const rzp = new window.Razorpay({
          key: keyId,
          amount: orderData.amount,
          currency: orderData.currency,
          order_id: orderData.order_id,
          name: 'AEGIS v1.0',
          description: planName.charAt(0).toUpperCase() + planName.slice(1) + ' Plan',
          prefill: { email: userEmail },
          theme: { color: '#00f2ff' },
          handler: async (response) => {
            try {
              const verifyResp = await fetch(`${API_BASE_URL}/api/verify-payment`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  razorpay_payment_id: response.razorpay_payment_id,
                  razorpay_order_id:   response.razorpay_order_id,
                  razorpay_signature:  response.razorpay_signature,
                  plan: planName,
                })
              });
              const verifyData = await verifyResp.json().catch(() => ({}));
              if (verifyResp.ok && verifyData.status === 'success') {
                alert('Payment successful! Your subscription is now active.');
                window.location.reload();
              } else {
                alert('Payment verification failed. Contact support with payment ID: ' + response.razorpay_payment_id);
              }
            } catch (verifyErr) {
              console.error('[Razorpay] Verify error:', verifyErr);
              alert('Network error during verification. Contact support with payment ID: ' + response.razorpay_payment_id);
            }
            resolve();
          },
          modal: {
            ondismiss: () => { resolve(); }
          }
        });
        rzp.on('payment.failed', (response) => {
          console.error('[Razorpay] Payment failed:', response.error);
          alert('Payment failed: ' + (response.error?.description || 'Please try again.'));
          resolve();
        });
        rzp.open();
      });

    } catch (error) {
      _hidePaymentLoader();
      console.error('Subscription error:', error);
      alert('An error occurred while processing payment. Please try again.');
    }
  }
};


// Show upgrade modal on dashboard if trial expired
function showUpgradeModal() {
  const modal = getUpgradeModal();
  if (modal) modal.style.display = 'flex';
}

// -------------------------------------------------------------------
// Logout
// -------------------------------------------------------------------
async function handleLogout() {
  try {
    // Close WebSocket
    if (ws) ws.close();

    // Unsubscribe from Firestore listeners
    if (signalsUnsubscribe) signalsUnsubscribe();
    if (tradesUnsubscribe) tradesUnsubscribe();

    // Clear local storage — including trial/session state so a new login starts clean
    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    localStorage.removeItem('trial_end_timestamp');
    localStorage.removeItem('trial_end_sig');
    localStorage.removeItem('cached_uid');
    Object.keys(localStorage).forEach(k => {
      if (k.startsWith('trialStart_')) localStorage.removeItem(k);
    });

    // Sign out from Firebase
    await signOut(auth);

    // Redirect to home
    window.location.href = '/web/src/pages/index.html';
  } catch (error) {
    console.error('Logout error:', error);
    window.location.href = '/web/src/pages/index.html';
  }
}

function redirectToLogin() {
  window.location.href = '/web/src/pages/index.html';
}

// -------------------------------------------------------------------
// Footer with Legal Compliance
// -------------------------------------------------------------------
function setupFooter() {
  const footer = document.querySelector('.footer');
  if (!footer) return;

  // Check if footer already has proprietor info
  if (!footer.innerHTML.includes('Proprietor')) {
    const legalHtml = `
      <div class="footer-links">
        <a href="/web/src/pages/terms.html">Terms of Service</a>
        <span class="separator">•</span>
        <a href="/web/src/pages/refund-policy.html">Refund Policy</a>
        <span class="separator">•</span>
        <a href="/web/src/pages/risk-disclosure.html">Risk Disclosure</a>
      </div>
      <div class="footer-proprietor">
        Proprietor: Animesh Kukreti | Dehradun, Uttarakhand, India
      </div>
      <p>© 2025 AEGIS v1.0 — Sovereign Intelligence Terminal</p>
    `;
    footer.innerHTML = legalHtml;
  }
}

// -------------------------------------------------------------------
// Missing User/Settings Utilities for app.js
// -------------------------------------------------------------------
export async function ensureUserDocument(user) {
  if (!user) return null;
  const userDocRef = doc(db, 'users', user.uid);
  const docSnap = await getDoc(userDocRef);
  if (!docSnap.exists()) {
    await setDoc(userDocRef, {
      uid: user.uid,
      email: user.email || '',
      joinDate: serverTimestamp(),
      lastLogin: serverTimestamp()
    });
  } else {
    await updateDoc(userDocRef, { lastLogin: serverTimestamp() });
  }
}

export function subscribeUserSettings(user, callback) {
  if (!user) return () => { };
  const ref = doc(db, 'users', user.uid, 'preferences', 'settings');
  
  let unsub = null;
  function connect() {
    if (unsub) unsub();
    unsub = onSnapshot(ref, (docSnap) => {
      if (docSnap.exists()) {
        callback(docSnap.data());
      } else {
        callback({ capital: 10000, risk_pct: 1 });
      }
    }, (error) => {
      console.error('User settings listener error:', error);
      if (error.code !== 'permission-denied') {
        setTimeout(connect, 5000);
      }
    });
  }
  
  connect();
  return () => { if (unsub) unsub(); };
}

export async function updateUserSetting(user, key, value) {
  if (!user) return;
  const ref = doc(db, 'users', user.uid, 'preferences', 'settings');
  await setDoc(ref, { [key]: value }, { merge: true });
}

export function getCurrentUserToken() {
  return AuthManager.getToken();
}

// -------------------------------------------------------------------
// Export for module usage
// -------------------------------------------------------------------
export {
  currentUser, currentUserData, userPlan, trialActive, allowedTokens,
  getUpgradeModal, showUpgradeModal, handleLogout as logout
};
