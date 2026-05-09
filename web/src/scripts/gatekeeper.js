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

// -------------------------------------------------------------------
// Firebase Configuration
// -------------------------------------------------------------------
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDtudUL2sE1_fKbzIro5d2IP0-M2dYI6x4",
  authDomain: "aegis-d78e1.firebaseapp.com",
  databaseURL: "https://aegis-d78e1-default-rtdb.asia-southeast1.firebasedatabase.app",
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

// -------------------------------------------------------------------
// API Base URL
// -------------------------------------------------------------------
const API_BASE_URL = window.location.origin;

// -------------------------------------------------------------------
// Global State
// -------------------------------------------------------------------
let currentUser = null;
let currentUserData = null;
let userPlan = 'trial';
let trialEnd = null;
let trialActive = true;
let allowedTokens = [];
let ws = null;
let signalsUnsubscribe = null;
let tradesUnsubscribe = null;
let countdownInterval = null;
let currentAlphaMode = false;

// DOM Elements
let signalsContainer, positionsContainer, balanceDisplay, capitalDisplay, riskDisplay;
let alphaToggleBtn, alphaStatus, upgradeBtn, logoutBtn, trialBanner, planBadge;
let capitalInput, riskInput, saveSettingsBtn;

// -------------------------------------------------------------------
// Token Lists
// -------------------------------------------------------------------
const BIG5_TOKENS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"];
const PRO_TOKENS = []; // Will be populated from backend

// -------------------------------------------------------------------
// Initialize Dashboard
// -------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  if (!window.location.pathname.includes('dashboard.html')) return;
  initializeElements();
  attachEventListeners();
  checkAuthAndLoad();
  setupFooter();
});

function initializeElements() {
  signalsContainer = document.getElementById('signalsContainer');
  positionsContainer = document.getElementById('positionsContainer');
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
  if (alphaToggleBtn) alphaToggleBtn.addEventListener('click', toggleAlphaMode);
  if (upgradeBtn) upgradeBtn.addEventListener('click', () => {
    window.location.href = '/web/src/pages/pricing.html';
  });
  if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);
  if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveUserSettings);
}

// -------------------------------------------------------------------
// Authentication & User Data
// -------------------------------------------------------------------
async function checkAuthAndLoad() {
  const token = localStorage.getItem('access_token');
  if (!token) {
    redirectToLogin();
    return;
  }

  // Check if token is expired
  if (isJWTExpired(token)) {
    localStorage.removeItem('access_token');
    redirectToLogin();
    return;
  }

  await loadUserFromBackend(token);
}

function isJWTExpired(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const exp = payload.exp * 1000;
    return Date.now() >= exp;
  } catch (e) {
    return true;
  }
}

async function loadUserFromBackend(token) {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const userData = await response.json();
      currentUser = { email: userData.email, token };
      currentUserData = userData;
      userPlan = userData.plan || 'trial';
      trialEnd = userData.trial_end ? new Date(userData.trial_end) : null;
      trialActive = userPlan === 'trial' && trialEnd && new Date() < trialEnd;

      await loadUserLimits();
      updateUI();
      startWebSocket(token);
      setupFirestoreListeners();
      startTrialCountdown();
      
      if (userPlan !== 'pro') {
        showUpgradePrompt();
      }
    } else if (response.status === 401) {
      localStorage.removeItem('access_token');
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

async function loadUserLimits() {
  const token = localStorage.getItem('access_token');
  if (!token) return;

  try {
    const response = await fetch(`${API_BASE_URL}/user/limits`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const limits = await response.json();
      allowedTokens = limits.allowed_tokens || BIG5_TOKENS;
      return limits;
    }
  } catch (error) {
    console.error('Load limits error:', error);
  }
  allowedTokens = BIG5_TOKENS;
  return null;
}

function updateUI() {
  // Update plan badge
  if (planBadge) {
    if (userPlan === 'pro') {
      planBadge.innerHTML = '<i class="fas fa-crown"></i> PRO PLAN';
      planBadge.className = 'plan-badge pro';
    } else if (trialActive) {
      planBadge.innerHTML = '<i class="fas fa-flask"></i> TRIAL ACTIVE';
      planBadge.className = 'plan-badge trial';
    } else {
      planBadge.innerHTML = '<i class="fas fa-clock"></i> TRIAL EXPIRED';
      planBadge.className = 'plan-badge expired';
    }
  }

  // Update trial banner
  if (trialBanner) {
    if (userPlan === 'pro') {
      trialBanner.style.display = 'none';
    } else if (trialActive && trialEnd) {
      trialBanner.style.display = 'block';
      trialBanner.innerHTML = `<i class="fas fa-hourglass-half"></i> Trial active until ${trialEnd.toLocaleDateString()} | <a href="/web/src/pages/pricing.html">Upgrade to Pro →</a>`;
    } else if (!trialActive && userPlan !== 'pro') {
      trialBanner.style.display = 'block';
      trialBanner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Your trial has expired. <a href="/web/src/pages/pricing.html">Subscribe now →</a>';
      trialBanner.classList.add('expired');
    }
  }

  // Update alpha mode visibility (Pro only)
  if (alphaToggleBtn) {
    alphaToggleBtn.style.display = userPlan === 'pro' ? 'flex' : 'none';
  }
}

// -------------------------------------------------------------------
// WebSocket Connection
// -------------------------------------------------------------------
function startWebSocket(token) {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/dashboard`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('✅ WebSocket connected');
    ws.send(JSON.stringify({ token }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboardData(data);
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting in 3s...');
    setTimeout(() => startWebSocket(token), 3000);
  };
}

function updateDashboardData(data) {
  // Update balance
  if (balanceDisplay && data.balance !== undefined) {
    balanceDisplay.textContent = `$${data.balance.toFixed(2)}`;
  }

  // Update signals with plan filtering
  if (signalsContainer && data.signals) {
    renderSignals(data.signals);
  }

  // Update open trades
  if (positionsContainer && data.open_trades) {
    renderTrades(data.open_trades);
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
}

// -------------------------------------------------------------------
// Signal Rendering with Plan Filtering
// -------------------------------------------------------------------
function renderSignals(signals) {
  if (!signalsContainer) return;

  const signalEntries = Object.entries(signals);
  
  // Filter signals based on user's allowed tokens
  const filteredEntries = signalEntries.filter(([symbol]) => {
    if (userPlan === 'pro') return true;
    return allowedTokens.includes(symbol);
  });

  if (filteredEntries.length === 0) {
    signalsContainer.innerHTML = `
      <div class="no-signals">
        <i class="fas fa-chart-line"></i>
        <p>No signals available for your plan</p>
        ${userPlan !== 'pro' ? '<a href="/web/src/pages/pricing.html" class="upgrade-link">Upgrade to Pro for 58 tokens →</a>' : ''}
      </div>
    `;
    return;
  }

  signalsContainer.innerHTML = filteredEntries.map(([symbol, signal]) => {
    const signalType = signal.signal || 'WAITING';
    const signalClass = getSignalClass(signalType);
    const confidence = (signal.ai_prob || 0) * 100;
    
    return `
      <div class="signal-card ${signalClass}">
        <div class="signal-header">
          <span class="signal-symbol">${symbol}</span>
          <span class="signal-badge ${signalClass}">${signalType}</span>
        </div>
        <div class="signal-details">
          <div class="signal-confidence">
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${confidence}%"></div>
            </div>
            <span>AI: ${confidence.toFixed(1)}%</span>
          </div>
          <div class="signal-meta">
            <span class="signal-strength ${signal.signal_strength?.toLowerCase()}">
              <i class="fas fa-bolt"></i> ${signal.signal_strength || 'NORMAL'}
            </span>
            <span class="signal-risk">
              <i class="fas fa-shield-alt"></i> Risk: ${signal.risk_pct || 2}%
            </span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function getSignalClass(signal) {
  const s = String(signal).toUpperCase();
  if (s.includes('BUY')) return 'buy';
  if (s.includes('SELL')) return 'sell';
  return 'neutral';
}

function renderTrades(trades) {
  if (!positionsContainer) return;

  if (!trades || trades.length === 0) {
    positionsContainer.innerHTML = '<div class="no-trades"><i class="fas fa-ban"></i> No active positions</div>';
    return;
  }

  positionsContainer.innerHTML = trades.map(trade => {
    const pnlClass = (trade.unrealized_pnl || 0) >= 0 ? 'positive' : 'negative';
    return `
      <div class="trade-card">
        <div class="trade-header">
          <span class="trade-symbol">${trade.symbol}</span>
          <span class="trade-side ${trade.side}">${trade.side}</span>
        </div>
        <div class="trade-details">
          <div>Entry: $${trade.entry_price?.toFixed(2)}</div>
          <div>Size: ${trade.position_size?.toFixed(4)}</div>
          <div>SL: $${trade.stop_loss?.toFixed(2)}</div>
          <div>TP: $${trade.take_profit?.toFixed(2)}</div>
        </div>
        <div class="trade-pnl ${pnlClass}">
          PnL: $${(trade.unrealized_pnl || 0).toFixed(2)}
        </div>
      </div>
    `;
  }).join('');
}

// -------------------------------------------------------------------
// Firestore Real-time Listeners
// -------------------------------------------------------------------
function setupFirestoreListeners() {
  const token = localStorage.getItem('access_token');
  if (!token) return;

  // Listen to signals collection
  const signalsQuery = query(collection(db, 'signals'), orderBy('timestamp', 'desc'), limit(50));
  
  signalsUnsubscribe = onSnapshot(signalsQuery, (snapshot) => {
    const signals = {};
    snapshot.forEach(doc => {
      const data = doc.data();
      const symbol = data.symbol || doc.id;
      
      // Apply plan filtering
      if (userPlan !== 'pro' && !allowedTokens.includes(symbol)) {
        return;
      }
      
      signals[symbol] = {
        signal: data.signal || 'WAITING',
        ai_prob: data.ai_prob || data.confidence || 0,
        signal_strength: data.signal_strength || 'NORMAL',
        risk_pct: data.risk_pct || 2,
        atr: data.atr || 0
      };
    });
    
    if (Object.keys(signals).length > 0) {
      renderSignals(signals);
    }
  }, (error) => {
    console.error('Signals listener error:', error);
  });

  // Listen to user's trades if exists
  if (currentUser?.email) {
    const tradesQuery = query(
      collection(db, 'trades'),
      where('email', '==', currentUser.email),
      where('status', '==', 'open'),
      orderBy('entry_time', 'desc')
    );
    
    tradesUnsubscribe = onSnapshot(tradesQuery, (snapshot) => {
      const trades = [];
      snapshot.forEach(doc => {
        trades.push({ id: doc.id, ...doc.data() });
      });
      renderTrades(trades);
    }, (error) => {
      console.error('Trades listener error:', error);
    });
  }
}

// -------------------------------------------------------------------
// Alpha Mode Toggle (Pro only)
// -------------------------------------------------------------------
async function toggleAlphaMode() {
  if (userPlan !== 'pro') {
    showUpgradeModal();
    return;
  }

  const token = localStorage.getItem('access_token');
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
    }
  } catch (error) {
    console.error('Toggle alpha error:', error);
  }
}

// -------------------------------------------------------------------
// User Settings
// -------------------------------------------------------------------
async function saveUserSettings() {
  const token = localStorage.getItem('access_token');
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

// -------------------------------------------------------------------
// Trial Countdown
// -------------------------------------------------------------------
function startTrialCountdown() {
  if (countdownInterval) clearInterval(countdownInterval);
  
  if (userPlan !== 'pro' && trialEnd && trialActive) {
    updateTrialTimer();
    countdownInterval = setInterval(updateTrialTimer, 60000);
  }
}

function updateTrialTimer() {
  if (!trialEnd) return;
  
  const now = new Date();
  const diff = trialEnd - now;
  
  if (diff <= 0) {
    if (countdownInterval) clearInterval(countdownInterval);
    trialActive = false;
    updateUI();
    if (trialBanner) {
      trialBanner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Your trial has expired. <a href="/web/src/pages/pricing.html">Subscribe now →</a>';
    }
    return;
  }
  
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (3600000)) / 60000);
  
  if (trialBanner && userPlan !== 'pro') {
    trialBanner.innerHTML = `<i class="fas fa-hourglass-half"></i> Trial expires in ${hours}h ${minutes}m | <a href="/web/src/pages/pricing.html">Upgrade to Pro →</a>`;
  }
}

// -------------------------------------------------------------------
// Upgrade Modal & Cashfree Integration
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
        <div class="pricing-options">
          <button onclick="window.AegisDashboard?.subscribeToPlan('basic')" class="btn-outline">Basic ($3.60/mo)</button>
          <button onclick="window.AegisDashboard?.subscribeToPlan('pro')" class="btn-pro">Pro ($24/mo)</button>
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

// Cashfree Subscription Integration
window.AegisDashboard = {
  subscribeToPlan: async (planType) => {
    const amount = planType === 'pro' ? 24.00 : 3.60;
    const planName = planType === 'pro' ? 'pro' : 'basic';
    const token = localStorage.getItem('access_token');
    
    if (!token) {
      alert('Please log in first');
      window.location.href = '/web/src/pages/index.html';
      return;
    }
    
    try {
      // Get user email from backend
      const userResponse = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const userData = await userResponse.json();
      
      const response = await fetch(`${API_BASE_URL}/create-subscription`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          plan_name: planName,
          amount: amount,
          email: userData.email,
          customer_phone: null
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.success && data.sub_auth_url) {
        // Redirect to Cashfree authorization page
        window.location.href = data.sub_auth_url;
      } else {
        alert(data.detail || 'Failed to create subscription. Please try again.');
      }
    } catch (error) {
      console.error('Subscription error:', error);
      alert('Network error. Please try again.');
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
    
    // Clear local storage
    localStorage.removeItem('access_token');
    
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
// Export for module usage
// -------------------------------------------------------------------
export { 
  currentUser, currentUserData, userPlan, trialActive, allowedTokens,
  getUpgradeModal, showUpgradeModal
};
