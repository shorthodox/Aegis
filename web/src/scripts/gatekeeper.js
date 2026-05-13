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
  onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log("Firebase user detected:", user.uid);
      // If user exists, proceed to load data
      const token = await user.getIdToken();
      await loadUserFromBackend(token);
    } else {
      // No Firebase user, fallback to manual token check
      checkAuthAndLoad(); 
    }
  });
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
  // Alpha toggle is handled via modal now
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

      if (requiresPro && userPlan !== 'pro') {
        showUpgradeModal();
        return;
      }

      currentTimeframe = tf;
      
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
        filterAndRenderSignals();
      }
    });
  });

  const strategySelect = document.getElementById('strategy-matchmaker');
  if (strategySelect) {
    strategySelect.addEventListener('change', () => {
      if (typeof window.latestSignals !== 'undefined' && Object.keys(window.latestSignals).length > 0) {
        filterAndRenderSignals();
      }
    });
  }

  const simSelect = document.getElementById('sim-symbol');
  if (simSelect) {
    simSelect.addEventListener('change', (e) => {
      const sym = e.target.value;
      if (sym && window.latestSignals) {
        // Find the corresponding signal (try 1h timeframe first, then fallback)
        const key1h = `${sym}_1h`;
        const key15m = `${sym}_15m`;
        if (window.latestSignals[key1h] || window.latestSignals[key15m]) {
          const timeframe = window.latestSignals[key1h] ? '1h' : '15m';
          window.selectSignal(sym, timeframe);
        }
      }
    });
  }
}

function filterAndRenderSignals() {
  const currentSignals = {};
  const strategySelect = document.getElementById('strategy-matchmaker');
  const currentStrategy = strategySelect ? strategySelect.value : '';

  Object.values(window.latestSignals || {}).forEach(sig => {
    if (sig.timeframe === currentTimeframe) {
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
  renderSignals(currentSignals);
}

// -------------------------------------------------------------------
// Authentication & User Data
// -------------------------------------------------------------------
async function checkAuthAndLoad() {
  const token = AuthManager.getToken();
  const bypassAuth = localStorage.getItem('dashboardAuthBypass') === 'true' || window.location.search.includes('bypassAuth=true');

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

async function loadUserFromBackend(token) {
  try {
    // First, get user data from backend
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
      
      document.dispatchEvent(new CustomEvent('dashboardUserLoaded', { detail: { userData: currentUserData } }));
    } else if (response.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('authToken');
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
  const token = AuthManager.getToken();
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

  // We leave trial banner manipulation to trial-countdown.js
  // Update alpha mode visibility (Pro only - or available for all if requested)
  if (alphaToggleBtn) {
    alphaToggleBtn.style.display = 'flex';
  }
}

// -------------------------------------------------------------------
// WebSocket Connection
// -------------------------------------------------------------------
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const baseReconnectDelay = 1000; // 1 second

function startWebSocket(token) {
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error('Max WebSocket reconnection attempts reached');
    updateConnectionStatus('DISCONNECTED', 'red');
    return;
  }

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/dashboard`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('✅ WebSocket connected');
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
        // Handle pong response
        console.log('🏓 WebSocket pong received');
        return;
      }
      
      if (data.type === 'error') {
        console.error('WebSocket error message:', data.message);
        return;
      }
      
      if (data.type === 'signals' || data.type === 'update') {
        updateDashboardData(data);
      } else {
        // Handle other message types
        updateDashboardData(data);
      }
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateConnectionStatus('ERROR', 'red');
  };

  ws.onclose = (event) => {
    console.log(`WebSocket disconnected (code: ${event.code}, reason: ${event.reason}), reconnecting...`);
    updateConnectionStatus('RECONNECTING', 'yellow');
    
    // Stop heartbeat
    stopHeartbeat();
    
    // Exponential backoff for reconnection
    const delay = Math.min(baseReconnectDelay * Math.pow(2, reconnectAttempts), 30000); // Max 30 seconds
    reconnectAttempts++;
    
    setTimeout(() => startWebSocket(token), delay);
  };
}

let heartbeatInterval = null;

function startHeartbeat() {
  if (heartbeatInterval) clearInterval(heartbeatInterval);
  heartbeatInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000); // Send ping every 30 seconds
}

function stopHeartbeat() {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
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

function updateDashboardData(data) {
  // Update balance
  if (balanceDisplay && data.balance !== undefined) {
    balanceDisplay.textContent = `$${data.balance.toFixed(2)}`;
  }

  // Handle live prices if provided
  if (data.tickers) {
    window.previousTickers = window.currentTickers || {};
    window.currentTickers = data.tickers;
    
    Object.entries(data.tickers).forEach(([sym, price]) => {
      const priceDisplays = document.querySelectorAll(`.live-price[data-symbol="${sym}"]`);
      priceDisplays.forEach(priceDisplay => {
          const currentPrice = parseFloat(price);
          const previousPrice = parseFloat(window.previousTickers[sym] || currentPrice);
          
          // Format based on price size
          const priceStr = currentPrice < 0.01 ? currentPrice.toFixed(6) : currentPrice.toFixed(4);
          priceDisplay.textContent = `$${priceStr}`;
          
          priceDisplay.classList.remove('price-up', 'price-down');
          if (currentPrice > previousPrice) {
            priceDisplay.classList.add('price-up');
          } else if (currentPrice < previousPrice) {
            priceDisplay.classList.add('price-down');
          }
      });
    });
  }

  // Update signals with plan filtering
  if (signalsContainer && data.signals) {
    // Populate window.latestSignals from WebSocket
    Object.entries(data.signals).forEach(([sym, sig]) => {
      const key = `${sym}_1h`; // Defaulting to 1h for ws
      window.latestSignals = window.latestSignals || {};
      window.latestSignals[key] = {
        symbol: sym,
        signal: sig.signal || 'WAITING',
        ai_prob: sig.ai_prob || sig.confidence || 0,
        signal_strength: sig.signal_strength || 'NORMAL',
        risk_pct: sig.risk_pct || 2,
        atr: sig.atr || 0,
        timeframe: '1h',
        direction: sig.direction || "NEUTRAL",
        entry_price: sig.entry_price || 0,
        sl: sig.sl || 0,
        tp: sig.tp || 0,
        confidence_score: sig.confidence_score || 0,
        signal_id: sig.signal_id || "",
        trading_accuracy: sig.trading_accuracy || 0.5,
        profitability_index: sig.profitability_index || 0
      };
    });
    filterAndRenderSignals();
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

  // Populate simulation select dropdown
  const simSelect = document.getElementById('sim-symbol');
  if (simSelect) {
    const currentSelection = simSelect.value;
    simSelect.innerHTML = '<option value="">Select a signal...</option>' + 
      filteredEntries.map(([sym]) => `<option value="${sym}">${sym}</option>`).join('');
    if (filteredEntries.some(([sym]) => sym === currentSelection)) {
      simSelect.value = currentSelection;
    }
  }

  const strategySelect = document.getElementById('strategy-matchmaker');
  const isStrategyActive = strategySelect && strategySelect.value !== '';

  signalsContainer.innerHTML = filteredEntries.map(([symbol, signal]) => {
    const signalType = signal.signal || 'WAITING';
    const timeframe = signal.timeframe || '1h'; // Default to 1h if not provided
    const signalClass = getSignalClass(signalType);
    const confidence = (signal.ai_prob || 0) * 100;
    
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

    let matchClasses = '';
    let matchBadge = '';
    if (isStrategyActive) {
      matchClasses = 'border-cyan shadow-[0_0_15px_rgba(0,242,255,0.4)]';
      matchBadge = '<span class="bg-cyan/20 text-cyan border border-cyan/50 px-2 py-0.5 rounded text-[10px] ml-2 font-bold tracking-wider animate-pulse">ALPHA MATCH</span>';
    }

    return `
      <div class="signal-card ${signalClass} cursor-pointer hover:shadow-[0_0_15px_rgba(0,242,255,0.2)] transition-all ${matchClasses}" onclick="window.selectSignal('${symbol}', '${timeframe}')" data-symbol="${symbol}">
        <div class="signal-header flex justify-between items-center">
          <div class="flex items-center">
            <span class="signal-symbol font-bold">${symbol}</span>
            <span class="signal-timeframe text-xs text-gray-500 ml-2">${timeframe}</span>
            ${directionBadge}
            ${matchBadge}
          </div>
          <div class="flex items-center gap-3">
            <div class="price-container text-sm hidden md:flex"><span class="live-price" data-symbol="${symbol}">${window.currentTickers && window.currentTickers[symbol] ? '$' + parseFloat(window.currentTickers[symbol]).toFixed(4) : '-'}</span></div>
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
        </div>
      </div>
    `;
  }).join('');
}

window.selectSignal = function(symbol, timeframe) {
  const key = `${symbol}_${timeframe}`;
  const sig = window.latestSignals && window.latestSignals[key];
  if (!sig) return;

  const simSelect = document.getElementById('sim-symbol');
  const simEntry = document.getElementById('sim-entry');
  const simSl = document.getElementById('sim-sl');
  const simTp = document.getElementById('sim-tp');
  const directionBadge = document.getElementById('direction-badge');

  if (simSelect) {
    if (!Array.from(simSelect.options).some(opt => opt.value === symbol)) {
       const newOpt = document.createElement('option');
       newOpt.value = symbol;
       newOpt.textContent = symbol;
       simSelect.appendChild(newOpt);
    }
    simSelect.value = symbol;
  }
  if (simEntry) simEntry.value = sig.entry_price || 0;
  if (simSl) simSl.value = sig.sl || 0;
  if (simTp) simTp.value = sig.tp || 0;
  
  if (directionBadge) {
    directionBadge.textContent = sig.direction || 'NEUTRAL';
    if (sig.direction === 'LONG') {
      directionBadge.className = 'text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400 font-bold border border-green-500/30';
    } else if (sig.direction === 'SHORT') {
      directionBadge.className = 'text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-400 font-bold border border-red-500/30';
    } else {
      directionBadge.className = 'text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400';
    }
  }

  if (typeof window.calculatePosition === 'function') {
    window.calculatePosition();
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
  const token = AuthManager.getToken();
  if (!token) return;

  // Listen to signals collection
  const signalsQuery = query(collection(db, 'signals'), orderBy('timestamp', 'desc'), limit(150));
  
  window.latestSignals = {}; // Accumulate signals to prevent flickering
  signalsUnsubscribe = onSnapshot(signalsQuery, (snapshot) => {
    snapshot.docChanges().forEach(change => {
      if (change.type === 'added' || change.type === 'modified') {
        const data = change.doc.data();
        const symbol = data.symbol || change.doc.id;
        const tf = data.timeframe || '1h';
        const key = `${symbol}_${tf}`;
        
        // Apply plan filtering
        if (userPlan !== 'pro' && !allowedTokens.includes(symbol)) {
          return;
        }
        
        window.latestSignals[key] = {
          symbol: symbol,
          signal: data.signal || 'WAITING',
          ai_prob: data.ai_prob || data.confidence || 0,
          signal_strength: data.signal_strength || 'NORMAL',
          risk_pct: data.risk_pct || 2,
          atr: data.atr || 0,
          timeframe: tf,
          direction: data.direction || "NEUTRAL",
          entry_price: data.entry_price || 0,
          sl: data.sl || 0,
          tp: data.tp || 0,
          confidence_score: (data.ai_prob || data.confidence || 0) * 100,
          signal_id: data.signal_id || "",
          trading_accuracy: data.trading_accuracy || 0.5,
          profitability_index: data.profitability_index || 0
        };
      } else if (change.type === 'removed') {
        const data = change.doc.data();
        const symbol = data.symbol || change.doc.id;
        const tf = data.timeframe || '1h';
        const key = `${symbol}_${tf}`;
        delete window.latestSignals[key];
      }
    });
    
    if (Object.keys(window.latestSignals).length > 0) {
      filterAndRenderSignals();
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
    }
  } catch (error) {
    console.error('Toggle alpha error:', error);
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
    const token = AuthManager.getToken();
    
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
    localStorage.removeItem('authToken');
    
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
    if (!user) return () => {};
    const ref = doc(db, 'users', user.uid, 'preferences', 'settings');
    return onSnapshot(ref, (docSnap) => {
        if (docSnap.exists()) {
            callback(docSnap.data());
        } else {
            callback({ capital: 10000, risk_pct: 1 });
        }
    });
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
