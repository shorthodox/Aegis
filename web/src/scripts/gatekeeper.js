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
  if (typeof showSubscriptionExpiredOverlay === 'function') {
    showSubscriptionExpiredOverlay();
  }
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
let signalsContainer, positionsContainer, balanceDisplay, capitalDisplay, riskDisplay;
let alphaToggleBtn, alphaStatus, upgradeBtn, logoutBtn, trialBanner, planBadge;
let capitalInput, riskInput, saveSettingsBtn;

// -------------------------------------------------------------------
// Token Lists
// -------------------------------------------------------------------
const BIG5_TOKENS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "AAVE/USDT"];
const PRO_TOKENS = []; // Will be populated from backend

// Global state for new features
let signalHistory = [];
let paperTrades = [];
let signalDebounceMap = new Map(); // Track last signal time per symbol
let currentRiskProfile = 'balanced';
let _lastRenderedTrades = [];

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
    if (!el.closest('#subscriptionExpiredOverlay') && !el.closest('#access-expired-card')) {
      el.disabled = true;
      el.style.pointerEvents = 'none';
      el.style.opacity = '0.5';
    }
  });
}

// -------------------------------------------------------------------
// Signal History Management
// -------------------------------------------------------------------
function addSignalToHistory(signal) {
  if (!signal || signal.signal === 'HOLD') return;

  const status = getSignalStatus(signal);
  const historyEntry = {
    symbol: signal.symbol,
    signal: signal.signal,
    entry_price: signal.entry_price,
    sl: signal.sl,
    tp: signal.tp,
    timestamp: new Date().toISOString(),
    signal_id: signal.signal_id,
    status: status,
    direction: signal.direction || 'NEUTRAL'
  };

  signalHistory.unshift(historyEntry);

  // Keep only last 100 entries
  if (signalHistory.length > 100) {
    signalHistory = signalHistory.slice(0, 100);
  }

  updateSignalHistoryUI();
  saveSignalHistoryToStorage();
}

function updateSignalHistoryUI() {
  const tbody = document.getElementById('signalHistoryTbody');
  if (!tbody) return;

  if (signalHistory.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="p-6 text-center text-gray-500">No signal history available</td></tr>';
    return;
  }

  // Update signal statuses based on current prices
  signalHistory.forEach(entry => {
    if (!entry.status || entry.status === 'ACTIVE') {
      entry.status = getSignalStatus(entry);
    }
  });

  tbody.innerHTML = signalHistory.map(entry => {
    const signalClass = getSignalClass(entry.signal, entry.status);
    const timestamp = new Date(entry.timestamp).toLocaleString();
    const status = entry.status || 'ACTIVE';

    // Determine status badge styling
    let statusBadgeClass = 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50';
    let statusText = 'ACTIVE';

    if (status === 'EXPIRED') {
      statusBadgeClass = 'bg-green-500/20 text-green-400 border border-green-500/50';
      statusText = '✓ TARGET HIT';
    } else if (status === 'STOPPED_OUT') {
      statusBadgeClass = 'bg-red-500/20 text-red-400 border border-red-500/50';
      statusText = '✗ STOPPED OUT';
    }

    // Win Rate Badge
    const perf = (window.tokenPerformance && window.tokenPerformance[entry.symbol]) || null;
    let winRateHtml = '';
    if (perf) {
      let wrClass = '';
      if (perf.winRate >= 70) wrClass = 'win-rate-high';
      else if (perf.winRate >= 50) wrClass = 'win-rate-medium';
      winRateHtml = `<span class="win-rate-badge ${wrClass} text-[9px] ml-2 font-normal">WR: ${perf.winRate}%</span>`;
    }

    return `
      <tr class="hover:bg-white/5">
        <td class="p-3 font-bold flex items-center h-full">${entry.symbol} ${winRateHtml}</td>
        <td class="p-3">
          <span class="signal-badge ${signalClass} text-xs">${entry.signal}</span>
        </td>
        <td class="p-3 font-mono">${entry.entry_price ? '$' + entry.entry_price.toFixed(4) : '-'}</td>
        <td class="p-3 font-mono">${entry.tp ? '$' + entry.tp.toFixed(4) : '-'}</td>
        <td class="p-3 text-xs text-gray-400">${timestamp}</td>
        <td class="p-3">
          <span class="text-xs font-bold px-2 py-1 rounded ${statusBadgeClass}">${statusText}</span>
        </td>
        <td class="p-3">
          <button onclick="initiatePaperTrade('${entry.symbol}', ${entry.entry_price || 0}, ${entry.sl || 0}, ${entry.tp || 0})" 
                  class="text-xs bg-cyan/20 hover:bg-cyan/40 text-cyan border border-cyan/50 px-2 py-1 rounded transition-colors">
            Paper Trade
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

function saveSignalHistoryToStorage() {
  try {
    localStorage.setItem('signalHistory', JSON.stringify(signalHistory));
  } catch (e) {
    console.warn('Failed to save signal history to localStorage:', e);
  }
}

function loadSignalHistoryFromStorage() {
  try {
    const stored = localStorage.getItem('signalHistory');
    if (stored) {
      signalHistory = JSON.parse(stored);
      updateSignalHistoryUI();
    }
  } catch (e) {
    console.warn('Failed to load signal history from localStorage:', e);
  }
}

function clearSignalHistory() {
  signalHistory = [];
  updateSignalHistoryUI();
  localStorage.removeItem('signalHistory');
}

// -------------------------------------------------------------------
// Paper Trading System
// -------------------------------------------------------------------
function initiatePaperTrade(symbol, entryPrice, sl, tp) {
  const modal = document.getElementById('paperTradeModal');
  const symbolSpan = document.getElementById('paperTradeSymbol');

  // Pre-set selection globals so executeTrade() passes its guard check even
  // if the user edits values before clicking confirm in the modal.
  window.selectedTradeToken = symbol;

  if (modal && symbolSpan) {
    symbolSpan.textContent = symbol;
    modal.classList.remove('hidden');

    // Store trade data for confirmation
    modal._tradeData = { symbol, entryPrice, sl, tp };
  }
}

function startPaperTrade(symbol, entryPrice, sl, tp) {
  const trade = {
    id: Date.now().toString(),
    symbol,
    entryPrice,
    sl,
    tp,
    startTime: new Date(),
    currentPrice: entryPrice,
    pnl: 0,
    status: 'open'   // must match Firestore listener query: where('status', '==', 'open')
  };

  // Set global selection state so dashboard.js executeTrade() can find the token.
  // Derive direction from SL position: SL < entry → LONG, SL > entry → SHORT.
  window.selectedTradeToken = symbol;
  window.selectedTrade = {
    symbol,
    direction: sl < entryPrice ? 'LONG' : 'SHORT',
    entry_price: entryPrice,
    sl,
    tp,
    signalId: null,
  };

  paperTrades.push(trade);
  updatePaperTradesUI();

  // Auto-fill terminal
  autoFillTerminal(trade);

  // Switch to terminal room
  if (typeof switchRoom === 'function') {
    switchRoom('terminal');
  }
}

function updatePaperTradesUI() {
  if (paperTrades.length > 0) {
    renderTrades(paperTrades);
  }
}

function autoFillTerminal(trade) {
  const symbolSelect = document.getElementById('sim-symbol');
  const entryInput = document.getElementById('sim-entry');
  const slInput = document.getElementById('sim-sl');
  const tpInput = document.getElementById('sim-tp');

  if (symbolSelect) {
    // Add option if not exists
    let option = Array.from(symbolSelect.options).find(opt => opt.value === trade.symbol);
    if (!option) {
      option = document.createElement('option');
      option.value = trade.symbol;
      option.textContent = trade.symbol;
      symbolSelect.appendChild(option);
    }
    symbolSelect.value = trade.symbol;
  }

  if (entryInput) entryInput.value = trade.entryPrice.toFixed(4);
  if (slInput) slInput.value = trade.sl.toFixed(4);
  if (tpInput) tpInput.value = trade.tp.toFixed(4);

  // Trigger calculations
  if (typeof window.calculatePosition === 'function') {
    window.calculatePosition();
  }
}

// -------------------------------------------------------------------
// Signal Debouncing & Risk-Based Sorting
// -------------------------------------------------------------------
function shouldShowSignal(symbol, signalData) {
  const now = Date.now();
  const lastSignalTime = signalDebounceMap.get(symbol) || 0;
  const debouncePeriod = 300000; // 5 minutes

  if (now - lastSignalTime < debouncePeriod) {
    console.log(`Signal for ${symbol} debounced (last signal ${Math.round((now - lastSignalTime) / 1000)}s ago)`);
    return false;
  }

  signalDebounceMap.set(symbol, now);
  return true;
}

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
  loadSignalHistoryFromStorage();

  // Global Event Listener for Real-Time Price Sync to drive Signal History Updates
  window.addEventListener('priceUpdate', (e) => {
    const { symbol, price } = e.detail;
    let historyChanged = false;

    // Check active signals in history and update status if target/stop hit
    signalHistory.forEach(entry => {
      if (entry.symbol === symbol && (!entry.status || entry.status === 'ACTIVE')) {
        const newStatus = getSignalStatus(entry);
        if (newStatus !== entry.status) {
          entry.status = newStatus;
          historyChanged = true;
          console.log(`[Status Change] ${symbol} signal is now ${newStatus}`);
        }
      }
    });

    if (historyChanged) {
      updateSignalHistoryUI();
      saveSignalHistoryToStorage();
    }
  });

  // Refresh trades PnL whenever a tracked symbol's price updates
  window.addEventListener('priceUpdate', (e) => {
    const { symbol } = e.detail;
    if (_lastRenderedTrades.length > 0 && _lastRenderedTrades.some(t => t.symbol === symbol)) {
      renderTrades(_lastRenderedTrades);
    }
  });

  onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log("Firebase user detected:", user.uid);
      const token = await user.getIdToken();
      // Persist Firebase token so executeTrade and other API calls can find it
      if (typeof AuthManager !== 'undefined') AuthManager.setToken(token);
      await loadUserFromBackend(token, user);
    } else {
      // No Firebase user, fallback to manual token check
      checkAuthAndLoad();
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

  // Paper Trade Modal bindings
  const paperTradeConfirm = document.getElementById('paperTradeConfirm');
  const paperTradeCancel = document.getElementById('paperTradeCancel');
  const paperTradeModal = document.getElementById('paperTradeModal');

  if (paperTradeConfirm && paperTradeModal) {
    paperTradeConfirm.addEventListener('click', () => {
      if (paperTradeModal._tradeData) {
        const { symbol, entryPrice, sl, tp } = paperTradeModal._tradeData;
        startPaperTrade(symbol, entryPrice, sl, tp);
        paperTradeModal.classList.add('hidden');
      }
    });
  }
  if (paperTradeCancel && paperTradeModal) {
    paperTradeCancel.addEventListener('click', () => {
      paperTradeModal.classList.add('hidden');
    });
  }

  // Clear History Button
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all signal history?')) {
        clearSignalHistory();
      }
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

  // Setup mobile optimizations
  setupMobileOptimizations();
}

// ============================================================
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

  loadUserLimits().then(() => {
    updateUI();
    startWebSocket(token);
    setupFirestoreListeners();
    loadGlobalPerformanceData();
    document.dispatchEvent(new CustomEvent('dashboardUserLoaded', { detail: { userData: currentUserData } }));
  });
}

async function provisionUserFromFirebase(firebaseUser, token, attempt = 1) {
  const MAX_ATTEMPTS = 3;

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
        display_name: firebaseUser.displayName || null
      })
    });

    if (response.ok) {
      hideProvisioningState();
      const userData = await response.json();
      applyUserData(userData, token);
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
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error('[WS] Max WebSocket reconnection attempts reached');
    updateConnectionStatus('DISCONNECTED', 'red');
    return;
  }

  // State Management: Clean up existing instance before creating a new one
  cleanupWebSocket();

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

      // NOTE: do NOT gate on data.timeframe here — the backend sends a full
      // timeframes map and the summary timeframe rarely matches the user's tab.
      if (data.type === 'signals' || data.type === 'update') {
        updateDashboardData(data);
      } else {
        // Handle other message types
        updateDashboardData(data);
      }
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

  // Handle live prices AFTER signals are rendered so DOM changes apply correctly
  if (data.tickers || window.currentTickers) {
    window.previousTickers = window.currentTickers || {};
    if (data.tickers && Object.keys(data.tickers).length > 0) {
      window.currentTickers = { ...window.currentTickers, ...data.tickers };
    }

    // Defer ticker updates slightly to ensure Reactivity cycle finishes if debouncedFilterAndRenderSignals just fired
    setTimeout(() => {
      Object.entries(window.currentTickers).forEach(([sym, price]) => {
        const idStr = sym.replace('/', '-');
        const priceDisplays = document.querySelectorAll(`.live-price[data-symbol="${idStr}"]`);

        // Dispatch global custom event for dynamic UI synchronization (e.g., signal history updates)
        const currentPrice = parseFloat(price);
        if (!isNaN(currentPrice)) {
          window.dispatchEvent(new CustomEvent('priceUpdate', { detail: { symbol: sym, price: currentPrice } }));
        }

        const previousPrice = parseFloat(window.previousTickers[sym] || currentPrice);

        priceDisplays.forEach(priceDisplay => {
          if (isNaN(currentPrice)) return;

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

        // Update market card change indicator with % change
        const changeSpan = document.getElementById(`market-card-change-${idStr}`);
        if (changeSpan && !isNaN(currentPrice) && previousPrice && previousPrice !== currentPrice) {
          const pct = ((currentPrice - previousPrice) / previousPrice) * 100;
          const sign = pct >= 0 ? '+' : '';
          changeSpan.textContent = `${sign}${pct.toFixed(2)}%`;
          changeSpan.className = `text-[10px] font-mono font-semibold ${
            pct >= 0 ? 'text-green-400' : 'text-red-400'
          }`;
        }
      });
    }, 50);
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
    renderTrades(incomingTrades);
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
    const expiredCard = document.getElementById('access-expired-card');
    const isOverlayVisible = expiredCard && !expiredCard.classList.contains('hidden');

    if (data.trial_expired && !isOverlayVisible) {
      if (typeof window.setExpiredView === 'function') window.setExpiredView();
    } else if (!data.trial_expired && isOverlayVisible) {
      if (typeof window.clearExpiredView === 'function') window.clearExpiredView();
    }
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

  // Set the global selectedTrade so terminal execution knows which trade is active
  window.selectedTrade = {
    symbol: sig.symbol,
    direction: sig.direction || (sig.signal && sig.signal.includes('BUY') ? 'LONG' : (sig.signal && sig.signal.includes('SELL') ? 'SHORT' : 'NEUTRAL')),
    signalId: sig.signal_id || sig.signalId,
    ...sig
  };

  // Delegate to dashboard.js to properly populate selectedTrade state if function exists
  if (typeof window.prefillTradeSim === 'function') {
    window.prefillTradeSim(symbol, sig.entry_price || 0, sig);
  }

  // Add to signal history
  addSignalToHistory(sig);

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

function renderTrades(trades) {
  // Ensure we only render trades that belong to the currently authenticated
  // user. Some backend systems (live_engine) may write global executions
  // which should not appear in a user's Analytics view.
  let tradesToProcess = trades || [];
  const uid = (auth && auth.currentUser && auth.currentUser.uid) || (currentUser && currentUser.uid);
  if (uid && tradesToProcess.length > 0) {
    const containsOtherUsers = tradesToProcess.some(tt => tt.userId && tt.userId !== uid);
    if (containsOtherUsers) {
      tradesToProcess = tradesToProcess.filter(tt => (tt.userId && tt.userId === uid) || String(tt.id || '').startsWith('sim-'));
    }
  }

  _lastRenderedTrades = tradesToProcess || [];
  let normalized = (tradesToProcess || []).map(t => ({
    id:           t.id || t.tradeId || '',
    symbol:       t.symbol || '—',
    side:         t.side || t.direction || 'LONG',
    entryPrice:   parseFloat(t.entryPrice   || t.entry_price  || 0),
    stopLoss:     parseFloat(t.stopLoss     || t.stop_loss    || t.sl || 0),
    takeProfit:   parseFloat(t.takeProfit   || t.take_profit  || t.tp || 0),
    positionUnits:parseFloat(t.positionUnits|| t.position_size|| 0),
    openTime:     t.openTime || t.entry_time || '',
  }));

  // If the live snapshot returned no trades, try to fall back to any
  // recently executed trade saved to localStorage (analyticsActiveTrade)
  // or the last known trades cache so the UI doesn't flash empty briefly.
  if (normalized.length === 0) {
    try {
      const stored = localStorage.getItem('analyticsActiveTrade');
      const lastKnown = localStorage.getItem('lastKnownTrades');
      let fallback = [];

      if (lastKnown) {
        const parsed = JSON.parse(lastKnown);
        if (Array.isArray(parsed) && parsed.length > 0) fallback = parsed.slice();
      }

      if (stored) {
        const t = JSON.parse(stored);
        if (t && (t.status === 'open' || !t.status)) {
          // Ensure simulated trade appears at the front
          fallback.unshift(Object.assign({ id: t.id || `sim-${Date.now()}` }, t));
        }
      }

      if (fallback.length > 0) {
        normalized = fallback.map(t => ({
          id:           t.id || t.tradeId || '',
          symbol:       t.symbol || '—',
          side:         t.side || t.direction || 'LONG',
          entryPrice:   parseFloat(t.entryPrice   || t.entry_price  || t.entry || 0),
          stopLoss:     parseFloat(t.stopLoss     || t.stop_loss    || t.sl || 0),
          takeProfit:   parseFloat(t.takeProfit   || t.take_profit  || t.tp || 0),
          positionUnits:parseFloat(t.positionUnits|| t.position_size|| t.position_units || 0),
          openTime:     t.openTime || t.entry_time || t.openTime || ''
        }));
      }
    } catch (err) {
      // ignore parsing errors and fall through to empty state
      console.warn('Fallback trades parse error:', err);
    }
  }

  // ── Position Cards (#positionsContainer) ─────────────────────────
  if (positionsContainer) {
    if (normalized.length === 0) {
      positionsContainer.innerHTML = `
        <div class="text-center py-8 text-gray-500">
          <i class="fas fa-chart-line text-2xl mb-3 opacity-30"></i>
          <p class="text-sm">No active positions. Execute a trade from the Terminal to track it here.</p>
        </div>`;
    } else {
      positionsContainer.innerHTML = normalized.map(trade => {
        const cur      = window.currentTickers?.[trade.symbol]
          ? parseFloat(window.currentTickers[trade.symbol]) : trade.entryPrice;
        const pnl      = trade.side === 'LONG'
          ? (cur - trade.entryPrice) * trade.positionUnits
          : (trade.entryPrice - cur) * trade.positionUnits;
        const pnlPct   = trade.entryPrice > 0
          ? (pnl / (trade.entryPrice * (trade.positionUnits || 1)) * 100) : 0;
        const pnlSign  = pnl >= 0 ? '+' : '';
        const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
        const sideCls  = trade.side === 'LONG'
          ? 'bg-green-500/20 text-green-400 border-green-500/30'
          : 'bg-red-500/20 text-red-400 border-red-500/30';
        const opened   = trade.openTime ? new Date(trade.openTime).toLocaleString() : '—';

        return `
          <div class="bg-black/40 p-4 rounded-xl border border-white/10 flex flex-wrap items-center gap-4">
            <span class="text-xs px-2 py-1 rounded border font-bold ${sideCls}">${trade.side}</span>
            <div class="flex-1 min-w-0">
              <div class="font-bold text-white">${trade.symbol}</div>
              <div class="text-xs text-gray-500 font-mono truncate">Entry: $${trade.entryPrice.toFixed(4)} &nbsp;·&nbsp; ${opened}</div>
            </div>
            <div class="text-right">
              <div class="font-mono font-bold ${pnlColor}">${pnlSign}$${pnl.toFixed(2)}</div>
              <div class="text-xs ${pnlColor} opacity-70">${pnlSign}${pnlPct.toFixed(2)}%</div>
            </div>
            <div class="text-right font-mono text-xs text-gray-500 hidden sm:block">
              <div>SL <span class="text-red-400">$${trade.stopLoss.toFixed(4)}</span></div>
              <div>TP <span class="text-green-400">$${trade.takeProfit.toFixed(4)}</span></div>
            </div>
            <button onclick="window.closeTrade('${trade.id}')"
              class="text-xs px-3 py-2 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/40 transition-colors font-bold whitespace-nowrap">
              <i class="fas fa-times-circle mr-1"></i>Close
            </button>
          </div>`;
      }).join('');
    }
  }

  // ── Active Executions Table (#trades-tbody) ───────────────────────
  const tbody = document.getElementById('trades-tbody');
  if (tbody) {
    if (normalized.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="p-6 text-center text-gray-500">No active trades</td></tr>';
    } else {
      tbody.innerHTML = normalized.map(trade => {
        const cur      = window.currentTickers?.[trade.symbol]
          ? parseFloat(window.currentTickers[trade.symbol]) : trade.entryPrice;
        const pnl      = trade.side === 'LONG'
          ? (cur - trade.entryPrice) * trade.positionUnits
          : (trade.entryPrice - cur) * trade.positionUnits;
        const pnlSign  = pnl >= 0 ? '+' : '';
        const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
        const sideCls  = trade.side === 'LONG' ? 'text-green-400' : 'text-red-400';

        return `
          <tr class="hover:bg-white/5">
            <td class="p-3 font-bold text-white">${trade.symbol}</td>
            <td class="p-3 font-bold ${sideCls}">${trade.side}</td>
            <td class="p-3 font-mono">$${trade.entryPrice.toFixed(4)}</td>
            <td class="p-3 font-mono text-red-400">$${trade.stopLoss.toFixed(4)}</td>
            <td class="p-3 font-mono text-green-400">$${trade.takeProfit.toFixed(4)}</td>
            <td class="p-3 font-mono font-bold ${pnlColor}">${pnlSign}$${pnl.toFixed(2)}</td>
            <td class="p-3">
              <button onclick="window.closeTrade('${trade.id}')"
                class="text-xs px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/40 transition-colors font-bold">
                Close
              </button>
            </td>
          </tr>`;
      }).join('');
    }
  }
}

window.closeTrade = async function (tradeId) {
  if (!tradeId) return;

  // Find the trade in the last rendered list to validate ownership
  const trade = _lastRenderedTrades.find(t => (t.id || t.tradeId) === tradeId);
  const uid = (auth && auth.currentUser && auth.currentUser.uid) || (currentUser && currentUser.uid);

  if (trade) {
    // If trade has a userId and it doesn't match current user, don't call backend
    if (trade.userId && uid && trade.userId !== uid) {
      console.warn('Attempted to close a trade that does not belong to the current user', { tradeId, owner: trade.userId, me: uid });
      alert('Cannot close this trade: it does not belong to your account. Refreshing trades.');
      if (typeof window.forceTradesRefresh === 'function') window.forceTradesRefresh();
      return;
    }
  }

  // Optimistic removal — drop the row from both tables immediately so the
  // user sees instant feedback rather than waiting for the Firestore round-trip.
  _lastRenderedTrades = _lastRenderedTrades.filter(t => (t.id || t.tradeId) !== tradeId);
  renderTrades(_lastRenderedTrades);

  // Prune from localStorage so the fallback renderer can't re-add the trade
  function pruneFromStorage(id) {
    try {
      const lk = localStorage.getItem('lastKnownTrades');
      if (lk) localStorage.setItem('lastKnownTrades', JSON.stringify(JSON.parse(lk).filter(t => (t.id || t.tradeId) !== id)));
    } catch (_) {}
    try {
      const at = localStorage.getItem('analyticsActiveTrade');
      if (at) {
        const t = JSON.parse(at);
        if (t && (t.id === id || t.tradeId === id)) localStorage.removeItem('analyticsActiveTrade');
      }
    } catch (_) {}
  }

  // Simulated trades (sim-*) only live in localStorage — no backend record exists
  if (String(tradeId).startsWith('sim-')) {
    pruneFromStorage(tradeId);
    return;
  }

  try {
    // Defensive: ensure simulated trades never reach the backend (double-check)
    if (String(tradeId).startsWith('sim-')) {
      pruneFromStorage(tradeId);
      return;
    }

    const token = typeof AuthManager !== 'undefined' ? AuthManager.getToken() : null;
    if (!token) { alert('Session expired. Please refresh and log in again.'); return; }

    console.log('Closing trade via API', { tradeId, uid });
    const url = `/api/trades/${encodeURIComponent(tradeId)}/close`;
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (r.status === 404) {
      // Trade not on server (already closed or never persisted) — just clean up locally
      console.warn('Backend returned 404 when closing trade', tradeId);
      pruneFromStorage(tradeId);
      return;
    }

    if (!r.ok) {
      const text = await r.text();
      throw new Error(`HTTP ${r.status} ${text}`);
    }

    pruneFromStorage(tradeId);
    // Firestore listener will reconcile the authoritative list on the next snapshot.
  } catch (e) {
    console.error('closeTrade error:', e);
    alert(`Could not close trade: ${e.message}`);
    // Re-fetch known trades to restore the list if the request failed
    if (typeof window.forceTradesRefresh === 'function') window.forceTradesRefresh();
  }
};

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
      renderTrades(trades);
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

    // Firestore-synced trades from the onSnapshot listener
    const known = localStorage.getItem('lastKnownTrades');
    if (known) {
      try {
        const parsed = JSON.parse(known);
        if (Array.isArray(parsed)) trades = parsed;
      } catch (_) {}
    }

    // Always include a freshly-executed trade that isn't already in the Firestore list
    const cached = localStorage.getItem('analyticsActiveTrade');
    if (cached) {
      try {
        const trade = JSON.parse(cached);
        if (trade && trade.status === 'open') {
          const alreadyIn = trades.some(t =>
            (trade.signalId && t.signalId === trade.signalId) ||
            (t.symbol === trade.symbol && String(t.entryPrice) === String(trade.entryPrice))
          );
          if (!alreadyIn) trades.unshift(trade);
        }
      } catch (_) {}
    }

    // Filter fallback trades to the current user as well
    const uid = (auth && auth.currentUser && auth.currentUser.uid) || (currentUser && currentUser.uid);
    if (uid && trades.length > 0) {
      const containsOtherUsers = trades.some(t => t.userId && t.userId !== uid);
      if (containsOtherUsers) {
        trades = trades.filter(t => (t.userId && t.userId === uid) || String(t.id || '').startsWith('sim-'));
      }
    }

    if (trades.length > 0) { renderTrades(trades); return; }
    if (paperTrades.length > 0) { renderTrades(paperTrades); }
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

// Cashfree Subscription Integration
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

    try {
      let amount = 3.60;
      if (planName === 'pro') amount = 40.00;
      else if (planName === 'intermediate') amount = 24.00;

      // 1. Fetch payment session from backend
      const response = await fetch(`${API_BASE_URL}/api/v1/create-payment-session`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ tier: planName, amount: amount })
      });

      if (!response.ok) {
        console.warn('Subscription backend not ready, using sandbox fallback mock.');
        await mockSuccessfulPayment(planName);
        return;
      }

      const data = await response.json();
      const paymentSessionId = data.payment_session_id;

      if (!data.success || !paymentSessionId) {
        console.warn('Invalid Cashfree session generated, using sandbox fallback mock. Error:', data.error);
        await mockSuccessfulPayment(planName);
        return;
      }

      // 2. Load Cashfree SDK on-demand (guarded — errors if blocked or timed out)
      try {
        await loadThirdPartyScript('https://sdk.cashfree.com/js/v3/cashfree.js');
      } catch (sdkErr) {
        console.error('[Cashfree] SDK failed to load:', sdkErr);
        alert('Payment gateway failed to load. Please check your connection and try again.');
        return;
      }

      // 3. Open Cashfree modal
      const cashfree = window.Cashfree({ mode: 'sandbox' }); // Change to 'production' in live
      const checkoutOptions = { paymentSessionId, redirectTarget: '_modal' };

      try {
        const result = await cashfree.checkout(checkoutOptions);
        if (result.error) {
          console.error('[Cashfree] Checkout error:', result.error);
          alert('Payment was cancelled or failed. Please try again.');
        } else if (result.paymentDetails) {
          console.log('[Cashfree] Payment successful');
          await mockSuccessfulPayment(planName);
        }
      } catch (checkoutErr) {
        console.error('[Cashfree] Checkout exception:', checkoutErr);
        alert('Payment gateway error. Please try again.');
      }
    } catch (error) {
      console.error('Subscription error:', error);
      // Fallback mock payment for current sandbox environment
      await mockSuccessfulPayment(planName);
    }
  }
};

async function mockSuccessfulPayment(planName) {
  try {
    // Update Firestore user status directly for demo purposes
    if (currentUser && currentUser.uid) {
      const userDocRef = doc(db, 'users', currentUser.uid);
      await updateDoc(userDocRef, {
        plan: planName,
        subscriptionStatus: 'active',
        trialActive: false
      });
    }

    // Clear expired view if it exists
    if (typeof window.clearExpiredView === 'function') {
      window.clearExpiredView();
    }

    alert('Payment successful! Your subscription is now active.');
    window.location.reload();
  } catch (err) {
    console.error("Error updating subscription status:", err);
  }
}

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

// Expose functions globally for HTML access
window.initiatePaperTrade = initiatePaperTrade;

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
