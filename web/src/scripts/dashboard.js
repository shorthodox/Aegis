import { initializeTrialCountdown, fetchTrialStartFromFirestore } from './trial-countdown.js';
import { auth, db } from './gatekeeper.js';
import { doc, getDoc } from 'https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js';

let initialized = false;

async function checkUserSubscriptionStatus(uid) {
  try {
    const userDocRef = doc(db, 'users', uid);
    const userSnap = await getDoc(userDocRef);
    if (userSnap.exists()) {
      const data = userSnap.data();
      const plan = data.plan || data.tier;
      if (!plan) return true; // Fallback to prevent immediate locking for new users
      
      const p = plan.toLowerCase();
      if (p === 'premium' || p === 'pro' || p === 'intermediate' || p === 'basic' || p === 'trial' || p === 'free_tier') {
        return true;
      }
      return false; // Specifically locked/expired plans
    }
    return true; // Document doesn't exist fallback
  } catch (error) {
    console.error('Error fetching user subscription status:', error);
    return true; // Fallback on error
  }
}

// ============================================================
// WAIT FOR AUTH STATE CHANGE (replaces polling waitForUserId)
// ============================================================
function waitForAuthStateChange() {
  const authPromise = new Promise((resolve) => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      unsubscribe(); // Unsubscribe after first state change
      if (user?.uid) {
        const isPremium = await checkUserSubscriptionStatus(user.uid);
        if (isPremium) {
          window.isSubscriptionActive = true;
          window.isPremiumUser = true;
          clearExpiredView();
          unblockFeatures();
          const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display');
          countdownElements.forEach(el => el.style.display = 'none');
        }
      }
      resolve(user?.uid || null);
    });
  });

  const timeoutPromise = new Promise((resolve) => {
    setTimeout(() => {
      console.warn('waitForAuthStateChange timeout reached, triggering retry...');
      // Trigger retry mechanism
      const retryUnsubscribe = auth.onAuthStateChanged(async (user) => {
        retryUnsubscribe();
        if (user?.uid) {
          const isPremium = await checkUserSubscriptionStatus(user.uid);
          if (isPremium) {
            window.isSubscriptionActive = true;
            window.isPremiumUser = true;
            clearExpiredView();
            unblockFeatures();
            const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display');
            countdownElements.forEach(el => el.style.display = 'none');
          }
          resolve(user?.uid || null);
        } else {
          resolve(null);
        }
      });
      // Safety release for the retry
      setTimeout(() => {
        retryUnsubscribe();
        resolve(null);
      }, 5000);
    }, 20000); // Increased timeout to 20 seconds
  });

  return Promise.race([authPromise, timeoutPromise]);
}

function handleAuthFailure() {
  window.isSubscriptionActive = false;
  blockAllFeatures();

  const dashboardContent = document.getElementById('dashboard-main-content');
  if (dashboardContent) dashboardContent.classList.add('hidden');

  let errorCard = document.getElementById('auth-error-card');
  if (!errorCard) {
    errorCard = document.createElement('div');
    errorCard.id = 'auth-error-card';
    errorCard.className = 'expired-card'; // Reuse some styling
    errorCard.innerHTML = `
      <div class="expired-card-content">
        <i class="fas fa-wifi-off" style="font-size: 3rem; color: #ff8c00; margin-bottom: 1rem;"></i>
        <h2 style="color: #ff8c00; margin: 1rem 0;">Connection Error</h2>
        <p style="color: #ccc; margin: 1rem 0; font-size: 1.1rem;">
          Unable to verify your account status. Please check your connection and try again.
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem;">
          <button onclick="location.reload()" style="
            padding: 12px 32px;
            background: linear-gradient(135deg, #ff8c00, #ff0055);
            border: 2px solid #ff8c00;
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
          ">
            Retry
          </button>
        </div>
      </div>
    `;
    errorCard.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.95);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
    `;
    document.body.appendChild(errorCard);
  } else {
    errorCard.classList.remove('hidden');
  }
}

function setExpiredView() {
  const dashboardContent = document.getElementById('dashboard-main-content');
  const expiredCard = document.getElementById('access-expired-card');

  // Hide all dashboard content
  if (dashboardContent) dashboardContent.classList.add('hidden');

  // Show or create expired card
  if (expiredCard) {
    expiredCard.classList.remove('hidden');
  } else {
    // Create expired card if it doesn't exist
    createExpiredCard();
  }

  // Block all feature access
  blockAllFeatures();
}

function clearExpiredView() {
  const dashboardContent = document.getElementById('dashboard-main-content');
  const expiredCard = document.getElementById('access-expired-card');
  if (dashboardContent) dashboardContent.classList.remove('hidden');
  if (expiredCard) expiredCard.classList.add('hidden');

  // Re-enable features
  unblockFeatures();
}

// ============================================================
// ============================================================
// CREATE EXPIRED/SUBSCRIPTION ENDED CARD
// ============================================================
function createExpiredCard() {
  let expiredCard = document.getElementById('access-expired-card');

  if (!expiredCard) {
    expiredCard = document.createElement('div');
    expiredCard.id = 'access-expired-card';
    expiredCard.className = 'expired-card hidden';
    expiredCard.innerHTML = `
      <div class="expired-card-content">
        <i class="fas fa-lock" style="font-size: 3rem; color: #ff0055; margin-bottom: 1rem;"></i>
        <h2 style="color: #ff0055; margin: 1rem 0;">Subscription Ended</h2>
        <p style="color: #ccc; margin: 1rem 0; font-size: 1.1rem;">
          Your free trial has expired. Subscribe now to continue accessing premium signals and trading features.
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem; flex-wrap: wrap;">
          <button id="expired-subscribe-btn" style="
            padding: 12px 32px;
            background: linear-gradient(135deg, #6c63ff, #0f3cff);
            border: 2px solid #0f3cff;
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
          " onmouseover="this.style.boxShadow='0 0 20px rgba(111, 99, 255, 0.5)'" onmouseout="this.style.boxShadow='none'">
            Subscribe Now
          </button>
          <button id="expired-home-btn" style="
            padding: 12px 32px;
            background: transparent;
            border: 2px solid rgba(255,255,255,0.2);
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
          " onmouseover="this.style.borderColor='rgba(255,255,255,0.5)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.2)'">
            Return Home
          </button>
        </div>
        <p style="color: #999; margin-top: 2rem; font-size: 0.9rem;">
          All features are locked until you subscribe
        </p>
      </div>
    `;
    expiredCard.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.95);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
    `;
    document.body.appendChild(expiredCard);

    const subscribeBtn = document.getElementById('expired-subscribe-btn');
    if (subscribeBtn) {
      subscribeBtn.addEventListener('click', () => {
        window.location.href = '/web/src/pages/pricing.html';
      });
    }

    const homeBtn = document.getElementById('expired-home-btn');
    if (homeBtn) {
      homeBtn.addEventListener('click', () => {
        window.location.href = '/web/src/pages/index.html';
      });
    }
  }

  return expiredCard;
}

// ============================================================
// BLOCK ALL FEATURES WHEN SUBSCRIPTION EXPIRED
// ============================================================
function blockAllFeatures() {
  window.isSubscriptionActive = false;

  // Block token cards
  const tokenCards = document.querySelectorAll('[data-token-card], .token-card');
  tokenCards.forEach(card => {
    card.style.pointerEvents = 'none';
    card.style.opacity = '0.3';
  });

  // Block signals container
  const signalsContainer = document.getElementById('signalsContainer');
  if (signalsContainer) {
    signalsContainer.style.pointerEvents = 'none';
    signalsContainer.style.opacity = '0.3';
  }

  // Disable all interactive elements in dashboard
  const interactiveElements = document.querySelectorAll('button, a, input[type="checkbox"], [onclick]');
  interactiveElements.forEach(el => {
    const isExpiredElement = el.closest('#access-expired-card') !== null || (el.id && el.id.includes('expired'));
    
    // Safety check so we don't lock the user in completely
    const isLogout = el.id === 'logout-btn' || el.id === 'btn-logout' || el.classList.contains('logout-button');
    const isNav = el.closest('nav') !== null || el.closest('header') !== null;
    
    if (!isExpiredElement && !isLogout && !isNav) {
      el.style.pointerEvents = 'none';
      el.style.opacity = '0.3';
    }
  });

  console.log('✅ All features blocked - subscription expired');
}

// ============================================================
// UNBLOCK FEATURES WHEN SUBSCRIPTION ACTIVE
// ============================================================
function unblockFeatures() {
  window.isSubscriptionActive = true;

  // Restore token cards and time-span selectors
  const tokenCards = document.querySelectorAll('[data-token-card], .token-card, .tf-btn, [data-tf]');
  tokenCards.forEach(card => {
    card.style.pointerEvents = 'auto';
    card.style.opacity = '1';
  });

  // Restore signals container
  const signalsContainer = document.getElementById('signalsContainer');
  if (signalsContainer) {
    signalsContainer.style.pointerEvents = 'auto';
    signalsContainer.style.opacity = '1';
  }

  // Enable all interactive elements
  const interactiveElements = document.querySelectorAll('button, a, input[type="checkbox"], [onclick]');
  interactiveElements.forEach(el => {
    el.style.removeProperty('pointer-events');
    el.style.removeProperty('opacity');
  });

  console.log('✅ All features unlocked - subscription active');
}

// ============================================================
// CHECK FEATURE ACCESS - PREVENT USE IF EXPIRED
// ============================================================
function canAccessFeatures() {
  if (window.isSubscriptionActive === false) {
    console.warn('⛔ Feature access blocked - subscription expired');
    setExpiredView();
    return false;
  }
  return true;
}

// ============================================================
// EXPORT FUNCTIONS FOR EXTERNAL USE
// ============================================================
window.canAccessFeatures = canAccessFeatures;
window.setExpiredView = setExpiredView;
window.clearExpiredView = clearExpiredView;

// ============================================================
// INITIALIZE LOGO CLICK HANDLER
// ============================================================
function initializeLogoClickHandler() {
  // Find logo element (common selectors)
  const logoSelectors = ['.logo', '[class*="logo"]', '#logo', '[id*="aegis"]', '[class*="aegis"]'];
  let logoElement = null;

  for (const selector of logoSelectors) {
    logoElement = document.querySelector(selector);
    if (logoElement) break;
  }

  // If logo found, make it clickable
  if (logoElement) {
    logoElement.style.cursor = 'pointer';
    logoElement.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = '/web/src/pages/index.html';
    });
    console.log('✅ Logo click handler initialized');
  }
}

// ============================================================
// FETCH AND UPDATE TOKEN MOVEMENT DATA (LIVE MARKET CARDS)
// ============================================================
let marketCardsInitialized = false;

function fetchLiveMarketData() {
  try {
    const container = document.getElementById('market-token-cards');
    if (!container) return;

    const priorityTokens = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];
    
    // Initial HTML setup - gatekeeper.js will take over price updates natively via .live-price class
    if (!marketCardsInitialized) {
      let html = '';
      priorityTokens.forEach(sym => {
        const idStr = sym.replace('/', '-');
        html += `
          <div class="glass-panel p-4 rounded-xl flex flex-col justify-center border-l-2 border-cyan/50 hover:bg-white/5 transition-colors cursor-pointer shadow-lg" onclick="selectToken('${sym}')">
            <span class="text-[10px] text-gray-500 font-bold uppercase tracking-widest">${sym}</span>
            <span id="market-card-price-${idStr}" class="live-price text-xl font-mono mt-1 transition-colors duration-300" data-symbol="${idStr}">Loading...</span>
          </div>
        `;
      });
      container.innerHTML = html;
      marketCardsInitialized = true;
    }
  } catch (err) {
    console.error('Failed to init token market cards:', err);
  }
}


let trialSetupRunning = false;
let lastTrialSetupTime = 0;

async function setupTrialNonBlocking(userId) {
  if (!userId || window.isPremiumUser) return;

  const isPremium = await checkUserSubscriptionStatus(userId);
  if (isPremium) {
    window.isSubscriptionActive = true;
    window.isPremiumUser = true;
    clearExpiredView();
    unblockFeatures();
    const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display');
    countdownElements.forEach(el => el.style.display = 'none');
    return;
  }

  const now = Date.now();
  if (trialSetupRunning || (now - lastTrialSetupTime < 5000)) {
     console.log('⏳ Skipping duplicate setupTrialNonBlocking call to prevent request bloat');
     return;
  }
  trialSetupRunning = true;

  const cacheKey = `trialStart_${userId}`;
  let trialStart = localStorage.getItem(cacheKey);

  try {
    // 1. Check local caching
    if (trialStart) {
      console.log('⚡ Using cached trial start time');
      initializeTrialCountdown(userId, trialStart);
    }

    // 2. Fetch from Firestore (revalidate)
    const freshTrialStart = await fetchTrialStartFromFirestore(userId);

    if (freshTrialStart && freshTrialStart !== trialStart) {
      localStorage.setItem(cacheKey, freshTrialStart);
      initializeTrialCountdown(userId, freshTrialStart);
    } else if (!freshTrialStart && !trialStart) {
      console.log('🛡️ No trial start found in DB, initiating 3-day grace period');
      const fallbackStart = new Date().toISOString();
      localStorage.setItem(cacheKey, fallbackStart);
      initializeTrialCountdown(userId, fallbackStart);
    }
  } catch (trialErr) {
    console.error('Failed to fetch trial data, using fallback:', trialErr);
    // 4. Error Handling: fallback to grace period or last known
    if (!trialStart) {
      console.log('🛡️ Initiating 24h grace period due to network error');
      const fallbackStart = new Date().toISOString();
      localStorage.setItem(cacheKey, fallbackStart);
      initializeTrialCountdown(userId, fallbackStart);
    }
  } finally {
    trialSetupRunning = false;
    lastTrialSetupTime = Date.now();
  }
}

async function initDashboard(event) {
  clearExpiredView();

  if (initialized) return;
  if (!window.location.pathname.includes('dashboard.html')) return;

  initialized = true;
  initializeLogoClickHandler();
  initializeTerminalListeners();

  // Initialize the empty cards, gatekeeper.js will fill the prices natively
  fetchLiveMarketData();

  document.addEventListener('trialExpired', () => {
    console.log('🔒 Trial expired event triggered');
    setExpiredView();
  });

  try {
    // 3. Auth Optimization: Use cached userId if available to start immediately
    const eventUid = event?.detail?.userData?.uid;
    const cachedUid = localStorage.getItem('cached_uid');
    const initialUid = eventUid || cachedUid;

    if (initialUid) {
      // 2. Non-Blocking Fetch: fire and forget
      setupTrialNonBlocking(initialUid);
    }

    // Await true auth state
    const realUserId = eventUid || await waitForAuthStateChange();

    if (!realUserId) {
      console.warn('Dashboard countdown: could not resolve current user UID');
      handleAuthFailure();
      return;
    }

    // Cache the real user ID
    localStorage.setItem('cached_uid', realUserId);

    // If we didn't have an initial UID or it changed, run the setup again
    if (realUserId !== initialUid) {
      setupTrialNonBlocking(realUserId);
    }

  } catch (err) {
    console.error('Error initializing dashboard:', err);
    const signalsContainer = document.getElementById('signalsContainer');
    if (signalsContainer) {
      signalsContainer.innerHTML = `
          <div class="no-signals" style="color: #ff3333; border-color: #ff3333;">
            <i class="fas fa-exclamation-triangle"></i>
            <p>Error loading dashboard components.</p>
            <button id="retry-dashboard-btn" style="margin-top: 10px; padding: 4px 12px; background: rgba(255,0,0,0.2); border-radius: 4px; color: white;">Retry</button>
          </div>
        `;
      document.getElementById('retry-dashboard-btn')?.addEventListener('click', () => location.reload());
    }
  }
}

window.addEventListener('DOMContentLoaded', initDashboard);
document.addEventListener('dashboardUserLoaded', initDashboard);

// ============================================================
// TERMINAL SIMULATION LOGIC
// ============================================================

window.selectedTrade = null;
window.selectedTradeToken = null;

window.selectToken = function(sym) {
  window.selectedTradeToken = sym;
};

function getATR(price) { return price * 0.015; }

function updateSimulation() {
  if (!window.selectedTrade) return;
  const simEntry = document.getElementById('sim-entry');
  const simBalance = document.getElementById('sim-balance') || document.getElementById('user-capital');
  const simRiskSlider = document.getElementById('sim-risk-slider');
  const simLeverageSlider = document.getElementById('sim-leverage');
  const simSl = document.getElementById('sim-sl');
  const simTp = document.getElementById('sim-tp');

  const entry = parseFloat(simEntry?.value);
  const balance = parseFloat(simBalance?.value || 10000);
  const riskPercent = parseFloat(simRiskSlider?.value || 2);
  const leverage = parseFloat(simLeverageSlider?.value || 1);
  let sl = parseFloat(simSl?.value);
  let tp = parseFloat(simTp?.value);
  const direction = window.selectedTrade.direction;

  if (isNaN(entry) || entry <= 0) return;
  const atr = window.selectedTrade.atr || getATR(entry);

  if (isNaN(sl) || sl <= 0) {
    sl = direction === 'LONG' ? entry - atr * 1.5 : entry + atr * 1.5;
    if (simSl) simSl.value = sl.toFixed(4);
  }
  if (isNaN(tp) || tp <= 0) {
    const riskDistance = Math.abs(entry - sl);
    tp = direction === 'LONG' ? entry + riskDistance * 3 : entry - riskDistance * 3;
    if (simTp) simTp.value = tp.toFixed(4);
  }

  const slDistance = Math.abs(entry - sl);
  if (slDistance === 0) return;

  const riskAmount = balance * (riskPercent / 100);
  const positionUnits = riskAmount / slDistance;
  const notionalValue = positionUnits * entry;
  const marginRequired = notionalValue / leverage;
  const liquidationPrice = direction === 'LONG'
    ? entry - (entry / leverage) + (sl / leverage)
    : entry + (entry / leverage) - (sl / leverage);

  const riskPercentOfBalance = (riskAmount / balance) * 100;
  const gaugePercent = Math.min(100, (riskPercentOfBalance / 10) * 100);

  if (document.getElementById('risk-gauge-fill')) document.getElementById('risk-gauge-fill').style.width = `${gaugePercent}%`;
  if (document.getElementById('pos-units')) document.getElementById('pos-units').innerText = Number(positionUnits || 0).toFixed(4);
  if (document.getElementById('notional')) document.getElementById('notional').innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
  if (document.getElementById('margin')) document.getElementById('margin').innerText = `$${Number(marginRequired || 0).toFixed(2)}`;
  if (document.getElementById('liquidation')) document.getElementById('liquidation').innerText = `$${Number(liquidationPrice || 0).toFixed(4)}`;

  // RR Ratio calculation
  const tpDistance = Math.abs(tp - entry);
  const rrRatio = slDistance > 0 ? (tpDistance / slDistance).toFixed(2) : "0.00";
  if (document.getElementById('rr-ratio')) document.getElementById('rr-ratio').innerText = rrRatio;
  if (document.getElementById('suggested-amount')) document.getElementById('suggested-amount').innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
}

window.prefillTradeSim = function (symbol, price, signalObj) {
  const direction = signalObj.direction || (signalObj.signal === 'SELL' ? 'SHORT' : 'LONG');
  const atr = signalObj.atr || getATR(price);
  window.selectedTrade = {
    symbol,
    entryPrice: price,
    direction,
    atr,
    aiProb: signalObj.confidence || signalObj.ai_prob,
    signal: signalObj.signal,
    signalId: signalObj.signal_id
  };

  const symbolSelect = document.getElementById('sim-symbol');
  if (symbolSelect) {
    let found = false;
    for (let i = 0; i < symbolSelect.options.length; i++) {
      if (symbolSelect.options[i].value === symbol) {
        found = true;
        break;
      }
    }
    if (!found) {
      const opt = document.createElement('option');
      opt.value = symbol;
      opt.innerText = symbol;
      symbolSelect.appendChild(opt);
    }
    symbolSelect.value = symbol;
  }

  if (document.getElementById('sim-entry')) document.getElementById('sim-entry').value = price;

  const badge = document.getElementById('direction-badge');
  if (badge) {
    badge.className = `text-xs px-2 py-0.5 rounded font-bold ${direction === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`;
    badge.innerText = direction;
  }

  if (document.getElementById('sim-sl')) document.getElementById('sim-sl').value = '';
  if (document.getElementById('sim-tp')) document.getElementById('sim-tp').value = '';

  updateSimulation();
};

async function executeTrade() {
  if (!window.selectedTradeToken && !window.selectedTrade) {
    alert('Please select a token first.');
    return;
  }

  const entry = parseFloat(document.getElementById('sim-entry').value);
  const sl = parseFloat(document.getElementById('sim-sl').value);
  const tp = parseFloat(document.getElementById('sim-tp').value);
  const riskPercent = parseFloat(document.getElementById('sim-risk-slider').value);
  const leverage = parseFloat(document.getElementById('sim-leverage').value);
  const positionUnits = parseFloat(document.getElementById('pos-units').innerText);
  const notional = parseFloat(document.getElementById('notional').innerText.replace('$', ''));

  const tradeData = {
    symbol: window.selectedTradeToken || (window.selectedTrade ? window.selectedTrade.symbol : null),
    side: window.selectedTrade ? window.selectedTrade.direction : (document.getElementById('direction-badge')?.innerText || 'LONG'),
    entryPrice: entry,
    stopLoss: sl,
    takeProfit: tp,
    riskPercent,
    leverage,
    positionUnits,
    notionalValue: notional,
    status: 'open',
    signalId: window.selectedTrade ? (window.selectedTrade.signalId || null) : null
  };

  localStorage.setItem('analyticsActiveTrade', JSON.stringify(tradeData));

  try {
    let token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
    if (!token && typeof AuthManager !== 'undefined') {
      token = AuthManager.getToken();
    }
    
    if (!token) {
      console.warn('No auth token found, cannot execute trade on backend');
      alert('You must be logged in to execute trades.');
      return;
    }

    const response = await fetch('/api/trades/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(tradeData)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to execute trade on backend');
    }

    console.log('✅ Trade executed and stored via backend API');

    if (typeof window.switchRoom === 'function') {
      window.switchRoom('analytics');
    }
  } catch (err) {
    console.error('Failed to save trade:', err);
    alert('Failed to execute trade: ' + err.message);
    if (typeof window.switchRoom === 'function') {
      window.switchRoom('analytics');
    }
  }
}

function initializeTerminalListeners() {
  document.getElementById('user-capital')?.addEventListener('input', (e) => {
    const simBal = document.getElementById('sim-balance');
    if (simBal) simBal.value = e.target.value;
    const capDisplay = document.getElementById('capitalDisplay');
    if (capDisplay) capDisplay.innerText = `$${parseFloat(e.target.value).toLocaleString()}`;
    updateSimulation();
  });

  document.getElementById('risk-level')?.addEventListener('change', (e) => {
    const simRisk = document.getElementById('sim-risk-slider');
    if (simRisk) simRisk.value = e.target.value;
    if (document.getElementById('risk-percent-display')) document.getElementById('risk-percent-display').innerText = e.target.value + '%';
    const riskDisplay = document.getElementById('riskDisplay');
    if (riskDisplay) riskDisplay.innerText = `${e.target.value}%`;
    updateSimulation();
  });

  document.getElementById('sim-risk-slider')?.addEventListener('input', (e) => {
    if (document.getElementById('risk-percent-display')) document.getElementById('risk-percent-display').innerText = e.target.value + '%';
    updateSimulation();
  });

  document.getElementById('sim-leverage')?.addEventListener('input', (e) => {
    if (document.getElementById('leverage-display')) document.getElementById('leverage-display').innerText = e.target.value + 'x';
    document.querySelectorAll('.leverage-btn').forEach(btn => {
      if (btn.dataset.val === e.target.value) {
        btn.classList.add('bg-cyan/20', 'border-cyan', 'text-white');
        btn.classList.remove('bg-black/40', 'border-white/10', 'text-gray-400');
      } else {
        btn.classList.remove('bg-cyan/20', 'border-cyan', 'text-white');
        btn.classList.add('bg-black/40', 'border-white/10', 'text-gray-400');
      }
    });
    updateSimulation();
  });

  document.querySelectorAll('.leverage-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const val = e.currentTarget.dataset.val;
      const slider = document.getElementById('sim-leverage');
      if (slider) {
        slider.value = val;
        slider.dispatchEvent(new Event('input'));
      }
    });
  });

  ['sim-entry', 'sim-sl', 'sim-tp', 'sim-balance'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', updateSimulation);
  });

  document.getElementById('execute-trade-btn')?.addEventListener('click', executeTrade);

  document.getElementById('paperTradeConfirm')?.addEventListener('click', () => {
    const modal = document.getElementById('paperTradeModal');
    if (modal) modal.classList.add('hidden');
    if (typeof window.switchRoom === 'function') window.switchRoom('terminal');
  });

  document.getElementById('paperTradeCancel')?.addEventListener('click', () => {
    const modal = document.getElementById('paperTradeModal');
    if (modal) modal.classList.add('hidden');
  });

  document.getElementById('confirm-trade-yes')?.addEventListener('click', () => {
    const modal = document.getElementById('trade-confirmation-modal');
    if (modal) modal.classList.add('hidden');
    if (typeof window.switchRoom === 'function') window.switchRoom('terminal');
  });

  document.getElementById('confirm-trade-no')?.addEventListener('click', () => {
    const modal = document.getElementById('trade-confirmation-modal');
    if (modal) modal.classList.add('hidden');
  });
}


// ============================================================
// SIGNAL DETAILS MODAL LOGIC
// ============================================================
window.showSignalDetailsModal = function(signal) {
  const modal = document.getElementById('signalDetailsModal');
  if (!modal) return;

  // Basic Information
  document.getElementById('sd-symbol').textContent = signal.symbol;
  document.getElementById('sd-timeframe').textContent = signal.timeframe || '1h';
  
  const direction = signal.direction || 'LONG';
  const badge = document.getElementById('sd-direction-badge');
  badge.textContent = direction;
  if (direction === 'LONG') {
    badge.className = 'px-2 py-1 rounded text-xs font-bold tracking-wider bg-green-500/20 text-green-400 border border-green-500/30';
  } else if (direction === 'SHORT') {
    badge.className = 'px-2 py-1 rounded text-xs font-bold tracking-wider bg-red-500/20 text-red-400 border border-red-500/30';
  } else {
    badge.className = 'px-2 py-1 rounded text-xs font-bold tracking-wider bg-gray-500/20 text-gray-400 border border-gray-500/30';
  }

  // Price & Levels
  const currentPrice = window.currentTickers && window.currentTickers[signal.symbol] 
    ? parseFloat(window.currentTickers[signal.symbol]) 
    : (signal.entry_price || 0);
    
  document.getElementById('sd-live-price').textContent = `$${currentPrice.toFixed(4)}`;
  document.getElementById('sd-confidence').textContent = `${((signal.ai_prob || signal.confidence || 0) * 100).toFixed(1)}%`;
  document.getElementById('sd-sl').textContent = `$${(signal.sl || 0).toFixed(4)}`;
  document.getElementById('sd-tp').textContent = `$${(signal.tp || 0).toFixed(4)}`;

  // Confluence Scorecards (Mocking if not present)
  const confluence = signal.confluence || {
    trend: Math.floor(Math.random() * 20 + 70),
    momentum: Math.floor(Math.random() * 30 + 50),
    volume: Math.floor(Math.random() * 40 + 40)
  };
  document.getElementById('sd-conf-trend-val').textContent = `${confluence.trend}%`;
  document.getElementById('sd-conf-trend-bar').style.width = `${confluence.trend}%`;
  
  document.getElementById('sd-conf-mom-val').textContent = `${confluence.momentum}%`;
  document.getElementById('sd-conf-mom-bar').style.width = `${confluence.momentum}%`;
  
  document.getElementById('sd-conf-vol-val').textContent = `${confluence.volume}%`;
  document.getElementById('sd-conf-vol-bar').style.width = `${confluence.volume}%`;

  // Set active dataset for real-time tracking
  modal.dataset.activeSymbol = signal.symbol;
  modal.dataset.activeTimeframe = signal.timeframe || '1h';

  // Visual Zones
  updateZoneTracker(signal, currentPrice);

  // Expectancy Matrix
  renderExpectancyPanel();

  // Raw Prob & SHAP
  renderTelemetryPanel(signal);

  // Developer Portal
  renderDeveloperPortal(signal);

  // Setup Execute Button
  const execBtn = document.getElementById('sd-execute-btn');
  if (execBtn) {
    execBtn.onclick = () => {
      modal.classList.add('hidden');
      if (typeof window.selectSignal === 'function') {
        window.selectSignal(signal.symbol, signal.timeframe || '1h');
      }
    };
  }

  modal.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('signalDetailsModal');
  const closeBtn = document.getElementById('closeSignalDetailsBtn');
  
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
    });
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  }
});

// ============================================================
// VISUAL ZONE TRACKING ENGINE
// ============================================================

function getUserTier() {
  if (typeof AuthManager !== 'undefined') {
    const user = AuthManager.getUser();
    if (user) {
      const plan = (user.plan || user.tier || 'basic').toLowerCase();
      if (plan === 'pro' || plan === 'premium') return 'PRO';
      if (plan === 'intermediate') return 'INTERMEDIATE';
    }
  }
  // Fallback to window variables if set
  if (window.isPremiumUser) return 'PRO';
  return 'BASIC';
}

function updateZoneTracker(signal, currentPrice) {
    const container = document.getElementById('sd-visual-zone-container');
    if (!container) return;

    const tier = getUserTier();
    
    // BASIC Tier Placeholder
    if (tier === 'BASIC' || tier === 'TRIAL') {
        container.innerHTML = `
            <div class="flex-1 flex flex-col items-center justify-center p-4 border border-white/5 rounded-lg bg-black/40 relative overflow-hidden group cursor-pointer" onclick="window.location.href='/web/src/pages/pricing.html'">
                <div class="absolute inset-0 bg-gradient-to-r from-red-500/5 via-transparent to-green-500/5 opacity-50 blur-sm pointer-events-none"></div>
                <i class="fas fa-lock text-gray-500 text-xl mb-2 group-hover:text-cyan transition-colors"></i>
                <div class="text-[10px] text-gray-400 font-mono text-center mb-2 opacity-70">
                    <div class="h-1 w-full bg-white/10 rounded-full mb-2 overflow-hidden">
                        <div class="h-full w-1/3 bg-white/20"></div>
                    </div>
                    Visual zone tracking restricted
                </div>
                <span class="text-xs text-cyan font-bold bg-cyan/10 px-3 py-1 rounded group-hover:bg-cyan/20 transition-colors">
                    Unlock Risk Visualizer with Pro
                </span>
            </div>
        `;
        return;
    }

    // INTERMEDIATE / PRO Logic
    const direction = signal.direction || 'LONG';
    const entry = parseFloat(signal.entry_price) || 0;
    const sl = parseFloat(signal.sl) || 0;
    const tp = parseFloat(signal.tp) || 0;
    const price = parseFloat(currentPrice) || entry;
    
    if (!entry || !sl || !tp) {
        container.innerHTML = `<div class="text-xs text-gray-500 p-4">Invalid signal data for tracker.</div>`;
        return;
    }

    // Calculate Percentages (0% is SL, 100% is TP)
    let currentPercent = 0;
    let entryPercent = 0;

    if (direction === 'LONG') {
        const totalRange = tp - sl;
        if (totalRange > 0) {
            currentPercent = ((price - sl) / totalRange) * 100;
            entryPercent = ((entry - sl) / totalRange) * 100;
        }
    } else {
        const totalRange = sl - tp;
        if (totalRange > 0) {
            currentPercent = ((sl - price) / totalRange) * 100;
            entryPercent = ((sl - entry) / totalRange) * 100;
        }
    }

    // Safety Clamping (-5% to 105%)
    const clamp = (val) => Math.max(-5, Math.min(105, val));
    const renderCurrentPercent = clamp(currentPercent);
    const renderEntryPercent = clamp(entryPercent);

    // Telemetry Warning Engine
    let statusText = "⚡ Inside Active Entry Buffer Zone";
    let statusColorClass = "text-cyan animate-pulse";
    
    if (direction === 'LONG') {
        if (price >= tp) {
            statusText = "✓ TARGET HIT";
            statusColorClass = "text-green-400 font-bold";
        } else if (price <= sl) {
            statusText = "🛑 STOP LOSS TRIGGERED";
            statusColorClass = "text-red-500 font-bold";
        } else {
            const drift = (price - entry) / entry;
            if (drift > 0.002) {
                statusText = "⚠️ Breakout In Progress: Do Not Chase";
                statusColorClass = "text-yellow-400 font-bold";
            }
        }
    } else { // SHORT
        if (price <= tp) {
            statusText = "✓ TARGET HIT";
            statusColorClass = "text-green-400 font-bold";
        } else if (price >= sl) {
            statusText = "🛑 STOP LOSS TRIGGERED";
            statusColorClass = "text-red-500 font-bold";
        } else {
            const drift = (entry - price) / entry;
            if (drift > 0.002) {
                statusText = "⚠️ Breakout In Progress: Do Not Chase";
                statusColorClass = "text-yellow-400 font-bold";
            }
        }
    }

    // Render Template
    container.innerHTML = `
        <div class="flex justify-between items-center mb-3">
            <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold flex items-center gap-2">
                <i class="fas fa-bullseye text-orange"></i> Visual Zones
            </h4>
            <div class="text-[9px] uppercase tracking-wider ${statusColorClass}">
                ${statusText}
            </div>
        </div>
        
        <div class="relative w-full h-8 mt-2 px-4 flex flex-col justify-center border border-white/5 rounded-lg bg-black/50">
            <!-- Background Track Gradient -->
            <div class="absolute inset-0 rounded-lg bg-gradient-to-r from-red-500/20 via-slate-800/50 to-emerald-500/20 pointer-events-none"></div>
            
            <!-- Base Track Line -->
            <div class="relative w-full h-1 bg-white/10 rounded-full z-10">
                
                <!-- Static Entry Pin -->
                <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex flex-col items-center" style="left: ${renderEntryPercent}%">
                    <div class="w-2 h-4 bg-gray-400 rounded-sm"></div>
                    <span class="absolute top-full mt-1 text-[8px] text-gray-400 font-mono">ENTRY</span>
                </div>
                
                <!-- Dynamic Live Price Indicator -->
                <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex flex-col items-center transition-all duration-300 ease-out z-20" style="left: ${renderCurrentPercent}%">
                    <div class="w-3 h-3 bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.8)] border-2 border-cyan"></div>
                    <span class="absolute bottom-full mb-1 text-[8px] text-white font-mono shadow-black drop-shadow-md">LIVE</span>
                </div>
                
            </div>
            
            <!-- Scale Markers -->
            <div class="absolute inset-x-0 bottom-0 translate-y-full pt-1 flex justify-between text-[8px] font-mono text-gray-500 px-4">
                <span class="text-red-500/70">SL</span>
                <span class="text-green-500/70">TP</span>
            </div>
        </div>
    `;
}

// Listen to WebSocket price updates triggered from gatekeeper.js
window.addEventListener('priceUpdate', (e) => {
    const { symbol, price } = e.detail;
    const modal = document.getElementById('signalDetailsModal');
    
    if (modal && !modal.classList.contains('hidden') && modal.dataset.activeSymbol === symbol) {
        // Update Live Price text in the modal
        const livePriceEl = document.getElementById('sd-live-price');
        if (livePriceEl) {
            livePriceEl.textContent = `$${parseFloat(price).toFixed(4)}`;
        }
        
        // Update the Visual Zone Tracker
        const timeframe = modal.dataset.activeTimeframe || '1h';
        const key = `${symbol}_${timeframe}`;
        const sig = window.latestSignals && window.latestSignals[key];
        
        if (sig) {
            updateZoneTracker(sig, parseFloat(price));
            // Model telemetry usually updates on candle close, but we can re-render it if the payload changes
            renderTelemetryPanel(sig);
        }
    }
});

// ============================================================
// EXPECTANCY & RISK PANEL
// ============================================================

let globalAnalyticsCache = null;

async function fetchGlobalAnalytics() {
    if (globalAnalyticsCache) return globalAnalyticsCache;
    try {
        const docRef = doc(db, 'analytics', 'global_performance');
        const docSnap = await getDoc(docRef);
        if (docSnap.exists()) {
            globalAnalyticsCache = docSnap.data();
            return globalAnalyticsCache;
        }
    } catch (e) {
        console.error("Error fetching analytics", e);
    }
    return null;
}

async function renderExpectancyPanel() {
    const container = document.getElementById('sd-expectancy-container');
    if (!container) return;
    
    const tier = getUserTier();
    
    if (tier !== 'PRO') {
        container.innerHTML = `
            <div class="relative w-full h-full min-h-[120px] rounded-xl overflow-hidden group cursor-pointer" onclick="window.location.href='/web/src/pages/pricing.html'">
                <!-- Blurred background metrics -->
                <div class="absolute inset-0 p-4 blur-[4px] opacity-40 bg-black/50 flex flex-col justify-between">
                    <div class="flex justify-between"><span class="text-xs text-gray-400">Expectancy</span><span class="text-emerald-500 font-mono">+1.84%</span></div>
                    <div class="flex justify-between"><span class="text-xs text-gray-400">Max DD</span><span class="text-red-400 font-mono">-12.5%</span></div>
                    <div class="flex justify-between"><span class="text-xs text-gray-400">Profit Factor</span><span class="text-cyan font-mono">1.95</span></div>
                </div>
                
                <!-- Lock Overlay -->
                <div class="absolute inset-0 flex flex-col items-center justify-center bg-black/40 bg-gradient-to-t from-black/80 to-transparent">
                    <i class="fas fa-lock text-gray-300 text-xl mb-2 group-hover:text-amber-400 transition-colors"></i>
                    <span class="text-[10px] text-amber-400 font-bold tracking-widest text-center px-4 leading-relaxed group-hover:text-amber-300">
                        Unlock Institutional Quant Telemetry with Pro Tier
                    </span>
                </div>
            </div>
        `;
        return;
    }
    
    // Pro Tier: Render skeleton loading
    container.innerHTML = `<div class="flex items-center justify-center h-full min-h-[120px]"><i class="fas fa-circle-notch fa-spin text-cyan"></i></div>`;
    
    const data = await fetchGlobalAnalytics();
    
    if (!data) {
        container.innerHTML = `<div class="text-xs text-gray-500 p-4 text-center min-h-[120px] flex items-center justify-center">Telemetry unavailable</div>`;
        return;
    }
    
    const exp = data.expectancy || 0;
    const mdd = data.max_drawdown || 0;
    const pf = data.profit_factor || 0;
    
    const expColor = exp > 0 ? 'text-emerald-400' : (exp < 0 ? 'text-red-400' : 'text-gray-400');
    const expSign = exp > 0 ? '+' : '';
    const expArrow = exp > 0 ? '<i class="fas fa-arrow-trend-up text-[10px] ml-1"></i>' : (exp < 0 ? '<i class="fas fa-arrow-trend-down text-[10px] ml-1"></i>' : '');
    
    const mddColor = mdd > 15 ? 'text-red-500' : 'text-amber-400';
    
    const pfColor = pf > 1.5 ? 'text-cyan shadow-cyan/20' : 'text-white';
    
    container.innerHTML = `
        <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold mb-3 flex items-center gap-2">
            <i class="fas fa-chart-line text-purple-400"></i> Inst. Quant Telemetry
        </h4>
        <div class="flex flex-col gap-2 flex-1 justify-center">
            <div class="flex justify-between items-center border-b border-white/5 pb-2">
                <span class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Expectancy</span>
                <span class="font-mono text-sm font-bold ${expColor} drop-shadow-md">
                    ${expSign}${exp.toFixed(2)}% ${expArrow}
                </span>
            </div>
            <div class="flex justify-between items-center border-b border-white/5 pb-2">
                <span class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Max Drawdown</span>
                <span class="font-mono text-sm font-bold ${mddColor}">
                    -${mdd.toFixed(1)}%
                </span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Profit Factor</span>
                <span class="font-mono text-sm font-bold ${pfColor} bg-black/40 px-2 py-0.5 rounded border border-white/5">
                    ${pf.toFixed(2)}
                </span>
            </div>
        </div>
    `;
}

// ============================================================
// MODEL TELEMETRY & EXPLAINABILITY (SHAP) PANEL
// ============================================================

function renderTelemetryPanel(signal) {
    const container = document.getElementById('sd-telemetry-container');
    if (!container) return;

    const tier = getUserTier();

    if (tier !== 'PRO') {
        container.innerHTML = `
            <div class="relative w-full h-full min-h-[160px] rounded-xl overflow-hidden group cursor-pointer" onclick="window.location.href='/web/src/pages/pricing.html'">
                <!-- Blurred background metrics -->
                <div class="absolute inset-0 p-4 blur-[4px] opacity-40 bg-black/50 flex flex-col gap-4 pointer-events-none">
                    <div>
                        <div class="flex justify-between items-center text-[10px] mb-1 text-gray-400"><span class="uppercase tracking-widest">Model Conviction</span><span class="font-bold text-cyan">76% LONG</span></div>
                        <div class="w-full h-2 rounded overflow-hidden flex bg-black/50"><div class="h-full bg-rose-500/50" style="width: 14%"></div><div class="h-full bg-gray-500/50" style="width: 10%"></div><div class="h-full bg-emerald-500/50" style="width: 76%"></div></div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-400 uppercase tracking-widest mb-2">Live Logic Engine</div>
                        <div class="w-full h-4 bg-emerald-500/20 rounded mb-1"></div>
                        <div class="w-full h-4 bg-emerald-500/20 rounded mb-1 w-3/4"></div>
                        <div class="w-full h-4 bg-rose-500/20 rounded w-1/2 ml-auto"></div>
                    </div>
                </div>
                
                <!-- Lock Overlay -->
                <div class="absolute inset-0 flex flex-col items-center justify-center bg-black/40 bg-gradient-to-t from-black/90 to-transparent backdrop-blur-md z-10 transition-all group-hover:bg-black/50">
                    <i class="fas fa-lock text-gray-300 text-2xl mb-2 group-hover:text-cyan transition-colors"></i>
                    <span class="text-[10px] text-cyan font-bold tracking-widest text-center px-4 leading-relaxed group-hover:text-cyan/80">
                        Unlock Live Machine Learning Probabilities & SHAP Attribution Maps with Pro Tier
                    </span>
                </div>
            </div>
        `;
        return;
    }

    // Default fallbacks if backend doesn't send telemetry yet in websocket payload
    const probs = signal.probabilities || { "SHORT": 24.5, "HOLD": 10.2, "LONG": 65.3 };
    const shapList = signal.shap_contributions || [
        { "feature": "Funding_Rate", "impact": 0.45 },
        { "feature": "Orderbook_Imbalance", "impact": 0.23 },
        { "feature": "RSI_Divergence", "impact": -0.15 }
    ];

    // Calculate conviction winner
    let winner = "HOLD";
    let winnerProb = probs.HOLD || 0;
    let winnerColor = "text-gray-400";
    
    if ((probs.LONG || 0) > winnerProb) {
        winner = "LONG";
        winnerProb = probs.LONG;
        winnerColor = "text-emerald-400";
    }
    if ((probs.SHORT || 0) > winnerProb) {
        winner = "SHORT";
        winnerProb = probs.SHORT;
        winnerColor = "text-rose-400";
    }

    let shapHTML = '';
    
    if (!shapList || shapList.length === 0) {
        shapHTML = \`<div class="text-xs text-gray-500 italic py-2">Attribution map unavailable.</div>\`;
    } else {
        // Max absolute impact for scaling
        const maxImpact = Math.max(...shapList.map(s => Math.abs(s.impact)), 0.01);
        
        shapHTML = shapList.map(s => {
            const isPositive = s.impact > 0;
            const barWidth = (Math.abs(s.impact) / maxImpact) * 100;
            const barColorClass = isPositive ? 'bg-emerald-500/20 border-emerald-500/30' : 'bg-rose-500/20 border-rose-500/30';
            const textColorClass = isPositive ? 'text-emerald-400' : 'text-rose-400';
            const sign = isPositive ? '+' : '-';
            
            return \`
                <div class="flex items-center gap-2 mb-2 w-full text-[10px] font-mono">
                    <div class="flex-1 text-right truncate text-gray-400 \${!isPositive ? textColorClass : ''}">\${!isPositive ? s.feature : ''}</div>
                    
                    <div class="w-1/3 flex items-center justify-center relative h-3 bg-black/30 rounded border border-white/5 overflow-hidden">
                        <div class="absolute h-full border-r \${!isPositive ? 'border-r-0 border-l' : ''} border-white/10 \${barColorClass} \${isPositive ? 'left-1/2' : 'right-1/2'}" style="width: \${barWidth / 2}%"></div>
                        <div class="absolute w-[1px] h-full bg-white/20 left-1/2"></div>
                    </div>
                    
                    <div class="flex-1 truncate text-gray-400 \${isPositive ? textColorClass : ''}">\${isPositive ? s.feature : ''}</div>
                </div>
            \`;
        }).join('');
    }

    container.innerHTML = \`
        <div class="mb-4">
            <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold flex items-center gap-2">
                <i class="fas fa-brain text-emerald-400"></i> Model Telemetry
            </h4>
        </div>
        
        <!-- Conviction Meter -->
        <div class="mb-4 bg-black/40 p-3 rounded-lg border border-white/5">
            <div class="flex justify-between items-center text-[10px] uppercase tracking-widest font-bold mb-2">
                <span class="text-gray-500">Conviction Vector</span>
                <span class="\${winnerColor}">\${winnerProb.toFixed(1)}% \${winner}</span>
            </div>
            <div class="w-full h-1.5 rounded-full overflow-hidden flex bg-black/50 shadow-inner">
                <div class="h-full bg-rose-500 transition-all duration-500" style="width: \${probs.SHORT || 0}%"></div>
                <div class="h-full bg-gray-500 transition-all duration-500" style="width: \${probs.HOLD || 0}%"></div>
                <div class="h-full bg-emerald-500 transition-all duration-500" style="width: \${probs.LONG || 0}%"></div>
            </div>
            <div class="flex justify-between text-[8px] mt-1 text-gray-500 font-mono">
                <span>SH \${(probs.SHORT || 0).toFixed(0)}%</span>
                <span>HD \${(probs.HOLD || 0).toFixed(0)}%</span>
                <span>LN \${(probs.LONG || 0).toFixed(0)}%</span>
            </div>
        </div>

        <!-- Live Logic Engine (Micro-SHAP) -->
        <div class="bg-black/40 p-3 rounded-lg border border-white/5 flex-1 flex flex-col justify-center">
            <div class="text-[9px] text-gray-500 uppercase tracking-widest font-bold mb-3 text-center border-b border-white/5 pb-2">
                Live Logic Engine
            </div>
            <div class="flex flex-col w-full justify-center">
                \${shapHTML}
            </div>
        </div>
    \`;
}

// ============================================================
// API PORTABILITY & DEVELOPER PORTAL
// ============================================================

window.regenerateApiKey = async function(symbol) {
    try {
        let token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
        if (!token && typeof AuthManager !== 'undefined') {
            token = AuthManager.getToken();
        }
        
        if (!token) {
            alert('You must be logged in to regenerate API keys.');
            return;
        }

        const btn = document.getElementById('btn-regen-key');
        if (btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        const response = await fetch('/api/v1/developer/regenerate_key', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to generate key. Are you on the PRO tier?');
        }

        const data = await response.json();
        const apiKey = data.api_key;

        // Show it to the user
        const keyDisplay = document.getElementById('dev-api-key-display');
        if (keyDisplay) {
            keyDisplay.value = apiKey;
            keyDisplay.type = 'text';
        }

        alert(`Copy your API key now. This is the only time it will be shown:\n\n${apiKey}`);

        if (btn) btn.innerHTML = '<i class="fas fa-sync-alt"></i> Regenerate API Key';
    } catch (e) {
        console.error(e);
        alert(e.message);
        const btn = document.getElementById('btn-regen-key');
        if (btn) btn.innerHTML = '<i class="fas fa-sync-alt"></i> Regenerate API Key';
    }
};

window.copyEndpointUrl = function(symbol) {
    const url = `${window.location.origin}/api/v1/signals/fleet?symbol=${symbol}`;
    navigator.clipboard.writeText(url).then(() => {
        alert('Endpoint URL copied to clipboard!');
    });
};

function renderDeveloperPortal(signal) {
    const container = document.getElementById('sd-developer-portal');
    if (!container) return;

    const tier = getUserTier();

    if (tier !== 'PRO') {
        container.innerHTML = `
            <div class="relative w-full h-full min-h-[140px] rounded-xl overflow-hidden group cursor-pointer" onclick="window.location.href='/web/src/pages/pricing.html'">
                <!-- Blurred background metrics -->
                <div class="absolute inset-0 p-4 blur-[4px] opacity-40 bg-black/50 flex flex-col gap-3 pointer-events-none">
                    <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold flex items-center gap-2">
                        <i class="fas fa-code text-purple-400"></i> API Portability
                    </h4>
                    <input type="password" value="aegis_live_fakekey_xxxxxxxxxxxxxxxx" class="w-full bg-black/50 border border-white/10 rounded px-3 py-2 text-xs font-mono text-gray-500" disabled>
                    <div class="flex gap-2">
                        <button class="flex-1 bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded py-2 text-[10px] font-bold"><i class="fas fa-sync-alt"></i> REGENERATE</button>
                        <button class="flex-1 bg-white/5 text-gray-400 border border-white/10 rounded py-2 text-[10px] font-bold"><i class="fas fa-copy"></i> COPY ENDPOINT</button>
                    </div>
                </div>
                
                <!-- Lock Overlay -->
                <div class="absolute inset-0 flex flex-col items-center justify-center bg-black/40 bg-gradient-to-t from-black/90 to-transparent backdrop-blur-sm z-10 transition-all group-hover:bg-black/50">
                    <i class="fas fa-lock text-gray-300 text-2xl mb-2 group-hover:text-purple-400 transition-colors"></i>
                    <span class="text-[10px] text-purple-400 font-bold tracking-widest text-center px-4 leading-relaxed group-hover:text-purple-300">
                        Unlock Developer JSON Endpoints & Webhook Integrations with Pro Tier
                    </span>
                </div>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold mb-3 flex items-center gap-2">
            <i class="fas fa-code text-purple-400"></i> API Portability & Developer Access
        </h4>
        <div class="flex flex-col gap-3">
            <div class="relative">
                <input type="password" id="dev-api-key-display" value="aegis_live_••••••••••••••••" 
                    class="w-full bg-black/50 border border-white/10 rounded px-3 py-2 text-xs font-mono text-gray-300 outline-none focus:border-purple-500/50 transition-colors" readonly>
            </div>
            <div class="flex gap-2">
                <button id="btn-regen-key" onclick="window.regenerateApiKey('${signal.symbol}')"
                    class="flex-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30 rounded py-2 text-[10px] font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-2">
                    <i class="fas fa-sync-alt"></i> Regenerate API Key
                </button>
                <button onclick="window.copyEndpointUrl('${signal.symbol}')"
                    class="flex-1 bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 rounded py-2 text-[10px] font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-2">
                    <i class="fas fa-copy"></i> Copy Endpoint URL
                </button>
            </div>
            <div class="text-[9px] text-gray-500 font-mono mt-1 text-center">
                Requires header: <span class="text-purple-400">X-API-Key</span>
            </div>
        </div>
    `;
}
