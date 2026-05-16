import { initializeTrialCountdown, fetchTrialStartFromFirestore } from './trial-countdown.js';
import { auth } from './gatekeeper.js';

let initialized = false;

// ============================================================
// WAIT FOR AUTH STATE CHANGE (replaces polling waitForUserId)
// ============================================================
function waitForAuthStateChange() {
  return new Promise((resolve) => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      unsubscribe(); // Unsubscribe after first state change
      resolve(user?.uid || null);
    });
  });
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
    if (!isExpiredElement) {
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

  // Restore token cards
  const tokenCards = document.querySelectorAll('[data-token-card], .token-card');
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


async function setupTrialNonBlocking(userId) {
  if (!userId) return;

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
      const countdownDisplay = document.getElementById('countdown-display');
      if (countdownDisplay) {
        countdownDisplay.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            Session Expired - Please log in again
          `;
        countdownDisplay.style.display = 'block';
        countdownDisplay.style.background = 'rgba(255, 0, 85, 0.15)';
        countdownDisplay.style.borderColor = 'rgba(255, 0, 85, 0.4)';
        countdownDisplay.style.color = '#ff0055';
      }
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
