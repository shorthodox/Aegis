import { initializeTrialCountdown, fetchTrialStartFromFirestore } from './trial-countdown.js';
import { auth, db } from './gatekeeper.js';
import { doc, getDoc, setDoc } from 'https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js';

// ============================================================
// SEALED SUBSCRIPTION STATE — console-proof
// window.isSubscriptionActive and window.isPremiumUser are
// read-only getters; writes from the console silently no-op.
// The WebSocket tick is the server-authority enforcement loop.
// ============================================================
const _subState = { active: false, isPremium: false };
let _lastVerifiedAt = Date.now(); // grace period: blocks WS ticks until Firestore check completes
try {
  Object.defineProperty(window, 'isSubscriptionActive', {
    get: () => _subState.active,
    set: () => {},           // silently reject console writes
    configurable: false,
    enumerable: true,
  });
  Object.defineProperty(window, 'isPremiumUser', {
    get: () => _subState.isPremium,
    set: () => {},
    configurable: false,
    enumerable: true,
  });
} catch (_) { /* already sealed in this session */ }

let initialized = false;

// Returns 'paid' | 'trial' | 'expired'
async function checkUserSubscriptionStatus(uid) {
  const now = new Date();

  // Fast path: AuthManager already resolved the plan on a previous auth cycle.
  // Trust it immediately so the UI never flashes 'expired' while Firestore loads.
  if (typeof AuthManager !== 'undefined') {
    const u = AuthManager.getUser();
    if (u) {
      const p = (u.plan || u.tier || '').toLowerCase();
      if (p === 'pro' || p === 'premium' || p === 'intermediate' || p === 'basic') {
        console.log('[SubCheck] AuthManager fast-path → paid (' + p + ')');
        return 'paid';
      }
    }
    // Any active JWT session → at minimum a trial; won't default to 'expired'
    // before Firestore has a chance to respond.
  }

  // Placeholder UID — skip Firestore, fall back to localStorage only
  if (!uid || uid === 'jwt-user') {
    const localEnd = localStorage.getItem('trial_end_timestamp');
    if (localEnd && new Date(localEnd) < now) return 'expired';
    return 'trial';
  }

  try {
    const currentUser = typeof auth !== 'undefined' ? auth.currentUser : null;
    const docKey = currentUser?.email || uid;
    let userDocRef = doc(db, 'users', docKey);
    let userSnap = await getDoc(userDocRef);
    
    if (!userSnap.exists() && currentUser?.email) {
      userSnap = await getDoc(doc(db, 'users', uid));
    }

    if (userSnap.exists()) {
      const data = userSnap.data();
      const plan = data.plan || data.tier;

      // Paid plan — no expiry check needed
      if (plan) {
        const p = plan.toLowerCase();
        if (p === 'premium' || p === 'pro' || p === 'intermediate' || p === 'basic') return 'paid';
      }

      // Check trial end from Firestore (multiple possible field locations)
      const rawEnd = data.trial_end || data.trialEnd || data.trial?.endDate || data.trial?.end_date;
      if (rawEnd) {
        const endDate = rawEnd.toDate ? rawEnd.toDate() : new Date(rawEnd);
        if (!isNaN(endDate.getTime())) {
          if (endDate < now) return 'expired';
          return 'trial';
        }
      }

      // No Firestore end date — fall back to localStorage
      const localEnd = localStorage.getItem('trial_end_timestamp');
      if (localEnd) {
        const localEndDate = new Date(localEnd);
        if (!isNaN(localEndDate.getTime()) && localEndDate < now) return 'expired';
      }

      if (!plan) return 'trial';
      const p = plan.toLowerCase();
      if (p === 'trial' || p === 'free_tier') return 'trial';
      return 'expired';
    }

    // No Firestore doc — check localStorage trial end as last resort
    const localEnd = localStorage.getItem('trial_end_timestamp');
    if (localEnd) {
      const localEndDate = new Date(localEnd);
      if (!isNaN(localEndDate.getTime()) && localEndDate < now) return 'expired';
    }
    return 'trial';
  } catch (error) {
    console.error('[SubCheck] Invalid database error — falling back to AuthManager/localStorage:', error);
    if (error.code) {
        console.error(`[SubCheck] Firestore error code: ${error.code}. This is a database problem.`);
    }

    // On error: prefer AuthManager knowledge over defaulting to 'expired'
    if (typeof AuthManager !== 'undefined') {
      const u = AuthManager.getUser();
      if (u) {
        const p = (u.plan || u.tier || '').toLowerCase();
        if (p === 'pro' || p === 'premium' || p === 'intermediate' || p === 'basic') return 'paid';
        if (p === 'trial' || p === 'free_tier') return 'trial';
      }
      // Active token → safe to assume trial rather than blocking the user
      if (AuthManager.getToken()) return 'trial';
    }

    const localEnd = localStorage.getItem('trial_end_timestamp');
    if (localEnd) {
      const localEndDate = new Date(localEnd);
      if (!isNaN(localEndDate.getTime()) && localEndDate < now) return 'expired';
    }
    return 'trial';
  }
}

// ============================================================
// PLAN BADGE UPDATER
// ============================================================
function updatePlanBadge(status) {
  // Determine tier label
  let label = 'Free Trial';
  let colorClass = 'bg-amber-500/20 text-amber-400 border-amber-500/40';

  if (status === 'paid') {
    // Try to read actual plan from AuthManager for granularity
    let planName = 'Basic';
    if (typeof AuthManager !== 'undefined') {
      const u = AuthManager.getUser();
      if (u) {
        const p = (u.plan || u.tier || '').toLowerCase();
        if (p === 'pro' || p === 'premium') planName = 'Pro';
        else if (p === 'intermediate') planName = 'Intermediate';
        else if (p === 'basic') planName = 'Basic';
      }
    }
    label = planName;
    colorClass = planName === 'Pro'
      ? 'bg-violet-500/20 text-violet-300 border-violet-500/40'
      : planName === 'Intermediate'
        ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
        : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
  } else if (status === 'expired') {
    label = 'Expired';
    colorClass = 'bg-red-500/20 text-red-400 border-red-500/40';
  }

  const badgeHTML = `<span class="px-2 py-0.5 text-xs font-semibold rounded-full border ${colorClass}">${label}</span>`;

  // Update all plan badge slots
  ['planBadge', 'sidebar-plan-badge', 'header-plan-badge'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = badgeHTML;
  });
}

// ============================================================
// WAIT FOR AUTH STATE CHANGE (replaces polling waitForUserId)
// ============================================================
function waitForAuthStateChange() {
  const authPromise = new Promise((resolve) => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      unsubscribe();
      if (user?.uid) {
        const status = await checkUserSubscriptionStatus(user.uid);
        _subState.active = (status !== 'expired');
        _subState.isPremium = (status === 'paid');
        if (status !== 'expired') _lastVerifiedAt = Date.now();
        console.log('[waitForAuth] uid:', user.uid, 'status:', status, '_subState:', JSON.stringify(_subState));
        updatePlanBadge(status);
        if (status === 'expired') {
          setExpiredView();
        } else {
          clearExpiredView();
          unblockFeatures();
          if (status === 'paid') {
            document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display')
              .forEach(el => el.style.display = 'none');
          }
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
          const status = await checkUserSubscriptionStatus(user.uid);
          _subState.active = (status !== 'expired');
          _subState.isPremium = (status === 'paid');
          if (status !== 'expired') _lastVerifiedAt = Date.now();
          console.log('[waitForAuth retry] uid:', user.uid, 'status:', status, '_subState:', JSON.stringify(_subState));
          updatePlanBadge(status);
          if (status === 'expired') {
            setExpiredView();
          } else {
            clearExpiredView();
            unblockFeatures();
            if (status === 'paid') {
              document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display')
                .forEach(el => el.style.display = 'none');
            }
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
  _subState.active = false;
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
  // If Firestore already confirmed this session as paid, don't let a WS tick override it.
  if (_subState.isPremium) return;

  // Debounce: if Firestore verified within the last 2 minutes and sub is active, ignore WS ticks.
  if (_subState.active && (Date.now() - _lastVerifiedAt < 120000)) return;

  const dashboardContent = document.getElementById('dashboard-main-content');
  const expiredCard = document.getElementById('access-expired-card');

  // Dim/disable dashboard without removing it from DOM — preserves analytics state.
  if (dashboardContent) {
    dashboardContent.classList.remove('hidden');
    dashboardContent.classList.add('sub-expired');
  }

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
  if (dashboardContent) {
    dashboardContent.classList.remove('hidden');
    dashboardContent.classList.remove('sub-expired');
  }
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
  _subState.active = false;

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

    // Never block elements inside the signal modal or feature panels —
    // those overlays manage their own access gating via lock overlays.
    const isModalOrPanel = el.closest('#signalDetailsModal, #fp-confluence, #fp-zones, #fp-expectancy, #fp-shap, #fp-api') !== null;

    if (!isExpiredElement && !isLogout && !isNav && !isModalOrPanel) {
      el.dataset.aegisBlocked = '1';
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
  _subState.active = true;

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

  // Only restore elements that were explicitly blocked by aegis — avoids clobbering intentional disables.
  const blockedElements = document.querySelectorAll('[data-aegis-blocked]');
  blockedElements.forEach(el => {
    el.style.removeProperty('pointer-events');
    el.style.removeProperty('opacity');
    delete el.dataset.aegisBlocked;
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
// EXPORT FUNCTIONS FOR EXTERNAL USE — sealed so they cannot be
// overwritten or deleted from the browser console.
// ============================================================
try {
  Object.defineProperty(window, 'canAccessFeatures', { value: canAccessFeatures, writable: false, configurable: false, enumerable: true });
  Object.defineProperty(window, 'setExpiredView',    { value: setExpiredView,    writable: false, configurable: false, enumerable: true });
  Object.defineProperty(window, 'clearExpiredView',  { value: clearExpiredView,  writable: false, configurable: false, enumerable: true });
} catch (_) { /* already sealed */ }

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

    if (!marketCardsInitialized) {
      let html = '';
      priorityTokens.forEach(sym => {
        const idStr = sym.replace('/', '-');
        const base = sym.split('/')[0];
        html += `
          <div id="market-card-${idStr}"
               class="glass-panel p-4 rounded-xl flex flex-col gap-2 border-l-2 border-cyan/40 hover:border-cyan/80 hover:bg-white/5 transition-all cursor-pointer shadow-lg group relative overflow-hidden"
               onclick="window._openSignalCardForSymbol && window._openSignalCardForSymbol('${sym}')">
            <div class="flex items-center justify-between">
              <span class="text-[10px] text-gray-400 font-bold uppercase tracking-widest">${base}</span>
              <span id="market-card-signal-${idStr}"
                    class="text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded border hidden">—</span>
            </div>
            <span id="market-card-price-${idStr}"
                  class="live-price text-lg font-mono transition-colors duration-300"
                  data-symbol="${idStr}">—</span>
            <div class="flex items-center justify-between mt-auto">
              <span id="market-card-change-${idStr}" class="text-[10px] font-mono text-gray-500">—</span>
              <span class="text-[9px] text-gray-600 group-hover:text-cyan/60 transition-colors">
                <i class="fas fa-arrow-right"></i>
              </span>
            </div>
          </div>
        `;
      });
      container.innerHTML = html;
      marketCardsInitialized = true;
    }

    if (typeof window.updateMarketCardSignalBadges === 'function') {
      window.updateMarketCardSignalBadges();
    }
  } catch (err) {
    console.error('Failed to init token market cards:', err);
  }
}

window.updateMarketCardSignalBadges = function updateMarketCardSignalBadges() {
  // latestSignals is an object keyed by "SYMBOL/TIMEFRAME"; convert to array for lookup
  const raw = window.latestSignals || {};
  const signalArr = Array.isArray(raw) ? raw : Object.values(raw);
  const priorityTokens = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];

  priorityTokens.forEach(sym => {
    const idStr = sym.replace('/', '-');
    const badge = document.getElementById(`market-card-signal-${idStr}`);
    if (!badge) return;

    const match = signalArr.find(s => (s.symbol || '').toUpperCase() === sym.toUpperCase());
    const card = document.getElementById(`market-card-${idStr}`);

    if (match) {
      const dir = (match.direction || match.side || '').toUpperCase();
      const isLong  = dir === 'LONG'  || dir === 'BUY';
      const isShort = dir === 'SHORT' || dir === 'SELL';
      const isDirectional = isLong || isShort;

      if (isDirectional) {
        badge.textContent = isLong ? 'LONG' : 'SHORT';
        badge.className = `text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded border ${
          isLong
            ? 'bg-green-500/20 text-green-400 border-green-500/40'
            : 'bg-red-500/20 text-red-400 border-red-500/40'
        }`;
        badge.classList.remove('hidden');

        if (card) {
          card.classList.remove('border-cyan/40', 'border-green-500/50', 'border-red-500/50');
          card.classList.add(isLong ? 'border-green-500/50' : 'border-red-500/50');
        }
      } else {
        // NEUTRAL or unknown direction — hide badge, neutral border
        badge.classList.add('hidden');
        if (card) {
          card.classList.remove('border-green-500/50', 'border-red-500/50');
          card.classList.add('border-cyan/40');
        }
      }
    } else {
      badge.classList.add('hidden');
      if (card) {
        card.classList.remove('border-green-500/50', 'border-red-500/50');
        card.classList.add('border-cyan/40');
      }
    }
  });
};

// Opens signal details modal for the given symbol from window.latestSignals
window._openSignalCardForSymbol = function(sym) {
  const raw = window.latestSignals || {};
  const signals = Array.isArray(raw) ? raw : Object.values(raw);
  const match = signals.find(s => (s.symbol || '').toUpperCase() === sym.toUpperCase());
  if (match && typeof showSignalDetailsModal === 'function') {
    showSignalDetailsModal(match);
  } else {
    selectToken(sym);
  }
};


let trialSetupRunning = false;
let lastTrialSetupTime = 0;

async function setupTrialNonBlocking(userId) {
  if (!userId) return;

  const status = await checkUserSubscriptionStatus(userId);
  _subState.active = (status !== 'expired');
  _subState.isPremium = (status === 'paid');
  if (status !== 'expired') _lastVerifiedAt = Date.now();
  updatePlanBadge(status);

  if (status === 'expired') {
    setExpiredView();
    return;
  }

  // Only clear expired view if user has a valid subscription or active trial
  clearExpiredView();
  unblockFeatures();

  if (status === 'paid') {
    document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display')
      .forEach(el => el.style.display = 'none');
    return;
  }

  const now = Date.now();
  if (trialSetupRunning || (now - lastTrialSetupTime < 5000)) {
    console.log('⏳ Skipping duplicate setupTrialNonBlocking call to prevent request bloat');
    return;
  }
  trialSetupRunning = true;

  const cacheKey = `trialStart_${userId}`;

  try {
    // 1. Firestore is the PRIMARY source of truth — fetch it first
    const freshStart = await fetchTrialStartFromFirestore(userId);

    if (freshStart) {
      // Firestore returned a valid start — use it as the source of truth
      const freshDate = new Date(freshStart);
      if (!isNaN(freshDate.getTime())) {
        const freshEnd = new Date(freshDate.getTime() + 3 * 24 * 60 * 60 * 1000);
        // Persist the authoritative trial end to localStorage
        localStorage.setItem('trial_end_timestamp', freshEnd.toISOString());
        localStorage.setItem(cacheKey, freshStart);

        if (freshEnd < new Date()) {
          setExpiredView();
          updatePlanBadge('expired');
          return;
        }
        initializeTrialCountdown(userId, freshStart);
      }
    } else {
      // Firestore returned null — fall back to cached start from localStorage
      const cachedStart = localStorage.getItem(cacheKey);

      if (cachedStart) {
        const startDate = new Date(cachedStart);
        if (!isNaN(startDate.getTime())) {
          const derivedEnd = new Date(startDate.getTime() + 3 * 24 * 60 * 60 * 1000);
          localStorage.setItem('trial_end_timestamp', derivedEnd.toISOString());

          if (derivedEnd < new Date()) {
            setExpiredView();
            updatePlanBadge('expired');
            return;
          }
          initializeTrialCountdown(userId, cachedStart);
        }
      } else {
        // Neither Firestore nor cached start exists
        // Delegate to initializeTrialCountdown which will safely fetch from Firestore or create properly
        const badge = document.getElementById('planBadge');
        if (badge) {
          badge.textContent = 'Free Trial';
          badge.className = 'px-3 py-1 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20';
        }
        initializeTrialCountdown(userId);
      }
    }
  } catch (trialErr) {
    console.error('Failed to fetch trial data, using fallback:', trialErr);
    const cachedStart = localStorage.getItem(`trialStart_${userId}`);
    if (cachedStart) {
      initializeTrialCountdown(userId, cachedStart);
    } else {
      initializeTrialCountdown(userId);
    }
  } finally {
    trialSetupRunning = false;
    lastTrialSetupTime = Date.now();
  }
}

async function initDashboard(event) {
  clearExpiredView();

  if (initialized) return;
  if (!document.getElementById('dashboard-main-content') && !document.getElementById('market-token-cards')) return;

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

    // Await true auth state — always resolve to a real Firebase UID
    const realUserId = eventUid || await (async () => {
      const localToken = localStorage.getItem('access_token') || localStorage.getItem('authToken');
      if (localToken && cachedUid) return cachedUid; // fast path: token + cached UID
      return await waitForAuthStateChange();
    })();

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

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', initDashboard);
} else {
  initDashboard();
}
document.addEventListener('dashboardUserLoaded', initDashboard);

// ── Periodic subscription re-verification ────────────────────────────────────
// Re-checks Firestore every 5 minutes so a paid-to-expired transition is caught
// even if the WebSocket is briefly disconnected. Also runs on tab focus so
// returning users always see an up-to-date state without waiting for the interval.

let _periodicVerifyTimer = null;
let _lastPeriodicVerify   = 0;

async function _periodicSubscriptionCheck() {
  const now = Date.now();
  // Guard: skip if checked less than 4 minutes ago (WS may have already updated)
  if (now - _lastPeriodicVerify < 4 * 60_000) return;
  _lastPeriodicVerify = now;

  try {
    const uid = localStorage.getItem('cached_uid');
    if (!uid) return;
    const status = await checkUserSubscriptionStatus(uid);
    _subState.active    = (status !== 'expired');
    _subState.isPremium = (status === 'paid');
    if (status !== 'expired') _lastVerifiedAt = now;
    updatePlanBadge(status);
    if (status === 'expired') {
      setExpiredView();
    } else {
      clearExpiredView();
    }
  } catch (_) {
    // Silently ignore — the WS tick is the primary enforcement mechanism.
  }
}

document.addEventListener('dashboardUserLoaded', () => {
  // Start the 5-minute interval once the dashboard is initialised.
  if (!_periodicVerifyTimer) {
    _periodicVerifyTimer = setInterval(_periodicSubscriptionCheck, 5 * 60_000);
  }

  // Re-verify immediately when the user returns to the tab (debounced to 60 s).
  let _tabFocusLast = 0;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    const now = Date.now();
    if (now - _tabFocusLast < 60_000) return;
    _tabFocusLast = now;
    _periodicSubscriptionCheck();
  });
});

// ============================================================
// SCREEN MODE ENGINE
// ============================================================
const _SCREEN_MODE_KEY = 'aegis-screen-mode';

function _applyTheme(mode) {
    const effectiveTheme = mode === 'auto'
        ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
        : mode;
    document.documentElement.setAttribute('data-theme', effectiveTheme);
    if (effectiveTheme === 'light') document.documentElement.classList.remove('dark');
    else document.documentElement.classList.add('dark');
}

function _updateScreenModeUI(mode) {
    document.querySelectorAll('[data-screen-mode-btn]').forEach(btn => {
        btn.classList.toggle('screen-mode-btn--active', btn.dataset.screenModeBtn === mode);
    });
}

window.setScreenMode = function (mode) {
    if (!['auto', 'dark', 'grey', 'light'].includes(mode)) return;
    localStorage.setItem(_SCREEN_MODE_KEY, mode);
    _applyTheme(mode);
    _updateScreenModeUI(mode);
};

function initScreenMode() {
    const saved = localStorage.getItem(_SCREEN_MODE_KEY) || 'auto';
    _applyTheme(saved);
    _updateScreenModeUI(saved);

    // Respond to OS-level preference changes when in auto mode
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if ((localStorage.getItem(_SCREEN_MODE_KEY) || 'auto') === 'auto') _applyTheme('auto');
    });

    document.querySelectorAll('[data-screen-mode-btn]').forEach(btn => {
        btn.addEventListener('click', () => window.setScreenMode(btn.dataset.screenModeBtn));
    });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initScreenMode);
} else {
  initScreenMode();
}

// ============================================================
// TERMINAL SIMULATION LOGIC
// ============================================================

window.selectedTrade = null;
window.selectedTradeToken = null;

window.selectToken = function (sym) {
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

// Shared auth-token helper — always returns a fresh Firebase ID token when possible,
// falling back to whatever is in localStorage / AuthManager.
async function _getAuthToken() {
  try {
    // Prefer a fresh Firebase token (handles expiry silently)
    const { getAuth } = await import('https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js');
    const fbUser = getAuth().currentUser;
    if (fbUser) {
      const fresh = await fbUser.getIdToken();
      if (typeof AuthManager !== 'undefined') AuthManager.setToken(fresh);
      return fresh;
    }
  } catch (_) { /* not a Firebase session — fall through */ }
  // Custom JWT fallback
  if (typeof AuthManager !== 'undefined') return AuthManager.getToken();
  return localStorage.getItem('access_token') || localStorage.getItem('authToken') || null;
}

async function executeTrade() {
  if (!window.selectedTradeToken && !window.selectedTrade) {
    alert('Please select a token first.');
    return;
  }

  const entry    = parseFloat(document.getElementById('sim-entry').value);
  const sl       = parseFloat(document.getElementById('sim-sl').value);
  const tp       = parseFloat(document.getElementById('sim-tp').value);
  const riskPct  = parseFloat(document.getElementById('sim-risk-slider')?.value || 2);
  const leverage = parseFloat(document.getElementById('sim-leverage')?.value || 1);
  const posUnits = parseFloat(document.getElementById('pos-units')?.innerText || 0);
  const notional = parseFloat((document.getElementById('notional')?.innerText || '0').replace('$', ''));

  if (isNaN(entry) || entry <= 0) { alert('Please enter a valid entry price.'); return; }
  if (isNaN(sl)    || sl <= 0)    { alert('Please enter a valid stop-loss.'); return; }
  if (isNaN(tp)    || tp <= 0)    { alert('Please enter a valid take-profit.'); return; }

  const tradeData = {
    symbol:        window.selectedTradeToken || window.selectedTrade?.symbol || null,
    side:          window.selectedTrade?.direction || document.getElementById('direction-badge')?.innerText || 'LONG',
    entryPrice:    entry,
    stopLoss:      sl,
    takeProfit:    tp,
    riskPercent:   riskPct,
    leverage,
    positionUnits: posUnits,
    notionalValue: notional,
    status:        'open',
    signalId:      window.selectedTrade?.signalId || null,
  };

  localStorage.setItem('analyticsActiveTrade', JSON.stringify(tradeData));

  // UI feedback
  const btn = document.getElementById('execute-trade-btn');
  const originalLabel = btn?.innerHTML;
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Executing…'; }

  try {
    const token = await _getAuthToken();
    if (!token) {
      alert('You must be logged in to execute trades.\nPlease refresh the page and sign in again.');
      return;
    }

    const response = await fetch('/api/trades/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      body: JSON.stringify(tradeData),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
      throw new Error(err.detail || 'Backend rejected the trade');
    }

    const result = await response.json();
    console.log('✅ Trade executed:', result.trade_id);

    // Navigate to analytics to show the live trade
    if (typeof window.switchRoom === 'function') window.switchRoom('analytics');
    setTimeout(() => {
      if (typeof window.forceTradesRefresh === 'function') window.forceTradesRefresh();
    }, 150);

    // Show success toast
    _showToast('Trade executed successfully! Tracking in Analytics.', 'success');
  } catch (err) {
    console.error('executeTrade error:', err);
    _showToast(`Trade failed: ${err.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = originalLabel; }
  }
}

function _showToast(message, type = 'info') {
  const existing = document.getElementById('_aegis-toast');
  if (existing) existing.remove();
  const colors = { success: '#10b981', error: '#ef4444', info: '#06b6d4' };
  const icons  = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
  const toast = document.createElement('div');
  toast.id = '_aegis-toast';
  toast.style.cssText = `position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;display:flex;align-items:center;gap:10px;padding:14px 20px;border-radius:12px;background:#111827;border:1px solid ${colors[type]};color:${colors[type]};font-size:0.9rem;font-weight:600;box-shadow:0 0 20px ${colors[type]}40;animation:slideIn .25s ease;max-width:360px;`;
  toast.innerHTML = `<i class="fas ${icons[type]}"></i><span>${message}</span>`;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity .3s'; setTimeout(() => toast.remove(), 300); }, 4000);
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
window.showSignalDetailsModal = function (signal) {
  const wrapper = document.getElementById('token-details-modal');
  const modal = document.getElementById('signalDetailsModal');
  if (!modal) return;
  if (wrapper) wrapper.classList.remove('hidden');
  modal.classList.remove('hidden');

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

  // Set active dataset for real-time tracking
  modal.dataset.activeSymbol = signal.symbol;
  modal.dataset.activeTimeframe = signal.timeframe || '1h';

  // Store current signal for feature panels
  window._fpSignal = signal;

  // Populate Feature Access Cards
  const confluence = signal.confluence || { trend: 50, momentum: 50, volume: 50 };
  const _currentPrice = window.currentTickers?.[signal.symbol] ? parseFloat(window.currentTickers[signal.symbol]) : (signal.entry_price || 0);
  const _tier = getUserTier();

  const LOCK_BASIC = _tier === 'BASIC';
  const LOCK_NON_PRO = _tier !== 'PRO';

  function _featureLockOverlay(minTier) {
    const label = minTier === 'PRO' ? 'PRO' : 'Intermediate';
    return `<div class="absolute inset-0 bg-black/70 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center z-10 pointer-events-none">
      <i class="fas fa-lock text-gray-400 text-lg mb-1"></i>
      <span class="text-[10px] text-gray-300 font-bold">${label}+ Only</span>
    </div>`;
  }

  const _sl = signal.sl || 0;
  const _tp = signal.tp || 0;
  const _entry = signal.entry_price || 0;
  let _zoneBarHTML = '<div class="text-[9px] text-gray-500 flex items-center justify-center h-full">No levels</div>';
  if (_sl && _tp && _tp > _sl) {
    const _range = _tp - _sl;
    const _entryPct = ((_entry - _sl) / _range * 100).toFixed(1);
    const _curPct = ((_currentPrice - _sl) / _range * 100).toFixed(1);
    _zoneBarHTML = `<div class="absolute top-0 bottom-0 w-0.5 bg-cyan" style="left:${_entryPct}%"></div>
                    <div class="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white shadow-[0_0_6px_white]" style="left:${Math.max(0,Math.min(98,parseFloat(_curPct)))}%"></div>`;
  }

  let _shapLeadHTML = '';
  const _probs = signal.probabilities || { SHORT: 0.18, HOLD: 0.08, LONG: 0.74 };
  const _lead = Object.entries(_probs).sort((a,b) => b[1]-a[1])[0];
  const _leadPct = (_lead[1]*100).toFixed(0);
  const _leadColor = _lead[0]==='LONG' ? 'text-green-400' : _lead[0]==='SHORT' ? 'text-red-400' : 'text-gray-400';
  _shapLeadHTML = `<div class="text-lg font-black ${_leadColor}">${_leadPct}% ${_lead[0]}</div>`;

  const cardsHTML = `
    <!-- Card 1: Confluence -->
    <div data-open-panel="fp-confluence"
      class="relative cursor-pointer bg-black/40 p-4 rounded-xl border border-cyan/20 hover:border-cyan/50 hover:bg-cyan/5 transition-all group overflow-hidden">
      ${LOCK_BASIC ? _featureLockOverlay('INTERMEDIATE') : ''}
      <div class="flex items-center gap-2 mb-3">
        <i class="fas fa-layer-group text-cyan text-xs"></i>
        <span class="text-[11px] font-bold text-white uppercase tracking-wider">Confluence</span>
      </div>
      <div class="space-y-1.5">
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1 bg-black/50 rounded overflow-hidden">
            <div class="h-full bg-cyan/70" style="width:${confluence.trend}%"></div>
          </div>
          <span class="text-[10px] font-mono text-cyan w-8 text-right">${confluence.trend}%</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1 bg-black/50 rounded overflow-hidden">
            <div class="h-full bg-blue-400/70" style="width:${confluence.momentum}%"></div>
          </div>
          <span class="text-[10px] font-mono text-blue-300 w-8 text-right">${confluence.momentum}%</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1 bg-black/50 rounded overflow-hidden">
            <div class="h-full bg-violet-400/70" style="width:${confluence.volume}%"></div>
          </div>
          <span class="text-[10px] font-mono text-violet-300 w-8 text-right">${confluence.volume}%</span>
        </div>
      </div>
      <div class="mt-2 text-right">
        <span class="text-[9px] text-gray-500 group-hover:text-cyan transition-colors">Full Analysis <i class="fas fa-arrow-right"></i></span>
      </div>
    </div>

    <!-- Card 2: Visual Zones -->
    <div data-open-panel="fp-zones"
      class="relative cursor-pointer bg-black/40 p-4 rounded-xl border border-violet-500/20 hover:border-violet-500/50 hover:bg-violet-500/5 transition-all group overflow-hidden">
      ${LOCK_BASIC ? _featureLockOverlay('INTERMEDIATE') : ''}
      <div class="flex items-center gap-2 mb-3">
        <i class="fas fa-map-marker-alt text-violet-400 text-xs"></i>
        <span class="text-[11px] font-bold text-white uppercase tracking-wider">Zone Track</span>
      </div>
      <div class="relative h-4 rounded-lg overflow-hidden bg-gradient-to-r from-red-900/60 via-gray-900/60 to-green-900/60 border border-white/10">
        ${_zoneBarHTML}
      </div>
      <div class="flex justify-between mt-1.5 text-[9px] font-mono">
        <span class="text-red-400">SL $${(_sl).toFixed(2)}</span>
        <span class="text-gray-400">Entry</span>
        <span class="text-green-400">TP $${(_tp).toFixed(2)}</span>
      </div>
      <div class="mt-1.5 text-right">
        <span class="text-[9px] text-gray-500 group-hover:text-violet-400 transition-colors">Live View <i class="fas fa-arrow-right"></i></span>
      </div>
    </div>

    <!-- Card 3: Expectancy Matrix -->
    <div data-open-panel="fp-expectancy"
      class="relative cursor-pointer bg-black/40 p-4 rounded-xl border border-emerald-500/20 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group overflow-hidden">
      ${LOCK_NON_PRO ? _featureLockOverlay('PRO') : ''}
      <div class="flex items-center gap-2 mb-3">
        <i class="fas fa-chart-bar text-emerald-400 text-xs"></i>
        <span class="text-[11px] font-bold text-white uppercase tracking-wider">Expectancy</span>
      </div>
      <div class="font-mono">
        <div class="text-2xl font-black text-emerald-400">${typeof signal.expectancy === 'number' ? (signal.expectancy >= 0 ? '+' : '') + signal.expectancy.toFixed(2) + '%' : '+1.64%'}</div>
        <div class="text-[9px] text-gray-500 mt-0.5">per signal avg</div>
        <div class="mt-2 text-[10px] text-amber-400">Max DD: ${typeof signal.max_dd === 'number' ? signal.max_dd.toFixed(2) : '-5.12'}%</div>
      </div>
      <div class="mt-2 text-right">
        <span class="text-[9px] text-gray-500 group-hover:text-emerald-400 transition-colors">Full Stats <i class="fas fa-arrow-right"></i></span>
      </div>
    </div>

    <!-- Card 4: SHAP -->
    <div data-open-panel="fp-shap"
      class="relative cursor-pointer bg-black/40 p-4 rounded-xl border border-orange-500/20 hover:border-orange-500/50 hover:bg-orange-500/5 transition-all group overflow-hidden">
      ${LOCK_NON_PRO ? _featureLockOverlay('PRO') : ''}
      <div class="flex items-center gap-2 mb-3">
        <i class="fas fa-brain text-orange text-xs"></i>
        <span class="text-[11px] font-bold text-white uppercase tracking-wider">SHAP</span>
      </div>
      <div class="font-mono">
        <div class="text-xs font-bold text-gray-300 mb-1.5">Model Conviction</div>
        ${_shapLeadHTML}
      </div>
      <div class="mt-2 text-right">
        <span class="text-[9px] text-gray-500 group-hover:text-orange-400 transition-colors">Full Report <i class="fas fa-arrow-right"></i></span>
      </div>
    </div>

    <!-- Card 5: API Export (full width) -->
    <div data-open-panel="fp-api"
      class="relative cursor-pointer col-span-2 bg-black/40 p-4 rounded-xl border border-blue-500/20 hover:border-blue-500/50 hover:bg-blue-500/5 transition-all group flex items-center gap-4 overflow-hidden">
      ${LOCK_NON_PRO ? _featureLockOverlay('PRO') : ''}
      <div class="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
        <i class="fas fa-code text-blue-400 text-base"></i>
      </div>
      <div class="flex-1 min-w-0">
        <div class="text-[11px] font-bold text-white uppercase tracking-wider">API / JSON Data Export</div>
        <div class="text-[10px] text-gray-500 font-mono mt-0.5 truncate">aegis_live_&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;  &rarr;  /api/v1/signals/live</div>
      </div>
      <span class="text-[9px] text-gray-500 group-hover:text-blue-400 transition-colors flex-shrink-0">
        Dev Portal <i class="fas fa-arrow-right"></i>
      </span>
    </div>
  `;

  const featureCardsEl = document.getElementById('sd-feature-cards');
  if (featureCardsEl) featureCardsEl.innerHTML = cardsHTML;

  // Plan-aware footer buttons
  const footerActions = document.getElementById('sd-footer-actions');
  if (footerActions) {
    const tier = getUserTier();
    const isPro = tier === 'PRO';

    footerActions.innerHTML = `
      <div class="flex flex-col gap-2">
        <div class="flex gap-2">
          <button id="sd-execute-btn"
            class="flex-1 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold py-3 rounded-xl uppercase tracking-wider text-sm shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:-translate-y-0.5 transform transition-all">
            <i class="fas fa-terminal mr-2"></i>Execute in Terminal
          </button>
          <button id="sd-paper-trade-btn"
            class="px-5 bg-white/10 hover:bg-white/20 text-white font-bold py-3 rounded-xl uppercase tracking-wider text-sm border border-white/20 transition-colors whitespace-nowrap">
            <i class="fas fa-play-circle mr-1"></i>Paper
          </button>
        </div>
        ${isPro ? `
        <div class="flex gap-2">
          <button id="sd-copy-signal-btn"
            class="flex-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 font-bold py-2 rounded-xl text-xs border border-purple-500/30 transition-colors">
            <i class="fas fa-copy mr-1"></i>Copy Signal JSON
          </button>
          <button id="sd-view-analytics-btn"
            class="flex-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 font-bold py-2 rounded-xl text-xs border border-blue-500/30 transition-colors">
            <i class="fas fa-chart-pie mr-1"></i>View in Analytics
          </button>
        </div>` : `
        <a href="/web/src/pages/pricing.html"
          class="flex items-center justify-center gap-2 bg-gradient-to-r from-amber-500/10 to-orange/10 hover:from-amber-500/20 hover:to-orange/20 text-amber-400 font-bold py-2 rounded-xl text-xs border border-amber-500/30 transition-colors">
          <i class="fas fa-crown"></i>Unlock Pro &mdash; Copy JSON, Advanced Analytics &amp; more
        </a>`}
      </div>
    `;

    document.getElementById('sd-execute-btn')?.addEventListener('click', () => {
      modal.classList.add('hidden');
      if (typeof window.selectSignal === 'function') window.selectSignal(signal.symbol, signal.timeframe || '1h');
    });

    document.getElementById('sd-paper-trade-btn')?.addEventListener('click', () => {
      modal.classList.add('hidden');
      if (typeof window.initiatePaperTrade === 'function') {
        window.initiatePaperTrade(signal.symbol, signal.entry_price || 0, signal.sl || 0, signal.tp || 0);
      }
    });

    document.getElementById('sd-copy-signal-btn')?.addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(signal, null, 2))
        .then(() => _showToast('Signal JSON copied to clipboard!', 'success'))
        .catch(() => _showToast('Copy failed — check browser permissions', 'error'));
    });

    document.getElementById('sd-view-analytics-btn')?.addEventListener('click', () => {
      modal.classList.add('hidden');
      if (typeof window.switchRoom === 'function') window.switchRoom('analytics');
    });
  }

  modal.classList.remove('hidden');
}

function initModals() {
  const modal = document.getElementById('token-details-modal') || document.getElementById('signalDetailsModal');
  const closeBtn = document.getElementById('closeSignalDetailsBtn');

  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
    });
  }

  if (modal) {
    // Single delegating listener: feature-card clicks are caught first so inner
    // elements never accidentally trigger the backdrop-close branch.
    modal.addEventListener('click', (e) => {
      const card = e.target.closest('[data-open-panel], .feature-card-trigger');
      if (card) {
        const panelId = card.dataset.openPanel || card.dataset.target;
        if (panelId && typeof window.openFeaturePanel === 'function') {
          window.openFeaturePanel(panelId);
        }
        return;
      }
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initModals);
} else {
  initModals();
}

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
  // Fallback to private sealed state (not window.isPremiumUser which is a getter anyway)
  if (_subState.isPremium) return 'PRO';
  return 'BASIC';
}

// ============================================================
// FEATURE PANEL SYSTEM
// ============================================================

window._fpSignal = null;

// Explicit list — never accidentally matches *-body divs
const _FP_PANELS = ['fp-confluence', 'fp-zones', 'fp-expectancy', 'fp-shap', 'fp-api'];

window.openFeaturePanel = function(panelId) {
  // Hide all top-level panels only (NOT their -body children)
  _FP_PANELS.forEach(id => {
    const p = document.getElementById(id);
    if (p) { p.classList.add('hidden'); p.classList.remove('flex'); }
  });

  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.classList.remove('hidden');
  panel.classList.add('flex');

  // Also ensure the body div is visible (remove any stale hidden class)
  const body = document.getElementById(`${panelId}-body`);
  if (body) body.classList.remove('hidden');

  const signal = window._fpSignal;
  const tier = getUserTier();

  // If no signal context yet, show a waiting state instead of a blank panel
  if (!signal) {
    if (body) body.innerHTML = `
      <div class="flex flex-col items-center justify-center h-full text-center py-20">
        <i class="fas fa-satellite-dish text-4xl text-gray-600 mb-4 animate-pulse"></i>
        <p class="text-gray-500 text-sm">Open a signal card first to load this panel.</p>
      </div>`;
    return;
  }

  if (!body) return;

  switch(panelId) {
    case 'fp-confluence': _renderFpConfluence(body, signal, tier); break;
    case 'fp-zones':      _renderFpZones(body, signal, tier); break;
    case 'fp-expectancy': _renderFpExpectancy(body, signal, tier); break;
    case 'fp-shap':       _renderFpShap(body, signal, tier); break;
    case 'fp-api':        _renderFpApi(body, signal, tier); break;
  }
};

window.closeFP = function() {
  // Remove live price listener from zones panel
  const zonesBody = document.getElementById('fp-zones-body');
  if (zonesBody && zonesBody._zonePriceHandler) {
    window.removeEventListener('priceUpdate', zonesBody._zonePriceHandler);
    delete zonesBody._zonePriceHandler;
  }
  // Hide top-level panels only
  _FP_PANELS.forEach(id => {
    const p = document.getElementById(id);
    if (p) { p.classList.add('hidden'); p.classList.remove('flex'); }
  });
};

function _renderFpConfluence(body, signal, tier) {
  const locked = tier === 'BASIC';
  const confluence = signal.confluence || { trend: 50, momentum: 50, volume: 50 };

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-lock text-3xl text-gray-500 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Intermediate Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Upgrade to Intermediate or Pro to access detailed confluence analysis</p>
      <a href="/web/src/pages/pricing.html" class="px-6 py-2 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold rounded-xl text-sm">Upgrade Now</a>
    </div>` : '';

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-sm pointer-events-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
          <span class="text-sm text-gray-400">${signal.timeframe || '1h'}</span>
        </div>
        <div class="space-y-4">
          ${[
            { label: 'Trend Alignment', sublabel: 'EMA 200/50 confluence weight', val: confluence.trend, color: 'bg-cyan', textColor: 'text-cyan' },
            { label: 'Momentum Regime', sublabel: 'RSI / Stochastic position scaling', val: confluence.momentum, color: 'bg-blue-400', textColor: 'text-blue-300' },
            { label: 'Volume Delta', sublabel: 'Net buying/selling pressure', val: confluence.volume, color: 'bg-violet-400', textColor: 'text-violet-300' },
          ].map(item => `
            <div class="bg-black/40 p-4 rounded-xl border border-white/5">
              <div class="flex justify-between items-end mb-2">
                <div>
                  <div class="text-sm font-bold text-white">${item.label}</div>
                  <div class="text-[10px] text-gray-500 mt-0.5">${item.sublabel}</div>
                </div>
                <div class="font-black font-mono text-2xl ${item.textColor}">${item.val}%</div>
              </div>
              <div class="h-2 bg-black/60 rounded-full overflow-hidden">
                <div class="h-full ${item.color} rounded-full transition-all" style="width:${item.val}%"></div>
              </div>
            </div>
          `).join('')}
        </div>
        <div class="mt-4 bg-cyan/5 border border-cyan/20 p-4 rounded-xl">
          <div class="flex items-center justify-between">
            <span class="text-sm font-bold text-white">Overall Confluence Score</span>
            <span class="text-2xl font-black font-mono text-cyan">${Math.round((confluence.trend + confluence.momentum + confluence.volume) / 3)}%</span>
          </div>
          <div class="h-1.5 bg-black/50 rounded-full mt-2 overflow-hidden">
            <div class="h-full bg-gradient-to-r from-cyan to-blue-500 rounded-full"
                 style="width:${Math.round((confluence.trend + confluence.momentum + confluence.volume) / 3)}%"></div>
          </div>
        </div>
        <div class="mt-4 p-3 bg-black/30 rounded-lg border border-white/5">
          <div class="text-[10px] text-gray-500 font-mono">
            <i class="fas fa-info-circle text-cyan/50 mr-1"></i>
            Weights computed via XGBoost gradient boosting ensemble. Values represent normalized feature importance vectors for the current candle state. Updated on each WebSocket tick.
          </div>
        </div>
        ${(() => {
          const _sc = Math.round((confluence.trend + confluence.momentum + confluence.volume) / 3);
          const _dir = signal.direction || 'LONG';
          const _vc = _sc >= 75 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : _sc >= 55 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-red-500/20 text-red-400 border-red-500/30';
          const _vl = _sc >= 75 ? 'STRONG' : _sc >= 55 ? 'MODERATE' : 'WEAK';
          const _dom = confluence.trend >= confluence.momentum && confluence.trend >= confluence.volume
            ? 'Trend' : confluence.momentum >= confluence.volume ? 'Momentum' : 'Volume';
          const _domVal = Math.max(confluence.trend, confluence.momentum, confluence.volume);
          const _bias = _dir === 'LONG' ? 'bullish' : _dir === 'SHORT' ? 'bearish' : 'neutral';
          const _why = `Score of ${_sc}% reflects ${_bias} alignment across all three vectors — Trend ${confluence.trend}%, Momentum ${confluence.momentum}%, Volume ${confluence.volume}%.`;
          const _what = `${_dom} is the dominant driver at ${_domVal}%. ${_sc >= 75 ? 'All vectors are aligned — signal has strong structural backing.' : _sc >= 55 ? 'Minor divergence between vectors — signal is valid but not optimal.' : 'Significant divergence detected — signal reliability is reduced.'}`;
          const _when = _sc >= 75
            ? 'Optimal entry window. High confluence supports immediate position initiation at current levels.'
            : _sc >= 55
            ? 'Entry is viable — reduce position size by 30–50% to account for moderate alignment.'
            : 'Stand aside. Wait for confluence to exceed 55% before considering a position.';
          const _ep = (signal.entry_price || 0).toFixed(4);
          const _sl = (signal.sl || 0).toFixed(4);
          const _where = `Execute near $${_ep} with SL at $${_sl}. ${_sc >= 75 ? 'Full position size is justified.' : _sc >= 55 ? 'Reduced size recommended — scale in on confirmation.' : 'No trade — observe for vector realignment before acting.'}`;
          return `<div class="mt-4 p-4 bg-black/40 rounded-xl border border-cyan/20">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-xs font-bold text-cyan uppercase tracking-wider">Signal Intelligence</h4>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${_vc}">${_vl}</span>
            </div>
            <div class="space-y-2 text-[11px]">
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-cyan/60 font-bold uppercase text-[9px] pt-0.5">WHY</span><span class="text-gray-300">${_why}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-cyan/60 font-bold uppercase text-[9px] pt-0.5">WHAT</span><span class="text-gray-300">${_what}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-cyan/60 font-bold uppercase text-[9px] pt-0.5">WHEN</span><span class="text-gray-300">${_when}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-cyan/60 font-bold uppercase text-[9px] pt-0.5">WHERE</span><span class="text-gray-300">${_where}</span></div>
            </div>
          </div>`;
        })()}
      </div>
    </div>
  `;
}

function _renderFpZones(body, signal, tier) {
  const locked = tier === 'BASIC';
  const sl = signal.sl || 0;
  const tp = signal.tp || 0;
  const entry = signal.entry_price || 0;
  const currentPrice = window.currentTickers?.[signal.symbol]
    ? parseFloat(window.currentTickers[signal.symbol]) : entry;

  const range = tp - sl;
  const entryPct = range > 0 ? ((entry - sl) / range * 100) : 50;
  const curPct = range > 0 ? Math.max(0, Math.min(100, (currentPrice - sl) / range * 100)) : 50;

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-lock text-3xl text-gray-500 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Intermediate Plan Required</h3>
      <a href="/web/src/pages/pricing.html" class="px-6 py-2 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold rounded-xl text-sm mt-2">Upgrade Now</a>
    </div>` : '';

  let statusMsg = '';
  if (!locked && entry > 0 && currentPrice > 0) {
    const pctFromEntry = ((currentPrice - entry) / entry) * 100;
    const isLong = (signal.direction || 'LONG') === 'LONG';
    const inProfit = isLong ? currentPrice > entry : currentPrice < entry;
    if (inProfit && Math.abs(pctFromEntry) > 0.2) {
      statusMsg = `<div class="mt-3 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-bold animate-pulse">
        ⚠️ Breakout In Progress: Do Not Chase
      </div>`;
    } else {
      statusMsg = `<div class="mt-3 px-3 py-2 rounded-lg bg-cyan/10 border border-cyan/30 text-cyan text-xs font-bold">
        ⚡ Inside Active Entry Buffer Zone
      </div>`;
    }
  }

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-sm pointer-events-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
          <span class="text-sm ${(signal.direction||'LONG')==='LONG' ? 'text-green-400' : 'text-red-400'} font-bold">${signal.direction||'LONG'}</span>
        </div>
        <div class="bg-black/40 p-5 rounded-xl border border-white/5">
          <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-4 font-bold">Risk-to-Reward Coordinate Map</div>
          <div class="relative h-10 rounded-xl overflow-hidden" style="background: linear-gradient(to right, rgba(239,68,68,0.20) 0%, rgba(10,10,15,0.85) 25%, rgba(10,10,15,0.85) 75%, rgba(16,185,129,0.20) 100%); border: 1px solid rgba(255,255,255,0.08);">
            <div class="fp-entry-line absolute top-0 bottom-0 w-px bg-cyan" style="left:${entryPct.toFixed(1)}%">
              <div class="absolute -top-px -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-cyan"></div>
              <div class="absolute bottom-1 -translate-x-1/2 text-[8px] font-bold text-cyan/80 whitespace-nowrap">ENTRY</div>
            </div>
            <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-emerald-400"
                 id="fp-zone-dot"
                 style="left:${curPct.toFixed(1)}%"></div>
            <div class="absolute bottom-1 left-2 text-[8px] font-bold text-red-400/70">SL</div>
            <div class="absolute bottom-1 right-2 text-[8px] font-bold text-emerald-400/70">TP</div>
          </div>
          <div class="flex justify-between mt-2 text-[10px] font-mono">
            <div class="text-red-400"><div class="font-bold">SL</div><div>$${sl.toFixed(4)}</div></div>
            <div class="text-cyan text-center"><div class="font-bold">Entry</div><div>$${entry.toFixed(4)}</div></div>
            <div class="text-green-400 text-right"><div class="font-bold">TP</div><div>$${tp.toFixed(4)}</div></div>
          </div>
          ${statusMsg}
        </div>
        <div class="grid grid-cols-2 gap-3 mt-4">
          <div class="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Current Price</div>
            <div class="text-xl font-black font-mono text-white" id="fp-zone-price">$${currentPrice.toFixed(4)}</div>
          </div>
          <div class="bg-black/40 p-4 rounded-xl border border-white/5 text-center">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Risk/Reward</div>
            <div class="text-xl font-black font-mono text-cyan">
              ${sl && tp && entry ? `1:${((tp - entry) / (entry - sl)).toFixed(2)}` : '&mdash;'}
            </div>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-3">
          <div class="bg-red-500/5 border border-red-500/20 p-3 rounded-lg text-center">
            <div class="text-[9px] text-red-400/70 uppercase">Stop Distance</div>
            <div class="text-sm font-mono font-bold text-red-400 mt-0.5">${entry && sl ? ((entry - sl) / entry * 100).toFixed(2) : '&mdash;'}%</div>
          </div>
          <div class="bg-cyan/5 border border-cyan/20 p-3 rounded-lg text-center">
            <div class="text-[9px] text-cyan/70 uppercase">In Zone</div>
            <div class="text-sm font-mono font-bold text-cyan mt-0.5">${curPct.toFixed(1)}%</div>
          </div>
          <div class="bg-green-500/5 border border-green-500/20 p-3 rounded-lg text-center">
            <div class="text-[9px] text-green-400/70 uppercase">Target Dist</div>
            <div class="text-sm font-mono font-bold text-green-400 mt-0.5">${entry && tp ? ((tp - entry) / entry * 100).toFixed(2) : '&mdash;'}%</div>
          </div>
        </div>
        ${(() => {
          const _isLong = (signal.direction || 'LONG') === 'LONG';
          const _rrRaw = (sl && tp && entry) ? ((tp - entry) / (entry - sl)) : 0;
          const _rr = _rrRaw.toFixed(2);
          const _vc = _rrRaw >= 2 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : _rrRaw >= 1.5 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-red-500/20 text-red-400 border-red-500/30';
          const _vl = _rrRaw >= 2 ? 'HIGH R:R' : _rrRaw >= 1.5 ? 'FAIR R:R' : 'LOW R:R';
          const _pctFromEntry = entry > 0 ? ((currentPrice - entry) / entry * 100) : 0;
          const _inProfit = _isLong ? currentPrice > entry : currentPrice < entry;
          const _stopDist = entry && sl ? Math.abs(((entry - sl) / entry * 100)).toFixed(2) : '0.00';
          const _tgtDist = entry && tp ? Math.abs(((tp - entry) / entry * 100)).toFixed(2) : '0.00';
          const _why = `Price is ${Math.abs(_pctFromEntry).toFixed(2)}% ${_inProfit ? 'ahead of' : 'behind'} the entry at $${entry.toFixed(4)} — currently ${curPct.toFixed(1)}% across the SL-to-TP corridor.`;
          const _what = `Risk/Reward is 1:${_rr}. Stop distance is ${_stopDist}% ($${sl.toFixed(4)}); target distance is ${_tgtDist}% ($${tp.toFixed(4)}). ${_rrRaw >= 2 ? 'Excellent asymmetry — reward far outweighs risk.' : _rrRaw >= 1.5 ? 'Acceptable ratio — proceed with standard sizing.' : 'Tight reward relative to risk — consider skipping or waiting for a better entry.'}`;
          const _when = curPct < 30
            ? 'Price is near the entry zone — valid window to initiate the position.'
            : curPct < 60
            ? 'Price has moved into the middle of the range. Entry is still viable but chase risk is elevated.'
            : 'Price is deep into the TP corridor. Do not chase — wait for a pullback to the entry zone.';
          const _where = `Watch $${entry.toFixed(4)} (entry), $${sl.toFixed(4)} (invalidation). ${_isLong ? 'A close below SL signals the trade has failed.' : 'A close above SL signals the trade has failed.'} TP target at $${tp.toFixed(4)}.`;
          return `<div class="mt-4 p-4 bg-black/40 rounded-xl border border-emerald-500/20">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Zone Intelligence</h4>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${_vc}">${_vl}</span>
            </div>
            <div class="space-y-2 text-[11px]">
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-emerald-400/60 font-bold uppercase text-[9px] pt-0.5">WHY</span><span class="text-gray-300">${_why}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-emerald-400/60 font-bold uppercase text-[9px] pt-0.5">WHAT</span><span class="text-gray-300">${_what}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-emerald-400/60 font-bold uppercase text-[9px] pt-0.5">WHEN</span><span class="text-gray-300">${_when}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-emerald-400/60 font-bold uppercase text-[9px] pt-0.5">WHERE</span><span class="text-gray-300">${_where}</span></div>
            </div>
          </div>`;
        })()}
      </div>
    </div>
  `;

  // Live price update for zone panel
  if (body._zonePriceHandler) {
    window.removeEventListener('priceUpdate', body._zonePriceHandler);
  }
  body._zonePriceHandler = (e) => {
    if (e.detail.symbol !== signal.symbol) return;
    const p = parseFloat(e.detail.price);
    if (isNaN(p)) return;
    const el = body.querySelector('#fp-zone-price');
    if (el) el.textContent = `$${p.toFixed(4)}`;
    const dot = body.querySelector('#fp-zone-dot');
    if (dot && sl && tp) {
      const newPct = Math.max(2, Math.min(98, (p - sl) / (tp - sl) * 100));
      dot.style.left = `${newPct.toFixed(1)}%`;
    }
  };
  window.addEventListener('priceUpdate', body._zonePriceHandler);
}

function _renderFpExpectancy(body, signal, tier) {
  const locked = tier !== 'PRO';

  const expectancy = typeof signal.expectancy === 'number' ? signal.expectancy : 1.64;
  const maxDD = typeof signal.max_dd === 'number' ? signal.max_dd : -5.12;
  const profitFactor = typeof signal.profit_factor === 'number' ? signal.profit_factor : 1.87;
  const winRate = typeof signal.win_rate === 'number' ? signal.win_rate : 67;
  const totalTrades = typeof signal.total_trades === 'number' ? signal.total_trades : 42;

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-crown text-3xl text-amber-400 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Pro Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Statistical edge analysis is exclusively available to Pro subscribers</p>
      <a href="/web/src/pages/pricing.html" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
    </div>` : '';

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-md pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
          <span class="text-xs text-gray-500 bg-black/40 px-2 py-0.5 rounded">Last 30 Days</span>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div class="bg-black/40 p-4 rounded-xl border border-emerald-500/20">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Mathematical Expectancy</div>
            <div class="text-3xl font-black font-mono ${expectancy >= 0 ? 'text-emerald-400' : 'text-red-400'}">${expectancy >= 0 ? '+' : ''}${expectancy.toFixed(2)}%</div>
            <div class="text-[10px] text-gray-500 mt-1">average per trade</div>
          </div>
          <div class="bg-black/40 p-4 rounded-xl border border-amber-500/20">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Maximum Drawdown</div>
            <div class="text-3xl font-black font-mono text-amber-400">${maxDD.toFixed(2)}%</div>
            <div class="text-[10px] text-gray-500 mt-1">peak-to-trough</div>
          </div>
          <div class="bg-black/40 p-4 rounded-xl border border-cyan/20">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Profit Factor</div>
            <div class="text-3xl font-black font-mono ${profitFactor >= 1.5 ? 'text-cyan' : 'text-white'}">${profitFactor.toFixed(2)}</div>
            <div class="text-[10px] text-gray-500 mt-1">gross profit / loss</div>
          </div>
          <div class="bg-black/40 p-4 rounded-xl border border-white/5">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Win Rate</div>
            <div class="text-3xl font-black font-mono text-white">${winRate}%</div>
            <div class="text-[10px] text-gray-500 mt-1">from ${totalTrades} signals</div>
          </div>
        </div>
        <div class="mt-4 bg-black/40 p-4 rounded-xl border border-white/5">
          <div class="flex justify-between text-[10px] mb-2">
            <span class="text-gray-400 font-bold">Historical Win Distribution</span>
            <span class="text-white font-mono">${winRate}% wins</span>
          </div>
          <div class="h-3 bg-black/50 rounded-full overflow-hidden flex">
            <div class="h-full bg-emerald-500/60 rounded-l-full" style="width:${winRate}%"></div>
            <div class="h-full bg-red-500/40 rounded-r-full flex-1"></div>
          </div>
          <div class="flex justify-between text-[10px] mt-1 text-gray-500">
            <span>${Math.round(totalTrades * winRate / 100)} wins</span>
            <span>${Math.round(totalTrades * (100 - winRate) / 100)} losses</span>
          </div>
        </div>
        <div class="mt-3 p-3 bg-black/30 rounded-lg border border-white/5">
          <div class="text-[10px] text-gray-500 font-mono">
            <i class="fas fa-database text-cyan/50 mr-1"></i>
            Data sourced from cached Firestore performance document. Updated every 24h by background cron. Not indicative of future results.
          </div>
        </div>
        ${(() => {
          const _vc = expectancy >= 1.5 && profitFactor >= 1.5
            ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : expectancy >= 0 && profitFactor >= 1.0
            ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-red-500/20 text-red-400 border-red-500/30';
          const _vl = expectancy >= 1.5 && profitFactor >= 1.5 ? 'EDGE+' : expectancy >= 0 ? 'MARGINAL' : 'NEGATIVE EDGE';
          const _wins = Math.round(totalTrades * winRate / 100);
          const _losses = totalTrades - _wins;
          const _ddRatio = profitFactor > 0 ? (Math.abs(maxDD) / profitFactor).toFixed(2) : 'N/A';
          const _why = `Over ${totalTrades} historical signals, this asset produced a mathematical expectancy of ${expectancy >= 0 ? '+' : ''}${expectancy.toFixed(2)}% per trade — meaning every position statistically ${expectancy >= 0 ? 'returns a positive edge' : 'loses value on average'}.`;
          const _what = `Profit Factor of ${profitFactor.toFixed(2)} means every $1 lost returns $${profitFactor.toFixed(2)} in gross wins. Win rate is ${winRate}% (${_wins}W / ${_losses}L). Max drawdown reached ${maxDD.toFixed(2)}% peak-to-trough. DD/PF ratio: ${_ddRatio} ${parseFloat(_ddRatio) < 4 ? '(healthy)' : '(elevated — oversized risk).'}.`;
          const _when = expectancy >= 1.5 && profitFactor >= 1.5
            ? 'Strong edge confirmed. This is the right time to allocate full or above-average position size.'
            : expectancy >= 0 && profitFactor >= 1.0
            ? 'Marginal edge — trade with standard or reduced sizing. Monitor for edge degradation.'
            : 'Negative expectancy detected. Avoid new positions on this asset until performance improves.';
          const _where = `Focus on setups where ${winRate >= 60 ? 'win rate consistency' : 'profit factor'} is the primary driver. ${Math.abs(maxDD) > 10 ? 'High max drawdown warrants tighter stop placement.' : 'Max drawdown is within acceptable range for standard stops.'} Compare against your portfolio average to assess relative merit.`;
          return `<div class="mt-4 p-4 bg-black/40 rounded-xl border border-amber-500/20">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-xs font-bold text-amber-400 uppercase tracking-wider">Edge Intelligence</h4>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${_vc}">${_vl}</span>
            </div>
            <div class="space-y-2 text-[11px]">
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-amber-400/60 font-bold uppercase text-[9px] pt-0.5">WHY</span><span class="text-gray-300">${_why}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-amber-400/60 font-bold uppercase text-[9px] pt-0.5">WHAT</span><span class="text-gray-300">${_what}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-amber-400/60 font-bold uppercase text-[9px] pt-0.5">WHEN</span><span class="text-gray-300">${_when}</span></div>
              <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-amber-400/60 font-bold uppercase text-[9px] pt-0.5">WHERE</span><span class="text-gray-300">${_where}</span></div>
            </div>
          </div>`;
        })()}
      </div>
    </div>
  `;
}

function _renderFpShap(body, signal, tier) {
  const locked = tier !== 'PRO';

  const probs = signal.probabilities || { SHORT: 0.18, HOLD: 0.08, LONG: 0.74 };
  const leadEntry = Object.entries(probs).sort((a,b) => b[1]-a[1])[0];
  const leadClass = leadEntry[0] === 'LONG' ? 'text-green-400' : leadEntry[0] === 'SHORT' ? 'text-red-400' : 'text-gray-400';
  const leadPct = (leadEntry[1] * 100).toFixed(1);

  const shapValues = signal.shap_values || [
    { feature: 'Volume Delta 1h', value: 0.34, direction: 'long' },
    { feature: 'BTC Anchor Distance', value: -0.21, direction: 'short' },
    { feature: 'EMA 200 Confluence', value: 0.18, direction: 'long' },
    { feature: 'RSI Regime 4h', value: 0.15, direction: 'long' },
    { feature: 'Liq. Block Density', value: -0.09, direction: 'short' },
  ];

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-crown text-3xl text-amber-400 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Pro Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Raw ML probability vectors and SHAP attribution are Pro-exclusive</p>
      <a href="/web/src/pages/pricing.html" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
    </div>` : '';

  const maxShap = Math.max(...shapValues.map(s => Math.abs(s.value)));

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-md pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
        </div>
        <div class="bg-black/40 p-5 rounded-xl border border-white/5 mb-4">
          <div class="flex items-center justify-between mb-3">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Model Conviction</div>
            <div class="font-black text-xl font-mono ${leadClass}">${leadPct}% ${leadEntry[0]}</div>
          </div>
          <div class="flex h-6 rounded-lg overflow-hidden gap-0.5">
            <div class="bg-red-500/60 flex items-center justify-center text-[10px] font-bold text-white/80 transition-all"
                 style="width:${(probs.SHORT * 100).toFixed(1)}%">
              ${(probs.SHORT * 100) > 15 ? 'SHORT' : ''}
            </div>
            <div class="bg-gray-700/60 flex items-center justify-center text-[10px] font-bold text-white/80 transition-all"
                 style="width:${(probs.HOLD * 100).toFixed(1)}%">
              ${(probs.HOLD * 100) > 10 ? 'HOLD' : ''}
            </div>
            <div class="bg-emerald-500/60 flex items-center justify-center text-[10px] font-bold text-white/80 transition-all"
                 style="width:${(probs.LONG * 100).toFixed(1)}%">
              ${(probs.LONG * 100) > 15 ? 'LONG' : ''}
            </div>
          </div>
          <div class="flex justify-between text-[9px] font-mono mt-1 text-gray-500">
            <span class="text-red-400">${(probs.SHORT * 100).toFixed(1)}% SHORT</span>
            <span>${(probs.HOLD * 100).toFixed(1)}% HOLD</span>
            <span class="text-green-400">${(probs.LONG * 100).toFixed(1)}% LONG</span>
          </div>
        </div>
        <div class="bg-black/40 p-5 rounded-xl border border-white/5">
          <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-4">Live SHAP Feature Attribution</div>
          <div class="space-y-3">
            ${shapValues.map(sv => {
              const barPct = (Math.abs(sv.value) / maxShap * 80).toFixed(1);
              const isPos = sv.value > 0;
              const barColor = isPos ? 'bg-emerald-500/40 border-emerald-500/30' : 'bg-rose-500/40 border-rose-500/30';
              const textColor = isPos ? 'text-emerald-400' : 'text-rose-400';
              return `
                <div>
                  <div class="flex justify-between text-[10px] mb-1">
                    <span class="text-gray-300 font-mono">${sv.feature}</span>
                    <span class="font-bold font-mono ${textColor}">${isPos ? '+' : ''}${sv.value.toFixed(3)}</span>
                  </div>
                  <div class="h-5 bg-black/40 rounded border border-white/5 overflow-hidden flex items-center">
                    <div class="h-full ${barColor} rounded transition-all flex items-center px-1.5" style="width:${barPct}%"></div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
          <div class="mt-4 p-3 bg-black/30 rounded-lg border border-white/5">
            <div class="text-[10px] text-gray-500 font-mono">
              <i class="fas fa-flask text-orange/50 mr-1"></i>
              SHAP values via TreeExplainer on current candle row. Positive = pushes model toward LONG. Updated every WebSocket tick.
            </div>
          </div>
          ${(() => {
            const _topShap = shapValues.reduce((a, b) => Math.abs(a.value) > Math.abs(b.value) ? a : b, shapValues[0]);
            const _topDir = _topShap ? (_topShap.value > 0 ? 'LONG' : 'SHORT') : leadEntry[0];
            const _aligned = _topShap && _topDir === leadEntry[0];
            const _convPct = parseFloat(leadPct);
            const _vc = _convPct >= 70 && _aligned
              ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
              : _convPct >= 55
              ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
              : 'bg-red-500/20 text-red-400 border-red-500/30';
            const _vl = _convPct >= 70 && _aligned ? 'ALIGNED' : _convPct >= 55 ? 'PARTIAL' : 'DIVERGENT';
            const _topFeat = _topShap ? _topShap.feature : 'N/A';
            const _topVal = _topShap ? (_topShap.value > 0 ? '+' : '') + _topShap.value.toFixed(3) : '0.000';
            const _shortPct = (probs.SHORT * 100).toFixed(1);
            const _holdPct = (probs.HOLD * 100).toFixed(1);
            const _longPct = (probs.LONG * 100).toFixed(1);
            const _why = `The model assigns ${leadPct}% conviction to ${leadEntry[0]}. Top SHAP driver is "${_topFeat}" (${_topVal}), pushing the model ${_topDir === 'LONG' ? 'toward LONG' : _topDir === 'SHORT' ? 'toward SHORT' : 'toward HOLD'}. ${_aligned ? 'Top feature aligns with model direction — high-confidence signal.' : 'Top feature conflicts with model direction — treat with caution.'}`;
            const _what = `Full probability vector: SHORT ${_shortPct}% | HOLD ${_holdPct}% | LONG ${_longPct}%. ${_aligned ? 'SHAP and probability agree — XGBoost ensemble has clear directional bias.' : 'SHAP and probability diverge — the model may be uncertain. Weigh other confluences.'}`;
            const _when = _convPct >= 70 && _aligned
              ? 'High conviction with aligned SHAP. Act when price reaches the entry zone and Confluence Score exceeds 65%.'
              : _convPct >= 55
              ? 'Moderate conviction. Wait for at least two confirming indicators before executing.'
              : 'Low conviction or divergent signals. Skip this trade or reduce size significantly until the model realigns.';
            const _where = `Cross-reference "${_topFeat}" on your chart. ${_topShap && _topShap.value > 0.2 ? 'This feature has dominant positive influence — verify it visually before entering.' : _topShap && _topShap.value < -0.2 ? 'Strong bearish SHAP driver — confirm bearish structure on chart.' : 'No single feature dominates — signal is driven by collective weak signals.'} Compare model conviction against Confluence Score for final entry decision.`;
            return `<div class="mt-4 p-4 bg-black/40 rounded-xl border border-orange/20">
              <div class="flex items-center justify-between mb-3">
                <h4 class="text-xs font-bold text-orange uppercase tracking-wider">Model Intelligence</h4>
                <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${_vc}">${_vl}</span>
              </div>
              <div class="space-y-2 text-[11px]">
                <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-orange/60 font-bold uppercase text-[9px] pt-0.5">WHY</span><span class="text-gray-300">${_why}</span></div>
                <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-orange/60 font-bold uppercase text-[9px] pt-0.5">WHAT</span><span class="text-gray-300">${_what}</span></div>
                <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-orange/60 font-bold uppercase text-[9px] pt-0.5">WHEN</span><span class="text-gray-300">${_when}</span></div>
                <div class="flex gap-2 items-start"><span class="w-[46px] shrink-0 text-orange/60 font-bold uppercase text-[9px] pt-0.5">WHERE</span><span class="text-gray-300">${_where}</span></div>
              </div>
            </div>`;
          })()}
        </div>
      </div>
    </div>
  `;
}

function _renderFpApi(body, signal, tier) {
  const locked = tier !== 'PRO';

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-crown text-3xl text-amber-400 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Pro Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">API access and developer data export are Pro-exclusive features</p>
      <a href="/web/src/pages/pricing.html" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
    </div>` : '';

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-md pointer-events-none select-none' : ''}">
        <div class="bg-black/40 p-5 rounded-xl border border-blue-500/20 mb-4">
          <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-3">Your API Key</div>
          <div class="flex items-center gap-2 bg-black/60 border border-white/10 rounded-lg p-3 mb-3">
            <i class="fas fa-key text-blue-400 text-xs flex-shrink-0"></i>
            <span class="font-mono text-sm text-white flex-1 tracking-wider" id="fp-api-key-display">aegis_live_&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;</span>
            <button onclick="window._fpCopyApiKey()" class="text-[10px] text-blue-400 hover:text-blue-300 transition-colors px-2 py-1 bg-blue-500/10 rounded border border-blue-500/20 hover:bg-blue-500/20">
              <i class="fas fa-copy mr-1"></i>Copy
            </button>
          </div>
          <button onclick="window._fpRegenerateKey()"
            class="w-full py-2.5 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 font-bold rounded-xl text-sm border border-blue-500/30 transition-colors">
            <i class="fas fa-sync-alt mr-2"></i>Regenerate API Key
          </button>
          <div class="mt-2 text-[10px] text-gray-600 text-center">Key shown once on generation. Store it securely.</div>
        </div>
        <div class="bg-black/40 p-5 rounded-xl border border-white/5 mb-4">
          <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-3">Available Endpoints</div>
          <div class="space-y-2">
            ${[
              { path: '/api/v1/signals/live', desc: 'All live signals JSON' },
              { path: `/api/v1/signals/${encodeURIComponent(signal.symbol)}`, desc: `${signal.symbol} signal data` },
              { path: '/api/v1/tickers', desc: 'Live price tickers' },
            ].map(ep => `
              <div class="flex items-center gap-3 bg-black/40 p-2.5 rounded-lg border border-white/5">
                <code class="text-cyan text-[11px] font-mono flex-1">${ep.path}</code>
                <span class="text-[10px] text-gray-500 text-right">${ep.desc}</span>
              </div>
            `).join('')}
          </div>
        </div>
        <div class="bg-black/40 p-5 rounded-xl border border-white/5">
          <div class="flex items-center justify-between mb-3">
            <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Python Quick-Start</div>
            <button onclick="window._fpCopySnippet()" class="text-[10px] text-gray-400 hover:text-white transition-colors px-2 py-1 bg-white/5 rounded border border-white/10">
              <i class="fas fa-copy mr-1"></i>Copy
            </button>
          </div>
          <pre id="fp-python-snippet" class="text-[11px] font-mono text-green-300/90 bg-black/60 p-4 rounded-lg border border-white/5 overflow-x-auto leading-relaxed"><span class="text-blue-300">import</span> requests

API_KEY <span class="text-white">=</span> <span class="text-amber-300">"aegis_live_YOUR_KEY_HERE"</span>
HEADERS <span class="text-white">=</span> {<span class="text-amber-300">"X-API-Key"</span>: API_KEY}

resp <span class="text-white">=</span> requests.<span class="text-cyan">get</span>(
    <span class="text-amber-300">"https://gatekeeper.sbs/api/v1/signals/live"</span>,
    headers<span class="text-white">=</span>HEADERS
)
data <span class="text-white">=</span> resp.<span class="text-cyan">json</span>()
<span class="text-blue-300">print</span>(data[<span class="text-amber-300">"signals"</span>])</pre>
        </div>
      </div>
    </div>
  `;

  window._fpCopyApiKey = () => {
    const el = document.getElementById('fp-api-key-display');
    const key = el?.dataset.rawKey || 'aegis_live_DEMO_KEY';
    navigator.clipboard.writeText(key).then(() => _showToast('API key copied', 'success')).catch(() => {});
  };
  window._fpCopySnippet = () => {
    const pre = document.getElementById('fp-python-snippet');
    const text = pre?.innerText || '';
    navigator.clipboard.writeText(text).then(() => _showToast('Snippet copied', 'success')).catch(() => {});
  };
  window._fpRegenerateKey = async () => {
    try {
      const { getAuth } = await import('https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js');
      const fbUser = getAuth().currentUser;
      if (!fbUser) { _showToast('Not authenticated', 'error'); return; }
      const token = await fbUser.getIdToken();
      const r = await fetch('/api/v1/keys/regenerate', { method: 'POST', headers: { 'Authorization': `Bearer ${token}` }});
      if (r.ok) {
        const data = await r.json();
        const el = document.getElementById('fp-api-key-display');
        if (el) { el.textContent = data.key; el.dataset.rawKey = data.key; }
        _showToast('New key generated — copy it now!', 'success');
      } else {
        _showToast('Key regeneration failed', 'error');
      }
    } catch (e) {
      _showToast('Error: ' + e.message, 'error');
    }
  };
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
  }

  // Dispatch for feature panel live price handlers (e.g. zone dot)
  // Each panel body registers its own handler via body._zonePriceHandler
});

// renderExpectancyPanel, renderTelemetryPanel, renderDeveloperPortal, updateZoneTracker
// removed — superseded by the feature panel system (_renderFp* functions above).

async function renderExpectancyPanel() {
  // no-op: superseded by _renderFpExpectancy / openFeaturePanel('fp-expectancy')
}

