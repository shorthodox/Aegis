import { initializeTrialCountdown, fetchTrialStartFromFirestore } from './trial-countdown.js';
import { auth, db } from './gatekeeper.js?v=79.5';
import { doc, getDoc, setDoc } from 'https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js';

// Confidence values arrive on two scales depending on source: 0-1 (legacy
// meta probability) or 0-100 (edge_score from the live engine). Normalise
// to a 0-100 integer so the UI never shows "8977.7%".
function _confPct(v, fallback = 0) {
  const n = parseFloat(v);
  if (!isFinite(n) || n <= 0) return fallback;
  return Math.round(n > 1.5 ? n : n * 100);
}

function _showToast(msg, type = 'info') {
  const existing = document.getElementById('_dash-toast');
  if (existing) existing.remove();
  const c = { success: '#10b981', error: '#ef4444', info: '#06b6d4' };
  const ic = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
  const el = document.createElement('div');
  el.id = '_dash-toast';
  el.style.cssText = `position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;display:flex;align-items:center;gap:10px;padding:14px 20px;border-radius:12px;background:#111827;border:1px solid ${c[type]};color:${c[type]};font-size:.9rem;font-weight:600;box-shadow:0 0 20px ${c[type]}40;max-width:380px;`;
  el.innerHTML = `<i class="fas ${ic[type]}"></i><span>${msg}</span>`;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

// ============================================================
// SEALED SUBSCRIPTION STATE â€” console-proof
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
  // Only trust it if subscription_active is explicitly true â€” never bypass on plan name alone.
  if (typeof AuthManager !== 'undefined') {
    const u = AuthManager.getUser();
    if (u && u.subscription_active === true) {
      const p = (u.plan || u.tier || '').toLowerCase();
      if (p === 'pro' || p === 'premium' || p === 'intermediate' || p === 'basic') {
        console.log('[SubCheck] AuthManager fast-path â†’ paid (' + p + ')');
        return 'paid';
      }
    }
  }

  // Placeholder UID â€” skip Firestore, fall back to localStorage only
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

      // Paid plan â€” only valid if subscription is actually active in Firestore
      if (plan) {
        const p = plan.toLowerCase();
        if (p === 'premium' || p === 'pro' || p === 'intermediate' || p === 'basic') {
          const sub = data.subscription || {};
          if (sub.status === 'active') return 'paid';
          // Plan name set but no active subscription â€” fall through to trial date check
        }
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

      // If trial_end is missing, derive it from trial_start
      const rawStart = data.trial_start || data.trialStart || data.createdAt || data.trial?.startDate;
      if (rawStart) {
        const startDate = rawStart.toDate ? rawStart.toDate() : new Date(rawStart);
        if (!isNaN(startDate.getTime())) {
          const derivedEnd = new Date(startDate.getTime() + 3 * 24 * 60 * 60 * 1000);
          if (derivedEnd < now) return 'expired';
        }
      }

      // No Firestore end date â€” fall back to localStorage
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

    // No Firestore doc â€” check localStorage trial end as last resort
    const localEnd = localStorage.getItem('trial_end_timestamp');
    if (localEnd) {
      const localEndDate = new Date(localEnd);
      if (!isNaN(localEndDate.getTime()) && localEndDate < now) return 'expired';
    }
    return 'trial';
  } catch (error) {
    console.error('[SubCheck] Invalid database error â€” falling back to AuthManager/localStorage:', error);
    if (error.code) {
        console.error(`[SubCheck] Firestore error code: ${error.code}. This is a database problem.`);
    }

    // ALWAYS enforce local expiry first, even if there's a database error
    const localEnd = localStorage.getItem('trial_end_timestamp');
    if (localEnd) {
      const localEndDate = new Date(localEnd);
      if (!isNaN(localEndDate.getTime()) && localEndDate < Date.now()) {
        console.warn('[SubCheck] Local storage confirms expired. Enforcing expiry despite DB error.');
        return 'expired';
      }
    }

    // On error: prefer AuthManager knowledge over defaulting to 'expired' ONLY if not locally expired
    if (typeof AuthManager !== 'undefined') {
      const u = AuthManager.getUser();
      if (u) {
        const p = (u.plan || u.tier || '').toLowerCase();
        if (p === 'pro' || p === 'premium' || p === 'intermediate' || p === 'basic') return 'paid';
        if (p === 'trial' || p === 'free_tier') return 'trial';
      }
      // Active token â€” safe to assume trial rather than blocking the user
      if (AuthManager.getToken()) return 'trial';
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
  let timeoutId;
  const authPromise = new Promise((resolve) => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      unsubscribe();
      if (timeoutId) clearTimeout(timeoutId);
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
    timeoutId = setTimeout(() => {
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

  // Fully hide background content â€” opacity:0 so signals are invisible behind the overlay.
  if (dashboardContent) {
    dashboardContent.classList.remove('hidden');
    dashboardContent.classList.add('sub-expired');
  }

  // Hide the trial countdown banner so "Loading..." never floats above the expired screen.
  const trialBanner = document.getElementById('trialBanner');
  if (trialBanner) trialBanner.classList.add('hidden');

  // Clear signals immediately so nothing bleeds through during the transition.
  const signalsContainer = document.getElementById('signalsContainer');
  if (signalsContainer) signalsContainer.innerHTML = '';

  // Show the full-screen opaque overlay (z-[9999] covers header and all panels).
  const overlay = document.getElementById('subscriptionExpiredOverlay');
  if (overlay) {
    overlay.classList.remove('hidden');
  } else {
    // Fallback: create the overlay programmatically if the HTML element is missing.
    createExpiredCard();
  }

  // Block all feature access
  blockAllFeatures();

  // Force click handlers and pointer-events on the overlay buttons to guarantee navigation
  const subOverlay = document.getElementById('subscriptionExpiredOverlay');
  if (subOverlay) {
    const subBtn = subOverlay.querySelector('a[href*="pricing"]');
    const homeBtn = document.getElementById('returnHomeBtn');

    if (subBtn) {
      subBtn.style.setProperty('pointer-events', 'auto', 'important');
      subBtn.addEventListener('click', (e) => {
        window.location.href = '/pricing';
      });
    }

    if (homeBtn) {
      homeBtn.style.setProperty('pointer-events', 'auto', 'important');
      homeBtn.addEventListener('click', (e) => {
        window.location.href = '/';
      });
    }
  }
}

function clearExpiredView() {
  const dashboardContent = document.getElementById('dashboard-main-content');
  if (dashboardContent) {
    dashboardContent.classList.remove('hidden');
    dashboardContent.classList.remove('sub-expired');
  }

  // Hide the full-screen overlay.
  const overlay = document.getElementById('subscriptionExpiredOverlay');
  if (overlay) overlay.classList.add('hidden');

  // Also hide the legacy card in case it was created by an older code path.
  const expiredCard = document.getElementById('access-expired-card');
  if (expiredCard) expiredCard.classList.add('hidden');

  // Restore any elements that were blocked by blockAllFeatures so UI is fully interactive again.
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
        window.location.href = '/pricing';
      });
    }

    const homeBtn = document.getElementById('expired-home-btn');
    if (homeBtn) {
      homeBtn.addEventListener('click', () => {
        window.location.href = '/';
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

  // Disconnect the live server WebSocket if it's running
  if (typeof window.disconnectGatekeeperWebSocket === 'function') {
    window.disconnectGatekeeperWebSocket();
  }

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
    const isExpiredElement = el.closest('#access-expired-card, #subscriptionExpiredOverlay') !== null || (el.id && el.id.includes('expired'));

    // Safety check so we don't lock the user in completely
    const isLogout = el.id === 'logout-btn' || el.id === 'btn-logout' || el.classList.contains('logout-button');
    const isNav = el.closest('nav') !== null || el.closest('header') !== null;

    // Never block elements inside the signal modal or feature panels â€”
    // those overlays manage their own access gating via lock overlays.
    const isModalOrPanel = el.closest('#signalDetailsModal, #fp-confluence, #fp-zones, #fp-expectancy, #fp-shap, #fp-api') !== null;

    // Never block the Guardian help panel â€” it must always be dismissible.
    const isGuardian = el.closest('#guardian-drawer') !== null;

    if (!isExpiredElement && !isLogout && !isNav && !isModalOrPanel && !isGuardian) {
      el.dataset.aegisBlocked = '1';
      el.style.pointerEvents = 'none';
      el.style.opacity = '0.3';
    }
  });

  console.log('âœ… All features blocked - subscription expired');
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

  // Only restore elements that were explicitly blocked by aegis â€” avoids clobbering intentional disables.
  const blockedElements = document.querySelectorAll('[data-aegis-blocked]');
  blockedElements.forEach(el => {
    el.style.removeProperty('pointer-events');
    el.style.removeProperty('opacity');
    delete el.dataset.aegisBlocked;
  });

  console.log('âœ… All features unlocked - subscription active');
}

// ============================================================
// CHECK FEATURE ACCESS - PREVENT USE IF EXPIRED
// ============================================================
function canAccessFeatures() {
  if (window.isSubscriptionActive === false) {
    console.warn('â›” Feature access blocked - subscription expired');
    setExpiredView();
    return false;
  }
  return true;
}

// ============================================================
// EXPORT FUNCTIONS FOR EXTERNAL USE â€” sealed so they cannot be
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
      window.location.href = '/';
    });
    console.log('âœ… Logo click handler initialized');
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
                    class="text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded border hidden">â€”</span>
            </div>
            <span id="market-card-price-${idStr}"
                  class="live-price text-lg font-mono transition-colors duration-300"
                  data-symbol="${idStr}">â€”</span>
            <div class="flex items-center justify-between mt-auto">
              <span id="market-card-change-${idStr}" class="text-[10px] font-mono text-gray-500">â€”</span>
              <span class="text-[9px] text-gray-600 group-hover:text-cyan/60 transition-colors">
                <i class="fas fa-arrow-right"></i>
              </span>
            </div>
          </div>
        `;
      });
      container.innerHTML = html;
      marketCardsInitialized = true;
      
      // Fallback: manually update market cards on priceUpdate if gatekeeper querySelector fails
      window.addEventListener('priceUpdate', (e) => {
        const { symbol, price } = e.detail;
        const id = symbol.replace('/', '-');
        const el = document.getElementById(`market-card-price-${id}`);
        if (el) {
          const priceStr = price < 0.01 ? price.toFixed(6) : price.toFixed(4);
          el.textContent = `$${priceStr}`;
        }
      });
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
      // Only a FIRED signal counts as a directional setup. The rest of the app
      // (SELL/BUY Setups, cockpit, card data-dir) all require signal.fire; showing
      // the raw bias here made the overview flag SHORT/LONG for tokens that never
      // fired (e.g. ETH/SOL SHORT bias), so it disagreed with the "1 signal" count
      // in SELL Setups. Gate on fire so the overview matches the fired-only views.
      const fired = match.fire === true;
      const dir = (match.direction || match.side || '').toUpperCase();
      const isLong  = fired && (dir === 'LONG'  || dir === 'BUY');
      const isShort = fired && (dir === 'SHORT' || dir === 'SELL');
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
        // NEUTRAL or unknown direction â€” hide badge, neutral border
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
    window.openSignalDetails(sym, window.activeTimeframe || '1h');
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

  if (status === 'paid') {
    document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display')
      .forEach(el => el.style.display = 'none');
    return;
  }

  const now = Date.now();
  if (trialSetupRunning || (now - lastTrialSetupTime < 5000)) {
    console.log('â³ Skipping duplicate setupTrialNonBlocking call to prevent request bloat');
    return;
  }
  trialSetupRunning = true;

  const cacheKey = `trialStart_${userId}`;

  try {
    // 1. Firestore is the PRIMARY source of truth â€” fetch it first
    const freshStart = await fetchTrialStartFromFirestore(userId);

    if (freshStart) {
      // Firestore returned a valid start â€” use it as the source of truth
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
      // Firestore returned null â€” fall back to cached start from localStorage
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
  if (initialized) return;
  if (!document.getElementById('dashboard-main-content') && !document.getElementById('market-token-cards')) return;

  initialized = true;
  initializeLogoClickHandler();

  // Initialize the empty cards, gatekeeper.js will fill the prices natively
  fetchLiveMarketData();

  document.addEventListener('trialExpired', () => {
    console.log('ðŸ”’ Trial expired event triggered');
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

    // Await true auth state â€” always resolve to a real Firebase UID
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

// â”€â”€ Periodic subscription re-verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
      unblockFeatures();
    }
  } catch (_) {
    // Silently ignore â€” the WS tick is the primary enforcement mechanism.
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

  // â”€â”€ Signal Status Banner (reversal anticipation states) â”€â”€
  const sigStatus = signal.signal_status || 'ACTIVE';
  let sdStatusEl = document.getElementById('sd-signal-status');
  if (!sdStatusEl) {
    sdStatusEl = document.createElement('div');
    sdStatusEl.id = 'sd-signal-status';
    const priceRow = document.getElementById('sd-live-price');
    if (priceRow && priceRow.parentElement && priceRow.parentElement.parentElement) {
      priceRow.parentElement.parentElement.insertBefore(sdStatusEl, priceRow.parentElement);
    }
  }
  sdStatusEl.innerHTML = '';
  sdStatusEl.className = 'hidden';
  if (sigStatus === 'EXPIRED') {
    sdStatusEl.innerHTML = '<i class="fas fa-clock-rotate-left mr-1"></i>SIGNAL EXPIRED â€” Move exceeded 60% of target before entry';
    sdStatusEl.className = 'w-full px-3 py-2 mb-3 rounded-lg text-xs font-bold tracking-wider bg-gray-700/40 text-gray-400 border border-gray-600/40 text-center';
  } else if (sigStatus === 'AWAITING_CONFIRMATION') {
    const cc = signal.candles_confirmed || 0;
    sdStatusEl.innerHTML = '<i class="fas fa-hourglass-half mr-1"></i>CONFIRMING REVERSAL â€” ' + cc + '/3 candles confirmed';
    sdStatusEl.className = 'w-full px-3 py-2 mb-3 rounded-lg text-xs font-bold tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/30 text-center';
  } else if (sigStatus === 'AWAITING_SR_BREAK') {
    sdStatusEl.innerHTML = '<i class="fas fa-lock mr-1"></i>WAITING FOR SUPPORT BREAK â€” SELL held until support breaks';
    sdStatusEl.className = 'w-full px-3 py-2 mb-3 rounded-lg text-xs font-bold tracking-wider bg-orange-500/10 text-orange-400 border border-orange-500/30 text-center';
  } else if (sigStatus === 'SR_BREAK_CONFIRMED') {
    sdStatusEl.innerHTML = '<i class="fas fa-unlock mr-1"></i>SUPPORT BROKEN â€” SELL confirmed after S/R breach';
    sdStatusEl.className = 'w-full px-3 py-2 mb-3 rounded-lg text-xs font-bold tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-center';
  } else if (sigStatus === 'CONFIRMED' && (signal.reversal_score || 0) > 0) {
    const rs = ((signal.reversal_score || 0) * 100).toFixed(0);
    const tags = (signal.reversal_signals || []).slice(0, 3).join(' Â· ');
    sdStatusEl.innerHTML = '<i class="fas fa-rotate-left mr-1"></i>EARLY REVERSAL â€” Score ' + rs + '% Â· ' + (tags || 'Technical Setup');
    sdStatusEl.className = 'w-full px-3 py-2 mb-3 rounded-lg text-xs font-bold tracking-wider bg-cyan/10 text-cyan border border-cyan/30 text-center';
  }

  // Price & Levels
  const currentPrice = window.currentTickers && window.currentTickers[signal.symbol]
    ? parseFloat(window.currentTickers[signal.symbol])
    : (signal.entry_price || 0);

  document.getElementById('sd-live-price').textContent = `$${currentPrice.toFixed(4)}`;
  document.getElementById('sd-confidence').textContent = `${_confPct(signal.ai_prob || signal.confidence)}%`;
  document.getElementById('sd-sl').textContent = `$${(signal.sl || 0).toFixed(4)}`;
  document.getElementById('sd-tp').textContent = `$${(signal.tp || 0).toFixed(4)}`;

  // Set active dataset for real-time tracking
  modal.dataset.activeSymbol = signal.symbol;
  modal.dataset.activeTimeframe = signal.timeframe || '1h';

  // Store current signal for feature panels
  window._fpSignal = signal;

  // Populate Feature Access Cards
  // Confluence values are in [0,10] scale (5=neutral); convert to display %
  // by mapping: 0â†’0%, 5â†’50%, 10â†’100%. Backward-compat: if abs(v)<=1.1 it's
  // old [-1,+1] scale â€” convert first.
  function _c10(v) {
    const n = parseFloat(v) || 5;
    if (Math.abs(n) <= 1.05) return parseFloat(((n + 1) / 2 * 10).toFixed(1));
    return Math.min(10, Math.max(0, n));
  }
  function _cPct(v) { return Math.round(_c10(v) * 10); }  // [0,10] â†’ [0,100]%

  const rawConf = signal.confluence || {};
  const _confDisp = {
    trend:       _cPct(rawConf.trend       ?? 5),
    momentum:    _cPct(rawConf.momentum    ?? 5),
    volume:      _cPct(rawConf.volume      ?? 5),
    smart_money: _cPct(rawConf.smart_money ?? 5),
    candle:      _cPct(rawConf.candle      ?? 5),
    total:       _cPct(rawConf.total       ?? 5),
    summary:     rawConf.summary || 'â€”',
  };
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

  // Correct SL/TP field names (suggested_sl/tp come from live_engine;
  // sl/tp are the normalised copies we now set in gatekeeper.js)
  const _sl = signal.suggested_sl || signal.sl || 0;
  const isLongSig = (signal.direction || '').toUpperCase() === 'LONG' || (signal.signal || '').toUpperCase() === 'BUY';
  const _tp = signal.suggested_tp || (isLongSig ? signal.bull_tp1 : signal.bear_tp1) || signal.tp || 0;
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
  // Use real p_buy/p_sell/p_hold; fall back to legacy raw_probabilities if present
  const _rp = signal.raw_probabilities;
  const _probs = (signal.p_buy || signal.p_sell)
    ? { SHORT: signal.p_sell || 0, HOLD: signal.p_hold || 0, LONG: signal.p_buy || 0 }
    : _rp
      ? { SHORT: (_rp.SHORT||0)/100, HOLD: (_rp.HOLD||0)/100, LONG: (_rp.LONG||0)/100 }
      : { SHORT: 0, HOLD: 0, LONG: 0 };
  const _lead = Object.entries(_probs).sort((a,b) => b[1]-a[1])[0];
  const _leadPct = (_lead[1]*100).toFixed(0);
  const _leadColor = _lead[0]==='LONG' ? 'text-green-400' : _lead[0]==='SHORT' ? 'text-red-400' : 'text-gray-400';
  const _metaConf = _confPct(signal.meta_confidence);
  _shapLeadHTML = `
    <div class="text-lg font-black ${_leadColor}">${_leadPct}% ${_lead[0]}</div>
    ${_metaConf > 0 ? `<div class="text-[9px] text-gray-500 mt-0.5">Meta: ${_metaConf}% vs ${_confPct(signal.threshold, 60)}% required</div>` : ''}`;

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
            <div class="h-full ${_confDisp.trend >= 55 ? 'bg-cyan/70' : 'bg-rose-400/70'}" style="width:${_confDisp.trend}%"></div>
          </div>
          <span class="text-[10px] font-mono ${_confDisp.trend >= 55 ? 'text-cyan' : 'text-rose-400'} w-8 text-right">${_confDisp.trend}%</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1 bg-black/50 rounded overflow-hidden">
            <div class="h-full ${_confDisp.momentum >= 55 ? 'bg-blue-400/70' : 'bg-rose-400/70'}" style="width:${_confDisp.momentum}%"></div>
          </div>
          <span class="text-[10px] font-mono ${_confDisp.momentum >= 55 ? 'text-blue-300' : 'text-rose-400'} w-8 text-right">${_confDisp.momentum}%</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1 bg-black/50 rounded overflow-hidden">
            <div class="h-full ${_confDisp.volume >= 55 ? 'bg-violet-400/70' : 'bg-rose-400/70'}" style="width:${_confDisp.volume}%"></div>
          </div>
          <span class="text-[10px] font-mono ${_confDisp.volume >= 55 ? 'text-violet-300' : 'text-rose-400'} w-8 text-right">${_confDisp.volume}%</span>
        </div>
      </div>
      <div class="mt-2 flex items-center justify-between">
        <span class="text-[10px] font-mono text-gray-500">Total: <b class="${_confDisp.total >= 55 ? 'text-cyan' : _confDisp.total <= 45 ? 'text-rose-400' : 'text-gray-400'}">${_confDisp.total}%</b></span>
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
          <button id="sd-paper-trade-btn"
            class="flex-1 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold py-3 rounded-xl uppercase tracking-wider text-sm shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:-translate-y-0.5 transform transition-all">
            <i class="fas fa-play-circle mr-2"></i>Paper Trade
          </button>
          <button id="sd-execute-btn"
            class="px-5 bg-white/10 hover:bg-white/20 text-white font-bold py-3 rounded-xl uppercase tracking-wider text-sm border border-white/20 transition-colors whitespace-nowrap">
            <i class="fas fa-satellite-dish mr-1"></i>Demat
          </button>
        </div>
        <div class="text-center text-[10px] text-gray-500 py-0.5">
          Prefer advanced charts? <a href="https://www.tradingview.com/paper-trading/" target="_blank" rel="noopener noreferrer" class="text-cyan/70 hover:text-cyan underline">Try TradingView Paper Trading</a>
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
        <a href="/pricing"
          class="flex items-center justify-center gap-2 bg-gradient-to-r from-amber-500/10 to-orange/10 hover:from-amber-500/20 hover:to-orange/20 text-amber-400 font-bold py-2 rounded-xl text-xs border border-amber-500/30 transition-colors">
          <i class="fas fa-crown"></i>Unlock Pro &mdash; Copy JSON, Advanced Analytics &amp; more
        </a>`}
      </div>
    `;

    document.getElementById('sd-execute-btn')?.addEventListener('click', async () => {
      try {
        let token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
        if (!token && typeof AuthManager !== 'undefined') {
          token = AuthManager.getToken();
        }
        if (!token) {
          alert('You must be logged in to execute trades via API.');
          return;
        }

        const direction = signal.direction || (signal.signal && signal.signal.includes('BUY') ? 'LONG' : (signal.signal && signal.signal.includes('SELL') ? 'SHORT' : 'NEUTRAL'));
        const tradeData = {
            symbol: signal.symbol,
            side: direction,
            entryPrice: signal.entry_price || 0,
            stopLoss: signal.sl || 0,
            takeProfit: signal.tp || 0,
            riskPercent: 2, // Default 2%
            leverage: 10,   // Default 10x
            positionUnits: 1, 
            notionalValue: (signal.entry_price || 0) * 10,
            status: 'open',
            signalId: signal.signal_id || signal.signalId || null
        };

        const btn = document.getElementById('sd-execute-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Executing...';
        btn.disabled = true;

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
            throw new Error(errorData.detail || 'Failed to execute trade on API');
        }
        
        btn.innerHTML = '<i class="fas fa-check mr-2"></i>Sent to Demat';
        setTimeout(() => {
            document.getElementById('token-details-modal')?.classList.add('hidden');
            document.getElementById('signalDetailsModal')?.classList.add('hidden');
        }, 1500);
      } catch (err) {
        console.error('Failed to execute trade:', err);
        alert('Trade API execution failed: ' + err.message);
        const btn = document.getElementById('sd-execute-btn');
        btn.innerHTML = '<i class="fas fa-satellite-dish mr-1"></i>Demat';
        btn.disabled = false;
      }
    });

    document.getElementById('sd-paper-trade-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      // Close both modal layers explicitly â€” don't rely on closure variable
      const _mw = document.getElementById('token-details-modal');
      const _mi = document.getElementById('signalDetailsModal');
      if (_mw) _mw.classList.add('hidden');
      if (_mi) _mi.classList.add('hidden');
      if (typeof window.addToSignalHistory === 'function') window.addToSignalHistory(signal);
      if (typeof window.switchRoom === 'function') window.switchRoom('terminal');
      // Prefill AFTER room switch so inputs are in the active room
      if (typeof window.prefillFromSignal === 'function') window.prefillFromSignal(signal);
    });

    document.getElementById('sd-copy-signal-btn')?.addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(signal, null, 2))
        .then(() => _showToast('Signal JSON copied to clipboard!', 'success'))
        .catch(() => _showToast('Copy failed â€” check browser permissions', 'error'));
    });

    document.getElementById('sd-view-analytics-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      document.getElementById('token-details-modal')?.classList.add('hidden');
      document.getElementById('signalDetailsModal')?.classList.add('hidden');
      if (typeof window.switchRoom === 'function') window.switchRoom('analytics');
    });
  }

  modal.classList.remove('hidden');
}

function _closeSignalModal() {
  document.getElementById('token-details-modal')?.classList.add('hidden');
  document.getElementById('signalDetailsModal')?.classList.add('hidden');
}

function initModals() {
  const wrapper = document.getElementById('token-details-modal');
  const closeBtn = document.getElementById('closeSignalDetailsBtn');

  if (closeBtn) {
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _closeSignalModal();
    });
  }

  if (wrapper) {
    wrapper.addEventListener('click', (e) => {
      const card = e.target.closest('[data-open-panel], .feature-card-trigger');
      if (card) {
        const panelId = card.dataset.openPanel || card.dataset.target;
        if (panelId && typeof window.openFeaturePanel === 'function') {
          window.openFeaturePanel(panelId);
        }
        return;
      }
      // Close when clicking the backdrop (not the inner card)
      if (e.target === wrapper) {
        _closeSignalModal();
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

// Explicit list â€” never accidentally matches *-body divs
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
  // Normalize confluence to display % scale using the same _c10 helper
  // defined in showSignalDetailsModal scope (available as closure or re-declared)
  function _c10fp(v) {
    const n = parseFloat(v);
    if (isNaN(n)) return 5;
    if (Math.abs(n) <= 1.05) return parseFloat(((n + 1) / 2 * 10).toFixed(1));
    return Math.min(10, Math.max(0, n));
  }
  const rawConf = signal.confluence || {};
  const confluence = {
    trend:       Math.round(_c10fp(rawConf.trend       ?? 5) * 10),
    momentum:    Math.round(_c10fp(rawConf.momentum    ?? 5) * 10),
    volume:      Math.round(_c10fp(rawConf.volume      ?? 5) * 10),
    smart_money: Math.round(_c10fp(rawConf.smart_money ?? 5) * 10),
    bands:       Math.round(_c10fp(rawConf.bands       ?? 5) * 10),
    candle:      Math.round(_c10fp(rawConf.candle      ?? 5) * 10),
    total:       Math.round(_c10fp(rawConf.total       ?? 5) * 10),
  };
  // Recompute displayed total from all 6 visible categories using the same
  // weights as compute_category_confluence â€” this ensures the total shown
  // always matches the sum of the bars the user can actually see.
  const _wTotal = (
    confluence.trend       * 2.0 +
    confluence.momentum    * 1.5 +
    confluence.volume      * 1.5 +
    confluence.smart_money * 1.5 +
    confluence.bands       * 1.0 +
    confluence.candle      * 0.5
  ) / (2.0 + 1.5 + 1.5 + 1.5 + 1.0 + 0.5);
  confluence.total = Math.round(_wTotal);

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-lock text-3xl text-gray-500 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Intermediate Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Upgrade to Intermediate or Pro to access detailed confluence analysis</p>
      <a href="/pricing" class="px-6 py-2 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold rounded-xl text-sm">Upgrade Now</a>
    </div>` : '';

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-sm pointer-events-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
          <span class="text-sm text-gray-400">${signal.timeframe || '1h'}</span>
        </div>
        <div class="space-y-3">
          ${[
            { label: 'Trend',          sublabel: 'EMA stack Â· macro Â· market structure',         val: confluence.trend,       weight: 'Ã—2.0', color: 'bg-cyan',        textColor: 'text-cyan' },
            { label: 'Momentum',       sublabel: 'RSI Â· MACD Â· Stochastic Â· CCI',               val: confluence.momentum,    weight: 'Ã—1.5', color: 'bg-blue-400',    textColor: 'text-blue-300' },
            { label: 'Volume / Flow',  sublabel: 'CMF Â· MFI Â· OBV delta',                       val: confluence.volume,      weight: 'Ã—1.5', color: 'bg-violet-400',  textColor: 'text-violet-300' },
            { label: 'Smart Money',    sublabel: 'BOS Â· CHoCH Â· S/R proximity',                 val: confluence.smart_money, weight: 'Ã—1.5', color: 'bg-amber-400',   textColor: 'text-amber-300' },
            { label: 'Price Position', sublabel: 'BB% Â· ATR band Â· Donchian Â· quantile',        val: confluence.bands,       weight: 'Ã—1.0', color: 'bg-teal-400',    textColor: 'text-teal-300' },
            { label: 'Candle Patt.',   sublabel: 'Hammer Â· Engulfing Â· Morning Star',            val: confluence.candle,      weight: 'Ã—0.5', color: 'bg-pink-400',    textColor: 'text-pink-300' },
          ].map(item => {
            const isBull = item.val >= 55;
            const isBear = item.val <= 45;
            const barColor = isBull ? item.color : isBear ? 'bg-rose-500/60' : 'bg-gray-600/60';
            const valColor = isBull ? item.textColor : isBear ? 'text-rose-400' : 'text-gray-500';
            return `
            <div class="bg-black/40 p-3 rounded-xl border border-white/5">
              <div class="flex justify-between items-center mb-1.5">
                <div class="flex items-center gap-2">
                  <div class="text-sm font-bold text-white">${item.label}</div>
                  <span class="text-[9px] text-amber-400/60 font-bold">${item.weight}</span>
                </div>
                <div class="flex items-center gap-1.5">
                  <span class="text-[9px] font-bold ${isBull ? 'text-emerald-400' : isBear ? 'text-rose-400' : 'text-gray-500'}">${isBull ? 'â–²' : isBear ? 'â–¼' : 'â‰ˆ'}</span>
                  <div class="font-black font-mono text-lg ${valColor}">${item.val}%</div>
                </div>
              </div>
              <div class="h-1.5 bg-black/60 rounded-full overflow-hidden">
                <div class="h-full ${barColor} rounded-full transition-all" style="width:${item.val}%"></div>
              </div>
              <div class="text-[9px] text-gray-600 mt-1">${item.sublabel}</div>
            </div>`;
          }).join('')}
        </div>
        <div class="mt-4 bg-cyan/5 border border-cyan/20 p-4 rounded-xl">
          <div class="flex items-center justify-between">
            <span class="text-sm font-bold text-white">Weighted Total Score</span>
            <span class="text-2xl font-black font-mono ${confluence.total >= 65 ? 'text-emerald-400' : confluence.total <= 40 ? 'text-rose-400' : 'text-cyan'}">${confluence.total}%</span>
          </div>
          <div class="h-1.5 bg-black/50 rounded-full mt-2 overflow-hidden">
            <div class="h-full ${confluence.total >= 65 ? 'bg-gradient-to-r from-cyan to-emerald-400' : confluence.total <= 40 ? 'bg-gradient-to-r from-rose-500 to-amber-400' : 'bg-gradient-to-r from-gray-500 to-gray-400'} rounded-full"
                 style="width:${confluence.total}%"></div>
          </div>
          <div class="text-[10px] text-gray-500 mt-2">${rawConf.summary || (confluence.total >= 65 ? 'Bullish' : confluence.total <= 40 ? 'Bearish' : 'Neutral')} Â· 50% = neutral</div>
        </div>
        <div class="mt-4 p-3 bg-black/30 rounded-lg border border-white/5">
          <div class="text-[10px] text-gray-500 font-mono">
            <i class="fas fa-info-circle text-cyan/50 mr-1"></i>
            Weights computed via XGBoost gradient boosting ensemble. Values represent normalized feature importance vectors for the current candle state. Updated on each WebSocket tick.
          </div>
        </div>
        ${(() => {
          const _sc = confluence.total;
          const _dir = (signal.direction || 'LONG').toUpperCase();
          const _alignment = _dir === 'LONG' ? _sc : _dir === 'SHORT' ? (100 - _sc) : 50;
          
          const _vc = _alignment >= 65 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : _alignment >= 50 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-red-500/20 text-red-400 border-red-500/30';
          const _vl = _alignment >= 65 ? 'STRONG' : _alignment >= 50 ? 'MODERATE' : 'WEAK';
          
          const _cats = [['Trend',confluence.trend],['Momentum',confluence.momentum],['Volume',confluence.volume],['Smart Money',confluence.smart_money],['Candle',confluence.candle]];
          const _dom = _cats.reduce((a,b)=>b[1]>a[1]?b:a, _cats[0]);
          const _bias = _dir === 'LONG' ? 'bullish' : _dir === 'SHORT' ? 'bearish' : 'neutral';
          const _agrCount = _cats.filter(c => _dir==='LONG' ? c[1]>=55 : c[1]<=45).length;
          
          const _why = `Weighted total is ${_sc}% (50% = neutral). ${_agrCount}/5 indicator groups support the ${_bias} ${_dir} direction. ${rawConf.summary || ''}.`;
          const _what = `Dominant driver: ${_dom[0]} at ${_dom[1]}%. ${_alignment >= 65 ? 'Strong multi-group alignment â€” signal has high structural backing.' : _alignment >= 50 ? 'Moderate alignment â€” signal is valid but not a textbook setup.' : 'Low alignment â€” major groups are disagreeing.'}`;
          const _when = _alignment >= 65
            ? 'Strong setup â€” full position size is justified at current price.'
            : _alignment >= 50
            ? 'Entry is viable â€” reduce position size by 30% to account for incomplete alignment.'
            : 'Stand aside â€” wait for confluence to improve before committing capital.';
          
          function _fmtPx(v) { v=parseFloat(v)||0; if(!v)return'â€”'; if(v>=100)return'$'+v.toLocaleString('en-US',{maximumFractionDigits:2}); if(v>=1)return'$'+v.toFixed(4); return'$'+v.toFixed(6); }
          const _ep = _fmtPx(signal.entry_price || signal.price);
          const _slVal = signal.suggested_sl || signal.sl || 0;
          const _slStr = _fmtPx(_slVal);
          
          const _where = `Execute near ${_ep} with SL at ${_slStr}. ${_alignment >= 65 ? 'Full position size justified by strong confluence.' : _alignment >= 50 ? 'Scale in with 50-70% size. Add on confirmation.' : 'Observe only â€” wait for groups to align before entering.'}`;
          
          return `<div class="mt-4 p-4 bg-black/40 rounded-xl border border-cyan/20">
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-xs font-bold text-cyan uppercase tracking-wider">Signal Intelligence</h4>
              <div class="flex items-center gap-2">
                <span class="text-[10px] text-gray-400">Directional Alignment: <strong class="${_alignment >= 65 ? 'text-emerald-400' : _alignment >= 50 ? 'text-amber-400' : 'text-red-400'}">${_alignment}%</strong></span>
                <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${_vc}">${_vl}</span>
              </div>
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
  const isLongDir = (signal.direction || 'LONG').toUpperCase() === 'LONG';
  // Use real field names from live_engine._build_signal_entry
  const sl  = signal.suggested_sl || signal.sl  || 0;
  const tp  = signal.suggested_tp || (isLongDir ? signal.bull_tp1 : signal.bear_tp1) || signal.tp  || 0;
  const tp2 = isLongDir ? (signal.bull_tp2 || 0) : (signal.bear_tp2 || 0);
  const entry = signal.entry_price || signal.price || 0;
  const currentPrice = window.currentTickers?.[signal.symbol]
    ? parseFloat(window.currentTickers[signal.symbol]) : entry;
  const hasZone = sl > 0 && tp > 0 && Math.abs(tp - sl) > 1e-10;
  const absRange = hasZone ? Math.abs(tp - sl) : 1;
  const entryPct = hasZone
    ? (isLongDir
        ? Math.max(2, Math.min(98, ((entry - sl) / absRange * 100)))
        : Math.max(2, Math.min(98, ((sl - entry) / absRange * 100))))
    : 50;
  const curPct = hasZone
    ? (isLongDir
        ? Math.max(2, Math.min(98, ((currentPrice - sl) / absRange * 100)))
        : Math.max(2, Math.min(98, ((sl - currentPrice) / absRange * 100))))
    : 50;

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-lock text-3xl text-gray-500 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Intermediate Plan Required</h3>
      <a href="/pricing" class="px-6 py-2 bg-gradient-to-r from-cyan to-blue-600 text-white font-bold rounded-xl text-sm mt-2">Upgrade Now</a>
    </div>` : '';

  let statusMsg = '';
  if (!locked && entry > 0 && currentPrice > 0) {
    const pctFromEntry = ((currentPrice - entry) / entry) * 100;
    const isLong = (signal.direction || 'LONG') === 'LONG';
    const inProfit = isLong ? currentPrice > entry : currentPrice < entry;
    if (inProfit && Math.abs(pctFromEntry) > 0.2) {
      statusMsg = `<div class="mt-3 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-bold animate-pulse">
        âš ï¸ Breakout In Progress: Do Not Chase
      </div>`;
    } else {
      statusMsg = `<div class="mt-3 px-3 py-2 rounded-lg bg-cyan/10 border border-cyan/30 text-cyan text-xs font-bold">
        âš¡ Inside Active Entry Buffer Zone
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
              ${hasZone && entry > sl ? `1:${((tp - entry) / (entry - sl)).toFixed(2)}` : '&mdash;'}
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
          const _rrRaw = hasZone && entry > sl ? ((tp - entry) / (entry - sl)) : 0;
          const _rr = _rrRaw.toFixed(2);
          const _vc = _rrRaw >= 2 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : _rrRaw >= 1.5 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-red-500/20 text-red-400 border-red-500/30';
          const _vl = _rrRaw >= 2 ? 'HIGH R:R' : _rrRaw >= 1.5 ? 'FAIR R:R' : 'LOW R:R';
          const _pctFromEntry = entry > 0 ? ((currentPrice - entry) / entry * 100) : 0;
          const _inProfit = _isLong ? currentPrice > entry : currentPrice < entry;
          const _stopDist = entry && sl ? Math.abs(((entry - sl) / entry * 100)).toFixed(2) : '0.00';
          const _tgtDist = entry && tp ? Math.abs(((tp - entry) / entry * 100)).toFixed(2) : '0.00';
          const _why = `Price is ${Math.abs(_pctFromEntry).toFixed(2)}% ${_inProfit ? 'ahead of' : 'behind'} the entry at $${entry.toFixed(4)} â€” currently ${curPct.toFixed(1)}% across the SL-to-TP corridor.`;
          const _what = `Risk/Reward is 1:${_rr}. Stop distance is ${_stopDist}% ($${sl.toFixed(4)}); target distance is ${_tgtDist}% ($${tp.toFixed(4)}). ${_rrRaw >= 2 ? 'Excellent asymmetry â€” reward far outweighs risk.' : _rrRaw >= 1.5 ? 'Acceptable ratio â€” proceed with standard sizing.' : 'Tight reward relative to risk â€” consider skipping or waiting for a better entry.'}`;
          const _when = curPct < 30
            ? 'Price is near the entry zone â€” valid window to initiate the position.'
            : curPct < 60
            ? 'Price has moved into the middle of the range. Entry is still viable but chase risk is elevated.'
            : 'Price is deep into the TP corridor. Do not chase â€” wait for a pullback to the entry zone.';
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
    if (dot) {
      const _sl2 = signal.suggested_sl || signal.sl || 0;
      const _tp2 = signal.suggested_tp || (isLongDir ? signal.bull_tp1 : signal.bear_tp1) || signal.tp || 0;
      const _hasZ = _sl2 > 0 && _tp2 > 0 && Math.abs(_tp2 - _sl2) > 1e-10;
      if (_hasZ) {
        const absR = Math.abs(_tp2 - _sl2);
        const newPct = isLongDir
          ? Math.max(2, Math.min(98, (p - _sl2) / absR * 100))
          : Math.max(2, Math.min(98, (_sl2 - p) / absR * 100));
        dot.style.left = `${newPct.toFixed(1)}%`;
      } else {
        dot.style.left = '50%';
      }
    }
  };
  window.addEventListener('priceUpdate', body._zonePriceHandler);
}

function _renderFpExpectancy(body, signal, tier) {
  const locked = tier !== 'PRO';

  // Use real signal fields from live_engine v3 first; fall back to track record
  const metaConf   = _confPct(signal.meta_confidence);
  const threshold  = _confPct(signal.threshold, 60);
  const expectedMv = parseFloat(signal.expected_move_pct || 0);
  const rr         = parseFloat(signal.risk_reward || 0);
  const atrPct     = parseFloat(signal.atr_pct || 0);
  const volRegime  = (signal.volatility_regime || 'MEDIUM').toUpperCase();

  // Historical stats â€” use signal values if present, else load from track record
  let expectancy   = typeof signal.expectancy    === 'number' ? signal.expectancy    : null;
  let maxDD        = typeof signal.max_dd        === 'number' ? signal.max_dd        : null;
  let profitFactor = typeof signal.profit_factor === 'number' ? signal.profit_factor : null;
  let winRate      = typeof signal.win_rate      === 'number' ? signal.win_rate      : null;
  let totalTrades  = typeof signal.total_trades  === 'number' ? signal.total_trades  : null;

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-crown text-3xl text-amber-400 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Pro Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Statistical edge analysis is exclusively available to Pro subscribers</p>
      <a href="/pricing" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
    </div>` : '';

  body.innerHTML = `
    <div class="relative">
      ${lockOverlay}
      <div class="${locked ? 'blur-md pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-3 mb-4">
          <span class="font-black text-2xl text-white">${signal.symbol}</span>
          <span class="text-xs ${metaConf >= threshold ? 'text-emerald-400' : 'text-rose-400'} bg-black/40 px-2 py-0.5 rounded font-bold">
            ${metaConf}% conf ${metaConf >= threshold ? 'âœ“' : 'âœ—'}
          </span>
        </div>

        <!-- Current signal quality (live data) -->
        <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-2">This Signal</div>
        <div class="grid grid-cols-2 gap-2 mb-4">
          <div class="bg-black/40 p-3 rounded-xl border ${metaConf >= threshold ? 'border-emerald-500/20' : 'border-rose-500/20'}">
            <div class="text-[9px] text-gray-500 uppercase mb-1">AI Confidence</div>
            <div class="text-2xl font-black font-mono ${metaConf >= threshold ? 'text-emerald-400' : 'text-rose-400'}">${metaConf}%</div>
            <div class="text-[9px] text-gray-600 mt-0.5">required: ${threshold}%</div>
          </div>
          <div class="bg-black/40 p-3 rounded-xl border border-cyan/20">
            <div class="text-[9px] text-gray-500 uppercase mb-1">Expected Move</div>
            <div class="text-2xl font-black font-mono ${expectedMv >= 2 ? 'text-emerald-400' : 'text-cyan'}">
              ${expectedMv > 0 ? '~' + expectedMv.toFixed(1) + '%' : atrPct > 0 ? '~' + (atrPct * 1.5).toFixed(1) + '%' : 'â€”'}
            </div>
            <div class="text-[9px] text-gray-600 mt-0.5">AI projection</div>
          </div>
          <div class="bg-black/40 p-3 rounded-xl border border-white/5">
            <div class="text-[9px] text-gray-500 uppercase mb-1">Risk / Reward</div>
            <div class="text-2xl font-black font-mono text-cyan">${rr > 0 ? '1:' + rr.toFixed(2) : 'â€”'}</div>
            <div class="text-[9px] text-gray-600 mt-0.5">TP1 vs SL</div>
          </div>
          <div class="bg-black/40 p-3 rounded-xl border border-white/5">
            <div class="text-[9px] text-gray-500 uppercase mb-1">Volatility</div>
            <div class="text-xl font-black font-mono ${volRegime === 'HIGH' ? 'text-rose-400' : volRegime === 'LOW' ? 'text-blue-400' : 'text-amber-400'}">${volRegime}</div>
            <div class="text-[9px] text-gray-600 mt-0.5">ATR ${atrPct > 0 ? atrPct.toFixed(2) + '%' : 'â€”'}</div>
          </div>
        </div>

        <!-- Historical performance (loaded async) -->
        <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-2">Track Record (AEGIS)</div>
        <div id="fp-exp-hist" class="bg-black/40 p-4 rounded-xl border border-white/5 text-[10px] text-gray-500 font-mono">
          Loadingâ€¦
        </div>

        <div class="mt-3 p-3 bg-black/30 rounded-lg border border-white/5">
          <div class="text-[10px] text-gray-500 font-mono">
            <i class="fas fa-database text-cyan/50 mr-1"></i>
            Historical stats from live track record. Signal quality metrics are real-time. Not indicative of future results.
          </div>
        </div>
        <div class="mt-4 p-4 bg-black/40 rounded-xl border border-amber-500/20">
          <div class="text-[10px] font-bold text-amber-400 uppercase tracking-wider mb-2">Edge Intelligence</div>
          <div class="text-[11px] text-gray-300 leading-relaxed space-y-2">
            <div><span class="text-amber-400/60 font-bold text-[9px] uppercase mr-2">THIS SIGNAL</span>
              AI confidence is ${metaConf}% vs ${threshold}% required.
              ${metaConf >= threshold
                ? `Signal is above the bar â€” model has sufficient evidence.${rr > 0 ? ` R:R of 1:${rr.toFixed(2)}.` : ''}`
                : 'Signal is below threshold â€” model is watching but not confirmed.'}
              ${expectedMv > 0 ? ` Expected move: ~${expectedMv.toFixed(1)}%.` : ''}
            </div>
            <div><span class="text-amber-400/60 font-bold text-[9px] uppercase mr-2">VOLATILITY</span>
              Regime is ${volRegime}${atrPct > 0 ? ` â€” ATR is ${atrPct.toFixed(2)}% of price` : ''}.
              ${volRegime === 'HIGH' ? 'High volatility â€” use smaller position size and wider stops.' : volRegime === 'LOW' ? 'Low volatility â€” tighter stops viable, but expect smaller moves.' : 'Normal volatility â€” standard sizing applies.'}
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Async load historical stats into #fp-exp-hist
  (async () => {
    const el = body.querySelector('#fp-exp-hist');
    if (!el) return;
    try {
      const r = await fetch('/api/track-record', { cache: 'no-cache' });
      if (!r.ok) throw new Error();
      const d = await r.json();
      const s = d.summary || {};
      const wr  = s.win_rate_pct != null ? s.win_rate_pct.toFixed(1) + '%' : 'â€”';
      const tot = s.total_signals ?? 'â€”';
      const w   = s.wins  ?? 'â€”';
      const l   = s.losses ?? 'â€”';
      const avg = s.avg_pnl_pct != null ? (s.avg_pnl_pct >= 0 ? '+' : '') + s.avg_pnl_pct.toFixed(2) + '%' : 'â€”';
      const ttl = s.total_pnl_pct != null ? (s.total_pnl_pct >= 0 ? '+' : '') + s.total_pnl_pct.toFixed(2) + '%' : 'â€”';
      const wrNum = s.win_rate_pct || 0;
      el.innerHTML = `
        <div class="grid grid-cols-3 gap-x-4 gap-y-2">
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Total</div><div class="font-bold text-white">${tot}</div></div>
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Wins</div><div class="font-bold text-emerald-400">${w}</div></div>
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Losses</div><div class="font-bold text-rose-400">${l}</div></div>
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Win Rate</div><div class="font-bold ${wrNum >= 50 ? 'text-emerald-400' : 'text-rose-400'}">${wr}</div></div>
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Avg PnL</div><div class="font-bold ${avg.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'}">${avg}</div></div>
          <div><div class="text-[9px] text-gray-600 uppercase mb-0.5">Total PnL</div><div class="font-bold ${ttl.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'}">${ttl}</div></div>
        </div>
        <div class="h-1.5 bg-black/50 rounded-full mt-3 overflow-hidden flex">
          <div class="h-full bg-emerald-500/60 rounded-l-full" style="width:${wrNum}%"></div>
          <div class="h-full bg-rose-500/40 rounded-r-full flex-1"></div>
        </div>`;
    } catch {
      el.innerHTML = '<span class="text-gray-600">Track record unavailable â€” refresh to retry.</span>';
    }
  })();
}

function _renderFpShap(body, signal, tier) {
  const locked = tier !== 'PRO';

  // Use real p_buy/p_sell/p_hold from signal (populated by gatekeeper.js fix)
  const probs = (signal.p_buy || signal.p_sell)
    ? { SHORT: parseFloat(signal.p_sell) || 0, HOLD: parseFloat(signal.p_hold) || 0, LONG: parseFloat(signal.p_buy) || 0 }
    : { SHORT: 0, HOLD: 0, LONG: 0 };
  const leadEntry = Object.entries(probs).sort((a,b) => b[1]-a[1])[0];
  const leadClass = leadEntry[0] === 'LONG' ? 'text-green-400' : leadEntry[0] === 'SHORT' ? 'text-red-400' : 'text-gray-400';
  const leadPct = (leadEntry[1] * 100).toFixed(1);
  const metaCf  = Math.round((signal.meta_confidence || 0) * 100);
  const thrCf   = Math.round((signal.threshold || 0.6) * 100);

  // Key indicator drivers built from real signal fields (no fake SHAP)
  const shapValues = [
    { feature: 'Trend Confluence',    value: signal.confluence ? ((parseFloat(signal.confluence.trend || 5) - 5) / 5 * 0.5) : 0 },
    { feature: 'RSI (' + Math.round(signal.rsi || 50) + ')',  value: signal.rsi ? ((signal.rsi - 50) / 50 * 0.4) : 0 },
    { feature: 'Volume: ' + (signal.volume_strength || 'AVG'), value: signal.volume_zscore ? Math.max(-0.5, Math.min(0.5, (signal.volume_zscore || 0) / 4)) : 0 },
    { feature: 'Smart Money',         value: signal.confluence ? ((parseFloat(signal.confluence.smart_money || 5) - 5) / 5 * 0.35) : 0 },
    { feature: 'Momentum (' + (signal.macd_signal || 'N') + ')', value: signal.macd_signal === 'BULLISH' ? 0.3 : signal.macd_signal === 'BEARISH' ? -0.3 : 0 },
    { feature: 'Funding: ' + (signal.funding_bias || 'NEUTRAL'), value: signal.funding_bias === 'SHORTS_PAYING' ? 0.2 : signal.funding_bias === 'LONGS_PAYING' ? -0.2 : 0 },
  ].map(d => ({ ...d, value: parseFloat(d.value.toFixed(3)) }))
   .sort((a,b) => Math.abs(b.value) - Math.abs(a.value))
   .slice(0, 5);

  const lockOverlay = locked ? `
    <div class="absolute inset-0 bg-black/80 backdrop-blur-md rounded-xl flex flex-col items-center justify-center z-10">
      <i class="fas fa-crown text-3xl text-amber-400 mb-3"></i>
      <h3 class="text-lg font-bold text-white mb-1">Pro Plan Required</h3>
      <p class="text-sm text-gray-400 mb-4 text-center px-4">Raw ML probability vectors and SHAP attribution are Pro-exclusive</p>
      <a href="/pricing" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
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
            <div>
              <div class="font-black text-xl font-mono ${leadClass}">${leadPct}% ${leadEntry[0]}</div>
              ${metaCf > 0 ? `<div class="text-[9px] text-gray-500 text-right">Meta: ${metaCf}% vs ${thrCf}% req</div>` : ''}
            </div>
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
          <div class="text-[10px] text-gray-500 uppercase tracking-widest font-bold mb-4">Key Indicator Drivers</div>
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
              Indicator contribution scores derived from live signal data. Positive = bullish lean, negative = bearish. Updated every scan.
            </div>
          </div>
          ${(() => {
            const _topShap = shapValues.reduce((a, b) => Math.abs(a.value) > Math.abs(b.value) ? a : b, shapValues[0]);
            const _topDir = _topShap ? (_topShap.value > 0 ? 'LONG' : 'SHORT') : leadEntry[0];
            const _aligned = _topShap && _topDir === leadEntry[0];
            // Use meta_confidence for the quality gate, not just p_buy pct
            const _convPct = metaCf > 0 ? metaCf : parseFloat(leadPct);
            const _vc = _convPct >= thrCf && _aligned
              ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
              : _convPct >= thrCf
              ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
              : 'bg-red-500/20 text-red-400 border-red-500/30';
            const _vl = _convPct >= thrCf && _aligned ? 'ALIGNED' : _convPct >= thrCf ? 'ABOVE THR' : 'BELOW THR';
            const _topFeat = _topShap ? _topShap.feature : 'N/A';
            const _topVal = _topShap ? (_topShap.value > 0 ? '+' : '') + _topShap.value.toFixed(3) : '0.000';
            const _shortPct = (probs.SHORT * 100).toFixed(1);
            const _holdPct  = (probs.HOLD  * 100).toFixed(1);
            const _longPct  = (probs.LONG  * 100).toFixed(1);
            const _why = `The primary model gives ${_longPct}% LONG, ${_shortPct}% SHORT, ${_holdPct}% HOLD. Meta gate score: ${metaCf}% (threshold: ${thrCf}%). ${_aligned ? 'Dominant indicator and model direction agree.' : 'Top driver conflicts with model direction â€” caution.'}`;
            const _what = `Top driver: "${_topFeat}" (${_topVal}). ${_aligned ? 'Indicators and model are aligned â€” consistent signal.' : 'Divergence detected â€” model may be picking up a pattern that indicators do not yet show.'} ${metaCf >= thrCf ? 'Signal is ABOVE the confidence threshold.' : 'Signal is BELOW the threshold â€” not a valid trade.'}`;
            const _when = metaCf >= thrCf && _aligned
              ? `High conviction (${metaCf}% > ${thrCf}% required). Act at the entry zone with standard sizing.`
              : metaCf >= thrCf
              ? 'Above threshold but indicators diverge. Reduce size by 50% and wait for confluence to align.'
              : `Below threshold (${metaCf}% < ${thrCf}%). Do not trade â€” wait until model confidence rises.`;
            const _where = `Entry: $${(() => { const v=signal.entry_price||signal.price||0; return v>=100?v.toFixed(2):v>=1?v.toFixed(4):v.toFixed(6); })()}. ${metaCf >= thrCf ? 'Signal is valid â€” check zone tracker for SL/TP placement.' : 'Not a valid trade â€” add to watchlist and revisit at next scan.'}`;
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
      <a href="/pricing" class="px-6 py-2 bg-gradient-to-r from-amber-500 to-orange text-white font-bold rounded-xl text-sm">Upgrade to Pro</a>
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
    <span class="text-amber-300">"https://aegisignal.pro/api/v1/signals/live"</span>,
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
        _showToast('New key generated â€” copy it now!', 'success');
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
// removed â€” superseded by the feature panel system (_renderFp* functions above).

async function renderExpectancyPanel() {
  // no-op: superseded by _renderFpExpectancy / openFeaturePanel('fp-expectancy')
}

// â”€â”€ TOKEN SEARCH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

let _tokenSearchQuery = '';

function _applyTokenSearch(query) {
  _tokenSearchQuery = (query || '').toLowerCase().trim();

  const container  = document.getElementById('signalsContainer');
  const countEl    = document.getElementById('token-search-count');
  const clearBtn   = document.getElementById('token-search-clear');
  if (!container) return;

  if (clearBtn) clearBtn.classList.toggle('hidden', !_tokenSearchQuery);

  const cards = container.querySelectorAll('.signal-card[data-symbol]');
  let visible = 0;

  cards.forEach(card => {
    const sym = (card.dataset.symbol || '').toLowerCase();
    const match = !_tokenSearchQuery || sym.includes(_tokenSearchQuery);
    card.style.display = match ? '' : 'none';
    if (match) visible++;
  });

  // Count label
  if (countEl) {
    if (cards.length > 0) {
      countEl.textContent = _tokenSearchQuery
        ? `${visible} / ${cards.length}`
        : `${cards.length} tokens`;
      countEl.classList.toggle('has-query', !!_tokenSearchQuery && visible < cards.length);
    } else {
      countEl.textContent = '';
      countEl.classList.remove('has-query');
    }
  }

  // "No results" placeholder
  let noResult = container.querySelector('.search-no-result');
  if (cards.length > 0 && visible === 0 && _tokenSearchQuery) {
    if (!noResult) {
      noResult = document.createElement('div');
      noResult.className = 'no-signals search-no-result';
      noResult.style.gridColumn = '1 / -1';
      container.appendChild(noResult);
    }
    noResult.innerHTML = `
      <i class="fas fa-search" style="font-size:1.6rem;opacity:.35;margin-bottom:10px"></i>
      <p>No token matches "<strong style="color:var(--primary-cyan)">${_tokenSearchQuery.toUpperCase()}</strong>"</p>`;
  } else if (noResult) {
    noResult.remove();
  }
}

// ============================================================
// DIRECTIONAL ROOMS â€” BUY / SELL filtered signal cockpits
// ============================================================

function _syncDirectionalRooms() {
  const source      = document.getElementById('signalsContainer');
  const buyContainer  = document.getElementById('buySignalsContainer');
  const sellContainer = document.getElementById('sellSignalsContainer');
  if (!source || !buyContainer || !sellContainer) return;

  const cards = Array.from(source.querySelectorAll('.signal-card[data-dir]'));

  const isBuy  = c => c.dataset.dir === 'buy';
  const isSell = c => c.dataset.dir === 'sell';

  const buyCards  = cards.filter(isBuy);
  const sellCards = cards.filter(isSell);

  if (buyCards.length > 0) {
    buyContainer.innerHTML = '';
    buyCards.forEach(card => buyContainer.appendChild(card.cloneNode(true)));
  } else {
    buyContainer.innerHTML = `
      <div class="no-signals" style="grid-column:1/-1">
        <i class="fas fa-arrow-trend-up" style="color:#4ade80;opacity:0.35;font-size:2.5rem;margin-bottom:1rem"></i>
        <p>No active BUY signals right now</p>
      </div>`;
  }

  if (sellCards.length > 0) {
    sellContainer.innerHTML = '';
    sellCards.forEach(card => sellContainer.appendChild(card.cloneNode(true)));
  } else {
    sellContainer.innerHTML = `
      <div class="no-signals" style="grid-column:1/-1">
        <i class="fas fa-arrow-trend-down" style="color:#f87171;opacity:0.35;font-size:2.5rem;margin-bottom:1rem"></i>
        <p>No active SELL signals right now</p>
      </div>`;
  }

  // Update header counts
  const buyCount  = document.getElementById('buy-cockpit-count');
  const sellCount = document.getElementById('sell-cockpit-count');
  if (buyCount)  buyCount.textContent  = buyCards.length  > 0 ? `â€” ${buyCards.length} signal${buyCards.length  !== 1 ? 's' : ''}` : '';
  if (sellCount) sellCount.textContent = sellCards.length > 0 ? `â€” ${sellCards.length} signal${sellCards.length !== 1 ? 's' : ''}` : '';

  // Update nav badge counts
  const navBuy  = document.getElementById('nav-buy-count');
  const navSell = document.getElementById('nav-sell-count');
  if (navBuy) {
    navBuy.textContent = buyCards.length;
    navBuy.classList.toggle('hidden', buyCards.length === 0);
  }
  if (navSell) {
    navSell.textContent = sellCards.length;
    navSell.classList.toggle('hidden', sellCards.length === 0);
  }
}

window.syncDirectionalRooms = _syncDirectionalRooms;

// Keep directional rooms in sync whenever signalsContainer updates
(function _initDirectionalRoomsSync() {
  function attach() {
    const source = document.getElementById('signalsContainer');
    if (!source) return;
    new MutationObserver(() => {
      // Always update nav badge counts (regardless of active room)
      const cards = Array.from(source.querySelectorAll('.signal-card[data-dir]'));
      const buyCount  = cards.filter(c => c.dataset.dir === 'buy').length;
      const sellCount = cards.filter(c => c.dataset.dir === 'sell').length;
      const navBuy  = document.getElementById('nav-buy-count');
      const navSell = document.getElementById('nav-sell-count');
      if (navBuy)  { navBuy.textContent  = buyCount;  navBuy.classList.toggle('hidden',  buyCount  === 0); }
      if (navSell) { navSell.textContent = sellCount; navSell.classList.toggle('hidden', sellCount === 0); }
      // Sync room content only if a directional room is currently visible
      const buysActive  = document.getElementById('room-buys')?.classList.contains('active');
      const sellsActive = document.getElementById('room-sells')?.classList.contains('active');
      if (buysActive || sellsActive) _syncDirectionalRooms();
    }).observe(source, { childList: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach);
  else attach();
}());

// Wire up once DOM is ready
(function _initTokenSearch() {
  function attach() {
    const input   = document.getElementById('token-search-input');
    const clear   = document.getElementById('token-search-clear');
    const container = document.getElementById('signalsContainer');

    if (input) {
      input.addEventListener('input', e => _applyTokenSearch(e.target.value));

      // Keyboard shortcut: Escape clears the search
      input.addEventListener('keydown', e => {
        if (e.key === 'Escape') { input.value = ''; _applyTokenSearch(''); }
      });
    }

    if (clear) {
      clear.addEventListener('click', () => {
        const inp = document.getElementById('token-search-input');
        if (inp) { inp.value = ''; inp.focus(); }
        _applyTokenSearch('');
      });
    }

    // Re-apply filter after every signal render (renderSignals replaces innerHTML)
    if (container) {
      new MutationObserver(() => {
        const countEl = document.getElementById('token-search-count');
        if (_tokenSearchQuery) {
          _applyTokenSearch(_tokenSearchQuery);
        } else if (countEl) {
          const cards = container.querySelectorAll('.signal-card[data-symbol]');
          countEl.textContent = cards.length > 0 ? `${cards.length} tokens` : '';
          countEl.classList.remove('has-query');
        }
      }).observe(container, { childList: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
}());

// ============================================================
// ADMIN PANEL â€” Settings room, owner-only
// ============================================================

const _ADMIN_PANEL_OWNER = 'animeshkukreti60@gmail.com';
const _ADMIN_KEY_STORAGE = 'aegis_admin_key';

// Show admin panel if the logged-in user is the owner
document.addEventListener('dashboardUserLoaded', (e) => {
  const email = e?.detail?.userData?.email || window.currentUserData?.email || '';
  if (email === _ADMIN_PANEL_OWNER) {
    const wrap = document.getElementById('admin-panel-wrap');
    if (wrap) wrap.classList.remove('hidden');
    // Restore saved key from sessionStorage and auto-fetch if present
    const saved = sessionStorage.getItem(_ADMIN_KEY_STORAGE);
    if (saved) {
      const input = document.getElementById('admin-key-input');
      if (input) input.value = saved;
      adminFetchCurrent();
      adminListCodes();
    }
  }
});

function _adminKey() {
  return sessionStorage.getItem(_ADMIN_KEY_STORAGE) || '';
}

window.adminSaveKey = async function() {
  const input = document.getElementById('admin-key-input');
  const statusEl = document.getElementById('admin-key-status');
  const key = (input?.value || '').trim();
  if (!key) return;

  // Validate key by hitting the list endpoint
  try {
    const res = await fetch('/admin/dev-codes?include_used=false', {
      headers: { 'X-Admin-Key': key }
    });
    if (res.ok) {
      sessionStorage.setItem(_ADMIN_KEY_STORAGE, key);
      if (statusEl) { statusEl.textContent = 'âœ“ Authenticated'; statusEl.className = 'mt-1.5 text-[11px] text-green-400'; }
      adminFetchCurrent();
      adminListCodes();
    } else {
      if (statusEl) { statusEl.textContent = 'âœ— Wrong key'; statusEl.className = 'mt-1.5 text-[11px] text-red-400'; }
    }
  } catch (e) {
    if (statusEl) { statusEl.textContent = 'âœ— Network error'; statusEl.className = 'mt-1.5 text-[11px] text-red-400'; }
  }
};

window.adminFetchCurrent = async function() {
  const key = _adminKey();
  if (!key) return;
  const tokenEl = document.getElementById('admin-current-token');
  const expiresEl = document.getElementById('admin-current-expires');
  try {
    const res = await fetch('/admin/dev-codes/current', { headers: { 'X-Admin-Key': key } });
    if (res.ok) {
      const data = await res.json();
      if (tokenEl) tokenEl.textContent = data.code || 'â€”';
      if (expiresEl) {
        try {
          const exp = new Date(data.expires_at);
          expiresEl.textContent = `Expires: ${exp.toLocaleDateString()} ${exp.toLocaleTimeString()}`;
        } catch { expiresEl.textContent = data.expires_at || ''; }
      }
    }
  } catch { /* silent */ }
};

window.adminGenerateCode = async function() {
  const key = _adminKey();
  if (!key) { _showToast('Authenticate first', 'error'); return; }
  const plan = document.getElementById('admin-plan-select')?.value || 'pro';
  const days = parseInt(document.getElementById('admin-days-input')?.value || '30');
  const label = (document.getElementById('admin-label-input')?.value || 'beta').trim();
  const resultEl = document.getElementById('admin-generate-result');

  try {
    const res = await fetch('/admin/dev-codes/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Admin-Key': key },
      body: JSON.stringify({ count: 1, plan, days, label }),
    });
    if (res.ok) {
      const data = await res.json();
      const code = data.codes?.[0]?.code || 'â€”';
      if (resultEl) {
        resultEl.textContent = code;
        resultEl.classList.remove('hidden');
        // Auto-hide after 30s so it doesn't linger
        setTimeout(() => resultEl.classList.add('hidden'), 30000);
      }
      adminListCodes();
    } else {
      const err = await res.json().catch(() => ({}));
      _showToast(err.detail || 'Generate failed', 'error');
    }
  } catch (e) {
    _showToast('Network error', 'error');
  }
};

window.adminListCodes = async function() {
  const key = _adminKey();
  if (!key) return;
  const listEl = document.getElementById('admin-codes-list');
  const includeUsed = document.getElementById('admin-show-used')?.checked || false;
  if (!listEl) return;
  listEl.innerHTML = '<div class="text-xs text-gray-500 text-center py-3">Loadingâ€¦</div>';
  try {
    const res = await fetch(`/admin/dev-codes?include_used=${includeUsed}`, {
      headers: { 'X-Admin-Key': key }
    });
    if (res.ok) {
      const data = await res.json();
      if (!data.codes || data.codes.length === 0) {
        listEl.innerHTML = '<div class="text-xs text-gray-600 text-center py-3">No active codes found</div>';
        return;
      }
      listEl.innerHTML = data.codes.map(c => {
        const used = c.used_by ? `<span class="text-gray-600 text-[10px] truncate max-w-[120px]">â†’ ${c.used_by}</span>` : '';
        const expired = c.expired ? '<span class="text-red-400/60 text-[10px]">expired</span>' : '';
        const badge = `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${c.plan === 'pro' ? 'bg-amber-500/20 text-amber-300' : c.plan === 'intermediate' ? 'bg-cyan/20 text-cyan' : 'bg-white/10 text-gray-400'}">${c.plan}</span>`;
        return `
          <div class="flex items-center gap-2 px-2.5 py-2 rounded-lg bg-black/30 border border-white/5 hover:border-white/10 group">
            <code class="font-mono text-[11px] text-green-300 flex-1 min-w-0 truncate cursor-pointer" onclick="navigator.clipboard.writeText('${c.code}');_showToast('Copied!','success')" title="Click to copy">${c.code}</code>
            ${badge}
            <span class="text-[10px] text-gray-600">${c.label || ''}</span>
            ${used}${expired}
            <button onclick="adminDeleteCode('${c.code}')" class="opacity-0 group-hover:opacity-100 ml-1 text-red-400/60 hover:text-red-400 transition-all text-[10px]" title="Delete">âœ•</button>
          </div>`;
      }).join('');
    } else {
      listEl.innerHTML = '<div class="text-xs text-red-400 text-center py-3">Failed to load â€” check admin key</div>';
    }
  } catch {
    listEl.innerHTML = '<div class="text-xs text-red-400 text-center py-3">Network error</div>';
  }
};

window.adminDeleteCode = async function(code) {
  const key = _adminKey();
  if (!key) return;
  // No dedicated delete endpoint yet â€” placeholder for future
  _showToast('Delete endpoint not implemented yet', 'info');
};

// â”€â”€ Notification Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function _notifToken() {
  return localStorage.getItem('access_token') || localStorage.getItem('authToken') || '';
}

window._loadNotifSettings = async function() {
  try {
    const r = await fetch('/api/notifications/settings', {
      headers: { 'Authorization': `Bearer ${_notifToken()}` }
    });
    if (!r.ok) return;
    const cfg = await r.json();
    const el = k => document.getElementById(k);

    if (el('notif-enabled'))     el('notif-enabled').checked    = cfg.enabled !== false;

    // Telegram status loaded separately
    _refreshTelegramStatus();

    // Discord
    if (el('notif-discord-url')) el('notif-discord-url').value = cfg.discord_webhook_url || '';

    // WhatsApp / Twilio
    if (el('notif-twilio-sid')) {
      const redacted = cfg.twilio_account_sid === '***configured***';
      el('notif-twilio-sid').value       = redacted ? '' : (cfg.twilio_account_sid || '');
      el('notif-twilio-sid').placeholder = redacted ? 'âœ“ configured (leave blank to keep)' : 'Twilio Account SID';
    }
    if (el('notif-twilio-token')) {
      el('notif-twilio-token').value = '';
      if (cfg.twilio_account_sid === '***configured***')
        el('notif-twilio-token').placeholder = 'âœ“ configured (leave blank to keep)';
    }
    if (el('notif-wa-from')) el('notif-wa-from').value = cfg.whatsapp_from || '';
    if (el('notif-wa-to'))   el('notif-wa-to').value   = cfg.whatsapp_to   || '';

    // Filters
    if (el('notif-min-conf')) {
      const pct = Math.round((cfg.min_confidence || 0.65) * 100);
      el('notif-min-conf').value = pct;
      const lbl = el('notif-min-conf-val');
      if (lbl) lbl.textContent = pct + '%';
    }
    if (el('notif-max-hr'))      el('notif-max-hr').value      = cfg.max_alerts_per_hour || 5;
    const qh = cfg.quiet_hours || {};
    if (el('notif-quiet-start')) el('notif-quiet-start').value = qh.start || '23:00';
    if (el('notif-quiet-end'))   el('notif-quiet-end').value   = qh.end   || '06:00';
  } catch (_) {}
};

window.saveNotifSettings = async function() {
  const statusEl = document.getElementById('notif-status');
  const el = k => document.getElementById(k);

  const body = {
    enabled:             el('notif-enabled')?.checked !== false,
    telegram_chat_id:    el('notif-tg-chat')?.value.trim()     || '',
    discord_webhook_url: el('notif-discord-url')?.value.trim() || '',
    whatsapp_from:       el('notif-wa-from')?.value.trim()     || '',
    whatsapp_to:         el('notif-wa-to')?.value.trim()       || '',
    min_confidence:      parseFloat(el('notif-min-conf')?.value || 65) / 100,
    max_alerts_per_hour: parseInt(el('notif-max-hr')?.value || 5),
    quiet_hours: {
      start: el('notif-quiet-start')?.value || '23:00',
      end:   el('notif-quiet-end')?.value   || '06:00',
    },
  };
  // Twilio only if typed (under construction â€” hidden inputs will be empty)
  const sid = el('notif-twilio-sid')?.value.trim();
  const tok = el('notif-twilio-token')?.value.trim();
  if (sid) body.twilio_account_sid = sid;
  if (tok) body.twilio_auth_token  = tok;

  if (statusEl) { statusEl.className = 'text-[11px] text-center text-cyan-400'; statusEl.textContent = 'Savingâ€¦'; }
  try {
    const r = await fetch('/api/notifications/settings', {
      method:  'POST',
      headers: { 'Authorization': `Bearer ${_notifToken()}`, 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (statusEl) {
      if (r.ok) {
        statusEl.className   = 'text-[11px] text-center text-green-400';
        statusEl.textContent = 'âœ“ Settings saved';
      } else {
        statusEl.className   = 'text-[11px] text-center text-red-400';
        statusEl.textContent = `Error ${r.status}`;
      }
    }
    setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 3000);
  } catch (e) {
    if (statusEl) { statusEl.className = 'text-[11px] text-center text-red-400'; statusEl.textContent = 'Network error'; }
  }
};

window.testNotifSettings = async function() {
  const statusEl = document.getElementById('notif-status');
  if (statusEl) { statusEl.className = 'text-[11px] text-center text-cyan-400'; statusEl.textContent = 'Sending test pingâ€¦'; }
  try {
    const r = await fetch('/api/notifications/test', {
      method:  'POST',
      headers: { 'Authorization': `Bearer ${_notifToken()}` },
    });
    const data = await r.json();
    const results = data.results || {};
    const parts = [];
    if ('discord'  in results) parts.push(`Discord: ${results.discord  ? 'âœ“ sent' : 'âœ— failed'}`);
    if ('whatsapp' in results) parts.push(`WhatsApp: ${results.whatsapp ? 'âœ“ sent' : 'âœ— failed'}`);
    if (statusEl) {
      const allOk = Object.values(results).some(Boolean);
      statusEl.className   = `text-[11px] text-center ${allOk ? 'text-green-400' : 'text-red-400'}`;
      statusEl.textContent = parts.length ? parts.join('  Â·  ') : 'No channels configured yet';
    }
    setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 8000);
  } catch (e) {
    if (statusEl) { statusEl.className = 'text-[11px] text-center text-red-400'; statusEl.textContent = 'Network error'; }
  }
};

// â”€â”€ Telegram one-tap connect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

let _tgPollTimer = null;

function _tgSetState(state, chatId) {
  const states = ['disconnected', 'waiting', 'connected'];
  states.forEach(s => {
    const el = document.getElementById(`tg-state-${s}`);
    if (el) el.classList.toggle('hidden', s !== state);
  });
  if (state === 'connected' && chatId) {
    const disp = document.getElementById('tg-chat-id-display');
    if (disp) disp.textContent = `(${chatId})`;
  }
  const banner = document.getElementById('tg-prompt-banner');
  if (banner) banner.style.display = state === 'connected' ? 'none' : '';
}

async function _refreshTelegramStatus() {
  try {
    const r = await fetch('/api/notifications/telegram/status', {
      headers: { 'Authorization': `Bearer ${_notifToken()}` }
    });
    if (!r.ok) return;
    const data = await r.json();
    _tgSetState(data.connected ? 'connected' : 'disconnected', data.chat_id);
  } catch (_) {}
}

window.connectTelegram = async function() {
  const btn     = document.querySelector('#tg-state-disconnected button');
  const statusEl = document.getElementById('notif-status');

  // Open the window SYNCHRONOUSLY before any await â€” prevents popup blocker
  const tgWin = window.open('', '_blank');
  if (tgWin) {
    tgWin.document.write('<!DOCTYPE html><html><head><meta charset="utf-8"><title>Connect Telegram</title></head><body style="background:#0e1621;color:#8b949e;font-family:sans-serif;text-align:center;padding-top:25vh;margin:0"><p style="font-size:16px">Connecting to Telegramâ€¦</p></body></html>');
  }

  // Show loading state on button
  if (btn) { btn.disabled = true; btn.innerHTML = '<svg class="animate-spin h-4 w-4 inline mr-1" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> Connectingâ€¦'; }

  try {
    const r = await fetch('/api/notifications/telegram/connect', {
      headers: { 'Authorization': `Bearer ${_notifToken()}` }
    });

    if (!r.ok) {
      if (tgWin) tgWin.close();
      if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fab fa-telegram text-base"></i> Connect Telegram'; }
      const err = await r.json().catch(() => ({}));
      if (statusEl) {
        statusEl.className   = 'text-[11px] text-center text-red-400';
        statusEl.textContent = err.detail || `Error ${r.status} â€” Telegram bot may not be configured on this server`;
        setTimeout(() => { statusEl.textContent = ''; }, 7000);
      }
      return;
    }

    const { deeplink, code, bot_username } = await r.json();

    // tg:// opens the Telegram desktop/mobile app directly.
    // https://t.me/ only opens the web version in the browser.
    const tgAppLink = `tg://resolve?domain=${bot_username}&start=${code}`;

    if (tgWin) {
      // Write a proper landing page with a visible "Open in Telegram" button.
      // A hidden anchor click triggers the tg:// protocol (opens the app).
      // The visible button is the fallback if the protocol handler didn't fire.
      tgWin.document.open();
      tgWin.document.write(
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Connect Telegram</title></head>' +
        '<body style="background:#0e1621;color:#c8d1d9;font-family:sans-serif;text-align:center;padding:20vh 24px 0;margin:0">' +
        '<p style="font-size:20px;margin:0 0 8px;font-weight:600">Opening Telegramâ€¦</p>' +
        '<p style="font-size:13px;color:#8b949e;margin:0 0 28px">Your Telegram app should open automatically.</p>' +
        '<a id="tg-btn" href="' + deeplink + '" ' +
        'style="display:inline-block;background:#2ea6ff;color:#fff;padding:13px 28px;border-radius:10px;text-decoration:none;font-size:15px;font-weight:600">' +
        'Open in Telegram</a>' +
        '<p style="font-size:11px;color:#656d76;margin-top:20px">Tap <b>START</b> in Telegram, then return here â€” you\'ll be connected automatically.</p>' +
        '<p style="font-size:11px;color:#656d76;margin-top:8px">You can close this tab once done.</p>' +
        '<script>setTimeout(function(){var a=document.createElement("a");a.href="' + tgAppLink + '";document.body.appendChild(a);a.click();},200);<\/script>' +
        '</body></html>'
      );
      tgWin.document.close();
    } else {
      // Popup was blocked â€” try opening the app link directly in current tab context,
      // then show a clickable button in the status area.
      window.open(tgAppLink, '_blank');
      if (statusEl) {
        statusEl.className   = 'text-[11px] text-center text-yellow-400';
        statusEl.innerHTML   = `Popup blocked &mdash; <a href="${deeplink}" target="_blank" style="color:#58a6ff;text-decoration:underline">click here to open Telegram</a>`;
        setTimeout(() => { if (statusEl) statusEl.innerHTML = ''; }, 15000);
      }
    }

    // Switch to waiting state
    _tgSetState('waiting');
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fab fa-telegram text-base"></i> Connect Telegram'; }

    // Poll for confirmation every 2.5s
    if (_tgPollTimer) clearInterval(_tgPollTimer);
    _tgPollTimer = setInterval(async () => {
      try {
        const s    = await fetch('/api/notifications/telegram/status', { headers: { 'Authorization': `Bearer ${_notifToken()}` } });
        const data = await s.json();
        if (data.connected) {
          clearInterval(_tgPollTimer);
          _tgPollTimer = null;
          _tgSetState('connected', data.chat_id);
          if (statusEl) {
            statusEl.className   = 'text-[11px] text-center text-green-400';
            statusEl.textContent = 'âœ“ Telegram connected! Signals will arrive as DMs.';
            setTimeout(() => { statusEl.textContent = ''; }, 5000);
          }
        }
      } catch (_) {}
    }, 2500);

    // Auto-stop polling after 3 minutes
    setTimeout(() => {
      if (_tgPollTimer) { clearInterval(_tgPollTimer); _tgPollTimer = null; }
      _refreshTelegramStatus();
    }, 180000);

  } catch (e) {
    if (tgWin) tgWin.close();
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fab fa-telegram text-base"></i> Connect Telegram'; }
    if (statusEl) { statusEl.className = 'text-[11px] text-center text-red-400'; statusEl.textContent = 'Network error â€” is the server running?'; }
  }
};

window.cancelTelegramConnect = function() {
  if (_tgPollTimer) { clearInterval(_tgPollTimer); _tgPollTimer = null; }
  _tgSetState('disconnected');
};

window.disconnectTelegram = async function() {
  try {
    await fetch('/api/notifications/telegram/disconnect', {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${_notifToken()}` }
    });
    _tgSetState('disconnected');
    const statusEl = document.getElementById('notif-status');
    if (statusEl) {
      statusEl.className = 'text-[11px] text-center text-gray-400';
      statusEl.textContent = 'Telegram disconnected';
      setTimeout(() => { statusEl.textContent = ''; }, 3000);
    }
  } catch (_) {}
};

// On page load, hide the Telegram banner immediately for already-connected users
setTimeout(async function _initTgBannerCheck() {
  try {
    const tok = _notifToken();
    if (!tok) return;
    const r = await fetch('/api/notifications/telegram/status', {
      headers: { Authorization: `Bearer ${tok}` }
    });
    if (r.ok) {
      const data = await r.json();
      _tgSetState(data.connected ? 'connected' : 'disconnected', data.chat_id);
    }
  } catch (_) {}
}, 1500);

// Hook into switchRoom so settings auto-load when the room opens
(function _patchSwitchForNotif() {
  const _orig = window.switchRoom;
  window.switchRoom = function(roomName) {
    if (typeof _orig === 'function') _orig(roomName);
    if (roomName === 'settings') setTimeout(window._loadNotifSettings, 80);
  };
  // Also expose for direct call
  window.switchRoom._notifPatched = true;
}());


