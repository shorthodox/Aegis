// ============================================================
// AEGIS Trial Countdown – 3-Day Free Trial Management
// ============================================================

import { db } from './gatekeeper.js';
import { doc, getDoc, updateDoc, serverTimestamp } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";
import { AuthManager } from '../auth/authManager.js';

const TrialManager = (() => {
  // ============================================================
  // STATE
  // ============================================================
  let currentUserId = null;
  let trialCheckInterval = null;
  let cachedTrialInfo = null;
  let lastFetchTime = 0;
  let explicitTrialStart = null;
  let authRetryCount = 0;
  let maxAuthRetries = 3;
  let loadingStartTime = null;
  let loadingTimeoutMs = 5000; // 5 second timeout for loading state
  let networkErrorState = false;

  // ============================================================
  // HELPER: Validate ISO String
  // ============================================================
  function isValidISOString(str) {
    if (typeof str !== 'string') return false;
    if (str === 'null' || str === 'undefined') return false;
    try {
      const date = new Date(str);
      // Check if date is valid and string matches ISO format
      if (Number.isNaN(date.getTime())) return false;
      // Verify it's a valid ISO string by attempting to parse it
      return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(str);
    } catch {
      return false;
    }
  }

  // ============================================================
  // HELPER: Format Time Remaining
  // ============================================================
  function formatTimeRemaining(expiryDate) {
    const now = new Date();
    const diff = expiryDate.getTime() - now.getTime();

    if (diff <= 0) {
      return { expired: true, display: 'Trial Expired' };
    }

    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);

    const pad = (num) => num.toString().padStart(2, '0');
    const displayHours = (days * 24) + hours;
    const display = `${pad(displayHours)}:${pad(minutes)}:${pad(seconds)}`;

    return { expired: false, display, days, hours, minutes, seconds };
  }  // ============================================================
  // HELPER: Validate JWT Expiry
  // ============================================================
  function isJWTExpired(token) {
    if (!token) return true;
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return Date.now() >= payload.exp * 1000;
    } catch {
      return true;
    }
  }

  // ============================================================
  // HELPER: Get User Trial Info (Stale-While-Revalidate)
  // ============================================================
  async function getUserTrialInfo(userId) {
    if (!userId) return null;
    const now = Date.now();

    // Stale-While-Revalidate: Quickly assemble a state from cache
    let cachedState = null;
    const jwtData = AuthManager.getUserData() || {};
    const localEndStr = cachedTrialInfo?.trial_end || localStorage.getItem('trial_end_timestamp');
    
    let cachedTokens = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ARB/USDT', 'AAVE/USDT'];
    try {
        const storedTokens = localStorage.getItem('cachedAllowedTokens');
        if (storedTokens) {
            const parsed = JSON.parse(storedTokens);
            if (Array.isArray(parsed) && parsed.length > 0) cachedTokens = parsed;
        }
    } catch (e) {}

    const hasValidLocalEnd = isValidISOString(localEndStr);

    if (cachedTrialInfo && cachedTrialInfo._error === 'network' && !hasValidLocalEnd) {
        return {
          active: false,
          expired: false,
          networkError: true,
          display: 'Connection Error - Access Restricted',
          allowedTokens: []
        };
    }
    if (cachedTrialInfo && cachedTrialInfo._error === 'auth' && !hasValidLocalEnd) {
        return {
          active: false,
          expired: false,
          authError: true,
          display: 'Authentication Error - Access Restricted',
          message: 'Unable to verify trial status. Please login again.',
          allowedTokens: []
        };
    }

    if (cachedTrialInfo?.expired) {
      return cachedTrialInfo;
    }

    const PAID_PLANS = ['pro', 'premium', 'active', 'basic', 'intermediate'];
    const serverPlan = (cachedTrialInfo?.plan || '').toLowerCase();
    const jwtPlan   = (jwtData.plan_type || '').toLowerCase();
    if (cachedTrialInfo?.subscription_active || PAID_PLANS.includes(serverPlan) || PAID_PLANS.includes(jwtPlan)) {
      cachedState = {
        active: true,
        isPremium: true,
        plan: serverPlan || jwtPlan || 'pro',
        display: 'Subscription Active',
        expired: false,
        days: 999, hours: 23, minutes: 59, seconds: 59,
        allowedTokens: cachedTokens,
        allowedTimeframes: ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
      };
    } else if (isValidISOString(localEndStr)) {
      const trialEnd = new Date(localEndStr);
      if (!isNaN(trialEnd.getTime())) {
        const timeInfo = formatTimeRemaining(trialEnd);
        if (timeInfo.expired) {
          // Trust the local timestamp — show expired immediately.
          // The background fetch still runs; if the user has since upgraded,
          // trial-status-updated will flip the display to "Subscription Active".
          localStorage.removeItem('trial_end_timestamp');
          cachedState = {
            active: false,
            expired: true,
            display: 'Trial Expired',
            allowedTokens: [],
            allowedTimeframes: [],
            plan: 'trial'
          };
          cachedTrialInfo = cachedState;
        } else {
          cachedState = {
            active: true,
            ...timeInfo,
            allowedTokens: cachedTokens,
            allowedTimeframes: ['15m', '30m'],
            plan: 'trial',
            trialEndDate: trialEnd
          };
        }
      }
    }

    if (!cachedState) {
      if (!loadingStartTime) {
        loadingStartTime = now;
      }
      const elapsedTime = now - loadingStartTime;
      if (elapsedTime > loadingTimeoutMs) {
        // Timeout: no trial data found and backend unreachable — restrict access
        console.warn(`[Trial] Loading timeout after ${elapsedTime}ms — restricting access`);
        loadingStartTime = null;
        cachedState = {
          active: false,
          expired: true,
          display: 'Trial Expired',
          allowedTokens: [],
          allowedTimeframes: [],
          plan: 'trial'
        };
      } else {
        cachedState = {
          active: false,
          isLoading: true,
          display: `Verifying Trial Status... (${Math.round((loadingTimeoutMs - elapsedTime) / 1000)}s)`,
          allowedTokens: [],
          allowedTimeframes: [],
          plan: 'trial'
        };
      }
    } else {
      loadingStartTime = null;
    }

    // Trigger background fetch if cache is old or missing
    if (!cachedTrialInfo || now - lastFetchTime > 60000) {
      const fetchPromise = (async () => {
        try {
          let authHeader = AuthManager.getAuthHeader();
          let token = AuthManager.getToken();
          
          if (token && isJWTExpired(token)) {
             console.warn('JWT expired, attempting silent refresh via Firebase...');
             try {
               const { getAuth } = await import("https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js");
               const auth = getAuth();
               if (auth.currentUser) {
                  token = await auth.currentUser.getIdToken(true);
                  authHeader = `Bearer ${token}`;
                  localStorage.setItem('access_token', token);
               } else {
                  networkErrorState = false;
                  throw new Error("Cannot refresh token - no current user");
               }
             } catch (refreshErr) {
               networkErrorState = false;
               throw new Error("Token refresh failed: " + refreshErr.message);
             }
          }

          if (!authHeader) {
            networkErrorState = false;
            throw new Error("Missing auth header");
          }

          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 5000);

          const userResponse = await fetch('/auth/me', {
            headers: { 'Authorization': authHeader },
            signal: controller.signal
          });

          clearTimeout(timeoutId);

          if (userResponse.ok) {
            const userData = await userResponse.json();
            cachedTrialInfo = userData;
            lastFetchTime = Date.now();
            networkErrorState = false;
            // Reset loading timeout when successful fetch completes
            loadingStartTime = null;
            if (userData.trial_end && isValidISOString(userData.trial_end)) {
              localStorage.setItem('trial_end_timestamp', userData.trial_end);
            }
            console.log('[Trial] Successfully fetched user trial info:', userData);
          } else {
             if (userResponse.status >= 400 && userResponse.status < 600) {
                 networkErrorState = true;
             } else {
                 networkErrorState = false;
             }
             throw new Error(`HTTP Error ${userResponse.status}`);
          }
        } catch (err) {
          console.warn("Background fetch failed:", err);
          if (err.name === 'AbortError') {
            networkErrorState = true;
            cachedTrialInfo = { _error: 'network' };
          } else if (networkErrorState) {
              cachedTrialInfo = { _error: 'network' };
          } else {
              cachedTrialInfo = { _error: 'auth' };
          }
        } finally {
          window.dispatchEvent(new CustomEvent('trial-status-updated'));
        }
      })();
    }

    return cachedState;
  }

  // ============================================================
  // FETCH TRIAL START FROM FIRESTORE
  // ============================================================
  async function fetchTrialStartFromFirestore(userId) {
    if (!userId) return null;
    return new Promise(async (resolve) => {
      try {
        const { getAuth, onAuthStateChanged } = await import("https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js");
        const auth = getAuth();
        onAuthStateChanged(auth, async (user) => {
          if (!user) {
             resolve(null);
             return;
          }
          try {
            const userDocRef = doc(db, 'users', user.uid);
            const userDoc = await getDoc(userDocRef);
            if (!userDoc.exists()) { resolve(null); return; }
            const data = userDoc.data();
            // Try explicit start-date fields first
            const trialStart = data?.trial?.startDate || data?.trial_start || data?.trialStart || data?.joinDate;
            if (trialStart) {
              resolve(trialStart.toDate ? trialStart.toDate() : new Date(trialStart));
              return;
            }
            // Backend stores trial_end as a top-level ISO string — derive start from it
            const trialEnd = data?.trial_end;
            if (trialEnd) {
              const endDate = new Date(trialEnd);
              if (!isNaN(endDate.getTime())) {
                resolve(new Date(endDate.getTime() - 3 * 24 * 60 * 60 * 1000));
                return;
              }
            }
            resolve(null);
          } catch (error) {
            console.error('Error fetching trial start from Firestore:', error);
            resolve(null);
          }
        });
      } catch (err) {
        console.error("Auth module load error:", err);
        resolve(null);
      }
    });
  }

  // ============================================================
  // UPDATE WARMUP STATUS DISPLAY
  // ============================================================
  function updateWarmupDisplay(warmupStatus) {
    const warmupElement = document.getElementById('warmup-status');
    if (warmupElement) {
      warmupElement.innerText = warmupStatus;
      // Color code the warmup status
      const [done, total] = warmupStatus.split('/').map(x => parseInt(x.trim()));
      if (done && total) {
        const percentage = (done / total) * 100;
        if (percentage < 30) {
          warmupElement.style.color = '#ff6b6b';
        } else if (percentage < 70) {
          warmupElement.style.color = '#ffa500';
        } else {
          warmupElement.style.color = '#51cf66';
        }
      }
    }
  }

  // ============================================================
  // UPDATE COUNTDOWN DISPLAY
  // ============================================================
  async function updateTrialDisplay() {
    if (!currentUserId) return;

    const trialInfo = await getUserTrialInfo(currentUserId);

    if (trialInfo?.expired && !window.trialExpiredTriggered) {
      window.trialExpiredTriggered = true;
      document.dispatchEvent(new CustomEvent('trialExpired', { detail: { userId: currentUserId } }));
    }

    // Find countdown elements on page
    const countdownElements = Array.from(document.querySelectorAll('.trial-countdown, [data-trial-countdown]'));
    const countdownDisplay = document.getElementById('countdown-display');

    if (countdownDisplay) {
      countdownElements.push(countdownDisplay);
    }

    countdownElements.forEach(element => {
      if (trialInfo?.isLoading && !trialInfo?.isLoadingFallback) {
        // Loading state - show with progress
        element.innerHTML = `
          <span class="sovereign-badge">SOVEREIGN</span>
          <i class="fas fa-spinner" style="animation: spin 1s linear infinite;"></i>
          ${trialInfo.display}
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(100, 150, 255, 0.1)';
        element.style.borderColor = 'rgba(100, 150, 255, 0.3)';
      } else if (trialInfo?.isLoadingFallback) {
        // Loading fallback - show trial with caveat
        element.innerHTML = `
          <span class="sovereign-badge">SOVEREIGN</span>
          <i class="fas fa-hourglass-end"></i>
          Free Trial: <strong>${trialInfo.display}</strong>
          <span style="font-size:0.85em;opacity:0.7;margin-left:10px;">(Offline mode)</span>
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(100, 150, 255, 0.1)';
        element.style.borderColor = 'rgba(100, 150, 255, 0.3)';
      } else if (trialInfo?.authError) {
        // Auth error state
        element.innerHTML = `
          <i class="fas fa-exclamation-circle"></i>
          ${trialInfo.display}
          <button onclick="location.reload()" style="margin-left: 10px; padding: 4px 8px; cursor: pointer;">Retry</button>
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(255, 100, 100, 0.15)';
        element.style.borderColor = 'rgba(255, 100, 100, 0.4)';
        element.style.color = '#ff6464';
      } else if (trialInfo?.networkError) {
        // Network error state with retry button
        element.innerHTML = `
          <i class="fas fa-wifi-off"></i>
          ${trialInfo.display}
          <button onclick="location.reload()" style="margin-left: 10px; padding: 4px 8px; cursor: pointer;">Retry</button>
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(255, 140, 0, 0.15)';
        element.style.borderColor = 'rgba(255, 140, 0, 0.4)';
        element.style.color = '#ff8c00';
      } else if (trialInfo?.isPremium) {
        // Show tier badge instead of hiding the element
        const plan = (trialInfo.plan || 'pro').toLowerCase();
        const TIER_CONFIG = {
          pro:          { label: 'Pro',          icon: 'fa-crown',      color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  border: 'rgba(245,158,11,0.35)' },
          premium:      { label: 'Pro',          icon: 'fa-crown',      color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  border: 'rgba(245,158,11,0.35)' },
          intermediate: { label: 'Intermediate', icon: 'fa-bolt',       color: '#a78bfa', bg: 'rgba(167,139,250,0.10)', border: 'rgba(167,139,250,0.35)' },
          basic:        { label: 'Basic',        icon: 'fa-shield-alt', color: '#60a5fa', bg: 'rgba(96,165,250,0.10)',  border: 'rgba(96,165,250,0.35)'  },
        };
        const cfg = TIER_CONFIG[plan] || TIER_CONFIG.pro;
        element.innerHTML = `
          <i class="fas ${cfg.icon}" style="color:${cfg.color};margin-right:8px;"></i>
          <strong style="color:${cfg.color};">${cfg.label} Plan Active</strong>
          <span style="margin-left:10px;font-size:0.75em;opacity:0.65;">Full access enabled</span>
        `;
        element.style.display = 'block';
        element.style.background = cfg.bg;
        element.style.borderColor = cfg.border;
        element.style.color = cfg.color;
      } else if (trialInfo?.active) {
        element.innerHTML = `
          <span class="sovereign-badge">SOVEREIGN</span>
          <i class="fas fa-hourglass-end"></i>
          Free Trial: <strong>${trialInfo.display}</strong>
        `;
        element.style.display = 'block';

        // Add urgency styling if less than 24h remaining
        if (trialInfo.hours < 24 && trialInfo.days === 0) {
          element.style.background = 'rgba(255, 140, 0, 0.15)';
          element.style.borderColor = 'rgba(255, 140, 0, 0.4)';
        }

        // Critical styling if less than 1h
        if (trialInfo.hours === 0 && trialInfo.minutes < 60) {
          element.style.background = 'rgba(255, 0, 85, 0.15)';
          element.style.borderColor = 'rgba(255, 0, 85, 0.4)';
          element.style.color = '#ff0055';
        }
      } else if (trialInfo?.expired) {
        element.innerHTML = `
          <i class="fas fa-exclamation-triangle"></i>
          Trial Expired - Upgrade to continue
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(255, 0, 85, 0.2)';
        element.style.borderColor = 'rgba(255, 0, 85, 0.5)';
        element.style.color = '#ff0055';
      } else if (trialInfo?.noTrial) {
        element.innerHTML = `
          <i class="fas fa-info-circle"></i>
          No Active Trial - Please select a plan
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(255, 140, 0, 0.15)';
        element.style.borderColor = 'rgba(255, 140, 0, 0.4)';
        element.style.color = '#ff8c00';
      } else {
        element.style.display = 'none';
      }
    });
  }

  // ============================================================
  // INITIALIZE TRIAL COUNTDOWN & WARMUP DISPLAY
  // ============================================================
  async function initializeTrialCountdown(userId, trialStart = null) {
    // Inject required CSS for animations
    injectLoadingStyles();
    
    // Reset state in case of re-initialization
    resetTrialState();
    
    let authWaitElapsed = 0;
    while (!AuthManager.getAuthHeader() && authWaitElapsed < 10000) {
      await new Promise(r => setTimeout(r, 500));
      authWaitElapsed += 500;
    }

    if (!AuthManager.getAuthHeader()) {
      console.warn("Auth timeout reached, defaulting to Restricted Trial");
      const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown], #countdown-display');
      countdownElements.forEach(element => {
        element.innerHTML = `
          <i class="fas fa-exclamation-triangle"></i>
          Restricted Trial - Sign in to access full features
        `;
        element.style.display = 'block';
        element.style.background = 'rgba(255, 140, 0, 0.15)';
        element.style.borderColor = 'rgba(255, 140, 0, 0.4)';
        element.style.color = '#ff8c00';
      });
      return;
    }

    currentUserId = userId;
    
    // Fallback if trialStart is not provided (e.g. from app.js)
    if (!trialStart) {
      const cachedStart = localStorage.getItem('trial_start_timestamp');
      if (cachedStart) {
        trialStart = new Date(cachedStart);
      } else {
        const freshStart = await fetchTrialStartFromFirestore(userId);
        trialStart = freshStart ? freshStart : new Date();
      }
    }

    if (trialStart) {
      explicitTrialStart = trialStart instanceof Date ? trialStart : new Date(trialStart);
      if (!Number.isNaN(explicitTrialStart.getTime())) {
        localStorage.setItem('trial_start_timestamp', explicitTrialStart.toISOString());
        const trialEnd = new Date(explicitTrialStart.getTime() + 3 * 24 * 60 * 60 * 1000);
        localStorage.setItem('trial_end_timestamp', trialEnd.toISOString());
      }
    }

    // Update countdown immediately
    await updateTrialDisplay();

    // Hide "Preparing your trial token cards" once trial data is loaded and countdown starts
    const signalsContainer = document.getElementById('signalsContainer');
    if (signalsContainer) {
        const noSignalsDiv = signalsContainer.querySelector('.no-signals');
        if (noSignalsDiv && noSignalsDiv.textContent.includes('Preparing your trial token cards')) {
            noSignalsDiv.style.display = 'none';
        }
    }

    // Update every second
    if (trialCheckInterval) clearInterval(trialCheckInterval);
    trialCheckInterval = setInterval(updateTrialDisplay, 1000);

    // Listen for warmup updates from engine
    document.addEventListener('warmupUpdate', (e) => {
      if (e.detail.warmup) {
        updateWarmupDisplay(e.detail.warmup);
      }
    });

    // Listen for trial status updates from background fetch
    window.addEventListener('trial-status-updated', () => {
      console.log('[Trial] Status updated event received, refreshing display');
      updateTrialDisplay();
    });

    console.log('✅ Trial countdown initialized');
  }

  // ============================================================
  // CHECK IF TRIAL EXPIRED
  // ============================================================
  async function isTrialExpired(userId) {
    const trialInfo = await getUserTrialInfo(userId);
    return trialInfo?.expired || false;
  }

  // ============================================================
  // GET TRIAL RESTRICTIONS
  // ============================================================
  async function getTrialRestrictions(userId) {
    const trialInfo = await getUserTrialInfo(userId);

    if (!trialInfo?.active) {
      return {
        maxTokens: 0,
        allowedTokens: [],
        allowedTimeframes: [],
        maxSignalsPerDay: 0
      };
    }

    return {
      maxTokens: 5,
      allowedTokens: trialInfo.allowedTokens || ['BTC', 'ETH', 'SOL', 'ARB', 'AAVE'],
      allowedTimeframes: trialInfo.allowedTimeframes || ['30m', '1h'],
      maxSignalsPerDay: 10,
      message: 'Trial user - Limited to 5 tokens, 30m/1h timeframes only'
    };
  }

  // ============================================================
  // SEND TRIAL EXPIRY NOTIFICATION
  // ============================================================
  async function checkAndSendTrialExpiryNotification(userId) {
    if (!userId) return;

    try {
      const userDocRef = doc(db, 'users', userId);
      const userDoc = await getDoc(userDocRef);

      if (!userDoc.exists()) return;

      const userData = userDoc.data();
      const trial = userData.trial || {};

      // Only send if trial expires within 24h and hasn't been notified yet
      if (trial.active && trial.endDate && !trial.expiryNotified) {
        const expiryDate = trial.endDate.toDate ? trial.endDate.toDate() : new Date(trial.endDate);
        const now = new Date();
        const hoursUntilExpiry = (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60);

        if (hoursUntilExpiry < 24 && hoursUntilExpiry > 0) {
          // Send notification (via email, push, or in-app)
          console.log('🔔 Trial expiring soon for user:', userId);

          // Update notification flag
          await updateDoc(userDocRef, {
            'trial.expiryNotified': true
          });

          return { notified: true, hoursRemaining: hoursUntilExpiry };
        }
      }

      return { notified: false };
    } catch (error) {
      console.error('❌ Error checking trial expiry:', error);
      return { notified: false };
    }
  }

  // ============================================================
  // RESET STATE FOR MANUAL RETRY
  // ============================================================
  function resetTrialState() {
    authRetryCount = 0;
    loadingStartTime = null;
    networkErrorState = false;
    cachedTrialInfo = null;
    lastFetchTime = 0;
    console.log('✅ Trial state reset - retrying...');
  }

  // ============================================================
  // INJECT LOADING ANIMATION STYLES
  // ============================================================
  function injectLoadingStyles() {
    if (document.getElementById('trial-countdown-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'trial-countdown-styles';
    style.textContent = `
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
      @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
      }
    `;
    document.head.appendChild(style);
  }

  // ============================================================
  // CLEANUP
  // ============================================================
  function stopTrialCountdown() {
    if (trialCheckInterval) {
      clearInterval(trialCheckInterval);
      trialCheckInterval = null;
    }
    currentUserId = null;
    window.trialExpiredTriggered = false;

    // Reset state variables
    authRetryCount = 0;
    loadingStartTime = null;
    networkErrorState = false;

    // Clean up any stray UI elements or listeners if necessary
    const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown]');
    countdownElements.forEach(el => el.style.display = 'none');
  }

  // ============================================================
  // HELPER: Add Trial Badge to UI
  // ============================================================
  function addTrialBadgeToUI(trialInfo) {
    if (!trialInfo?.active) return;

    const badgeHTML = `
      <div class="trial-badge" style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid rgba(0, 242, 255, 0.3);
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--primary-cyan);
        margin-left: 0.5rem;
      ">
        <i class="fas fa-flask"></i>
        ${trialInfo.display}
      </div>
    `;

    return badgeHTML;
  }

  // ============================================================
  // CHECK SIGNAL ACCESS FOR TRIAL USER
  // ============================================================
  async function canAccessSignal(userId, symbol, timeframe) {
    const restrictions = await getTrialRestrictions(userId);

    // Non-trial users can access everything
    if (!restrictions.allowedTokens.length) {
      return { allowed: true };
    }

    // Check token
    if (!restrictions.allowedTokens.includes(symbol)) {
      return {
        allowed: false,
        reason: `Trial users limited to: ${restrictions.allowedTokens.join(', ')}`
      };
    }

    // Check timeframe
    if (!restrictions.allowedTimeframes.includes(timeframe)) {
      return {
        allowed: false,
        reason: `Trial users limited to: ${restrictions.allowedTimeframes.join(', ')}`
      };
    }

    return { allowed: true };
  }

  // ============================================================
  // FILTER SIGNALS FOR TRIAL USER
  // ============================================================
  async function filterSignalsForUser(userId, signals) {
    const restrictions = await getTrialRestrictions(userId);

    // Non-trial users see all signals
    if (!restrictions.allowedTokens.length) {
      return signals;
    }

    // Filter signals
    return signals.filter(signal => {
      const symbol = signal.symbol || signal.asset || '';
      const timeframe = signal.timeframe || '1h';

      return restrictions.allowedTokens.includes(symbol) &&
        restrictions.allowedTimeframes.includes(timeframe);
    });
  }

  // ============================================================
  // DISPLAY TRIAL RESTRICTION MESSAGE
  // ============================================================
  function showTrialRestrictionMessage(reason) {
    const messageEl = document.createElement('div');
    messageEl.style.cssText = `
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: rgba(255, 140, 0, 0.15);
      border: 1px solid rgba(255, 140, 0, 0.4);
      border-radius: 12px;
      padding: 1rem;
      max-width: 300px;
      color: #ff8c00;
      font-size: 0.9rem;
      z-index: 1000;
      animation: slideIn 0.3s ease;
    `;

    messageEl.innerHTML = `
      <i class="fas fa-info-circle" style="margin-right: 0.5rem;"></i>
      ${reason}
    `;

    document.body.appendChild(messageEl);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      messageEl.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => messageEl.remove(), 300);
    }, 5000);
  }

  return {
    getUserTrialInfo,
    fetchTrialStartFromFirestore,
    initializeTrialCountdown,
    isTrialExpired,
    getTrialRestrictions,
    checkAndSendTrialExpiryNotification,
    stopTrialCountdown,
    resetTrialState,
    injectLoadingStyles,
    addTrialBadgeToUI,
    canAccessSignal,
    filterSignalsForUser,
    showTrialRestrictionMessage
  };
})();

export const getUserTrialInfo = TrialManager.getUserTrialInfo;
export const fetchTrialStartFromFirestore = TrialManager.fetchTrialStartFromFirestore;
export const initializeTrialCountdown = TrialManager.initializeTrialCountdown;
export const isTrialExpired = TrialManager.isTrialExpired;
export const getTrialRestrictions = TrialManager.getTrialRestrictions;
export const checkAndSendTrialExpiryNotification = TrialManager.checkAndSendTrialExpiryNotification;
export const stopTrialCountdown = TrialManager.stopTrialCountdown;
export const resetTrialState = TrialManager.resetTrialState;
export const injectLoadingStyles = TrialManager.injectLoadingStyles;
export const addTrialBadgeToUI = TrialManager.addTrialBadgeToUI;
export const canAccessSignal = TrialManager.canAccessSignal;
export const filterSignalsForUser = TrialManager.filterSignalsForUser;
export const showTrialRestrictionMessage = TrialManager.showTrialRestrictionMessage;

// Animation keyframes (add to CSS if not present)
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes slideOut {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(20px); }
  }
  .sovereign-badge {
    display: inline-block;
    background: rgba(0, 242, 255, 0.15);
    color: var(--neon-blue, #00f2ff);
    border: 1px solid var(--neon-blue, #00f2ff);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: bold;
    letter-spacing: 1px;
    margin-right: 10px;
    animation: pulseBadge 2s infinite;
  }
  @keyframes pulseBadge {
    0% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(0, 242, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0); }
  }
`;
document.head.appendChild(style);




