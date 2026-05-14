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
        <button onclick="handleSubscribeClick()" style="
          margin-top: 2rem;
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
        <p style="color: #999; margin-top: 2rem; font-size: 0.9rem;">
          All features are locked until you subscribe
        </p>
      </div>
    `;
    expiredCard.style.cssText = `
      display: none;
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
    if (!el.id || !el.id.includes('expired')) {
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
    el.style.pointerEvents = 'auto';
    el.style.opacity = '1';
  });
  
  console.log('✅ All features unlocked - subscription active');
}

// ============================================================
// SUBSCRIBE BUTTON HANDLER
// ============================================================
function handleSubscribeClick() {
  // Navigate to subscription page or open subscription modal
  window.location.href = '/pricing.html' || '/subscribe.html';
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
      window.location.href = '/index.html';
    });
    console.log('✅ Logo click handler initialized');
  }
}

// ============================================================
// FETCH AND UPDATE TOKEN MOVEMENT DATA
// ============================================================
async function updateTokenMovement() {
  try {
    const response = await fetch('web/src/data/live_signals.json');
    if (!response.ok) throw new Error('Network response was not ok');
    
    const data = await response.json();
    
    // Update the UI elements (example selectors)
    const movementElement = document.getElementById('token-movement-info');
    if (movementElement && data.movement) {
      movementElement.textContent = `${data.movement}%`;
      movementElement.className = data.movement >= 0 ? 'text-green' : 'text-red';
    }
    
    // Update other parts of the "token card" here
  } catch (err) {
    console.error('Failed to update token data:', err);
  }
}

async function initializeDashboard(event) {
  clearExpiredView();

  if (initialized) return;

  if (!window.location.pathname.includes('dashboard.html')) return;

  try {
    const userId = event?.detail?.userData?.uid || await waitForAuthStateChange();
    if (!userId) {
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

    initialized = true;
    
    // Initialize logo click handler
    initializeLogoClickHandler();
    
    // Wrap trial fetch in try/catch to allow updateTokenMovement to run even on failure
    let trialStart = null;
    try {
      trialStart = await fetchTrialStartFromFirestore(userId);
      await initializeTrialCountdown(userId, trialStart);
    } catch (trialErr) {
      console.error('Failed to fetch or initialize trial countdown:', trialErr);
      // Trial initialization failed, but dashboard will continue
    }

    // Always start fetching token movement data, regardless of trial status
    await updateTokenMovement();
    setInterval(updateTokenMovement, 30000);

    document.addEventListener('trialExpired', () => {
      console.log('🔒 Trial expired event triggered');
      setExpiredView();
    });
  } catch (err) {
    console.error('Error initializing dashboard:', err);
    const signalsContainer = document.getElementById('signalsContainer');
    if (signalsContainer) {
        signalsContainer.innerHTML = `
          <div class="no-signals" style="color: #ff3333; border-color: #ff3333;">
            <i class="fas fa-exclamation-triangle"></i>
            <p>Error loading dashboard components.</p>
            <button onclick="location.reload()" style="margin-top: 10px; padding: 4px 12px; background: rgba(255,0,0,0.2); border-radius: 4px; color: white;">Retry</button>
          </div>
        `;
    }
    const countdownDisplay = document.getElementById('countdown-display');
    if (countdownDisplay) {
        countdownDisplay.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Error loading trial data`;
        countdownDisplay.style.display = 'block';
        countdownDisplay.style.background = 'rgba(255, 0, 85, 0.15)';
        countdownDisplay.style.color = '#ff0055';
        countdownDisplay.style.borderColor = 'rgba(255, 0, 85, 0.4)';
    }
  }
}

window.addEventListener('DOMContentLoaded', initializeDashboard);
document.addEventListener('dashboardUserLoaded', initializeDashboard);
