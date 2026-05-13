import { initializeTrialCountdown, fetchTrialStartFromFirestore } from './trial-countdown.js';
import { auth } from './gatekeeper.js';

let initialized = false;

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForUserId(timeoutMs = 5000) {
  let elapsed = 0;
  while (elapsed < timeoutMs) {
    const uid = auth.currentUser?.uid;
    if (uid) return uid;
    await sleep(100);
    elapsed += 100;
  }
  return null;
}

function setExpiredView() {
  const dashboardContent = document.getElementById('dashboard-main-content');
  const expiredCard = document.getElementById('access-expired-card');
  if (dashboardContent) dashboardContent.classList.add('hidden');
  if (expiredCard) expiredCard.classList.remove('hidden');
}

function clearExpiredView() {
  const dashboardContent = document.getElementById('dashboard-main-content');
  const expiredCard = document.getElementById('access-expired-card');
  if (dashboardContent) dashboardContent.classList.remove('hidden');
  if (expiredCard) expiredCard.classList.add('hidden');
}

async function initializeDashboard(event) {
  clearExpiredView();

  if (initialized) return;

  if (!window.location.pathname.includes('dashboard.html')) return;

  try {
    const userId = event?.detail?.userData?.uid || await waitForUserId();
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
    const trialStart = await fetchTrialStartFromFirestore(userId);
    await initializeTrialCountdown(userId, trialStart);

    document.addEventListener('trialExpired', () => {
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
