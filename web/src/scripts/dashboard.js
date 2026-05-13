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

  const userId = event?.detail?.userData?.uid || await waitForUserId();
  if (!userId) {
    console.warn('Dashboard countdown: could not resolve current user UID');
    return;
  }

  initialized = true;
  const trialStart = await fetchTrialStartFromFirestore(userId);
  if (trialStart?.error) {
    console.error('Dashboard init failed on trial fetch:', trialStart.message);
    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    window.location.href = '/login.html';
    return;
  }

  await initializeTrialCountdown(userId, trialStart);

  document.addEventListener('trialExpired', () => {
    setExpiredView();
  });
}

window.addEventListener('DOMContentLoaded', initializeDashboard);
document.addEventListener('dashboardUserLoaded', initializeDashboard);
