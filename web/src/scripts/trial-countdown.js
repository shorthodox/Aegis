// ============================================================
// AEGIS Trial Countdown – 3-Day Free Trial Management
// ============================================================

import { db } from './gatekeeper.js';
import { doc, getDoc, updateDoc, serverTimestamp } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";

// ============================================================
// STATE
// ============================================================
let currentUserId = null;
let trialCheckInterval = null;

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
  
  let display = '';
  if (days > 0) {
    display = `${days}d ${hours}h remaining`;
  } else if (hours > 0) {
    display = `${hours}h ${minutes}m remaining`;
  } else if (minutes > 0) {
    display = `${minutes}m ${seconds}s remaining`;
  } else {
    display = `${seconds}s remaining`;
  }
  
  return { expired: false, display, days, hours, minutes, seconds };
}

// ============================================================
// HELPER: Get User Trial Info
// ============================================================
export async function getUserTrialInfo(userId) {
  if (!userId) return null;
  
  try {
    const userDocRef = doc(db, 'users', userId);
    const userDoc = await getDoc(userDocRef);
    
    if (!userDoc.exists()) return null;
    
    const userData = userDoc.data();
    const trial = userData.trial || {};
    
    if (!trial.active || !trial.endDate) {
      return {
        active: false,
        message: 'Trial not active'
      };
    }
    
    const expiryDate = trial.endDate.toDate ? trial.endDate.toDate() : new Date(trial.endDate);
    const timeInfo = formatTimeRemaining(expiryDate);
    
    return {
      active: true,
      ...timeInfo,
      allowedTokens: trial.allowedTokens || [],
      allowedTimeframes: trial.allowedTimeframes || ['30m', '1h'],
      plan: userData.plan,
      trialEndDate: expiryDate
    };
  } catch (error) {
    console.error('❌ Error fetching trial info:', error);
    return null;
  }
}

// ============================================================
// INITIALIZE TRIAL COUNTDOWN DISPLAY
// ============================================================
export function initializeTrialCountdown(userId) {
  currentUserId = userId;
  
  // Update countdown immediately
  updateTrialDisplay();
  
  // Update every second
  if (trialCheckInterval) clearInterval(trialCheckInterval);
  trialCheckInterval = setInterval(updateTrialDisplay, 1000);
  
  console.log('✅ Trial countdown initialized');
}

// ============================================================
// UPDATE COUNTDOWN DISPLAY
// ============================================================
async function updateTrialDisplay() {
  if (!currentUserId) return;
  
  const trialInfo = await getUserTrialInfo(currentUserId);
  
  // Find countdown element on page
  const countdownElements = document.querySelectorAll('.trial-countdown, [data-trial-countdown]');
  
  countdownElements.forEach(element => {
    if (trialInfo?.active) {
      element.innerHTML = `
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
    } else {
      element.style.display = 'none';
    }
  });
}

// ============================================================
// CHECK IF TRIAL EXPIRED
// ============================================================
export async function isTrialExpired(userId) {
  const trialInfo = await getUserTrialInfo(userId);
  return trialInfo?.expired || false;
}

// ============================================================
// GET TRIAL RESTRICTIONS
// ============================================================
export async function getTrialRestrictions(userId) {
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
export async function checkAndSendTrialExpiryNotification(userId) {
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
// CLEANUP
// ============================================================
export function stopTrialCountdown() {
  if (trialCheckInterval) {
    clearInterval(trialCheckInterval);
    trialCheckInterval = null;
  }
}

// ============================================================
// HELPER: Add Trial Badge to UI
// ============================================================
export function addTrialBadgeToUI(trialInfo) {
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
export async function canAccessSignal(userId, symbol, timeframe) {
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
export async function filterSignalsForUser(userId, signals) {
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
export function showTrialRestrictionMessage(reason) {
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
`;
document.head.appendChild(style);
