// Lightweight auth client for pages (server-backed login + subscription helper)
// Works with server endpoints: POST /auth/login and POST /create-subscription
import { loadThirdPartyScript } from './iframeGuard.js';

function createModalIfMissing() {
  if (document.getElementById('simpleAuthModal')) return;
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div id="simpleAuthModal" class="modal" style="display:none;">
      <div class="modal-card">
        <span class="close-modal" id="simpleAuthClose">&times;</span>
        <div style="text-align:center; margin-bottom:1rem;">
          <div style="font-size:1.6rem; font-weight:800; background:linear-gradient(135deg,#fff,#00f2ff); -webkit-background-clip:text; color:transparent;">⚡ AEGIS</div>
          <p style="color:var(--dim); margin-top:0.4rem;">Sign in to access your dashboard and subscriptions</p>
        </div>

        <form id="simpleLoginForm">
          <label style="font-size:0.8rem; color:var(--dim);">Email</label>
          <input id="simpleEmail" type="email" class="auth-input" required placeholder="you@domain.com">
          <label style="font-size:0.8rem; color:var(--dim); margin-top:0.5rem;">Password</label>
          <input id="simplePassword" type="password" class="auth-input" required placeholder="••••••••">
          <button id="simpleLoginSubmit" class="auth-submit-btn" style="margin-top:0.8rem;">Sign In</button>
        </form>

        <div style="margin-top:0.6rem; text-align:center;">
          <a id="simpleGoogleLink" class="provider-btn" href="/auth/google" style="text-decoration:none; display:inline-block;"> <i class="fab fa-google"></i> Continue with Google</a>
        </div>

        <div id="simpleAuthError" style="display:none; color:#ff7b7b; margin-top:0.8rem; text-align:center;"></div>
      </div>
    </div>
  `;
  document.body.appendChild(wrap);

  const modal = document.getElementById('simpleAuthModal');
  const close = document.getElementById('simpleAuthClose');
  close?.addEventListener('click', () => closeModal());
  modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });

  const form = document.getElementById('simpleLoginForm');
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    await doLogin();
  });
}

function openModal() {
  createModalIfMissing();
  const modal = document.getElementById('simpleAuthModal');
  if (!modal) return;
  modal.style.display = 'flex';
  document.body.classList.add('modal-open');
}

function closeModal() {
  const modal = document.getElementById('simpleAuthModal');
  if (!modal) return;
  modal.style.display = 'none';
  document.body.classList.remove('modal-open');
}

async function doLogin() {
  const email = document.getElementById('simpleEmail')?.value?.trim();
  const password = document.getElementById('simplePassword')?.value;
  const errEl = document.getElementById('simpleAuthError');
  if (errEl) { errEl.style.display = 'none'; errEl.textContent = ''; }
  if (!email || !password) {
    if (errEl) { errEl.style.display = 'block'; errEl.textContent = 'Email and password are required'; }
    return;
  }
  const btn = document.getElementById('simpleLoginSubmit');
  try {
    btn.disabled = true;
    btn.innerText = 'Signing in...';
    const resp = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await resp.json().catch(() => ({}));
    if (resp.ok && data.access_token) {
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('authenticated', 'true');
      closeModal();
      // try to update any UI provided by auth-modal
      try { const m = await import('./auth-modal.js'); if (m.updateAuthButtonState) m.updateAuthButtonState(); } catch(e) {}
      window.location.reload();
      return;
    }
    const message = data.detail || data.message || 'Invalid credentials';
    if (errEl) { errEl.style.display = 'block'; errEl.textContent = message; }
  } catch (err) {
    if (errEl) { errEl.style.display = 'block'; errEl.textContent = 'Network error'; }
  } finally {
    btn.disabled = false;
    btn.innerText = 'Sign In';
  }
}

async function subscribeToPlan(planType) {
  const token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
  if (!token) {
    openModal();
    return;
  }

  try {
    // Prefer locally-stored profile (Firebase auth flow populates this via AuthManager.setUser).
    // Avoids hitting /auth/me with a Firebase ID token that the custom server endpoint rejects.
    let userEmail = null, userId = null;
    try {
      const profileRaw = localStorage.getItem('user_profile');
      if (profileRaw) {
        const profile = JSON.parse(profileRaw);
        userEmail = profile.email || null;
        userId = profile.uid || null;
      }
    } catch (_) {}

    // Get a fresh Firebase token (handles 1-hour expiry; Firebase SDK auto-refreshes).
    // Also fills userEmail/userId if the stored profile was missing.
    let freshToken = token;
    try {
      const { auth } = await import('./gatekeeper.js');
      if (auth?.currentUser) {
        freshToken = await auth.currentUser.getIdToken();
        localStorage.setItem('access_token', freshToken);
        if (!userEmail) userEmail = auth.currentUser.email;
        if (!userId)   userId   = auth.currentUser.uid;
      }
    } catch (_) {}

    // Hard fallback: server-issued JWT users who never went through Firebase login
    if (!userEmail) {
      const meResp = await fetch('/auth/me', { headers: { 'Authorization': `Bearer ${freshToken}` } });
      if (!meResp.ok) {
        openModal();
        return;
      }
      const me = await meResp.json();
      userEmail = me.email;
      userId = me.uid || me.id || 'user_unknown';
    }

    if (!userEmail) {
      openModal();
      return;
    }

    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const isIndia = timeZone === 'Asia/Calcutta' || timeZone === 'Asia/Kolkata';
    const currency = isIndia ? 'INR' : 'USD';

    let amount = isIndia ? 299.00 : 3.60;
    if (planType === 'pro') amount = isIndia ? 3499.00 : 40.00;
    else if (planType === 'intermediate') amount = isIndia ? 1999.00 : 24.00;

    const body = { tier: planType, amount: amount, currency: currency, email: userEmail, user_id: userId || 'user_unknown' };
    const resp = await fetch('/api/v1/create-payment-session', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${freshToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await resp.json().catch(() => ({}));
    if (!data.success || !data.payment_session_id) {
      alert('Subscription API failed. Falling back to mock (development mode). Error: ' + (data.error || data.detail || 'Invalid session'));
      // Simulate mock success here if desired or simply return
      return;
    }
    
    // Load Cashfree SDK on-demand (guarded — errors if blocked or timed out)
    try {
      await loadThirdPartyScript('https://sdk.cashfree.com/js/v3/cashfree.js');
    } catch (sdkErr) {
      console.error('[Cashfree] SDK failed to load:', sdkErr);
      alert('Payment gateway failed to load. Please disable ad-blockers and refresh.');
      return;
    }

    // Use "sandbox" for TEST or "production" for LIVE. Hardcoded to sandbox for demo/safety.
    const cashfree = window.Cashfree({ mode: 'sandbox' });

    // Mobile browsers (iOS Safari / Android Chrome) block cross-origin iframes opened
    // after async operations because the user-gesture context is lost by then.
    // Use _self (full-page redirect) on mobile — return_url in the order already
    // points back to dashboard.html?order_id=... so the user lands there after payment.
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    if (isMobile) {
      cashfree.checkout({ paymentSessionId: data.payment_session_id, redirectTarget: '_self' });
      return; // page navigates away — nothing below runs
    }

    try {
      const result = await cashfree.checkout({ paymentSessionId: data.payment_session_id, redirectTarget: '_modal' });
      if (result.error) {
        console.error('[Cashfree] Checkout error:', result.error);
        alert('Payment was cancelled or failed. Please try again.');
      } else if (result.paymentDetails) {
        console.log('[Cashfree] Payment successful');
        alert('Payment successful! Redirecting to your dashboard...');
        window.location.href = '/src/pages/dashboard.html';
      }
    } catch (checkoutErr) {
      console.error('[Cashfree] Checkout exception:', checkoutErr);
      alert('Payment gateway error. Please try again.');
    }
  } catch (err) {
    console.error('subscribeToPlan error', err);
    alert('Network error while creating subscription');
  }
}

function attachHandlers() {
  // Portal button
  const portalBtn = document.getElementById('portalBtn');
  if (portalBtn) {
    portalBtn.removeEventListener('click', portalClickHandler);
    portalBtn.addEventListener('click', portalClickHandler);
  }

  // Trial button
  const trialBtn = document.getElementById('trialBtn');
  if (trialBtn) {
    trialBtn.removeEventListener('click', trialClickHandler);
    trialBtn.addEventListener('click', trialClickHandler);
  }

  // [data-plan] buttons are handled entirely by pricing.html's own module script
  // which owns the pendingPlan + authStateChange resume flow. Adding listeners
  // here would create a double-handler and open two modals simultaneously.
}

function portalClickHandler(e) {
  e.preventDefault();
  const token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
  if (token) {
    window.location.href = '/src/pages/dashboard.html';
  } else {
    openModal();
  }
}

function trialClickHandler(e) {
  e.preventDefault();
  const token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
  if (token) {
    window.location.href = '/src/pages/dashboard.html';
  } else {
    openModal();
  }
}


// Auto init
document.addEventListener('DOMContentLoaded', () => {
  createModalIfMissing();
  attachHandlers();
});

export { openModal, closeModal, subscribeToPlan };
