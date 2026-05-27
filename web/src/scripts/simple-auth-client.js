// Lightweight auth client for pages (server-backed login + subscription helper)
// Works with server endpoints: POST /auth/login and POST /create-subscription
import { loadThirdPartyScript } from './iframeGuard.js';

function showPaymentLoader() {
  if (document.getElementById('aegis-pay-loader')) return;
  const el = document.createElement('div');
  el.id = 'aegis-pay-loader';
  el.innerHTML = `
    <style>
      #aegis-pay-loader {
        position: fixed; inset: 0; z-index: 99999;
        background: rgba(0,0,0,0.82); backdrop-filter: blur(6px);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 20px;
      }
      #aegis-pay-loader .apl-ring {
        width: 56px; height: 56px;
        border: 3px solid rgba(0,242,255,0.15);
        border-top-color: #00f2ff;
        border-radius: 50%;
        animation: apl-spin 0.75s linear infinite;
      }
      #aegis-pay-loader .apl-text {
        font-family: monospace; font-size: 13px;
        color: rgba(0,242,255,0.85); letter-spacing: 0.12em;
        text-transform: uppercase; font-weight: 700;
      }
      #aegis-pay-loader .apl-sub {
        font-family: monospace; font-size: 10px;
        color: rgba(255,255,255,0.3); letter-spacing: 0.08em;
      }
      @keyframes apl-spin { to { transform: rotate(360deg); } }
    </style>
    <div class="apl-ring"></div>
    <div class="apl-text">Connecting to Gateway</div>
    <div class="apl-sub">Please wait&hellip;</div>
  `;
  document.body.appendChild(el);
}

function hidePaymentLoader() {
  const el = document.getElementById('aegis-pay-loader');
  if (el) el.remove();
}

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

  showPaymentLoader();

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
        hidePaymentLoader();
        openModal();
        return;
      }
      const me = await meResp.json();
      userEmail = me.email;
      userId = me.uid || me.id || 'user_unknown';
    }

    if (!userEmail) {
      hidePaymentLoader();
      openModal();
      return;
    }

    // Detect currency from timezone; backend converts from USD at live rate
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const currency = (timeZone === 'Asia/Calcutta' || timeZone === 'Asia/Kolkata') ? 'INR' : 'USD';

    // 1. Fetch Razorpay key_id from backend config (keeps secret off the frontend)
    const configResp = await fetch('/payment/config');
    const config = await configResp.json().catch(() => ({}));
    const keyId = config?.razorpay?.key_id;
    if (!keyId) {
      hidePaymentLoader();
      alert('Payment gateway is not configured. Please contact support.');
      return;
    }

    // 2. Create Razorpay order on the backend
    const orderResp = await fetch('/api/create-order', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${freshToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan: planType, currency })
    });
    if (!orderResp.ok) {
      hidePaymentLoader();
      if (orderResp.status === 401 || orderResp.status === 403) {
        alert('Your session has expired. Please sign in again.');
        window.location.href = '/web/src/pages/index.html';
        return;
      }
      const err = await orderResp.json().catch(() => ({}));
      alert('Could not create payment order. ' + (err.detail || 'Please try again.'));
      return;
    }
    const orderData = await orderResp.json();

    // 3. Load Razorpay checkout.js on-demand
    try {
      await loadThirdPartyScript('https://checkout.razorpay.com/v1/checkout.js');
    } catch (sdkErr) {
      hidePaymentLoader();
      console.error('[Razorpay] Checkout script failed to load:', sdkErr);
      alert('Payment gateway failed to load. Please disable ad-blockers and refresh.');
      return;
    }

    // 4. Open Razorpay modal
    hidePaymentLoader();
    await new Promise((resolve) => {
      const rzp = new window.Razorpay({
        key: keyId,
        amount: orderData.amount,
        currency: orderData.currency,
        order_id: orderData.order_id,
        name: 'AEGIS v1.0',
        description: `${planType.charAt(0).toUpperCase() + planType.slice(1)} Plan`,
        prefill: { email: userEmail },
        theme: { color: '#00f2ff' },
        handler: async (response) => {
          // 5. Verify signature on the backend
          try {
            const verifyResp = await fetch('/api/verify-payment', {
              method: 'POST',
              headers: { 'Authorization': `Bearer ${freshToken}`, 'Content-Type': 'application/json' },
              body: JSON.stringify({
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_order_id:   response.razorpay_order_id,
                razorpay_signature:  response.razorpay_signature,
                plan: planType,
              })
            });
            const verifyData = await verifyResp.json().catch(() => ({}));
            if (verifyResp.ok && verifyData.status === 'success') {
              alert('Payment successful! Redirecting to your dashboard...');
              window.location.href = '/web/src/pages/dashboard.html';
            } else {
              alert('Payment verification failed. Please contact support with your payment ID: ' + response.razorpay_payment_id);
            }
          } catch (verifyErr) {
            console.error('[Razorpay] Verify error:', verifyErr);
            alert('Network error during verification. Contact support with payment ID: ' + response.razorpay_payment_id);
          }
          resolve();
        },
        modal: {
          ondismiss: () => {
            console.log('[Razorpay] Payment modal dismissed by user');
            resolve();
          }
        }
      });
      rzp.on('payment.failed', (response) => {
        console.error('[Razorpay] Payment failed:', response.error);
        alert('Payment failed: ' + (response.error?.description || 'Please try again.'));
        resolve();
      });
      rzp.open();
    });
  } catch (err) {
    hidePaymentLoader();
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
    window.location.href = '/web/src/pages/dashboard.html';
  } else {
    openModal();
  }
}

function trialClickHandler(e) {
  e.preventDefault();
  const token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
  if (token) {
    window.location.href = '/web/src/pages/dashboard.html';
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
