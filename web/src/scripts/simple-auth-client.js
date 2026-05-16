// Lightweight auth client for pages (server-backed login + subscription helper)
// Works with server endpoints: POST /auth/login and POST /create-subscription

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
    const meResp = await fetch('/auth/me', { headers: { 'Authorization': `Bearer ${token}` } });
    if (!meResp.ok) {
      openModal();
      return;
    }
    const me = await meResp.json();
    let amount = 3.60;
    if (planType === 'pro') amount = 40.00;
    else if (planType === 'intermediate') amount = 24.00;
    
    const body = { plan_name: planType, amount: amount, email: me.email };
    const resp = await fetch('/create-subscription', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await resp.json().catch(() => ({}));
    if (resp.ok && data.success && data.sub_auth_url) {
      window.location.href = data.sub_auth_url;
      return;
    }
    alert('Subscription failed: ' + (data.detail || data.message || resp.status));
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

  // Subscription buttons
  document.querySelectorAll('[data-plan]').forEach(btn => {
    btn.removeEventListener('click', planClickHandler);
    btn.addEventListener('click', planClickHandler);
  });
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

function planClickHandler(e) {
  e.preventDefault();
  const plan = e.currentTarget.dataset.plan;
  if (!plan) return;
  subscribeToPlan(plan);
}

// Auto init
document.addEventListener('DOMContentLoaded', () => {
  createModalIfMissing();
  attachHandlers();
});

export { openModal, closeModal, subscribeToPlan };

