// ============================================================
// SIGNIN UI – Professional Login Interface
// ============================================================

import {
  handleGoogleAuth,
  handleEmailLogin,
  handlePasswordReset,
  isUserAuthenticated
} from './auth.js';

export function createSignInModal() {
  return `
    <div id="signinModal" class="auth-modal-overlay">
      <div class="auth-modal-container">
        <button class="auth-modal-close" id="signinClose">
          <i class="fas fa-times"></i>
        </button>

        <!-- SIGNIN FORM -->
        <div class="auth-header">
          <div class="auth-logo">⚡ AEGIS</div>
          <h2>Welcome Back</h2>
          <p>Sign in to your trading account</p>
        </div>

        <form id="signinEmailForm" class="auth-form">
          <div class="form-group">
            <label>Email Address</label>
            <input 
              type="email" 
              id="signinEmail" 
              placeholder="your@email.com"
              required
            >
            <span class="error-msg" id="signinEmailError"></span>
          </div>

          <div class="form-group">
            <label>Password</label>
            <input 
              type="password" 
              id="signinPassword" 
              placeholder="••••••••"
              required
            >
            <span class="error-msg" id="signinPasswordError"></span>
          </div>

          <button type="submit" class="auth-btn-primary">Sign In</button>
          <span class="auth-error" id="signinFormError"></span>
        </form>

        <div class="auth-divider">OR</div>

        <button type="button" class="auth-btn-social" id="googleSigninBtn">
          <i class="fab fa-google"></i> Continue with Google
        </button>

        <div class="auth-footer">
          <p>Don't have an account? <a href="#" id="toSignupLink" class="auth-link">Create one</a></p>
          <p><a href="#" id="forgotPasswordLink" class="auth-link">Forgot password?</a></p>
        </div>

        <div class="auth-security-badge">
          <i class="fas fa-lock"></i> Secured by Firebase
        </div>
      </div>
    </div>

    <!-- FORGOT PASSWORD MODAL -->
    <div id="forgotPasswordModal" class="auth-modal-overlay" style="display: none;">
      <div class="auth-modal-container">
        <button class="auth-modal-close" id="forgotPasswordClose">
          <i class="fas fa-times"></i>
        </button>

        <div class="auth-header">
          <h2>Reset Password</h2>
          <p>Enter your email to receive a reset link</p>
        </div>

        <form id="forgotPasswordForm" class="auth-form">
          <div class="form-group">
            <label>Email Address</label>
            <input 
              type="email" 
              id="forgotEmail" 
              placeholder="your@email.com"
              required
            >
            <span class="error-msg" id="forgotEmailError"></span>
          </div>

          <button type="submit" class="auth-btn-primary">Send Reset Link</button>
          <span class="auth-error" id="forgotPasswordError"></span>
          <span class="auth-success" id="forgotPasswordSuccess"></span>
        </form>

        <div class="auth-footer">
          <p><a href="#" id="backToSigninLink" class="auth-link">Back to Sign In</a></p>
        </div>
      </div>
    </div>

    <!-- LOADING OVERLAY -->
    <div id="signinLoadingOverlay" class="auth-loading-overlay hidden">
      <div class="spinner"></div>
      <p>Signing in...</p>
    </div>
  `;
}

export function initSignInUI() {
  const modalContainer = document.createElement('div');
  modalContainer.innerHTML = createSignInModal();
  document.body.appendChild(modalContainer);

  attachSignInEventListeners();
  console.log('✅ Sign In UI initialized');
}

function attachSignInEventListeners() {
  // Close buttons
  document.getElementById('signinClose')?.addEventListener('click', closeSignInModal);
  document.getElementById('forgotPasswordClose')?.addEventListener('click', closeForgotPasswordModal);

  // Form submissions
  document.getElementById('signinEmailForm')?.addEventListener('submit', performEmailSignin);
  document.getElementById('forgotPasswordForm')?.addEventListener('submit', performForgotPassword);

  // Navigation links
  document.getElementById('toSignupLink')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeSignInModal();
    // Emit event to switch to signup
    window.dispatchEvent(new CustomEvent('openSignup'));
  });

  document.getElementById('forgotPasswordLink')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeSignInModal();
    openForgotPasswordModal();
  });

  document.getElementById('backToSigninLink')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeForgotPasswordModal();
    openSignInModal();
  });

  // Google auth
  document.getElementById('googleSigninBtn')?.addEventListener('click', performGoogleSignin);

  // Close on outside click
  document.getElementById('signinModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'signinModal') closeSignInModal();
  });
}

async function performEmailSignin(e) {
  e.preventDefault();
  const email = document.getElementById('signinEmail')?.value?.trim();
  const password = document.getElementById('signinPassword')?.value;

  if (!email || !password) {
    showError('signinFormError', 'Email and password required');
    return;
  }

  showLoading();
  try {
    const result = await handleEmailLogin(email, password);
    if (result.success) {
      showSuccess('Sign in successful!');
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      setTimeout(() => {
        closeSignInModal();
        if (!window.pendingPlan && !window.pendingAction) {
          window.location.href = '/web/src/pages/dashboard.html';
        }
      }, 1000);
    } else if (result.needsVerification) {
      showError('signinFormError', '📧 ' + result.message);
    } else {
      showError('signinFormError', result.message);
    }
  } catch (error) {
    showError('signinFormError', error.message);
  } finally {
    hideLoading();
  }
}

async function performGoogleSignin() {
  showLoading();
  try {
    const result = await handleGoogleAuth();
    if (result.success) {
      showSuccess('Google sign in successful!');
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      setTimeout(() => {
        closeSignInModal();
        if (!window.pendingPlan && !window.pendingAction) {
          window.location.href = '/web/src/pages/dashboard.html';
        }
      }, 1000);
    } else {
      showError('signinFormError', result.message);
    }
  } catch (error) {
    showError('signinFormError', error.message);
  } finally {
    hideLoading();
  }
}

async function performForgotPassword(e) {
  e.preventDefault();
  const email = document.getElementById('forgotEmail')?.value?.trim();

  if (!email) {
    showError('forgotEmailError', 'Email required');
    return;
  }

  showLoading();
  try {
    const result = await handlePasswordReset(email);
    if (result.success) {
      document.getElementById('forgotPasswordSuccess').textContent = result.message;
      document.getElementById('forgotPasswordError').textContent = '';
      document.getElementById('forgotEmail').value = '';
    } else {
      showError('forgotPasswordError', result.message);
    }
  } catch (error) {
    showError('forgotPasswordError', error.message);
  } finally {
    hideLoading();
  }
}

export function openSignInModal() {
  const modal = document.getElementById('signinModal');
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

export function closeSignInModal() {
  const modal = document.getElementById('signinModal');
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
    // Reset form
    document.getElementById('signinEmailForm')?.reset();
    document.getElementById('signinFormError').textContent = '';
  }
}

function openForgotPasswordModal() {
  const modal = document.getElementById('forgotPasswordModal');
  if (modal) modal.style.display = 'flex';
}

function closeForgotPasswordModal() {
  const modal = document.getElementById('forgotPasswordModal');
  if (modal) modal.style.display = 'none';
}

function showLoading() {
  document.getElementById('signinLoadingOverlay')?.classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('signinLoadingOverlay')?.classList.add('hidden');
}

function showError(elementId, message) {
  const el = document.getElementById(elementId);
  if (el) {
    el.textContent = message;
    el.style.display = 'block';
  }
}

function showSuccess(message) {
  console.log('✅', message);
}

export function updateSignInButtonState() {
  const btn = document.getElementById('portalBtn');
  if (!btn) return;

  if (isUserAuthenticated()) {
    btn.textContent = '👤 Account';
    btn.onclick = (e) => {
      e.preventDefault();
      // Show account menu or navigate to dashboard
      window.location.href = '/web/src/pages/dashboard.html';
    };
  } else {
    btn.textContent = '🔐 Client Login';
    btn.onclick = (e) => {
      e.preventDefault();
      openSignInModal();
    };
  }
}
