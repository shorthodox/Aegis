// ============================================================
// SIGNUP UI – Professional Registration Interface
// ============================================================

import {
  handleGoogleAuth,
  handleEmailSignup,
  sendPhoneOTP,
  verifyPhoneOTP,
  isUserAuthenticated
} from './auth.js';

export function createSignUpModal() {
  return `
    <div id="signupModal" class="auth-modal-overlay">
      <div class="auth-modal-container">
        <button class="auth-modal-close" id="signupClose">
          <i class="fas fa-times"></i>
        </button>

        <!-- SIGNUP FORM -->
        <div class="auth-header">
          <div class="auth-logo">⚡ AEGIS</div>
          <h2>Create Account</h2>
          <p>Join our trading community and get 3-day free trial</p>
        </div>

        <form id="signupEmailForm" class="auth-form">
          <div class="form-group">
            <label>Full Name</label>
            <input 
              type="text" 
              id="signupName" 
              placeholder="John Doe"
              required
            >
            <span class="error-msg" id="signupNameError"></span>
          </div>

          <div class="form-group">
            <label>Email Address</label>
            <input 
              type="email" 
              id="signupEmail" 
              placeholder="your@email.com"
              required
            >
            <span class="error-msg" id="signupEmailError"></span>
          </div>

          <div class="form-group">
            <label>Password (min 8 characters)</label>
            <input 
              type="password" 
              id="signupPassword" 
              placeholder="••••••••"
              minlength="8"
              required
            >
            <span class="error-msg" id="signupPasswordError"></span>
          </div>

          <div class="form-group">
            <label>Confirm Password</label>
            <input 
              type="password" 
              id="signupPasswordConfirm" 
              placeholder="••••••••"
              required
            >
            <span class="error-msg" id="signupPasswordConfirmError"></span>
          </div>

          <div class="form-group checkbox">
            <input 
              type="checkbox" 
              id="termsAgree" 
              required
            >
            <label for="termsAgree">I agree to Terms of Service and Privacy Policy</label>
          </div>

          <button type="submit" class="auth-btn-primary">Create Account</button>
          <span class="auth-error" id="signupFormError"></span>
        </form>

        <div class="auth-divider">OR</div>

        <button type="button" class="auth-btn-social" id="googleSignupBtn">
          <i class="fab fa-google"></i> Sign Up with Google
        </button>

        <div class="auth-footer">
          <p>Already have an account? <a href="#" id="toSigninLink" class="auth-link">Sign In</a></p>
          <p style="font-size: 0.7rem; color: #6b7280;">
            <i class="fas fa-info-circle"></i> By signing up, you'll get instant access to 3-day free trial
          </p>
        </div>

        <div class="auth-security-badge">
          <i class="fas fa-lock"></i> Your data is encrypted and secure
        </div>
      </div>
    </div>

    <!-- LOADING OVERLAY -->
    <div id="signupLoadingOverlay" class="auth-loading-overlay hidden">
      <div class="spinner"></div>
      <p>Creating your account...</p>
    </div>
  `;
}

export function initSignUpUI() {
  const modalContainer = document.createElement('div');
  modalContainer.innerHTML = createSignUpModal();
  document.body.appendChild(modalContainer);

  attachSignUpEventListeners();
  listenForSignupEvent();
  console.log('✅ Sign Up UI initialized');
}

function attachSignUpEventListeners() {
  // Close button
  document.getElementById('signupClose')?.addEventListener('click', closeSignUpModal);

  // Form submission
  document.getElementById('signupEmailForm')?.addEventListener('submit', performEmailSignup);

  // Navigation
  document.getElementById('toSigninLink')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeSignUpModal();
    // Emit event to switch to signin
    window.dispatchEvent(new CustomEvent('openSignin'));
  });

  // Google auth
  document.getElementById('googleSignupBtn')?.addEventListener('click', performGoogleSignup);

  // Close on outside click
  document.getElementById('signupModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'signupModal') closeSignUpModal();
  });

  // Password match validation
  document.getElementById('signupPasswordConfirm')?.addEventListener('change', () => {
    const pwd = document.getElementById('signupPassword')?.value;
    const confirm = document.getElementById('signupPasswordConfirm')?.value;
    if (pwd && confirm && pwd !== confirm) {
      showError('signupPasswordConfirmError', 'Passwords do not match');
    } else {
      document.getElementById('signupPasswordConfirmError').textContent = '';
    }
  });
}

function listenForSignupEvent() {
  window.addEventListener('openSignup', () => {
    closeSignUpModal();
    openSignUpModal();
  });
}

async function performEmailSignup(e) {
  e.preventDefault();

  const name = document.getElementById('signupName')?.value?.trim();
  const email = document.getElementById('signupEmail')?.value?.trim();
  const password = document.getElementById('signupPassword')?.value;
  const confirmPassword = document.getElementById('signupPasswordConfirm')?.value;
  const termsAgree = document.getElementById('termsAgree')?.checked;

  // Validation
  if (!name || !email || !password || !confirmPassword) {
    showError('signupFormError', 'All fields are required');
    return;
  }

  if (password.length < 8) {
    showError('signupPasswordError', 'Password must be at least 8 characters');
    return;
  }

  if (password !== confirmPassword) {
    showError('signupPasswordConfirmError', 'Passwords do not match');
    return;
  }

  if (!termsAgree) {
    showError('signupFormError', 'You must agree to the Terms of Service');
    return;
  }

  showLoading();
  try {
    console.log('📝 Creating account for:', email);
    const result = await handleEmailSignup(email, password, name);

    if (result.success) {
      showSuccess('Account created! Welcome to AEGIS! Redirecting...');
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      setTimeout(() => {
        closeSignUpModal();
        window.location.href = '/web/src/pages/pricing.html?newUser=1';
      }, 1500);
    } else {
      showError('signupFormError', result.message);
    }
  } catch (error) {
    console.error('Signup error:', error);
    showError('signupFormError', error.message || 'Failed to create account');
  } finally {
    hideLoading();
  }
}

async function performGoogleSignup() {
  showLoading();
  try {
    const result = await handleGoogleAuth();
    if (result.success) {
      showSuccess('Account created with Google!');
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      setTimeout(() => {
        closeSignUpModal();
        window.location.href = '/web/src/pages/pricing.html?newUser=1';
      }, 1000);
    } else {
      showError('signupFormError', result.message);
    }
  } catch (error) {
    showError('signupFormError', error.message);
  } finally {
    hideLoading();
  }
}

export function openSignUpModal() {
  const modal = document.getElementById('signupModal');
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

export function closeSignUpModal() {
  const modal = document.getElementById('signupModal');
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
    // Reset form
    document.getElementById('signupEmailForm')?.reset();
    document.getElementById('signupFormError').textContent = '';
  }
}

function showLoading() {
  document.getElementById('signupLoadingOverlay')?.classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('signupLoadingOverlay')?.classList.add('hidden');
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
