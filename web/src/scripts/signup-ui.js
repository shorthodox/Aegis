// ============================================================
// SIGNUP UI – Two-step OTP registration flow
// Step 1: name / email / password form  → sends OTP
// Step 2: 6-digit OTP input             → verifies + creates account
// ============================================================

import {
  handleGoogleAuth,
  handleEmailSignup,
  sendOTPForSignup,
  verifyOTPForSignup
} from './auth.js';

// Pending form data stored between step 1 and step 2
let _pending = { name: '', email: '', password: '' };
let _resendTimer = null;

// ============================================================
// Modal HTML
// ============================================================
export function createSignUpModal() {
  return `
    <div id="signupModal" class="auth-modal-overlay">
      <div class="auth-modal-container">
        <button class="auth-modal-close" id="signupClose">
          <i class="fas fa-times"></i>
        </button>

        <!-- ── STEP 1: registration form ── -->
        <div id="signupStep1">
          <div class="auth-header">
            <div class="auth-logo">⚡ AEGIS</div>
            <h2>Create Account</h2>
            <p>Join our trading community and get 3-day free trial</p>
          </div>

          <form id="signupEmailForm" class="auth-form">
            <div class="form-group">
              <label>Full Name</label>
              <input type="text" id="signupName" placeholder="John Doe" required>
              <span class="error-msg" id="signupNameError"></span>
            </div>

            <div class="form-group">
              <label>Email Address</label>
              <input type="email" id="signupEmail" placeholder="your@email.com" required>
              <span class="error-msg" id="signupEmailError"></span>
            </div>

            <div class="form-group">
              <label>Password (min 8 characters)</label>
              <input type="password" id="signupPassword" placeholder="••••••••" minlength="8" required>
              <span class="error-msg" id="signupPasswordError"></span>
            </div>

            <div class="form-group">
              <label>Confirm Password</label>
              <input type="password" id="signupPasswordConfirm" placeholder="••••••••" required>
              <span class="error-msg" id="signupPasswordConfirmError"></span>
            </div>

            <div class="form-group checkbox">
              <input type="checkbox" id="termsAgree" required>
              <label for="termsAgree">I agree to Terms of Service and Privacy Policy</label>
            </div>

            <button type="submit" class="auth-btn-primary" id="sendOtpBtn">
              Send Verification Code
            </button>
            <span class="auth-error" id="signupFormError"></span>
          </form>

          <div class="auth-divider">OR</div>
          <button type="button" class="auth-btn-social" id="googleSignupBtn">
            <i class="fab fa-google"></i> Sign Up with Google
          </button>

          <div class="auth-footer">
            <p>Already have an account? <a href="#" id="toSigninLink" class="auth-link">Sign In</a></p>
          </div>

          <div class="auth-security-badge">
            <i class="fas fa-lock"></i> Your data is encrypted and secure
          </div>
        </div>

        <!-- ── STEP 2: OTP verification ── -->
        <div id="signupStep2" style="display:none;">
          <div class="auth-header">
            <div class="auth-logo">⚡ AEGIS</div>
            <h2>Verify Your Email</h2>
            <p>Enter the 6-digit code sent to<br>
               <strong id="otpEmailDisplay" style="color:#00f2ff;"></strong>
            </p>
          </div>

          <div class="otp-box-row" id="otpBoxRow"
               style="display:flex;gap:8px;justify-content:center;margin:1.5rem 0;">
            ${[0,1,2,3,4,5].map(i => `
              <input
                class="otp-box"
                id="otpBox${i}"
                type="text"
                inputmode="numeric"
                maxlength="1"
                pattern="[0-9]"
                autocomplete="one-time-code"
                style="width:44px;height:52px;text-align:center;font-size:1.4rem;font-weight:700;
                       background:#0a0a0c;border:1px solid rgba(255,255,255,0.15);border-radius:10px;
                       color:#fff;outline:none;transition:border-color .2s;"
              >
            `).join('')}
          </div>

          <span class="auth-error" id="otpError"></span>

          <button id="verifyOtpBtn" class="auth-btn-primary" style="margin-top:.5rem;">
            Verify &amp; Create Account
          </button>

          <div style="text-align:center;margin-top:1.25rem;font-size:0.82rem;color:#6b7280;">
            <span id="resendCountdown"></span>
            <a href="#" id="resendOtpLink" class="auth-link" style="display:none;">
              Resend code
            </a>
          </div>

          <div style="text-align:center;margin-top:1rem;">
            <a href="#" id="backToStep1Link" class="auth-link" style="font-size:0.82rem;">
              ← Change email
            </a>
          </div>
        </div>
      </div>
    </div>

    <!-- LOADING OVERLAY -->
    <div id="signupLoadingOverlay" class="auth-loading-overlay hidden">
      <div class="spinner"></div>
      <p id="signupLoadingMsg">Creating your account...</p>
    </div>
  `;
}

// ============================================================
// Init
// ============================================================
export function initSignUpUI() {
  const wrap = document.createElement('div');
  wrap.innerHTML = createSignUpModal();
  document.body.appendChild(wrap);
  attachStep1Listeners();
  attachStep2Listeners();
  listenForSignupEvent();
}

// ============================================================
// Step 1 listeners
// ============================================================
function attachStep1Listeners() {
  document.getElementById('signupClose')?.addEventListener('click', closeSignUpModal);

  document.getElementById('signupEmailForm')?.addEventListener('submit', handleStep1Submit);

  document.getElementById('toSigninLink')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeSignUpModal();
    window.dispatchEvent(new CustomEvent('openSignin'));
  });

  document.getElementById('googleSignupBtn')?.addEventListener('click', performGoogleSignup);

  document.getElementById('signupModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'signupModal') closeSignUpModal();
  });

  document.getElementById('signupPasswordConfirm')?.addEventListener('input', () => {
    const pwd = document.getElementById('signupPassword')?.value;
    const conf = document.getElementById('signupPasswordConfirm')?.value;
    const el = document.getElementById('signupPasswordConfirmError');
    if (el) el.textContent = (pwd && conf && pwd !== conf) ? 'Passwords do not match' : '';
  });
}

// ============================================================
// Step 1 submit — validate form then send OTP
// ============================================================
async function handleStep1Submit(e) {
  e.preventDefault();

  const name     = document.getElementById('signupName')?.value?.trim();
  const email    = document.getElementById('signupEmail')?.value?.trim();
  const password = document.getElementById('signupPassword')?.value;
  const confirm  = document.getElementById('signupPasswordConfirm')?.value;
  const terms    = document.getElementById('termsAgree')?.checked;

  clearError('signupFormError');

  if (!name || !email || !password || !confirm) {
    return showError('signupFormError', 'All fields are required');
  }
  if (password.length < 8) {
    return showError('signupPasswordError', 'Password must be at least 8 characters');
  }
  if (password !== confirm) {
    return showError('signupPasswordConfirmError', 'Passwords do not match');
  }
  if (!terms) {
    return showError('signupFormError', 'You must agree to the Terms of Service');
  }

  setLoading(true, 'Sending verification code…');
  const result = await sendOTPForSignup(email);
  setLoading(false);

  if (!result.success) {
    return showError('signupFormError', result.message);
  }

  // Store for step 2
  _pending = { name, email, password };

  showStep2(email);
  startResendCountdown(60);
}

// ============================================================
// Step 2 listeners
// ============================================================
function attachStep2Listeners() {
  // Auto-advance between digit boxes
  document.addEventListener('input', (e) => {
    if (!e.target.classList.contains('otp-box')) return;
    const idx = parseInt(e.target.id.replace('otpBox', ''));
    const val = e.target.value.replace(/\D/g, '');
    e.target.value = val.slice(-1);
    if (val && idx < 5) document.getElementById(`otpBox${idx + 1}`)?.focus();
  });

  document.addEventListener('keydown', (e) => {
    if (!e.target.classList.contains('otp-box')) return;
    const idx = parseInt(e.target.id.replace('otpBox', ''));
    if (e.key === 'Backspace' && !e.target.value && idx > 0) {
      document.getElementById(`otpBox${idx - 1}`)?.focus();
    }
  });

  // Paste handler — spread digits across boxes
  document.getElementById('otpBoxRow')?.addEventListener('paste', (e) => {
    e.preventDefault();
    const digits = (e.clipboardData.getData('text') || '').replace(/\D/g, '').slice(0, 6);
    digits.split('').forEach((d, i) => {
      const box = document.getElementById(`otpBox${i}`);
      if (box) box.value = d;
    });
    document.getElementById(`otpBox${Math.min(digits.length, 5)}`)?.focus();
  });

  document.getElementById('verifyOtpBtn')?.addEventListener('click', handleStep2Verify);

  document.getElementById('resendOtpLink')?.addEventListener('click', async (e) => {
    e.preventDefault();
    clearError('otpError');
    setLoading(true, 'Resending code…');
    const result = await sendOTPForSignup(_pending.email);
    setLoading(false);
    if (result.success) {
      clearOtpBoxes();
      startResendCountdown(60);
    } else {
      showError('otpError', result.message);
    }
  });

  document.getElementById('backToStep1Link')?.addEventListener('click', (e) => {
    e.preventDefault();
    clearResendTimer();
    showStep1();
  });
}

// ============================================================
// Step 2 verify OTP then create account
// ============================================================
async function handleStep2Verify() {
  clearError('otpError');
  const otp = [0,1,2,3,4,5].map(i => document.getElementById(`otpBox${i}`)?.value || '').join('');

  if (otp.length !== 6 || !/^\d{6}$/.test(otp)) {
    return showError('otpError', 'Please enter all 6 digits');
  }

  setLoading(true, 'Verifying code…');
  const verifyResult = await verifyOTPForSignup(_pending.email, otp);

  if (!verifyResult.success) {
    setLoading(false);
    return showError('otpError', verifyResult.message);
  }

  // OTP passed — create Firebase account + Firestore doc
  setLoading(true, 'Creating your account…');
  const signupResult = await handleEmailSignup(_pending.email, _pending.password, _pending.name);
  setLoading(false);

  if (!signupResult.success) {
    return showError('otpError', signupResult.message);
  }

  clearResendTimer();
  _pending = { name: '', email: '', password: '' };
  window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
  closeSignUpModal();
  window.location.href = '/web/src/pages/pricing.html?newUser=1';
}

// ============================================================
// Google signup (unchanged path — Google verifies email itself)
// ============================================================
async function performGoogleSignup() {
  setLoading(true, 'Connecting to Google…');
  try {
    const result = await handleGoogleAuth();
    if (result.success) {
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      closeSignUpModal();
      window.location.href = '/web/src/pages/pricing.html?newUser=1';
    } else {
      showError('signupFormError', result.message);
    }
  } catch (error) {
    showError('signupFormError', error.message);
  } finally {
    setLoading(false);
  }
}

// ============================================================
// Step transitions
// ============================================================
function showStep1() {
  document.getElementById('signupStep1').style.display = '';
  document.getElementById('signupStep2').style.display = 'none';
}

function showStep2(email) {
  document.getElementById('signupStep1').style.display = 'none';
  document.getElementById('signupStep2').style.display = '';
  document.getElementById('otpEmailDisplay').textContent = email;
  clearOtpBoxes();
  clearError('otpError');
  document.getElementById('otpBox0')?.focus();
}

function clearOtpBoxes() {
  [0,1,2,3,4,5].forEach(i => {
    const box = document.getElementById(`otpBox${i}`);
    if (box) box.value = '';
  });
}

// ============================================================
// Resend countdown timer
// ============================================================
function startResendCountdown(seconds) {
  const countdown = document.getElementById('resendCountdown');
  const link = document.getElementById('resendOtpLink');
  if (link) link.style.display = 'none';
  clearResendTimer();

  let remaining = seconds;
  const tick = () => {
    if (countdown) countdown.textContent = `Resend code in ${remaining}s`;
    if (remaining <= 0) {
      if (countdown) countdown.textContent = '';
      if (link) link.style.display = 'inline';
      return;
    }
    remaining--;
    _resendTimer = setTimeout(tick, 1000);
  };
  tick();
}

function clearResendTimer() {
  if (_resendTimer) { clearTimeout(_resendTimer); _resendTimer = null; }
}

// ============================================================
// Modal open / close
// ============================================================
export function openSignUpModal() {
  const modal = document.getElementById('signupModal');
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    showStep1();
  }
}

export function closeSignUpModal() {
  const modal = document.getElementById('signupModal');
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('signupEmailForm')?.reset();
    clearError('signupFormError');
    clearResendTimer();
    showStep1();
  }
}

function listenForSignupEvent() {
  window.addEventListener('openSignup', openSignUpModal);
}

// ============================================================
// UI helpers
// ============================================================
function setLoading(on, msg = 'Creating your account…') {
  const overlay = document.getElementById('signupLoadingOverlay');
  const msgEl = document.getElementById('signupLoadingMsg');
  if (msgEl) msgEl.textContent = msg;
  overlay?.classList.toggle('hidden', !on);
}

function showError(id, message) {
  const el = document.getElementById(id);
  if (el) { el.textContent = message; el.style.display = 'block'; }
}

function clearError(id) {
  const el = document.getElementById(id);
  if (el) { el.textContent = ''; el.style.display = ''; }
}
