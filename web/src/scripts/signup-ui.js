// ============================================================
// SIGNUP UI – Two-step OTP registration flow
// Step 1: name / email / password form  → sends OTP
// Step 2: 6-digit OTP input             → verifies + creates account
// ============================================================

import {
  handleGoogleAuth,
  handleEmailSignup,
  sendOTPForSignup,
  verifyOTPForSignup,
  completeGoogleSignupWithPhone,
  cancelIncompleteGoogleSignup,
} from './auth.js';

// Pending form data stored between step 1 and step 2
let _pending = { name: '', email: '', password: '', mobile: '' };
let _resendTimer = null;
let _flow = 'email'; // 'email' | 'google'
let _pendingGoogleUser = null;
let _otpEmail = ''; // email used to send OTP (for verification + resend)
let _otpVia = 'email'; // 'email' | 'sms' — where OTP was delivered

function normalizeMobile(raw, countryCodeSelectId = 'signupCountryCode') {
  if (!raw) return null;
  const stripped = raw.replace(/[\s\-\.\(\)]/g, '');
  // Already full E.164 (user typed it manually with +)
  if (/^\+\d{7,15}$/.test(stripped)) return stripped;
  const digits = stripped.replace(/\D/g, '');
  if (!digits) return null;
  const cc = document.getElementById(countryCodeSelectId)?.value || '+91';
  const ccDigits = cc.replace('+', '');
  // Subscriber number only — prepend selected country code
  if (digits.length >= 7 && digits.length <= 12) {
    // Avoid double-prepending country code (e.g. user typed 919876543210)
    if (digits.startsWith(ccDigits) && digits.length > ccDigits.length + 5) {
      return '+' + digits;
    }
    return cc + digits;
  }
  return null;
}

async function checkPhoneUnique(phone) {
  try {
    const res = await fetch('/auth/check-phone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone })
    });
    const data = await res.json();
    return data.available !== false;
  } catch {
    return true; // fail open — backend will enforce on provision
  }
}

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
            <div class="auth-logo">AEGIS · v1.0</div>
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
              <label>Mobile Number</label>
              <div style="display:flex;align-items:stretch;border:1px solid rgba(184,150,106,0.25);border-radius:8px;overflow:hidden;background:rgba(255,255,255,0.04);">
                <select id="signupCountryCode"
                  style="border:none;border-right:1px solid rgba(184,150,106,0.25);background:rgba(184,150,106,0.08);color:#B8966A;padding:0 10px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;cursor:pointer;outline:none;min-width:90px;">
                  <option value="+91">🇮🇳 +91</option>
                  <option value="+1">🇺🇸 +1</option>
                  <option value="+44">🇬🇧 +44</option>
                  <option value="+61">🇦🇺 +61</option>
                  <option value="+971">🇦🇪 +971</option>
                  <option value="+65">🇸🇬 +65</option>
                  <option value="+60">🇲🇾 +60</option>
                </select>
                <input type="tel" id="signupMobile" placeholder="9876543210" required inputmode="numeric"
                  style="border:none;background:transparent;flex:1;padding:12px 14px;color:inherit;font-size:inherit;outline:none;">
              </div>
              <span style="font-size:0.72rem;color:#6b7280;margin-top:4px;display:block;">Select country code, then enter your number — used for account security</span>
              <span class="error-msg" id="signupMobileError"></span>
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

        <!-- ── STEP 1b: Phone for Google users ── -->
        <div id="signupStep1b" style="display:none;">
          <div class="auth-header">
            <div class="auth-logo">AEGIS · v1.0</div>
            <h2>One Last Step</h2>
            <p>Add your mobile number to secure your account</p>
          </div>
          <form id="googlePhoneForm" class="auth-form">
            <div class="form-group">
              <label>Mobile Number</label>
              <div style="display:flex;align-items:stretch;border:1px solid rgba(184,150,106,0.25);border-radius:8px;overflow:hidden;background:rgba(255,255,255,0.04);">
                <select id="googleCountryCode"
                  style="border:none;border-right:1px solid rgba(184,150,106,0.25);background:rgba(184,150,106,0.08);color:#B8966A;padding:0 10px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;cursor:pointer;outline:none;min-width:90px;">
                  <option value="+91">🇮🇳 +91</option>
                  <option value="+1">🇺🇸 +1</option>
                  <option value="+44">🇬🇧 +44</option>
                  <option value="+61">🇦🇺 +61</option>
                  <option value="+971">🇦🇪 +971</option>
                  <option value="+65">🇸🇬 +65</option>
                  <option value="+60">🇲🇾 +60</option>
                </select>
                <input type="tel" id="googleMobile" placeholder="9876543210" required inputmode="numeric"
                  style="border:none;background:transparent;flex:1;padding:12px 14px;color:inherit;font-size:inherit;outline:none;">
              </div>
              <span style="font-size:0.72rem;color:#6b7280;margin-top:4px;display:block;">Select country code, then enter your number</span>
              <span class="error-msg" id="googleMobileError"></span>
            </div>
            <button type="submit" class="auth-btn-primary" id="googlePhoneSubmitBtn">
              Complete Sign Up
            </button>
            <span class="auth-error" id="googlePhoneFormError"></span>
          </form>
        </div>

        <!-- ── STEP 2: OTP verification ── -->
        <div id="signupStep2" style="display:none;">
          <div class="auth-header">
            <div class="auth-logo">AEGIS · v1.0</div>
            <h2>Enter Verification Code</h2>
            <p id="otpDeliveryMsg">Enter the 6-digit code sent to<br>
               <strong id="otpEmailDisplay" style="color: var(--ae-gold, #B8966A);font-family:'JetBrains Mono',monospace;"></strong>
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
                       font-family:'JetBrains Mono',monospace;
                       background:rgba(255,255,255,0.04);border:1px solid rgba(184,150,106,0.2);border-radius:8px;
                       color:var(--ae-text-1,#EAE6DF);outline:none;transition:border-color .2s,box-shadow .2s;"
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
              ← Change details
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

  attachGooglePhoneListeners();
}

// ============================================================
// Step 1 submit — validate form then send OTP
// ============================================================
async function handleStep1Submit(e) {
  e.preventDefault();

  const name     = document.getElementById('signupName')?.value?.trim();
  const email    = document.getElementById('signupEmail')?.value?.trim();
  const mobileRaw = document.getElementById('signupMobile')?.value?.trim();
  const password = document.getElementById('signupPassword')?.value;
  const confirm  = document.getElementById('signupPasswordConfirm')?.value;
  const terms    = document.getElementById('termsAgree')?.checked;

  clearError('signupFormError');
  clearError('signupMobileError');

  if (!name || !email || !mobileRaw || !password || !confirm) {
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

  const mobile = normalizeMobile(mobileRaw);
  if (!mobile) {
    return showError('signupMobileError', 'Enter a valid mobile number (e.g. 9876543210 or +91 9876543210)');
  }

  setLoading(true, 'Checking mobile number…');
  const phoneAvailable = await checkPhoneUnique(mobile);
  if (!phoneAvailable) {
    setLoading(false);
    return showError('signupMobileError', 'This mobile number is already registered to another account');
  }

  setLoading(true, 'Sending verification code…');
  const result = await sendOTPForSignup(email, mobile);
  setLoading(false);

  if (!result.success) {
    return showError('signupFormError', result.message || 'Failed to send code. Please try again.');
  }

  // Store for step 2
  _pending = { name, email, password, mobile };
  _otpEmail = email;
  _otpVia = result.via || 'email';

  showStep2(_otpVia, mobile, email);
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
    const result = await sendOTPForSignup(_otpEmail, _pending.mobile);
    setLoading(false);
    if (result.success) {
      _otpVia = result.via || _otpVia;
      clearOtpBoxes();
      startResendCountdown(60);
    } else {
      showError('otpError', result.message || 'Failed to resend. Please try again.');
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
  const verifyResult = await verifyOTPForSignup(_otpEmail, otp, _pending.mobile);
  if (!verifyResult.success) {
    setLoading(false);
    return showError('otpError', verifyResult.message);
  }

  if (_flow === 'google') {
    setLoading(true, 'Creating your account…');
    const signupResult = await completeGoogleSignupWithPhone(_pendingGoogleUser, _pending.mobile);
    setLoading(false);
    if (!signupResult.success) {
      return showError('otpError', signupResult.message);
    }
    clearResendTimer();
    _pending = { name: '', email: '', password: '', mobile: '' };
    _pendingGoogleUser = null;
    _otpEmail = '';
    _flow = 'email';
    finishGoogleSignup();
    return;
  }

  // Email/password flow
  setLoading(true, 'Creating your account…');
  const signupResult = await handleEmailSignup(
    _pending.email, _pending.password, _pending.name, verifyResult.signup_token, _pending.mobile
  );
  setLoading(false);

  if (!signupResult.success) {
    return showError('otpError', signupResult.message);
  }

  clearResendTimer();
  _pending = { name: '', email: '', password: '', mobile: '' };
  _otpEmail = '';
  _flow = 'email';
  window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
  closeSignUpModal();
  window.location.href = '/pricing?newUser=1';
}

// ============================================================
// Google signup — new users must verify phone via OTP
// ============================================================
async function performGoogleSignup() {
  setLoading(true, 'Connecting to Google…');
  try {
    const result = await handleGoogleAuth();
    setLoading(false);
    if (!result.success) {
      return showError('signupFormError', result.message);
    }
    if (!result.isNewUser) {
      // Returning Google user — already authenticated, go to dashboard
      window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
      closeSignUpModal();
      window.location.href = '/dashboard';
      return;
    }
    // New Google user — require phone + OTP before creating account
    _pendingGoogleUser = result.user;
    _flow = 'google';
    showGooglePhoneStep();
  } catch (error) {
    setLoading(false);
    showError('signupFormError', error.message);
  }
}

function showGooglePhoneStep() {
  document.getElementById('signupStep1').style.display = 'none';
  document.getElementById('signupStep1b').style.display = '';
  document.getElementById('googleMobile')?.focus();
}

function finishGoogleSignup() {
  window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
  closeSignUpModal();
  window.location.href = '/pricing?newUser=1';
}

function attachGooglePhoneListeners() {
  document.getElementById('googlePhoneForm')?.addEventListener('submit', handleGooglePhoneSubmit);
}

async function handleGooglePhoneSubmit(e) {
  e.preventDefault();
  clearError('googleMobileError');
  clearError('googlePhoneFormError');

  const raw = document.getElementById('googleMobile')?.value?.trim();
  const mobile = normalizeMobile(raw || '', 'googleCountryCode');
  if (!mobile) {
    return showError('googleMobileError', 'Enter a valid mobile number (e.g. 9876543210 or +91 9876543210)');
  }

  setLoading(true, 'Checking mobile number…');
  const available = await checkPhoneUnique(mobile);
  if (!available) {
    setLoading(false);
    // Sign out the Firebase session for this new Google user. If we don't,
    // onAuthStateChanged fires as "authenticated" and breaks the page UI state
    // (e.g. the Sign Up button disappears and buttons stop responding).
    cancelIncompleteGoogleSignup().catch(() => {});
    _pendingGoogleUser = null;
    _flow = 'email';
    return showError('googleMobileError', 'This mobile number is already registered to another account. Please sign in to the existing account instead.');
  }

  // Send OTP via backend (email fallback if SMS unavailable)
  const googleEmail = _pendingGoogleUser?.email || '';
  const otpResult = await sendOTPForSignup(googleEmail, mobile);
  setLoading(false);
  if (!otpResult.success) {
    return showError('googlePhoneFormError', otpResult.message || 'Failed to send code. Please try again.');
  }

  _pending.mobile = mobile;
  _otpEmail = googleEmail;
  _otpVia = otpResult.via || 'email';
  document.getElementById('signupStep1b').style.display = 'none';
  showStep2(_otpVia, mobile, googleEmail);
  startResendCountdown(60);
}

// ============================================================
// Step transitions
// ============================================================
function showStep1() {
  document.getElementById('signupStep1').style.display = '';
  document.getElementById('signupStep1b').style.display = 'none';
  document.getElementById('signupStep2').style.display = 'none';
}

function showStep2(via, phone, email) {
  document.getElementById('signupStep1').style.display = 'none';
  document.getElementById('signupStep1b').style.display = 'none';
  document.getElementById('signupStep2').style.display = '';
  const dest = via === 'sms' ? phone : email;
  const channelLabel = via === 'sms' ? 'SMS sent to' : 'Email sent to';
  const msgEl = document.getElementById('otpDeliveryMsg');
  if (msgEl) {
    msgEl.innerHTML = `${channelLabel}<br><strong id="otpEmailDisplay" style="color:var(--ae-gold,#B8966A);font-family:'JetBrains Mono',monospace;">${dest}</strong>`;
  }
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
    // Always clear the loading overlay — prevents it from blocking the page
    // if the modal is closed while a request is in-flight.
    setLoading(false);

    // If a Google signup was started but never completed (user got blocked on
    // phone step or manually closed), sign out the Firebase session so it doesn't
    // leak into the page's auth state and lock up navigation/buttons.
    if (_flow === 'google' && _pendingGoogleUser) {
      cancelIncompleteGoogleSignup().catch(() => {});
    }

    modal.classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('signupEmailForm')?.reset();
    document.getElementById('googlePhoneForm')?.reset();
    clearError('signupFormError');
    clearError('googleMobileError');
    clearError('googlePhoneFormError');
    clearResendTimer();
    _flow = 'email';
    _pendingGoogleUser = null;
    _otpEmail = '';
    _otpVia = 'email';
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
