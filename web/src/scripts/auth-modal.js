// ============================================================
// AEGIS Auth Modal – Mobile-Responsive Login/Signup UI
// ============================================================

import {
  handleGoogleAuth,
  handleEmailSignup,
  handleEmailLogin,
  sendPhoneOTP,
  verifyPhoneOTP,
  handleLogout,
  handlePasswordReset,
  isUserAuthenticated,
  getCurrentUserId
} from './auth.js';

// ============================================================
// STATE
// ============================================================
let authModal = null;
let currentAuthStep = 'landing'; // landing | email-login | email-signup | phone-otp | verify-otp | forgot-password
let otpSent = false;

// ============================================================
// CREATE AUTH MODAL HTML
// ============================================================
function createAuthModalHTML() {
  return `
    <div id="authModal" class="auth-modal-overlay">
      <div class="auth-modal-container">
        <!-- CLOSE BUTTON -->
        <button class="auth-modal-close" id="authModalClose">
          <i class="fas fa-times"></i>
        </button>

        <!-- LANDING PAGE -->
        <div class="auth-step" id="authStep-landing">
          <div class="auth-header">
            <div class="auth-logo">⚡ AEGIS</div>
            <h2>Access Sovereign Intelligence</h2>
            <p>Trading signals with real-time risk management</p>
          </div>

          <div class="auth-options">
            <button class="auth-btn-primary" id="googleAuthBtn">
              <i class="fab fa-google"></i> Continue with Google
            </button>

            <button class="auth-btn-secondary" id="emailAuthBtn">
              <i class="fas fa-envelope"></i> Continue with Email
            </button>

            <button class="auth-btn-secondary" id="phoneAuthBtn">
              <i class="fas fa-mobile-alt"></i> Continue with Phone
            </button>
          </div>

          <div class="auth-footer">
            <p>All new users get <strong>3-day free trial</strong> with basic features</p>
          </div>
        </div>

        <!-- EMAIL LOGIN PAGE -->
        <div class="auth-step hidden" id="authStep-email-login">
          <div class="auth-header">
            <button class="auth-back-btn" data-step="landing">
              <i class="fas fa-arrow-left"></i>
            </button>
            <h2>Login with Email</h2>
            <p>Enter your credentials</p>
          </div>

          <form id="emailLoginForm" class="auth-form">
            <div class="form-group">
              <label>Email Address</label>
              <input 
                type="email" 
                id="loginEmail" 
                placeholder="your@email.com"
                required
              >
              <span class="error-msg" id="loginEmailError"></span>
            </div>

            <div class="form-group">
              <label>Password</label>
              <input 
                type="password" 
                id="loginPassword" 
                placeholder="••••••••"
                required
              >
              <span class="error-msg" id="loginPasswordError"></span>
            </div>

            <button type="submit" class="auth-btn-primary">Login</button>

            <div class="auth-divider">OR</div>

            <button type="button" class="auth-btn-secondary" id="toSignupBtn">
              Don't have an account? Sign up
            </button>

            <button 
              type="button" 
              class="auth-link-btn" 
              id="toForgotPasswordBtn"
            >
              Forgot password?
            </button>
          </form>

          <span class="auth-error" id="emailLoginError"></span>
        </div>

        <!-- EMAIL SIGNUP PAGE -->
        <div class="auth-step hidden" id="authStep-email-signup">
          <div class="auth-header">
            <button class="auth-back-btn" data-step="landing">
              <i class="fas fa-arrow-left"></i>
            </button>
            <h2>Create Account</h2>
            <p>Join Aegis trading community</p>
          </div>

          <form id="emailSignupForm" class="auth-form">
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

            <button type="submit" class="auth-btn-primary">Create Account</button>

            <div class="auth-divider">OR</div>

            <button type="button" class="auth-btn-secondary" id="toLoginBtn">
              Already have an account? Login
            </button>
          </form>

          <span class="auth-error" id="emailSignupError"></span>
        </div>

        <!-- PHONE OTP PAGE -->
        <div class="auth-step hidden" id="authStep-phone-otp">
          <div class="auth-header">
            <button class="auth-back-btn" data-step="landing">
              <i class="fas fa-arrow-left"></i>
            </button>
            <h2>Create Account via Phone</h2>
            <p>Sign up with your mobile number</p>
          </div>

          <form id="phoneOtpForm" class="auth-form">
            <div class="form-group">
              <label>Full Name</label>
              <input 
                type="text" 
                id="phoneName" 
                placeholder="John Doe"
                required
              >
            </div>

            <div class="form-group">
              <label>Phone Number</label>
              <div class="phone-input-group">
                <span class="country-code">+91</span>
                <input 
                  type="tel" 
                  id="phoneNumber" 
                  placeholder="98765 43210"
                  pattern="[0-9]{10}"
                  required
                >
              </div>
              <span class="error-msg" id="phoneNumberError"></span>
              <small>We'll send you an OTP to verify</small>
            </div>

            <div id="recaptcha-container"></div>

            <button type="submit" class="auth-btn-primary" id="sendOtpBtn">
              Send OTP
            </button>

            <span class="auth-error" id="phoneOtpError"></span>
          </form>
        </div>

        <!-- VERIFY OTP PAGE -->
        <div class="auth-step hidden" id="authStep-verify-otp">
          <div class="auth-header">
            <button class="auth-back-btn" data-step="phone-otp">
              <i class="fas fa-arrow-left"></i>
            </button>
            <h2>Verify OTP</h2>
            <p id="otpMessage">Enter the code sent to your phone</p>
          </div>

          <form id="verifyOtpForm" class="auth-form">
            <div class="form-group">
              <label>Enter 6-digit OTP</label>
              <input 
                type="text" 
                id="otpCode" 
                placeholder="000000"
                maxlength="6"
                pattern="[0-9]{6}"
                required
              >
              <span class="error-msg" id="otpCodeError"></span>
            </div>

            <button type="submit" class="auth-btn-primary">Verify & Create Account</button>

            <button 
              type="button" 
              class="auth-link-btn" 
              id="resendOtpBtn"
            >
              Didn't receive? Resend OTP
            </button>

            <span class="auth-error" id="verifyOtpError"></span>
          </form>
        </div>

        <!-- FORGOT PASSWORD PAGE -->
        <div class="auth-step hidden" id="authStep-forgot-password">
          <div class="auth-header">
            <button class="auth-back-btn" data-step="email-login">
              <i class="fas fa-arrow-left"></i>
            </button>
            <h2>Reset Password</h2>
            <p>Enter your email to receive reset link</p>
          </div>

          <form id="forgotPasswordForm" class="auth-form">
            <div class="form-group">
              <label>Email Address</label>
              <input 
                type="email" 
                id="resetEmail" 
                placeholder="your@email.com"
                required
              >
            </div>

            <button type="submit" class="auth-btn-primary">Send Reset Link</button>

            <span class="auth-error" id="resetPasswordError"></span>
            <span class="auth-success" id="resetPasswordSuccess"></span>
          </form>
        </div>
      </div>
    </div>

    <!-- LOADING OVERLAY -->
    <div id="authLoadingOverlay" class="auth-loading-overlay hidden">
      <div class="spinner"></div>
      <p>Processing...</p>
    </div>
  `;
}

// ============================================================
// INITIALIZE AUTH MODAL
// ============================================================
export function initAuthModal() {
  // Inject modal HTML into DOM
  const modalContainer = document.createElement('div');
  modalContainer.innerHTML = createAuthModalHTML();
  document.body.appendChild(modalContainer);
  
  authModal = document.getElementById('authModal');
  
  // Attach event listeners
  attachAuthEventListeners();
  
  console.log('✅ Auth modal initialized');
}

// ============================================================
// ATTACH EVENT LISTENERS
// ============================================================
function attachAuthEventListeners() {
  // Close button
  document.getElementById('authModalClose').addEventListener('click', closeAuthModal);
  
  // Google auth
  document.getElementById('googleAuthBtn').addEventListener('click', performGoogleAuth);
  
  // Email/Phone buttons from landing
  document.getElementById('emailAuthBtn').addEventListener('click', () => switchAuthStep('email-login'));
  document.getElementById('phoneAuthBtn').addEventListener('click', () => switchAuthStep('phone-otp'));
  
  // Back buttons
  document.querySelectorAll('.auth-back-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const step = btn.dataset.step;
      switchAuthStep(step);
    });
  });
  
  // Email Login form
  document.getElementById('emailLoginForm').addEventListener('submit', performEmailLogin);
  document.getElementById('toSignupBtn').addEventListener('click', (e) => {
    e.preventDefault();
    switchAuthStep('email-signup');
  });
  document.getElementById('toForgotPasswordBtn').addEventListener('click', (e) => {
    e.preventDefault();
    switchAuthStep('forgot-password');
  });
  
  // Email Signup form
  document.getElementById('emailSignupForm').addEventListener('submit', performEmailSignup);
  document.getElementById('toLoginBtn').addEventListener('click', (e) => {
    e.preventDefault();
    switchAuthStep('email-login');
  });
  
  // Phone OTP form
  document.getElementById('phoneOtpForm').addEventListener('submit', performPhoneOTP);
  
  // Verify OTP form
  document.getElementById('verifyOtpForm').addEventListener('submit', performVerifyOTP);
  document.getElementById('resendOtpBtn').addEventListener('click', (e) => {
    e.preventDefault();
    performPhoneOTP(null);
  });
  
  // Forgot password form
  document.getElementById('forgotPasswordForm').addEventListener('submit', performPasswordReset);
  
  // Auto-format OTP input (numbers only)
  document.getElementById('otpCode')?.addEventListener('input', (e) => {
    e.target.value = e.target.value.replace(/[^0-9]/g, '');
  });
  
  // Close modal when clicking outside
  authModal?.addEventListener('click', (e) => {
    if (e.target === authModal) {
      closeAuthModal();
    }
  });
}

// ============================================================
// SWITCH AUTH STEP
// ============================================================
function switchAuthStep(step) {
  document.querySelectorAll('.auth-step').forEach(el => {
    el.classList.add('hidden');
  });
  
  const targetStep = document.getElementById(`authStep-${step}`);
  if (targetStep) {
    targetStep.classList.remove('hidden');
    currentAuthStep = step;
    
    // Scroll to top
    authModal?.querySelector('.auth-modal-container').scrollTop = 0;
  }
}

// ============================================================
// SHOW LOADING
// ============================================================
function showLoading() {
  document.getElementById('authLoadingOverlay').classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('authLoadingOverlay').classList.add('hidden');
}

// ============================================================
// PERFORM GOOGLE AUTH
// ============================================================
async function performGoogleAuth() {
  try {
    showLoading();
    const result = await handleGoogleAuth();
    
    if (result.success) {
      showSuccessMessage('Google login successful!');
      setTimeout(() => {
        closeAuthModal();
        window.location.reload();
      }, 1000);
    } else {
      showErrorMessage('googleAuthBtn', result.message);
    }
  } catch (error) {
    showErrorMessage('googleAuthBtn', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// PERFORM EMAIL LOGIN
// ============================================================
async function performEmailLogin(e) {
  e.preventDefault();
  
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;
  
  try {
    showLoading();
    const result = await handleEmailLogin(email, password);
    
    if (result.success) {
      showSuccessMessage('Login successful!');
      setTimeout(() => {
        closeAuthModal();
        window.location.reload();
      }, 1000);
    } else {
      showErrorMessage('emailLoginError', result.message);
    }
  } catch (error) {
    showErrorMessage('emailLoginError', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// PERFORM EMAIL SIGNUP
// ============================================================
async function performEmailSignup(e) {
  e.preventDefault();
  
  const name = document.getElementById('signupName').value;
  const email = document.getElementById('signupEmail').value;
  const password = document.getElementById('signupPassword').value;
  const confirmPassword = document.getElementById('signupPasswordConfirm').value;
  
  if (password !== confirmPassword) {
    showErrorMessage('emailSignupError', 'Passwords do not match');
    return;
  }
  
  try {
    showLoading();
    const result = await handleEmailSignup(email, password, name);
    
    if (result.success) {
      showSuccessMessage('Account created! Welcome to Aegis!');
      setTimeout(() => {
        closeAuthModal();
        window.location.reload();
      }, 1000);
    } else {
      showErrorMessage('emailSignupError', result.message);
    }
  } catch (error) {
    showErrorMessage('emailSignupError', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// PERFORM PHONE OTP
// ============================================================
async function performPhoneOTP(e) {
  if (e) e.preventDefault();
  
  const name = document.getElementById('phoneName').value;
  const phone = document.getElementById('phoneNumber').value;
  
  if (!name || !phone) {
    showErrorMessage('phoneOtpError', 'Please fill in all fields');
    return;
  }
  
  try {
    showLoading();
    const result = await sendPhoneOTP('+91' + phone, name);
    
    if (result.success) {
      otpSent = true;
      switchAuthStep('verify-otp');
      document.getElementById('otpMessage').textContent = `OTP sent to +91 ${phone}`;
    } else {
      showErrorMessage('phoneOtpError', result.message);
    }
  } catch (error) {
    showErrorMessage('phoneOtpError', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// PERFORM VERIFY OTP
// ============================================================
async function performVerifyOTP(e) {
  e.preventDefault();
  
  const otpCode = document.getElementById('otpCode').value;
  
  if (!otpCode || otpCode.length !== 6) {
    showErrorMessage('verifyOtpError', 'Please enter valid 6-digit OTP');
    return;
  }
  
  try {
    showLoading();
    const result = await verifyPhoneOTP(otpCode);
    
    if (result.success) {
      showSuccessMessage('Account created via phone!');
      setTimeout(() => {
        closeAuthModal();
        window.location.reload();
      }, 1000);
    } else {
      showErrorMessage('verifyOtpError', result.message);
    }
  } catch (error) {
    showErrorMessage('verifyOtpError', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// PERFORM PASSWORD RESET
// ============================================================
async function performPasswordReset(e) {
  e.preventDefault();
  
  const email = document.getElementById('resetEmail').value;
  
  try {
    showLoading();
    const result = await handlePasswordReset(email);
    
    if (result.success) {
      document.getElementById('resetPasswordSuccess').textContent = result.message;
      document.getElementById('resetPasswordError').textContent = '';
      document.getElementById('resetEmail').value = '';
    } else {
      showErrorMessage('resetPasswordError', result.message);
    }
  } catch (error) {
    showErrorMessage('resetPasswordError', error.message);
  } finally {
    hideLoading();
  }
}

// ============================================================
// HELPER: Show Error Message
// ============================================================
function showErrorMessage(elementId, message) {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = message;
    element.style.display = 'block';
  }
}

// ============================================================
// HELPER: Show Success Message
// ============================================================
function showSuccessMessage(message) {
  alert(message); // Replace with toast notification if preferred
}

// ============================================================
// OPEN AUTH MODAL
// ============================================================
export function openAuthModal(step = 'landing') {
  if (!authModal) {
    initAuthModal();
  }
  
  authModal.classList.add('active');
  document.body.style.overflow = 'hidden';
  switchAuthStep(step);
}

// ============================================================
// CLOSE AUTH MODAL
// ============================================================
export function closeAuthModal() {
  if (authModal) {
    authModal.classList.remove('active');
    document.body.style.overflow = 'auto';
  }
}

// ============================================================
// TOGGLE AUTH MODAL
// ============================================================
export function toggleAuthModal() {
  if (authModal?.classList.contains('active')) {
    closeAuthModal();
  } else {
    openAuthModal();
  }
}

// ============================================================
// HANDLE LOGOUT FROM UI
// ============================================================
export async function performLogoutUI() {
  try {
    showLoading();
    const result = await handleLogout();
    
    if (result.success) {
      showSuccessMessage('Logged out successfully!');
      setTimeout(() => {
        window.location.href = '/';
      }, 1000);
    } else {
      alert('Logout failed: ' + result.message);
    }
  } finally {
    hideLoading();
  }
}

// ============================================================
// UPDATE AUTH BUTTON STATE
// ============================================================
export function updateAuthButtonState() {
  const loginBtn = document.getElementById('loginBtn') || document.getElementById('portalBtn');
  
  if (!loginBtn) return;
  
  if (isUserAuthenticated()) {
    loginBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
    loginBtn.onclick = performLogoutUI;
  } else {
    loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
    loginBtn.onclick = () => openAuthModal();
  }
}
