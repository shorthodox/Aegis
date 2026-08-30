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

// Full world dial-code list — India first, rest sorted by dial code
const _COUNTRY_CODES = [
  { code: '+91',   flag: '🇮🇳', name: 'India' },
  { code: '+1',    flag: '🇺🇸', name: 'United States / Canada' },
  { code: '+7',    flag: '🇷🇺', name: 'Russia / Kazakhstan' },
  { code: '+20',   flag: '🇪🇬', name: 'Egypt' },
  { code: '+27',   flag: '🇿🇦', name: 'South Africa' },
  { code: '+30',   flag: '🇬🇷', name: 'Greece' },
  { code: '+31',   flag: '🇳🇱', name: 'Netherlands' },
  { code: '+32',   flag: '🇧🇪', name: 'Belgium' },
  { code: '+33',   flag: '🇫🇷', name: 'France' },
  { code: '+34',   flag: '🇪🇸', name: 'Spain' },
  { code: '+36',   flag: '🇭🇺', name: 'Hungary' },
  { code: '+39',   flag: '🇮🇹', name: 'Italy' },
  { code: '+40',   flag: '🇷🇴', name: 'Romania' },
  { code: '+41',   flag: '🇨🇭', name: 'Switzerland' },
  { code: '+43',   flag: '🇦🇹', name: 'Austria' },
  { code: '+44',   flag: '🇬🇧', name: 'United Kingdom' },
  { code: '+45',   flag: '🇩🇰', name: 'Denmark' },
  { code: '+46',   flag: '🇸🇪', name: 'Sweden' },
  { code: '+47',   flag: '🇳🇴', name: 'Norway' },
  { code: '+48',   flag: '🇵🇱', name: 'Poland' },
  { code: '+49',   flag: '🇩🇪', name: 'Germany' },
  { code: '+51',   flag: '🇵🇪', name: 'Peru' },
  { code: '+52',   flag: '🇲🇽', name: 'Mexico' },
  { code: '+53',   flag: '🇨🇺', name: 'Cuba' },
  { code: '+54',   flag: '🇦🇷', name: 'Argentina' },
  { code: '+55',   flag: '🇧🇷', name: 'Brazil' },
  { code: '+56',   flag: '🇨🇱', name: 'Chile' },
  { code: '+57',   flag: '🇨🇴', name: 'Colombia' },
  { code: '+58',   flag: '🇻🇪', name: 'Venezuela' },
  { code: '+60',   flag: '🇲🇾', name: 'Malaysia' },
  { code: '+61',   flag: '🇦🇺', name: 'Australia' },
  { code: '+62',   flag: '🇮🇩', name: 'Indonesia' },
  { code: '+63',   flag: '🇵🇭', name: 'Philippines' },
  { code: '+64',   flag: '🇳🇿', name: 'New Zealand' },
  { code: '+65',   flag: '🇸🇬', name: 'Singapore' },
  { code: '+66',   flag: '🇹🇭', name: 'Thailand' },
  { code: '+81',   flag: '🇯🇵', name: 'Japan' },
  { code: '+82',   flag: '🇰🇷', name: 'South Korea' },
  { code: '+84',   flag: '🇻🇳', name: 'Vietnam' },
  { code: '+86',   flag: '🇨🇳', name: 'China' },
  { code: '+90',   flag: '🇹🇷', name: 'Turkey' },
  { code: '+92',   flag: '🇵🇰', name: 'Pakistan' },
  { code: '+93',   flag: '🇦🇫', name: 'Afghanistan' },
  { code: '+94',   flag: '🇱🇰', name: 'Sri Lanka' },
  { code: '+95',   flag: '🇲🇲', name: 'Myanmar' },
  { code: '+98',   flag: '🇮🇷', name: 'Iran' },
  { code: '+212',  flag: '🇲🇦', name: 'Morocco' },
  { code: '+213',  flag: '🇩🇿', name: 'Algeria' },
  { code: '+216',  flag: '🇹🇳', name: 'Tunisia' },
  { code: '+218',  flag: '🇱🇾', name: 'Libya' },
  { code: '+220',  flag: '🇬🇲', name: 'Gambia' },
  { code: '+221',  flag: '🇸🇳', name: 'Senegal' },
  { code: '+222',  flag: '🇲🇷', name: 'Mauritania' },
  { code: '+223',  flag: '🇲🇱', name: 'Mali' },
  { code: '+224',  flag: '🇬🇳', name: 'Guinea' },
  { code: '+225',  flag: '🇨🇮', name: "Côte d'Ivoire" },
  { code: '+226',  flag: '🇧🇫', name: 'Burkina Faso' },
  { code: '+227',  flag: '🇳🇪', name: 'Niger' },
  { code: '+228',  flag: '🇹🇬', name: 'Togo' },
  { code: '+229',  flag: '🇧🇯', name: 'Benin' },
  { code: '+230',  flag: '🇲🇺', name: 'Mauritius' },
  { code: '+231',  flag: '🇱🇷', name: 'Liberia' },
  { code: '+232',  flag: '🇸🇱', name: 'Sierra Leone' },
  { code: '+233',  flag: '🇬🇭', name: 'Ghana' },
  { code: '+234',  flag: '🇳🇬', name: 'Nigeria' },
  { code: '+235',  flag: '🇹🇩', name: 'Chad' },
  { code: '+236',  flag: '🇨🇫', name: 'Central African Republic' },
  { code: '+237',  flag: '🇨🇲', name: 'Cameroon' },
  { code: '+238',  flag: '🇨🇻', name: 'Cape Verde' },
  { code: '+239',  flag: '🇸🇹', name: 'São Tomé & Príncipe' },
  { code: '+240',  flag: '🇬🇶', name: 'Equatorial Guinea' },
  { code: '+241',  flag: '🇬🇦', name: 'Gabon' },
  { code: '+242',  flag: '🇨🇬', name: 'Republic of Congo' },
  { code: '+243',  flag: '🇨🇩', name: 'DR Congo' },
  { code: '+244',  flag: '🇦🇴', name: 'Angola' },
  { code: '+245',  flag: '🇬🇼', name: 'Guinea-Bissau' },
  { code: '+248',  flag: '🇸🇨', name: 'Seychelles' },
  { code: '+249',  flag: '🇸🇩', name: 'Sudan' },
  { code: '+250',  flag: '🇷🇼', name: 'Rwanda' },
  { code: '+251',  flag: '🇪🇹', name: 'Ethiopia' },
  { code: '+252',  flag: '🇸🇴', name: 'Somalia' },
  { code: '+253',  flag: '🇩🇯', name: 'Djibouti' },
  { code: '+254',  flag: '🇰🇪', name: 'Kenya' },
  { code: '+255',  flag: '🇹🇿', name: 'Tanzania' },
  { code: '+256',  flag: '🇺🇬', name: 'Uganda' },
  { code: '+257',  flag: '🇧🇮', name: 'Burundi' },
  { code: '+258',  flag: '🇲🇿', name: 'Mozambique' },
  { code: '+260',  flag: '🇿🇲', name: 'Zambia' },
  { code: '+261',  flag: '🇲🇬', name: 'Madagascar' },
  { code: '+262',  flag: '🇷🇪', name: 'Réunion / Mayotte' },
  { code: '+263',  flag: '🇿🇼', name: 'Zimbabwe' },
  { code: '+264',  flag: '🇳🇦', name: 'Namibia' },
  { code: '+265',  flag: '🇲🇼', name: 'Malawi' },
  { code: '+266',  flag: '🇱🇸', name: 'Lesotho' },
  { code: '+267',  flag: '🇧🇼', name: 'Botswana' },
  { code: '+268',  flag: '🇸🇿', name: 'Eswatini' },
  { code: '+269',  flag: '🇰🇲', name: 'Comoros' },
  { code: '+290',  flag: '🇸🇭', name: 'Saint Helena' },
  { code: '+291',  flag: '🇪🇷', name: 'Eritrea' },
  { code: '+297',  flag: '🇦🇼', name: 'Aruba' },
  { code: '+298',  flag: '🇫🇴', name: 'Faroe Islands' },
  { code: '+299',  flag: '🇬🇱', name: 'Greenland' },
  { code: '+350',  flag: '🇬🇮', name: 'Gibraltar' },
  { code: '+351',  flag: '🇵🇹', name: 'Portugal' },
  { code: '+352',  flag: '🇱🇺', name: 'Luxembourg' },
  { code: '+353',  flag: '🇮🇪', name: 'Ireland' },
  { code: '+354',  flag: '🇮🇸', name: 'Iceland' },
  { code: '+355',  flag: '🇦🇱', name: 'Albania' },
  { code: '+356',  flag: '🇲🇹', name: 'Malta' },
  { code: '+357',  flag: '🇨🇾', name: 'Cyprus' },
  { code: '+358',  flag: '🇫🇮', name: 'Finland' },
  { code: '+359',  flag: '🇧🇬', name: 'Bulgaria' },
  { code: '+370',  flag: '🇱🇹', name: 'Lithuania' },
  { code: '+371',  flag: '🇱🇻', name: 'Latvia' },
  { code: '+372',  flag: '🇪🇪', name: 'Estonia' },
  { code: '+373',  flag: '🇲🇩', name: 'Moldova' },
  { code: '+374',  flag: '🇦🇲', name: 'Armenia' },
  { code: '+375',  flag: '🇧🇾', name: 'Belarus' },
  { code: '+376',  flag: '🇦🇩', name: 'Andorra' },
  { code: '+377',  flag: '🇲🇨', name: 'Monaco' },
  { code: '+378',  flag: '🇸🇲', name: 'San Marino' },
  { code: '+380',  flag: '🇺🇦', name: 'Ukraine' },
  { code: '+381',  flag: '🇷🇸', name: 'Serbia' },
  { code: '+382',  flag: '🇲🇪', name: 'Montenegro' },
  { code: '+383',  flag: '🇽🇰', name: 'Kosovo' },
  { code: '+385',  flag: '🇭🇷', name: 'Croatia' },
  { code: '+386',  flag: '🇸🇮', name: 'Slovenia' },
  { code: '+387',  flag: '🇧🇦', name: 'Bosnia & Herzegovina' },
  { code: '+389',  flag: '🇲🇰', name: 'North Macedonia' },
  { code: '+420',  flag: '🇨🇿', name: 'Czech Republic' },
  { code: '+421',  flag: '🇸🇰', name: 'Slovakia' },
  { code: '+423',  flag: '🇱🇮', name: 'Liechtenstein' },
  { code: '+500',  flag: '🇫🇰', name: 'Falkland Islands' },
  { code: '+501',  flag: '🇧🇿', name: 'Belize' },
  { code: '+502',  flag: '🇬🇹', name: 'Guatemala' },
  { code: '+503',  flag: '🇸🇻', name: 'El Salvador' },
  { code: '+504',  flag: '🇭🇳', name: 'Honduras' },
  { code: '+505',  flag: '🇳🇮', name: 'Nicaragua' },
  { code: '+506',  flag: '🇨🇷', name: 'Costa Rica' },
  { code: '+507',  flag: '🇵🇦', name: 'Panama' },
  { code: '+509',  flag: '🇭🇹', name: 'Haiti' },
  { code: '+590',  flag: '🇬🇵', name: 'Guadeloupe' },
  { code: '+591',  flag: '🇧🇴', name: 'Bolivia' },
  { code: '+592',  flag: '🇬🇾', name: 'Guyana' },
  { code: '+593',  flag: '🇪🇨', name: 'Ecuador' },
  { code: '+595',  flag: '🇵🇾', name: 'Paraguay' },
  { code: '+597',  flag: '🇸🇷', name: 'Suriname' },
  { code: '+598',  flag: '🇺🇾', name: 'Uruguay' },
  { code: '+599',  flag: '🇨🇼', name: 'Curaçao' },
  { code: '+670',  flag: '🇹🇱', name: 'Timor-Leste' },
  { code: '+673',  flag: '🇧🇳', name: 'Brunei' },
  { code: '+675',  flag: '🇵🇬', name: 'Papua New Guinea' },
  { code: '+676',  flag: '🇹🇴', name: 'Tonga' },
  { code: '+677',  flag: '🇸🇧', name: 'Solomon Islands' },
  { code: '+678',  flag: '🇻🇺', name: 'Vanuatu' },
  { code: '+679',  flag: '🇫🇯', name: 'Fiji' },
  { code: '+680',  flag: '🇵🇼', name: 'Palau' },
  { code: '+682',  flag: '🇨🇰', name: 'Cook Islands' },
  { code: '+685',  flag: '🇼🇸', name: 'Samoa' },
  { code: '+686',  flag: '🇰🇮', name: 'Kiribati' },
  { code: '+687',  flag: '🇳🇨', name: 'New Caledonia' },
  { code: '+688',  flag: '🇹🇻', name: 'Tuvalu' },
  { code: '+689',  flag: '🇵🇫', name: 'French Polynesia' },
  { code: '+691',  flag: '🇫🇲', name: 'Micronesia' },
  { code: '+692',  flag: '🇲🇭', name: 'Marshall Islands' },
  { code: '+850',  flag: '🇰🇵', name: 'North Korea' },
  { code: '+852',  flag: '🇭🇰', name: 'Hong Kong' },
  { code: '+853',  flag: '🇲🇴', name: 'Macau' },
  { code: '+855',  flag: '🇰🇭', name: 'Cambodia' },
  { code: '+856',  flag: '🇱🇦', name: 'Laos' },
  { code: '+880',  flag: '🇧🇩', name: 'Bangladesh' },
  { code: '+886',  flag: '🇹🇼', name: 'Taiwan' },
  { code: '+960',  flag: '🇲🇻', name: 'Maldives' },
  { code: '+961',  flag: '🇱🇧', name: 'Lebanon' },
  { code: '+962',  flag: '🇯🇴', name: 'Jordan' },
  { code: '+963',  flag: '🇸🇾', name: 'Syria' },
  { code: '+964',  flag: '🇮🇶', name: 'Iraq' },
  { code: '+965',  flag: '🇰🇼', name: 'Kuwait' },
  { code: '+966',  flag: '🇸🇦', name: 'Saudi Arabia' },
  { code: '+967',  flag: '🇾🇪', name: 'Yemen' },
  { code: '+968',  flag: '🇴🇲', name: 'Oman' },
  { code: '+970',  flag: '🇵🇸', name: 'Palestine' },
  { code: '+971',  flag: '🇦🇪', name: 'UAE' },
  { code: '+972',  flag: '🇮🇱', name: 'Israel' },
  { code: '+973',  flag: '🇧🇭', name: 'Bahrain' },
  { code: '+974',  flag: '🇶🇦', name: 'Qatar' },
  { code: '+975',  flag: '🇧🇹', name: 'Bhutan' },
  { code: '+976',  flag: '🇲🇳', name: 'Mongolia' },
  { code: '+977',  flag: '🇳🇵', name: 'Nepal' },
  { code: '+992',  flag: '🇹🇯', name: 'Tajikistan' },
  { code: '+993',  flag: '🇹🇲', name: 'Turkmenistan' },
  { code: '+994',  flag: '🇦🇿', name: 'Azerbaijan' },
  { code: '+995',  flag: '🇬🇪', name: 'Georgia' },
  { code: '+996',  flag: '🇰🇬', name: 'Kyrgyzstan' },
  { code: '+998',  flag: '🇺🇿', name: 'Uzbekistan' },
  { code: '+1242', flag: '🇧🇸', name: 'Bahamas' },
  { code: '+1246', flag: '🇧🇧', name: 'Barbados' },
  { code: '+1264', flag: '🇦🇮', name: 'Anguilla' },
  { code: '+1268', flag: '🇦🇬', name: 'Antigua & Barbuda' },
  { code: '+1284', flag: '🇻🇬', name: 'British Virgin Islands' },
  { code: '+1340', flag: '🇻🇮', name: 'US Virgin Islands' },
  { code: '+1345', flag: '🇰🇾', name: 'Cayman Islands' },
  { code: '+1441', flag: '🇧🇲', name: 'Bermuda' },
  { code: '+1473', flag: '🇬🇩', name: 'Grenada' },
  { code: '+1649', flag: '🇹🇨', name: 'Turks & Caicos' },
  { code: '+1664', flag: '🇲🇸', name: 'Montserrat' },
  { code: '+1670', flag: '🇲🇵', name: 'Northern Mariana Islands' },
  { code: '+1671', flag: '🇬🇺', name: 'Guam' },
  { code: '+1684', flag: '🇦🇸', name: 'American Samoa' },
  { code: '+1758', flag: '🇱🇨', name: 'Saint Lucia' },
  { code: '+1767', flag: '🇩🇲', name: 'Dominica' },
  { code: '+1784', flag: '🇻🇨', name: 'St. Vincent & Grenadines' },
  { code: '+1787', flag: '🇵🇷', name: 'Puerto Rico' },
  { code: '+1809', flag: '🇩🇴', name: 'Dominican Republic' },
  { code: '+1868', flag: '🇹🇹', name: 'Trinidad & Tobago' },
  { code: '+1869', flag: '🇰🇳', name: 'Saint Kitts & Nevis' },
  { code: '+1876', flag: '🇯🇲', name: 'Jamaica' },
];

function _buildCountryOptions(selectedCode = '+91') {
  return _COUNTRY_CODES
    .map(c => `<option value="${c.code}"${c.code === selectedCode ? ' selected' : ''}>${c.flag} ${c.code} ${c.name}</option>`)
    .join('');
}

function _populateCountrySelects() {
  const ids = ['signupCountryCode', 'googleCountryCode'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = _buildCountryOptions(el.value || '+91');
  });
}

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
                  <option value="+91">🇮🇳 +91 India</option>
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
                  <option value="+91">🇮🇳 +91 India</option>
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
  _populateCountrySelects();
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
    showError('signupMobileError', 'This mobile number is already registered to another account.');
    _appendSignInLink('signupMobileError');
    return;
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
// Submit as soon as the sixth digit lands, from typing, pasting or the
// one-time-code autofill. Making someone type six digits and THEN hunt for a
// button is the kind of friction that gets blamed on the code being wrong —
// especially here, where a wrong guess costs an attempt and a 60s cooldown.
let _autoVerifying = false;
function _maybeAutoVerify() {
  if (_autoVerifying) return;
  const otp = [0, 1, 2, 3, 4, 5]
    .map(i => document.getElementById(`otpBox${i}`)?.value || '').join('');
  if (!/^\d{6}$/.test(otp)) return;
  _autoVerifying = true;
  // Let the last keystroke paint before the button goes into its loading state.
  setTimeout(() => {
    Promise.resolve(handleStep2Verify()).finally(() => { _autoVerifying = false; });
  }, 80);
}

let _step2Bound = false;
function attachStep2Listeners() {
  // These are DOCUMENT-level listeners, so binding them twice would fire
  // auto-advance twice per keystroke and skip a box.
  if (_step2Bound) return;
  _step2Bound = true;
  // Auto-advance between digit boxes
  document.addEventListener('input', (e) => {
    if (!e.target.classList.contains('otp-box')) return;
    const idx = parseInt(e.target.id.replace('otpBox', ''));
    const val = e.target.value.replace(/\D/g, '');
    e.target.value = val.slice(-1);
    if (val && idx < 5) document.getElementById(`otpBox${idx + 1}`)?.focus();
    _maybeAutoVerify();
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
    _maybeAutoVerify();
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
      if (signupResult.needsSignin) {
        showStep1();
        showError('signupFormError', signupResult.message);
        _appendSignInLink('signupFormError');
        return;
      }
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
    if (signupResult.needsSignin) {
      // Account already exists (duplicate email or phone) — go back to step 1 so
      // the error appears in context and the "Sign In" link is reachable.
      showStep1();
      showError('signupFormError', signupResult.message);
      _appendSignInLink('signupFormError');
      return;
    }
    return showError('otpError', signupResult.message);
  }

  clearResendTimer();
  _pending = { name: '', email: '', password: '', mobile: '' };
  _otpEmail = '';
  _flow = 'email';
  window.dispatchEvent(new CustomEvent('authStateChange', { detail: { authenticated: true } }));
  closeSignUpModal();
  // The account now exists - OTP verified and the record created. Fired
  // here rather than on OTP request, so an abandoned signup is never
  // counted as a registration. Both the email and the Google path reach
  // this; aegisTrack de-dupes by name, so it fires exactly once.
  if (window.aegisTrack) window.aegisTrack('complete_registration');
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
  // The account now exists - OTP verified and the record created. Fired
  // here rather than on OTP request, so an abandoned signup is never
  // counted as a registration. Both the email and the Google path reach
  // this; aegisTrack de-dupes by name, so it fires exactly once.
  if (window.aegisTrack) window.aegisTrack('complete_registration');
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
    showError('googleMobileError', 'This mobile number is already registered to another account.');
    _appendSignInLink('googleMobileError');
    return;
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

/** Verify an account that already exists but never completed verification.
 *
 * Sign-in used to dead-end these: "Your account setup is incomplete. Please sign
 * up and complete phone verification", with a link that reopened a BLANK signup
 * form. The user then retyped everything they had just typed, to create an
 * account that already existed. Nothing about that asked them for the one thing
 * actually missing — the code.
 *
 * This opens the modal straight at the OTP step for the address they just tried
 * to sign in with. On verify, handleStep2Verify runs the normal completion path;
 * createUserWithEmailAndPassword returns auth/email-already-in-use and the
 * recovery branch in auth.js adopts the existing account, rebuilds its profile
 * and backend record, and signs them in.
 *
 * The password comes from the sign-in attempt that just succeeded against
 * Firebase Auth, so it is known-good — which is exactly what the recovery branch
 * needs to prove ownership before adopting.
 */
export async function openVerifyExistingAccount({ email, password, mobile = '', name = '' } = {}) {
  if (!email) return;
  const modal = document.getElementById('signupModal');
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
  _flow = 'email';
  _pending = { name: name || email.split('@')[0], email, password, mobile };
  _otpEmail = email;

  showStep1();
  setLoading(true, 'Sending verification code…');
  const result = await sendOTPForSignup(email, mobile || null);
  setLoading(false);

  if (!result.success) {
    showError('signupFormError', result.message || 'Could not send a verification code. Please try again.');
    return;
  }
  _otpVia = result.via || 'email';
  showStep2(_otpVia, mobile, email);
  startResendCountdown(60);
}

function listenForSignupEvent() {
  window.addEventListener('openSignup', openSignUpModal);
  window.addEventListener('verifyExistingAccount', (ev) => {
    openVerifyExistingAccount(ev.detail || {});
  });
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

function _appendSignInLink(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const link = document.createElement('a');
  link.href = '#';
  link.textContent = ' Sign in instead →';
  link.className = 'auth-link';
  link.style.display = 'inline';
  link.addEventListener('click', (ev) => {
    ev.preventDefault();
    closeSignUpModal();
    window.dispatchEvent(new CustomEvent('openSignin'));
  });
  el.appendChild(link);
}
