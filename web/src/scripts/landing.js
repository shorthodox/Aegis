// AEGIS v1.0 – Sovereign Terminal Core (Production‑Ready for Railway)
import { 
    auth, db, ensureUserDocument, subscribeUserSettings, updateUserSetting,
    getCurrentUserToken, logout, isTokenVisible, getUpgradeModal,
    registerWithEmail, signInWithEmail, sendPhoneOTP, verifyPhoneOTP, signInWithGoogle,
    setupRecaptcha, checkAccountExists, updateUserOnboarding
} from './gatekeeper.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";
import { collection, addDoc, onSnapshot, doc, updateDoc, query, where } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";

// Scroll lock helpers (for alpha modal)
function lockBodyScroll() {
    document.body.classList.add('modal-open');
}
function unlockBodyScroll() {
    document.body.classList.remove('modal-open');
}

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    // ========== DOM Elements ==========
    const signalGrid = document.getElementById('signal-grid');
    const capitalInput = document.getElementById('user-capital');
    const riskSelect = document.getElementById('risk-level');
    const suggestedSpan = document.getElementById('suggested-amount');
    const alphaToggleDiv = document.getElementById('alphaToggle');
    const statusDot = document.getElementById('ws-status-dot');
    const statusText = document.getElementById('ws-status-text');
    const warmupSpan = document.getElementById('warmup-status');
    const simSymbol = document.getElementById('sim-symbol');
    const simEntry = document.getElementById('sim-entry');
    const simBalance = document.getElementById('sim-balance');
    const simRiskSlider = document.getElementById('sim-risk-slider');
    const riskPercentDisplay = document.getElementById('risk-percent-display');
    const simLeverageSlider = document.getElementById('sim-leverage');
    const leverageDisplay = document.getElementById('leverage-display');
    const simSl = document.getElementById('sim-sl');
    const simTp = document.getElementById('sim-tp');
    const posUnitsSpan = document.getElementById('pos-units');
    const notionalSpan = document.getElementById('notional');
    const marginSpan = document.getElementById('margin');
    const liquidationSpan = document.getElementById('liquidation');
    const rrRatioSpan = document.getElementById('rr-ratio');
    const riskGaugeFill = document.getElementById('risk-gauge-fill');
    const executeBtn = document.getElementById('execute-trade-btn');
    const copyBtn = document.getElementById('copy-execution-btn');
    const tradesTbody = document.getElementById('trades-tbody');
    const directionBadge = document.getElementById('direction-badge');
    const logoutBtn = document.getElementById('logoutBtn');

    // ========== Global State ==========
    let currentUser = null;
    let userPlan = 'trial';
    let trialActive = false;
    let userSettings = { capital: 10000, risk_pct: 2 };
    let alphaMode = false;
    let warmupActive = true;
    let btcSignalSafe = false;
    let ws = null;
    let reconnectAttempts = 0;
    let currentSignals = new Map();      // symbol (normalized with '/') -> signal object
    let selectedTrade = null;
    let activeTrades = new Map();         // from Firestore (user executed trades)
    let unsubscribeTrades = null;
    let currentPrices = new Map();        // symbol (normalized with '/') -> latest price
    let unsubscribeSettings = null;
    let engineConnected = false;           // WebSocket open state
    let firstSignalsReceived = false;      // at least one signal packet arrived

    // ========== Symbol Normalization ==========
    // Backend sends symbols with '/' (e.g., "BTC/USDT"). Ensure all keys use this format.
    function normalizeSymbol(sym) {
        if (!sym) return sym;
        // Replace any underscore with slash (e.g., "BTC_USDT" → "BTC/USDT")
        return String(sym).replace(/_/g, '/');
    }

    // ========== Helper: ATR ==========
    function getATR(price) { return price * 0.015; }

    // ========== Connection Status ==========
    function setStatus(connected, message = '') {
        if (statusDot) statusDot.className = connected ? 'ws-dot connected' : 'ws-dot disconnected';
        if (statusText) statusText.innerText = message || (connected ? 'Live' : 'Connecting...');
        engineConnected = connected;
        if (connected) {
            // Force UI redraw to show "Synchronizing..." instead of connection error
            renderAllSignals();
        }
    }

    // ========== Simulator Update ==========
    function updateSimulation() {
        if (!selectedTrade) return;
        const entry = parseFloat(simEntry.value);
        const balance = parseFloat(simBalance.value);
        const riskPercent = parseFloat(simRiskSlider.value);
        const leverage = parseFloat(simLeverageSlider.value);
        let sl = parseFloat(simSl.value);
        let tp = parseFloat(simTp.value);
        const direction = selectedTrade.direction;
        if (isNaN(entry) || entry <= 0) return;
        const atr = selectedTrade.atr || getATR(entry);
        if (isNaN(sl) || sl <= 0) {
            sl = direction === 'LONG' ? entry - atr * 1.5 : entry + atr * 1.5;
            simSl.value = sl.toFixed(4);
        }
        if (isNaN(tp) || tp <= 0) {
            const riskDistance = Math.abs(entry - sl);
            tp = direction === 'LONG' ? entry + riskDistance * 3 : entry - riskDistance * 3;
            simTp.value = tp.toFixed(4);
        }
        const slDistance = Math.abs(entry - sl);
        if (slDistance === 0) return;
        const riskAmount = balance * (riskPercent / 100);
        const positionUnits = riskAmount / slDistance;
        const notionalValue = positionUnits * entry;
        const marginRequired = notionalValue / leverage;
        const liquidationPrice = direction === 'LONG'
            ? entry - (entry / leverage) + (sl / leverage)
            : entry + (entry / leverage) - (sl / leverage);
        const riskReward = 3;
        const riskPercentOfBalance = (riskAmount / balance) * 100;
        const gaugePercent = Math.min(100, (riskPercentOfBalance / 10) * 100);
        if (riskGaugeFill) riskGaugeFill.style.width = `${gaugePercent}%`;
        if (posUnitsSpan) posUnitsSpan.innerText = Number(positionUnits || 0).toFixed(4);
        if (notionalSpan) notionalSpan.innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
        if (marginSpan) marginSpan.innerText = `$${Number(marginRequired || 0).toFixed(2)}`;
        if (liquidationSpan) liquidationSpan.innerText = `$${Number(liquidationPrice || 0).toFixed(4)}`;
        if (rrRatioSpan) rrRatioSpan.innerText = Number(riskReward).toFixed(2);
        if (suggestedSpan) suggestedSpan.innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
    }

    // ========== Pre-fill Simulator ==========
    function prefillTradeSim(symbol, price, signalObj) {
        const direction = (signalObj.signal === 'BUY' || signalObj.signal === 'STRONG_BUY') ? 'LONG' : 'SHORT';
        const atr = getATR(price);
        selectedTrade = { symbol, entryPrice: price, direction, atr, aiProb: signalObj.ai_prob, signal: signalObj.signal };
        simSymbol.value = symbol;
        simEntry.value = price;
        directionBadge.className = `direction-badge ${direction === 'LONG' ? 'long' : 'short'}`;
        directionBadge.innerText = direction;
        simSl.value = '';
        simTp.value = '';
        updateSimulation();
    }

    // ========== Safety Brake ==========
    function updateSafetyBrake() {
        if (!executeBtn) return;
        const warmupOk = !warmupActive;
        const btcOk = btcSignalSafe;
        const disabled = (!alphaMode && (!warmupOk || !btcOk));
        if (disabled) {
            executeBtn.classList.add('disabled');
            executeBtn.disabled = true;
            executeBtn.title = alphaMode ? '' : (warmupActive ? 'Safety Brake: Warmup in progress' : 'BTC Signal not safe (HOLD/SELL)');
        } else {
            executeBtn.classList.remove('disabled');
            executeBtn.disabled = false;
            executeBtn.title = 'Execute trade';
        }
    }

    // ========== Execute Trade ==========
    async function executeTrade() {
        if (!selectedTrade || !currentUser || executeBtn.disabled) return;
        const entry = parseFloat(simEntry.value);
        const sl = parseFloat(simSl.value);
        const tp = parseFloat(simTp.value);
        const riskPercent = parseFloat(simRiskSlider.value);
        const leverage = parseFloat(simLeverageSlider.value);
        const positionUnits = parseFloat(posUnitsSpan.innerText);
        const notional = parseFloat(notionalSpan.innerText.replace('$', ''));
        const tradeData = {
            symbol: selectedTrade.symbol,
            side: selectedTrade.direction,
            entryPrice: entry,
            stopLoss: sl,
            takeProfit: tp,
            riskPercent,
            leverage,
            positionUnits,
            notionalValue: notional,
            status: 'open',
            openTime: new Date(),
            userId: currentUser.uid
        };
        try {
            const tradesRef = collection(db, 'users', currentUser.uid, 'trades');
            await addDoc(tradesRef, tradeData);
            console.log('Trade executed');
        } catch (err) {
            console.error('Failed to save trade:', err);
            alert('Trade execution failed: ' + err.message);
        }
    }

    // ========== Trades Table ==========
    function renderTradesTable() {
        if (!tradesTbody) return;
        if (activeTrades.size === 0) {
            tradesTbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No active trades</td></tr>';
            return;
        }
        let html = '';
        for (let [id, trade] of activeTrades.entries()) {
            const currentPrice = currentPrices.get(trade.symbol) || trade.entryPrice || 0;
            let pnl = 0;
            const entryPrice = Number(trade.entryPrice || 0);
            const positionUnits = Number(trade.positionUnits || 0);
            if (trade.side === 'LONG') pnl = (currentPrice - entryPrice) * positionUnits;
            else pnl = (entryPrice - currentPrice) * positionUnits;
            const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            html += `
                <tr data-trade-id="${id}">
                    <td>${trade.symbol}</td>
                    <td>${trade.side}</td>
                    <td>${entryPrice.toFixed(4)}</td>
                    <td>${Number(trade.stopLoss || 0).toFixed(4)}</td>
                    <td>${Number(trade.takeProfit || 0).toFixed(4)}</td>
                    <td class="${pnlClass}">$${Number(pnl).toFixed(2)}</td>
                    <td><button class="close-trade-btn" data-id="${id}">Close</button></td>
                </tr>
            `;
        }
        tradesTbody.innerHTML = html;
        document.querySelectorAll('.close-trade-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const tradeId = btn.dataset.id;
                await closeTrade(tradeId);
            });
        });
    }

    async function closeTrade(tradeId) {
        if (!currentUser) return;
        const tradeRef = doc(db, 'users', currentUser.uid, 'trades', tradeId);
        await updateDoc(tradeRef, { status: 'closed', closeTime: new Date() });
    }

    function subscribeToTrades(userId) {
        const tradesRef = collection(db, 'users', userId, 'trades');
        const q = query(tradesRef, where('status', '==', 'open'));
        if (unsubscribeTrades) unsubscribeTrades();
        unsubscribeTrades = onSnapshot(q, (snapshot) => {
            activeTrades.clear();
            snapshot.forEach(doc => {
                activeTrades.set(doc.id, { id: doc.id, ...doc.data() });
            });
            renderTradesTable();
        }, (error) => console.error('Trade subscription error:', error));
    }

    // ========== Signal Cards ==========
    function getSignalClass(signalStrength) {
        switch (signalStrength) {
            case 'STRONG': return 'signal-strong';
            case 'NORMAL': return 'signal-buy';
            case 'HOLD': return 'signal-hold';
            case 'SELL':
            case 'STRONG_SELL': return 'signal-avoid';
            default: return 'signal-hold';
        }
    }

    function renderSignalCard(symbol, price, sig, isLocked = false) {
        const card = document.createElement('div');
        card.className = `signal-card ${isLocked ? 'is-locked' : ''}`;
        if (isLocked) {
            card.innerHTML = `
                <div class="card-header">
                    <span class="symbol">${symbol} <i class="fas fa-lock"></i></span>
                    <span class="mode-badge">🔒 PRO</span>
                </div>
                <div class="price-row">
                    <span>💰 ${price.toFixed(4)}</span>
                    <span class="delta-up">+0.00%</span>
                </div>
                <div class="price-row">
                    <span>🤖 AI --%</span>
                    <span class="signal-tag signal-hold">LOCKED</span>
                </div>
                <div class="sl-tp">
                    <span>🛡️ Upgrade to unlock</span>
                    <span>🎯 full fleet</span>
                </div>
            `;
            card.addEventListener('click', (e) => {
                e.stopPropagation();
                const modal = getUpgradeModal();
                if (modal) modal.style.display = 'flex';
            });
            return card;
        }

        const signalStrength = sig.signal_strength || (sig.signal === 'BUY' ? 'NORMAL' : (sig.signal === 'STRONG_BUY' ? 'STRONG' : (sig.signal === 'HOLD' ? 'HOLD' : 'SELL')));
        const signalClass = getSignalClass(signalStrength);
        const displaySignal = sig.signal || 'WAITING';
        const delta = ((currentPrices.get(symbol) || price) - price) / price * 100;
        const deltaClass = delta >= 0 ? 'delta-up' : 'delta-down';
        const deltaSign = delta >= 0 ? '+' : '';
        const aiProb = (sig.ai_prob * 100).toFixed(0);
        const atrVal = sig.atr || price * 0.015;
        const stopLoss = price - atrVal * 1.5;
        const takeProfit = price + atrVal * 4.5;
        card.innerHTML = `
            <div class="card-header">
                <span class="symbol">${symbol}</span>
                <span class="mode-badge">AGGR</span>
            </div>
            <div class="price-row">
                <span>💰 ${price.toFixed(4)}</span>
                <span class="${deltaClass}">${deltaSign}${delta.toFixed(2)}%</span>
            </div>
            <div class="price-row">
                <span>🤖 AI ${aiProb}%</span>
                <span class="signal-tag ${signalClass}">${displaySignal}</span>
                <span>📉 ${sig.risk_pct || 2}%</span>
            </div>
            <div class="sl-tp">
                <span>🛡️ SL: ${stopLoss.toFixed(4)}</span>
                <span>🎯 TP: ${takeProfit.toFixed(4)}</span>
            </div>
        `;
        card.addEventListener('click', () => prefillTradeSim(symbol, price, sig));
        return card;
    }

    function renderAllSignals() {
        if (!signalGrid) return;
        const trialValid = (userPlan === 'trial' && trialActive);

        // No data yet – distinguish between connection problem and engine warmup
        if (!firstSignalsReceived) {
            if (engineConnected) {
                signalGrid.innerHTML = '<div class="loading-spinner">⚡ Synchronizing with Engine... (signals incoming)</div>';
            } else {
                signalGrid.innerHTML = '<div class="loading-spinner">❌ Waiting for engine connection...</div>';
            }
            return;
        }

        if (currentSignals.size === 0) {
            signalGrid.innerHTML = '<div class="loading-spinner">📡 No signal data yet. Engine may be warming up.</div>';
            return;
        }

        signalGrid.innerHTML = '';
        for (let [rawSym, sig] of currentSignals.entries()) {
            const sym = normalizeSymbol(rawSym);
            const price = currentPrices.get(sym) || 0;
            const visible = isTokenVisible(sym, userPlan, trialValid);
            signalGrid.appendChild(renderSignalCard(sym, price, sig, !visible));
        }
    }

    // ========== WebSocket (dynamic URL with protocol detection) ==========
    function connectWebSocket() {
        if (ws && ws.readyState === WebSocket.OPEN) return;

        // Determine secure protocol based on page origin
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
        console.log(`🔄 Connecting WebSocket to ${wsUrl}`);

        ws = new WebSocket(wsUrl);
        ws.onopen = () => { 
            setStatus(true, 'Live'); 
            reconnectAttempts = 0;
            console.log('✅ WebSocket connected');
            // Reset data on reconnect to avoid stale signals
            firstSignalsReceived = false;
            currentSignals.clear();
            currentPrices.clear();
            renderAllSignals();
        };
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('📡 Aegis Data Received:', data);
                // Mark that we have received at least one data packet
                if (!firstSignalsReceived && (Object.keys(data.tickers || {}).length > 0 || Object.keys(data.signals || {}).length > 0)) {
                    firstSignalsReceived = true;
                }
                // Warmup status
                if (data.warmup) {
                    warmupActive = data.warmup !== 'Active' && !data.warmup.includes('Active');
                    if (warmupSpan) warmupSpan.innerText = data.warmup;
                }
                // Alpha mode
                alphaMode = data.alpha_mode || false;
                if (alphaToggleDiv) {
                    alphaToggleDiv.innerText = alphaMode ? '⚡ ON' : 'OFF';
                    alphaToggleDiv.classList.toggle('active', alphaMode);
                }
                // Tickers: normalize keys and store
                for (const [sym, price] of Object.entries(data.tickers || {})) {
                    const norm = normalizeSymbol(sym);
                    currentPrices.set(norm, price);
                }
                // Signals: normalize keys and store enriched object
                for (const [sym, sig] of Object.entries(data.signals || {})) {
                    const norm = normalizeSymbol(sym);
                    currentSignals.set(norm, {
                        ai_prob: sig.ai_prob || 0,
                        signal: sig.signal || 'WAITING',
                        signal_strength: sig.signal_strength || 'HOLD',
                        atr: sig.atr || 0,
                        risk_pct: sig.risk_pct || 2
                    });
                }
                // BTC safety
                const btcSig = currentSignals.get('BTC/USDT');
                if (btcSig) {
                    btcSignalSafe = (btcSig.signal === 'BUY' || btcSig.signal === 'STRONG_BUY');
                }
                // Refresh UI
                renderAllSignals();
                updateSafetyBrake();
                renderTradesTable();
            } catch (err) {
                console.error('WebSocket message error:', err);
            }
        };
        ws.onclose = (event) => {
            setStatus(false, 'Reconnecting...');
            console.warn(`WebSocket closed: code=${event.code}, reason=${event.reason}, wasClean=${event.wasClean}`);
            const delay = Math.min(30000, 1000 * Math.pow(2, reconnectAttempts));
            reconnectAttempts++;
            setTimeout(connectWebSocket, delay);
        };
        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setStatus(false, 'Error');
        };
    }

    // ========== Logout Handler ==========
    async function handleLogout() {
        console.log("🔴 Logout initiated");
        try {
            if (unsubscribeTrades) unsubscribeTrades();
            if (unsubscribeSettings) unsubscribeSettings();
            if (ws && ws.readyState === WebSocket.OPEN) ws.close();
            const result = await logout();
            if (result.success) {
                console.log("✅ Logout successful, redirecting...");
                window.location.replace('./index.html');
            } else {
                console.error('Logout failed:', result.error);
                alert('Logout failed: ' + result.error);
            }
        } catch (err) {
            console.error('Logout error:', err);
            alert('Logout error: ' + err.message);
        }
    }

    // ========== Auth Init ==========
    onAuthStateChanged(auth, async (user) => {
        // Only redirect unauthenticated users away when on protected pages
        const isProtectedPage = document.body.classList.contains('dashboard-body') || window.location.pathname.includes('dashboard.html');
        if (!user) {
            if (isProtectedPage) {
                window.location.replace('./index.html');
            }
            // If not on a protected page, allow public view without redirect
            return;
        }
        currentUser = user;
        try {
            const userData = await ensureUserDocument(user);
            userPlan = userData.plan || 'trial';
            const joinDate = userData.join_date?.toDate() || new Date();
            trialActive = (userPlan === 'trial' && (Date.now() - joinDate < 72 * 3600000));
            unsubscribeSettings = subscribeUserSettings(user, (settings) => {
                userSettings = settings;
                if (capitalInput) capitalInput.value = settings.capital;
                if (riskSelect) riskSelect.value = settings.risk_pct;
                if (simBalance) simBalance.value = settings.capital;
                if (simRiskSlider) simRiskSlider.value = settings.risk_pct;
                updateSimulation();
            });
            subscribeToTrades(user.uid);
            connectWebSocket();
        } catch (err) {
            console.error('Auth init error:', err);
        }
    });

    // ========== Event Listeners ==========
    // --- Modal controls (signup/login) ---
    const portalBtn = document.getElementById('portalBtn');
    const heroMathBtn = document.getElementById('heroMathBtn');
    const loginModalEl = document.getElementById('loginModal');
    const step1El = document.getElementById('step1');
    const step2El = document.getElementById('step2');
    const step3El = document.getElementById('step3');
    const continueToOtpBtnEl = document.getElementById('continueToOtpBtn');
    const backToEmailBtnEl = document.getElementById('backToEmailBtn');
    const completeOnboardingBtn = document.getElementById('completeOnboarding');
    const googleStepBtnEl = document.getElementById('googleStepBtn');
    const emailInputEl = document.getElementById('emailInput');
    const otpContainerEl = document.getElementById('otpContainer');
    const otpInputs = document.querySelectorAll('.otp-digit');
    const otpEmailDisplayEl = document.getElementById('otpEmailDisplay');
    const fullNameEl = document.getElementById('fullName');
    const locationEl = document.getElementById('location');
    const regPasswordEl = document.getElementById('regPassword');
    const regConfirmPasswordEl = document.getElementById('regConfirmPassword');
    const avatarFileEl = document.getElementById('avatarFile');

    let modalAuthMode = 'register'; // 'register' or 'login'
    let modalUsePhone = false;
    let pendingPhone = null;

    function showModalStep(n) {
        step1El.classList.remove('active');
        step2El.classList.remove('active');
        step3El.classList.remove('active');
        if (n === 1) step1El.classList.add('active');
        if (n === 2) step2El.classList.add('active');
        if (n === 3) step3El.classList.add('active');
    }

    function openAuthModal() {
        if (loginModalEl) {
            loginModalEl.style.display = 'flex';
            lockBodyScroll();
            showModalStep(1);
        }
    }

    function closeAuthModal() {
        if (loginModalEl) {
            loginModalEl.style.display = 'none';
            unlockBodyScroll();
        }
    }

    if (portalBtn) portalBtn.addEventListener('click', openAuthModal);
    if (heroMathBtn) heroMathBtn.addEventListener('click', openAuthModal);

    // Google button
    if (googleStepBtnEl) {
        googleStepBtnEl.addEventListener('click', async () => {
            try {
                const res = await signInWithGoogle();
                if (res && res.success) {
                    // Redirect existing users to dashboard; new users continue to pricing
                    // We cannot reliably detect 'new' here, so send to dashboard for now
                    window.location.replace('/web/src/pages/dashboard.html');
                } else {
                    alert(res.error || 'Google sign-in failed');
                }
            } catch (err) {
                console.error('Google sign-in error:', err);
                alert('Google sign-in error: ' + (err.message || err));
            }
        });
    }
    if (capitalInput) capitalInput.addEventListener('change', () => { if (currentUser) updateUserSetting(currentUser, 'capital', parseFloat(capitalInput.value)); });
    if (riskSelect) riskSelect.addEventListener('change', () => { if (currentUser) updateUserSetting(currentUser, 'risk_pct', parseInt(riskSelect.value)); });
    if (simRiskSlider) simRiskSlider.addEventListener('input', () => { riskPercentDisplay.innerText = simRiskSlider.value + '%'; updateSimulation(); });
    if (simLeverageSlider) simLeverageSlider.addEventListener('input', () => { leverageDisplay.innerText = simLeverageSlider.value + 'x'; updateSimulation(); });
    if (simEntry) simEntry.addEventListener('input', updateSimulation);
    if (simSl) simSl.addEventListener('input', updateSimulation);
    if (simTp) simTp.addEventListener('input', updateSimulation);
    if (simBalance) simBalance.addEventListener('input', updateSimulation);
    if (executeBtn) executeBtn.addEventListener('click', executeTrade);
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            if (selectedTrade) {
                const str = `${selectedTrade.direction} ${selectedTrade.symbol} | Entry: ${simEntry.value} | SL: ${simSl.value} | TP: ${simTp.value} | Risk: ${simRiskSlider.value}%`;
                navigator.clipboard.writeText(str);
                alert('Copied to clipboard');
            }
        });
    }
    
    // Alpha modal with scroll lock
    if (alphaToggleDiv) {
        alphaToggleDiv.addEventListener('click', () => {
            const modal = document.getElementById('alpha-modal');
            if (modal) {
                modal.style.display = 'flex';
                lockBodyScroll();
            }
        });
    }
    
    const alphaConfirm = document.getElementById('alpha-confirm');
    const alphaCancel = document.getElementById('alpha-cancel');
    if (alphaConfirm) {
        alphaConfirm.addEventListener('click', async () => {
            const token = await getCurrentUserToken();
            if (token) {
                // Relative path (works on any origin)
                const response = await fetch('/alpha/toggle', {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (!response.ok) {
                    console.warn('Alpha toggle failed:', response.status);
                }
            }
            const modal = document.getElementById('alpha-modal');
            if (modal) {
                modal.style.display = 'none';
                unlockBodyScroll();
            }
        });
    }
    if (alphaCancel) {
        alphaCancel.addEventListener('click', () => {
            const modal = document.getElementById('alpha-modal');
            if (modal) {
                modal.style.display = 'none';
                unlockBodyScroll();
            }
        });
    }
    
    // ========== Login modal close fix ==========
    const loginModal = document.getElementById('loginModal');
    const closeModalBtn = document.querySelector('.close-modal');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            if (loginModal) {
                loginModal.style.display = 'none';
                unlockBodyScroll();
            }
        });
    }
    window.addEventListener('click', (e) => {
        if (e.target === loginModal) {
            loginModal.style.display = 'none';
            unlockBodyScroll();
        }
    });
    // ========== Auth modal interactions ==========
    if (backToEmailBtnEl) backToEmailBtnEl.addEventListener('click', () => showModalStep(1));

    if (continueToOtpBtnEl) {
        continueToOtpBtnEl.addEventListener('click', async () => {
            const val = (emailInputEl && emailInputEl.value || '').trim();
            if (!val) { alert('Enter an email or phone number'); return; }

            // Simple phone detection: starts with + or digits and contains no @
            const looksLikePhone = /^\+?\d[\d\s().-]{6,}$/.test(val) && !val.includes('@');
            if (looksLikePhone) {
                // Phone OTP flow
                modalUsePhone = true;
                pendingPhone = val;
                try {
                    setupRecaptcha && setupRecaptcha('recaptcha-container');
                    const r = await sendPhoneOTP(val, 'recaptcha-container');
                    if (!r.success) throw new Error(r.error || 'Failed to send OTP');
                    otpEmailDisplayEl.innerText = val;
                    showModalStep(2);
                    // focus first OTP input
                    if (otpInputs && otpInputs[0]) otpInputs[0].focus();
                } catch (err) {
                    console.error('sendPhoneOTP error:', err);
                    alert('Failed to send OTP: ' + (err.message || err));
                }
                return;
            }

            // Email path – check if account exists to choose login vs register
            modalUsePhone = false;
            try {
                const existsResp = await checkAccountExists(val);
                if (existsResp && existsResp.exists) {
                    modalAuthMode = 'login';
                } else {
                    modalAuthMode = 'register';
                }
                // Pre-fill Step3 fields
                if (fullNameEl) fullNameEl.value = '';
                if (locationEl) locationEl.value = '';
                showModalStep(3);
            } catch (err) {
                console.error('Account check failed:', err);
                alert('Account check failed: ' + (err.message || err));
            }
        });
    }

    // OTP input wiring (auto-advance)
    if (otpInputs && otpInputs.length) {
        otpInputs.forEach((inp, idx) => {
            inp.addEventListener('input', () => {
                const v = inp.value.replace(/[^0-9]/g, ''); inp.value = v;
                if (v && idx + 1 < otpInputs.length) otpInputs[idx + 1].focus();
                // if last and filled, attempt verify
                const completed = Array.from(otpInputs).every(i => i.value.trim().length === 1);
                if (completed) {
                    const code = Array.from(otpInputs).map(i => i.value).join('');
                    (async () => {
                        try {
                            const resp = await verifyPhoneOTP(code);
                            if (resp && resp.success) {
                                // After successful phone verification, redirect to pricing for onboarding
                                closeAuthModal();
                                window.location.replace('/web/src/pages/pricing.html');
                            } else {
                                alert(resp.error || 'Invalid OTP');
                            }
                        } catch (err) {
                            console.error('verifyPhoneOTP error:', err);
                            alert('OTP verification failed: ' + (err.message || err));
                        }
                    })();
                }
            });
            inp.addEventListener('keydown', (ev) => {
                if (ev.key === 'Backspace' && !inp.value && idx > 0) {
                    otpInputs[idx - 1].focus();
                }
            });
        });
    }

    // Complete onboarding / register or login
    if (completeOnboardingBtn) {
        completeOnboardingBtn.addEventListener('click', async () => {
            const email = emailInputEl && emailInputEl.value && emailInputEl.value.trim();
            const pwd = regPasswordEl && regPasswordEl.value;
            const pwd2 = regConfirmPasswordEl && regConfirmPasswordEl.value;
            const name = fullNameEl && fullNameEl.value.trim();
            const loc = locationEl && locationEl.value.trim();
            if (!email) { alert('Email required'); return; }
            if (!pwd || pwd.length < 6) { alert('Password must be at least 6 characters'); return; }
            if (pwd !== pwd2) { alert('Passwords do not match'); return; }

            if (modalAuthMode === 'login') {
                try {
                    const r = await signInWithEmail(email, pwd);
                    if (r && r.success) {
                        closeAuthModal();
                        window.location.replace('/web/src/pages/dashboard.html');
                    } else {
                        alert(r.error || 'Login failed');
                    }
                } catch (err) {
                    console.error('Login error:', err);
                    alert('Login error: ' + (err.message || err));
                }
                return;
            }

            // Register flow
            try {
                const res = await registerWithEmail(email, pwd, name || '');
                if (res && res.success) {
                    // Save onboarding metadata
                    try {
                        await updateUserOnboarding(res.user.uid, { fullName: name, location: loc, avatarUrl: null });
                    } catch (e) {
                        console.warn('Onboarding save failed:', e);
                    }
                    closeAuthModal();
                    // Redirect newly registered users to pricing to pick a plan
                    window.location.replace('/web/src/pages/pricing.html');
                } else {
                    alert(res.error || 'Registration failed');
                }
            } catch (err) {
                console.error('Registration error:', err);
                alert('Registration error: ' + (err.message || err));
            }
        });
    }
    if (logoutBtn) {
        console.log("Attaching logout listener");
        logoutBtn.addEventListener('click', handleLogout);
    } else {
        if (!window.__logoutWarned) {
            console.warn("Logout button not found in DOM");
            window.__logoutWarned = true;
        }
    }
});
