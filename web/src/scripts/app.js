import { 
    auth, db, ensureUserDocument, subscribeUserSettings, updateUserSetting,
    getCurrentUserToken, logout, getUpgradeModal
} from './gatekeeper.js';
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js";
import { collection, addDoc, onSnapshot, doc, updateDoc, query, where } from "https://www.gstatic.com/firebasejs/12.12.1/firebase-firestore.js";

import { AuthManager } from '../auth/authManager.js';
import { SignalStore } from '../stores/signalStore.js';
import { WebSocketManager } from '../websocket/websocketManager.js';
import { RenderEngine } from '../components/renderEngine.js';
import { initializeTrialCountdown } from './trial-countdown.js';

// Scroll lock helpers
function lockBodyScroll() { document.body.classList.add('modal-open'); }
function unlockBodyScroll() { document.body.classList.remove('modal-open'); }

document.addEventListener('DOMContentLoaded', () => {
    // ========== Modules ==========
    const signalStore = new SignalStore();
    const renderEngine = new RenderEngine('signal-grid', signalStore);
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
    
    const wsManager = new WebSocketManager(wsUrl, signalStore, (msg) => {
        // Dashboard specific updates
        if (msg.warmup && document.getElementById('warmup-status')) {
            document.getElementById('warmup-status').innerText = msg.warmup;
        }
        if (msg.alpha_mode !== undefined && document.getElementById('alphaToggle')) {
            const alphaToggleDiv = document.getElementById('alphaToggle');
            alphaToggleDiv.innerText = msg.alpha_mode ? '⚡ ON' : 'OFF';
            alphaToggleDiv.classList.toggle('active', msg.alpha_mode);
        }
    });

    // Listen for warmup updates from the public signals endpoint
    document.addEventListener('warmupUpdate', (e) => {
        if (e.detail.warmup && document.getElementById('warmup-status')) {
            document.getElementById('warmup-status').innerText = e.detail.warmup;
        }
    });

    // ========== Simulator Global State ==========
    let currentUser = null;
    let selectedTrade = null;
    let activeTrades = new Map();
    let unsubscribeTrades = null;
    let unsubscribeSettings = null;

    // Listen to row clicks for simulation
    document.addEventListener('signalRowClicked', (e) => {
        const sig = e.detail;
        prefillTradeSim(sig.pair, sig.entry, sig);
    });

    // ========== Helper ==========
    function getATR(price) { return price * 0.015; }

    // ========== Simulator Update ==========
    function updateSimulation() {
        if (!selectedTrade) return;
        const simEntry = document.getElementById('sim-entry');
        const simBalance = document.getElementById('sim-balance');
        const simRiskSlider = document.getElementById('sim-risk-slider');
        const simLeverageSlider = document.getElementById('sim-leverage');
        const simSl = document.getElementById('sim-sl');
        const simTp = document.getElementById('sim-tp');
        
        const entry = parseFloat(simEntry?.value);
        const balance = parseFloat(simBalance?.value);
        const riskPercent = parseFloat(simRiskSlider?.value);
        const leverage = parseFloat(simLeverageSlider?.value);
        let sl = parseFloat(simSl?.value);
        let tp = parseFloat(simTp?.value);
        const direction = selectedTrade.direction;
        
        if (isNaN(entry) || entry <= 0) return;
        const atr = selectedTrade.atr || getATR(entry);
        
        if (isNaN(sl) || sl <= 0) {
            sl = direction === 'LONG' ? entry - atr * 1.5 : entry + atr * 1.5;
            if (simSl) simSl.value = sl.toFixed(4);
        }
        if (isNaN(tp) || tp <= 0) {
            const riskDistance = Math.abs(entry - sl);
            tp = direction === 'LONG' ? entry + riskDistance * 3 : entry - riskDistance * 3;
            if (simTp) simTp.value = tp.toFixed(4);
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
            
        const riskPercentOfBalance = (riskAmount / balance) * 100;
        const gaugePercent = Math.min(100, (riskPercentOfBalance / 10) * 100);
        
        if (document.getElementById('risk-gauge-fill')) document.getElementById('risk-gauge-fill').style.width = `${gaugePercent}%`;
        if (document.getElementById('pos-units')) document.getElementById('pos-units').innerText = Number(positionUnits || 0).toFixed(4);
        if (document.getElementById('notional')) document.getElementById('notional').innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
        if (document.getElementById('margin')) document.getElementById('margin').innerText = `$${Number(marginRequired || 0).toFixed(2)}`;
        if (document.getElementById('liquidation')) document.getElementById('liquidation').innerText = `$${Number(liquidationPrice || 0).toFixed(4)}`;
        if (document.getElementById('rr-ratio')) document.getElementById('rr-ratio').innerText = "3.00";
        if (document.getElementById('suggested-amount')) document.getElementById('suggested-amount').innerText = `$${Number(notionalValue || 0).toFixed(2)}`;
    }

    // ========== Pre-fill Simulator ==========
    function prefillTradeSim(symbol, price, signalObj) {
        const direction = signalObj.direction || (signalObj.signal === 'SELL' ? 'SHORT' : 'LONG');
        const atr = signalObj.atr || getATR(price);
        selectedTrade = { 
            symbol, 
            entryPrice: price, 
            direction, 
            atr, 
            aiProb: signalObj.confidence || signalObj.ai_prob, 
            signal: signalObj.signal,
            signalId: signalObj.signal_id
        };
        
        if (document.getElementById('sim-symbol')) document.getElementById('sim-symbol').value = symbol;
        if (document.getElementById('sim-entry')) document.getElementById('sim-entry').value = price;
        
        const badge = document.getElementById('direction-badge');
        if (badge) {
            badge.className = `direction-badge ${direction === 'LONG' ? 'long' : 'short'}`;
            badge.innerText = direction;
        }
        
        if (document.getElementById('sim-sl')) document.getElementById('sim-sl').value = '';
        if (document.getElementById('sim-tp')) document.getElementById('sim-tp').value = '';
        
        updateSimulation();
    }

    // ========== Execute Trade (Firestore) ==========
    async function executeTrade() {
        if (!selectedTrade || !currentUser) return;
        const entry = parseFloat(document.getElementById('sim-entry').value);
        const sl = parseFloat(document.getElementById('sim-sl').value);
        const tp = parseFloat(document.getElementById('sim-tp').value);
        const riskPercent = parseFloat(document.getElementById('sim-risk-slider').value);
        const leverage = parseFloat(document.getElementById('sim-leverage').value);
        const positionUnits = parseFloat(document.getElementById('pos-units').innerText);
        const notional = parseFloat(document.getElementById('notional').innerText.replace('$', ''));
        
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
            userId: currentUser.uid,
            signalId: selectedTrade.signalId || null
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

    // ========== Trades Table (Firestore) ==========
    function renderTradesTable() {
        const tbody = document.getElementById('trades-tbody');
        if (!tbody) return;
        
        if (activeTrades.size === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No active trades</td></tr>';
            return;
        }
        let html = '';
        for (let [id, trade] of activeTrades.entries()) {
            const currentSignal = signalStore.signals[trade.symbol];
            const currentPrice = currentSignal ? currentSignal.entry : (trade.entryPrice || 0);
            let pnl = 0;
            const entryPrice = Number(trade.entryPrice || 0);
            const positionUnits = Number(trade.positionUnits || 0);
            if (trade.side === 'LONG') pnl = (currentPrice - entryPrice) * positionUnits;
            else pnl = (entryPrice - currentPrice) * positionUnits;
            
            const pnlClass = pnl >= 0 ? 'color: #00ff88;' : 'color: #ff3333;';
            html += `
                <tr data-trade-id="${id}">
                    <td>${trade.symbol}</td>
                    <td>${trade.side}</td>
                    <td>${entryPrice.toFixed(4)}</td>
                    <td>${Number(trade.stopLoss || 0).toFixed(4)}</td>
                    <td>${Number(trade.takeProfit || 0).toFixed(4)}</td>
                    <td style="${pnlClass}">$${Number(pnl).toFixed(2)}</td>
                    <td><button class="close-trade-btn btn-primary-glow" data-id="${id}" style="padding: 0.2rem 0.5rem; font-size: 0.8rem;">Close</button></td>
                </tr>
            `;
        }
        tbody.innerHTML = html;
        document.querySelectorAll('.close-trade-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                await closeTrade(btn.dataset.id);
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

    // ========== Auth Init ==========
    onAuthStateChanged(auth, async (user) => {
        let jwt = AuthManager.getToken();
        
        // Validate state
        const isAuth = AuthManager.isAuthenticated();
        const isTrialValid = AuthManager.isTrialValid();

        if (!user && !jwt) {
            wsManager.connect();
            return;
        }

        // Token refresh logic
        if (isAuth && !AuthManager.isTokenValid() && isTrialValid) {
            const refreshed = await AuthManager.refreshToken();
            if (refreshed) {
                jwt = AuthManager.getToken();
                console.log('Token successfully refreshed');
            } else {
                console.warn('Token refresh failed');
            }
        }

        if (!user && jwt) {
            try {
                const resp = await fetch('/auth/me', { 
                    method: 'GET',
                    credentials: 'include',
                    headers: { 
                        'Authorization': `Bearer ${jwt}`,
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    } 
                });
                if (!resp.ok) console.warn('Could not fetch /auth/me');
            } catch (err) {}
            wsManager.connect();
            return;
        }

        currentUser = user;
        try {
            await ensureUserDocument(user);
            initializeTrialCountdown(user.uid);
            unsubscribeSettings = subscribeUserSettings(user, (settings) => {
                const capital = settings.capital ?? 10000;
                const risk_pct = settings.risk_pct ?? 2.0;
                if (document.getElementById('user-capital')) document.getElementById('user-capital').value = capital;
                if (document.getElementById('risk-level')) document.getElementById('risk-level').value = risk_pct;
                if (document.getElementById('sim-balance')) document.getElementById('sim-balance').value = capital;
                if (document.getElementById('sim-risk-slider')) document.getElementById('sim-risk-slider').value = risk_pct;
                updateSimulation();
            });
            subscribeToTrades(user.uid);
            wsManager.connect();
        } catch (err) {
            console.error('Auth init error:', err);
        }
    });

    // ========== Event Listeners ==========
    document.getElementById('user-capital')?.addEventListener('input', (e) => {
        if (currentUser) updateUserSetting(currentUser, 'capital', parseFloat(e.target.value));
        const simBal = document.getElementById('sim-balance');
        if (simBal) simBal.value = e.target.value;
        updateSimulation();
    });

    document.getElementById('risk-level')?.addEventListener('change', (e) => {
        if (currentUser) updateUserSetting(currentUser, 'risk_pct', parseFloat(e.target.value));
        const simRisk = document.getElementById('sim-risk-slider');
        if (simRisk) simRisk.value = e.target.value;
        if (document.getElementById('risk-percent-display')) document.getElementById('risk-percent-display').innerText = e.target.value + '%';
        updateSimulation();
    });
    
    document.getElementById('sim-risk-slider')?.addEventListener('input', (e) => { 
        if (document.getElementById('risk-percent-display')) document.getElementById('risk-percent-display').innerText = e.target.value + '%'; 
        updateSimulation(); 
    });
    
    document.getElementById('sim-leverage')?.addEventListener('input', (e) => { 
        if (document.getElementById('leverage-display')) document.getElementById('leverage-display').innerText = e.target.value + 'x'; 
        updateSimulation(); 
    });
    
    ['sim-entry', 'sim-sl', 'sim-tp', 'sim-balance'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', updateSimulation);
    });
    
    document.getElementById('execute-trade-btn')?.addEventListener('click', executeTrade);
    
    // Logout
    document.getElementById('logoutBtn')?.addEventListener('click', async () => {
        try {
            if (unsubscribeTrades) unsubscribeTrades();
            if (unsubscribeSettings) unsubscribeSettings();
            AuthManager.clearToken();
            await logout();
            window.location.replace('/web/src/pages/index.html');
        } catch (err) {
            console.error(err);
        }
    });

    // Modal fixes
    const loginModal = document.getElementById('loginModal');
    document.querySelector('.close-modal')?.addEventListener('click', () => {
        if (loginModal) { loginModal.style.display = 'none'; unlockBodyScroll(); }
    });
    window.addEventListener('click', (e) => {
        if (e.target === loginModal) { loginModal.style.display = 'none'; unlockBodyScroll(); }
    });

    // ========== Mobile Menu Logic ==========
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const closeMenuBtn = document.getElementById('closeMenuBtn');
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');

    function toggleSidebar() {
        if (sidebar) sidebar.classList.toggle('-translate-x-full');
        if (sidebarOverlay) sidebarOverlay.classList.toggle('hidden');
    }

    if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', toggleSidebar);
    if (closeMenuBtn) closeMenuBtn.addEventListener('click', toggleSidebar);
    if (sidebarOverlay) sidebarOverlay.addEventListener('click', toggleSidebar);

    const mobileStatusBtn = document.getElementById('mobileStatusBtn');
    const statusDropdown = document.getElementById('statusDropdown');
    
    if (mobileStatusBtn && statusDropdown) {
        mobileStatusBtn.addEventListener('click', () => {
            statusDropdown.classList.toggle('hidden');
            statusDropdown.classList.toggle('flex');
        });
    }
});

