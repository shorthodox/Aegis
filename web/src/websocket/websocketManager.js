import { AuthManager } from '../auth/authManager.js';

export class WebSocketManager {
    constructor(url, signalStore, onDashboardUpdate) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectDelay = 30000;
        this.signalStore = signalStore;
        this.onDashboardUpdate = onDashboardUpdate;
        this.pingInterval = null;
        
        // 3. Cleanup Routine
        this.cleanupInterval = setInterval(() => this.cleanupExpiredSignals(), 30000); // Check every 30 seconds
    }

    isSignalExpired(signal) {
        if (!signal || !signal.time) return false;
        const sigTime = new Date(signal.time).getTime();
        const now = Date.now();
        let durationMs = 3600000; // default 1 hour
        if (signal.timeframe) {
            const tf = signal.timeframe;
            if (tf.endsWith('m')) durationMs = parseInt(tf) * 60000;
            else if (tf.endsWith('h')) durationMs = parseInt(tf) * 3600000;
            else if (tf.endsWith('d')) durationMs = parseInt(tf) * 86400000;
        }
        return now > (sigTime + durationMs);
    }

    handleExpirations(signalsObj) {
        if (!signalsObj) return signalsObj;
        
        const validSignals = {};
        let historyChanged = false;
        let history = [];
        try {
            const hStr = localStorage.getItem('aegis_signal_history');
            if (hStr) history = JSON.parse(hStr);
        } catch(e) {}

        for (const [pair, sig] of Object.entries(signalsObj)) {
            // 1. Expiration Logic
            if (this.isSignalExpired(sig)) {
                // 2. JSON State Management: if expired, mark history
                history.forEach(h => {
                    if ((h.symbol === pair || h.symbol === sig.pair) && h.status !== 'signal expired') {
                        h.status = 'signal expired';
                        historyChanged = true;
                    }
                });
            } else {
                validSignals[pair] = sig;
            }
        }
        
        if (historyChanged) {
            localStorage.setItem('aegis_signal_history', JSON.stringify(history));
            if (typeof window.updateSignalHistoryUI === 'function') {
                window.updateSignalHistoryUI();
            }
        }
        
        return validSignals;
    }

    cleanupExpiredSignals() {
        if (!this.signalStore || !this.signalStore.signals) return;
        
        let changed = false;
        let historyChanged = false;
        let history = [];
        try {
            const hStr = localStorage.getItem('aegis_signal_history');
            if (hStr) history = JSON.parse(hStr);
        } catch(e) {}

        for (const [pair, sig] of Object.entries(this.signalStore.signals)) {
            if (this.isSignalExpired(sig)) {
                delete this.signalStore.signals[pair];
                changed = true;
                
                history.forEach(h => {
                    if ((h.symbol === pair || h.symbol === sig.pair) && h.status !== 'signal expired') {
                        h.status = 'signal expired';
                        historyChanged = true;
                    }
                });
            }
        }
        
        if (historyChanged) {
            localStorage.setItem('aegis_signal_history', JSON.stringify(history));
            if (typeof window.updateSignalHistoryUI === 'function') {
                window.updateSignalHistoryUI();
            }
        }
        
        if (changed) {
            this.signalStore.notify();
            // 4. UI Feedback
            if (this.onDashboardUpdate) {
                this.onDashboardUpdate({ type: 'dashboard_update', signals: this.signalStore.signals });
            }
        }
    }

    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log("WebSocket connected");
            this.reconnectAttempts = 0;
            const token = AuthManager.getToken();
            if (token) {
                this.ws.send(JSON.stringify({ token: token, timeframe: this.signalStore.timeframe }));
            }
            
            this.pingInterval = setInterval(() => {
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: "ping" }));
                }
            }, 15000);
            
            const statusText = document.getElementById('ws-status-text');
            const statusDot = document.getElementById('ws-status-dot');
            if (statusText) statusText.textContent = 'Connected Live';
            if (statusDot) statusDot.className = 'ws-dot pulse';
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                
                // Step 4: Error Logging
                console.log(`[WS Manager] Message Received. Type: ${msg.type || 'NO_TYPE'}`);
                if (msg.data) console.log(`[WS Manager] Data keys: ${Object.keys(msg.data)}`);
                if (msg.tickers) console.log(`[WS Manager] Tickers count: Object.keys(msg.tickers).length`);
                
                if (msg.type === "signal_update") {
                    if (this.isSignalExpired(msg.data)) {
                        this.handleExpirations({ [msg.data.pair]: msg.data });
                        // Ensure cleanup is synced
                        this.cleanupExpiredSignals();
                    } else {
                        this.signalStore.updateSignal(msg.data);
                    }
                } else if (msg.type === "dashboard_update" || (!msg.type && (msg.signals || msg.timeframes))) {
                    // Unified backend fallback
                    if (msg.signals) {
                        msg.signals = this.handleExpirations(msg.signals);
                        this.signalStore.updateMultiple(msg.signals);
                        // Also sweep any old signals currently in the store
                        this.cleanupExpiredSignals();
                    }
                    if (msg.timeframes) {
                        // timeframes is nested: { symbol: { tf: sig, ... } }
                        this.signalStore.updateMultiple(msg.timeframes);
                        this.cleanupExpiredSignals();
                    }
                    if (this.onDashboardUpdate) {
                        // Pass through the message to the dashboard callback
                        this.onDashboardUpdate(msg);
                    }
                }
            } catch (err) {
                console.error("WS Parse error", err);
            }
        };

        this.ws.onclose = () => {
            console.log("WebSocket disconnected");
            clearInterval(this.pingInterval);
            if (this.cleanupInterval) clearInterval(this.cleanupInterval);
            const statusText = document.getElementById('ws-status-text');
            const statusDot = document.getElementById('ws-status-dot');
            if (statusText) statusText.textContent = 'Disconnected';
            if (statusDot) statusDot.className = 'ws-dot disconnected';
            this.scheduleReconnect();
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error", err);
            this.ws.close();
        };
    }

    scheduleReconnect() {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
        this.reconnectAttempts++;
        console.log(`Reconnecting in ${delay}ms...`);
        const statusText = document.getElementById('ws-status-text');
        if (statusText) statusText.textContent = `Reconnecting... (${this.reconnectAttempts})`;
        setTimeout(() => this.connect(), delay);
    }
}
