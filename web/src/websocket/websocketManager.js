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
                if (msg.type === "signal_update") {
                    this.signalStore.updateSignal(msg.data);
                } else if (msg.type === "dashboard_update" || (!msg.type && msg.signals)) {
                    // Unified backend fallback
                    if (msg.signals) {
                        this.signalStore.updateMultiple(msg.signals);
                    }
                    if (this.onDashboardUpdate) {
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
