import { AuthManager } from '../auth/authManager.js';

export class SignalStore {
    constructor() {
        this.signals = {}; // Map of pair -> signal object
        this.timeframe = localStorage.getItem('selected_timeframe') || '15m';
        this.listeners = [];
        this.searchQuery = "";
        this.fetchInterval = null;
        this.startAutoFetch();
    }

    startAutoFetch() {
        // Fetch signals every 2 seconds from public endpoint
        this.fetchInterval = setInterval(() => this.fetchLiveSignals(), 2000);
        // Also fetch immediately on init
        this.fetchLiveSignals();
    }

    async fetchLiveSignals() {
        try {
            const headers = {
                'Content-Type': 'application/json'
            };
            const authHeader = AuthManager.getAuthHeader();
            if (authHeader) {
                headers['Authorization'] = authHeader;
            }

            // Using public endpoint with Authorization header
            const response = await fetch('/api/public/signals', {
                method: 'GET',
                headers: headers
            });
            
            if (response.status === 401 || response.status === 403) {
                // Dispatch event to show expired UI
                window.dispatchEvent(new Event('aegis:auth:expired'));
                return;
            }

            if (response.ok) {
                const data = await response.json();
                
                if (data.expired) {
                    window.dispatchEvent(new Event('aegis:auth:expired'));
                    return;
                }

                if (data.signals && typeof data.signals === 'object') {
                    this.updateMultiple(data.signals);
                }
                // Notify listeners of warmup status
                if (data.warmup !== undefined) {
                    const event = new CustomEvent('warmupUpdate', { detail: { warmup: data.warmup } });
                    document.dispatchEvent(event);
                }
            }
        } catch (err) {
            console.debug('Failed to fetch live signals:', err);
        }
    }

    stopAutoFetch() {
        if (this.fetchInterval) {
            clearInterval(this.fetchInterval);
            this.fetchInterval = null;
        }
    }

    setTimeframe(tf) {
        this.timeframe = tf;
        localStorage.setItem('selected_timeframe', tf);
        this.notify();
    }

    setSearchQuery(query) {
        this.searchQuery = query.toLowerCase();
        this.notify();
    }

    updateSignal(payload) {
        this.signals[payload.pair] = payload;
        this.notify();
    }
    
    updateMultiple(signalsObj) {
        let changed = false;
        for (const [pair, data] of Object.entries(signalsObj)) {
            // Support nested timeframe map: { symbol: { '1m': {...}, '1h': {...} } }
            if (data && typeof data === 'object' && Object.keys(data).every(k => /^(1m|5m|15m|30m|1h|4h|1d|1w)$/.test(k))) {
                for (const [tf, tfdata] of Object.entries(data)) {
                    const key = `${pair}_${tf}`;
                    const payload = tfdata || {};
                    const entry = {
                        pair: key,
                        signal: payload.signal || "WAITING",
                        entry: payload.price || payload.entry_price || payload.entry || 0,
                        sl: payload.sl || 0,
                        tp: payload.tp || 0,
                        status: payload.status || "OPEN",
                        time: payload.time || new Date().toISOString(),
                        timeframe: tf,
                        confidence: payload.ai_prob !== undefined ? payload.ai_prob : (payload.confidence || 0),
                        rr: payload.rr || 0,
                        atr: payload.atr || 0,
                        signal_status: payload.signal_status || 'ACTIVE',
                        candles_confirmed: payload.candles_confirmed || 0,
                        reversal_score: payload.reversal_score || 0,
                        reversal_signals: payload.reversal_signals || [],
                        direction: payload.direction || 'NEUTRAL',
                    };
                    if (!this.signals[key] || JSON.stringify(this.signals[key]) !== JSON.stringify(entry)) {
                        this.signals[key] = entry;
                        changed = true;
                    }
                }
                continue;
            }

            if (!this.signals[pair] || JSON.stringify(this.signals[pair]) !== JSON.stringify(data)) {
                this.signals[pair] = {
                    pair: pair,
                    signal: data.signal || "WAITING",
                    entry: data.price || data.entry_price || data.entry || 0,
                    sl: data.sl || 0,
                    tp: data.tp || 0,
                    status: data.status || "OPEN",
                    time: data.time || new Date().toISOString(),
                    timeframe: data.timeframe || this.timeframe,
                    confidence: data.ai_prob !== undefined ? data.ai_prob : (data.confidence || 0),
                    rr: data.rr || 0,
                    atr: data.atr || 0,
                    signal_status: data.signal_status || 'ACTIVE',
                    candles_confirmed: data.candles_confirmed || 0,
                    reversal_score: data.reversal_score || 0,
                    reversal_signals: data.reversal_signals || [],
                    direction: data.direction || 'NEUTRAL',
                };
                changed = true;
            }
        }
        if (changed) this.notify();
    }

    subscribe(callback) {
        this.listeners.push(callback);
    }

    notify() {
        let sorted = Object.values(this.signals).sort((a, b) => {
            return new Date(b.time) - new Date(a.time);
        });

        if (this.searchQuery) {
            sorted = sorted.filter(s => s.pair.toLowerCase().includes(this.searchQuery));
        }

        this.listeners.forEach(cb => cb(sorted));
    }
}
