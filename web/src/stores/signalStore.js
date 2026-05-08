export class SignalStore {
    constructor() {
        this.signals = {}; // Map of pair -> signal object
        this.timeframe = localStorage.getItem('selected_timeframe') || '15m';
        this.listeners = [];
        this.searchQuery = "";
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
            if (!this.signals[pair] || JSON.stringify(this.signals[pair]) !== JSON.stringify(data)) {
                this.signals[pair] = {
                    pair: pair,
                    signal: data.signal || "WAITING",
                    entry: data.entry || 0,
                    sl: data.sl || 0,
                    tp: data.tp || 0,
                    status: data.status || "OPEN",
                    time: data.time || new Date().toISOString(),
                    timeframe: data.timeframe || this.timeframe,
                    confidence: data.ai_prob !== undefined ? data.ai_prob : (data.confidence || 0),
                    rr: data.rr || 0,
                    atr: data.atr || 0
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
