export class RenderEngine {
    constructor(containerId, signalStore) {
        this.container = document.getElementById(containerId);
        this.signalStore = signalStore;
        
        if (this.container) {
            this.initTable();
            this.signalStore.subscribe((signals) => this.renderSignals(signals));
        }
    }

    initTable() {
        this.container.innerHTML = `
            <div class="table-controls" style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem; gap: 1rem; flex-wrap: wrap;">
                <div class="timeframe-selector" style="display:flex; gap:0.5rem; background: rgba(0,0,0,0.5); padding:0.5rem; border-radius:12px; border:1px solid rgba(255,255,255,0.1);">
                    ${['1m','3m','5m','15m','30m','1h','4h','1d'].map(tf => 
                        `<button class="tf-btn ${this.signalStore.timeframe === tf ? 'active' : ''}" data-tf="${tf}">${tf}</button>`
                    ).join('')}
                </div>
                <div class="signal-filters" style="display:flex; gap:1rem;">
                    <input type="text" id="signalSearch" placeholder="Search pairs..." class="sidebar-input" style="width:200px; margin-bottom:0;">
                    <button id="exportCsvBtn" class="btn-primary-glow" style="padding: 0.5rem 1rem;"><i class="fas fa-download"></i> Export CSV</button>
                </div>
            </div>
            <div id="signalCardsContainer" class="signal-grid"></div>
        `;

        // Event listeners
        const tfBtns = this.container.querySelectorAll('.tf-btn');
        tfBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                tfBtns.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.signalStore.setTimeframe(e.target.dataset.tf);
            });
        });

        const searchInput = document.getElementById('signalSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.signalStore.setSearchQuery(e.target.value);
            });
        }
        
        const exportBtn = document.getElementById('exportCsvBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportCsv());
        }
    }

    renderSignals(signals) {
        const container = document.getElementById('signalCardsContainer');
        if (!container) return;
        
        let html = '';
        signals.forEach(sig => {
            const sigClass = sig.signal === 'BUY' ? 'signal-buy' : (sig.signal === 'SELL' ? 'signal-avoid' : 'signal-hold');
            const confColor = sig.confidence > 0.8 ? 'var(--success-green)' : (sig.confidence > 0.5 ? 'var(--warning-orange)' : 'var(--danger-red)');
            const timeAgoStr = this.timeAgo(sig.time);
            const isRecent = timeAgoStr.includes('s') || timeAgoStr.includes('m ago');
            const rowAnim = isRecent ? 'animation: pulse-border 2s;' : '';
            
            const sigJsonStr = JSON.stringify(sig).replace(/'/g, "\\'");
            
            html += `
                <div class="signal-card" style="${rowAnim}" onclick='document.dispatchEvent(new CustomEvent("signalRowClicked", {detail: ${sigJsonStr}}))'>
                    <div class="card-header">
                        <span class="symbol">${sig.pair}</span>
                        <span class="signal-tag ${sigClass}">${sig.signal}</span>
                    </div>
                    <div class="price-row">
                        <span>Timeframe</span>
                        <span style="color: var(--text-dim);">${sig.timeframe}</span>
                    </div>
                    <div class="price-row">
                        <span>Entry Price</span>
                        <span>${sig.entry ? sig.entry.toFixed(4) : '-'}</span>
                    </div>
                    <div class="price-row" style="margin-top: 0.8rem;">
                        <span style="color: ${confColor}; font-weight: bold;">AI Conviction: ${(sig.confidence * 100).toFixed(0)}%</span>
                        <span style="color: var(--text-dim);">${timeAgoStr}</span>
                    </div>
                    <div class="sl-tp">
                        <span>SL: ${sig.sl ? sig.sl.toFixed(4) : '-'}</span>
                        <span>TP: ${sig.tp ? sig.tp.toFixed(4) : '-'}</span>
                    </div>
                </div>
            `;
        });
        
        if (!html) {
            html = `<div style="grid-column: 1 / -1; text-align:center; padding: 2rem; color:var(--text-dim);">No signals matching criteria</div>`;
        }
        
        container.innerHTML = html;
    }
    
    timeAgo(dateStr) {
        if (!dateStr) return "Just now";
        try {
            const date = new Date(dateStr.replace(' UTC', 'Z'));
            const seconds = Math.floor((new Date() - date) / 1000);
            
            if (seconds < 60) return Math.floor(seconds) + "s ago";
            const interval = seconds / 60;
            if (interval < 60) return Math.floor(interval) + "m ago";
            const hrs = interval / 60;
            if (hrs < 24) return Math.floor(hrs) + "h ago";
            return Math.floor(hrs / 24) + "d ago";
        } catch {
            return "Just now";
        }
    }

    exportCsv() {
        const signals = Object.values(this.signalStore.signals);
        if (signals.length === 0) return;
        
        const headers = ["Pair", "Timeframe", "Signal", "Entry", "SL", "TP", "Confidence", "RR", "Status", "Time"];
        const rows = signals.map(s => [
            s.pair, s.timeframe, s.signal, s.entry, s.sl, s.tp, s.confidence, s.rr, s.status, s.time
        ]);
        
        let csvContent = "data:text/csv;charset=utf-8," 
            + headers.join(",") + "\n"
            + rows.map(e => e.join(",")).join("\n");
            
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `aegis_signals_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}
