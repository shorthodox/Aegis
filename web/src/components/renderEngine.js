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
                <div class="timeframe-selector" style="display:flex; gap:0.5rem; background:#0f141e; padding:0.5rem; border-radius:12px; border:1px solid #1a1f2e;">
                    ${['1m','3m','5m','15m','30m','1h','4h','1d'].map(tf => 
                        `<button class="tf-btn ${this.signalStore.timeframe === tf ? 'active' : ''}" data-tf="${tf}">${tf}</button>`
                    ).join('')}
                </div>
                <div class="signal-filters" style="display:flex; gap:1rem;">
                    <input type="text" id="signalSearch" placeholder="Search pairs..." class="sidebar-input" style="width:200px; margin-bottom:0;">
                    <button id="exportCsvBtn" class="btn-primary-glow" style="padding: 0.5rem 1rem;"><i class="fas fa-download"></i> Export CSV</button>
                </div>
            </div>
            <div class="table-responsive" style="overflow-x: auto; background:#0a0a0c; border:1px solid #1a1f2e; border-radius:12px;">
                <table class="professional-table" style="width:100%; border-collapse:collapse; text-align:left;">
                    <thead style="background:#0f141e; border-bottom:1px solid #1a1f2e;">
                        <tr>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Pair</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Timeframe</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Signal</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Entry</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Stop Loss</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Take Profit</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Confidence</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">RR</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Status</th>
                            <th style="padding:1rem; color:var(--dim); font-size:0.85rem; text-transform:uppercase; font-weight:600;">Age</th>
                        </tr>
                    </thead>
                    <tbody id="signalTableBody">
                    </tbody>
                </table>
            </div>
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
        const tbody = document.getElementById('signalTableBody');
        if (!tbody) return;
        
        let html = '';
        signals.forEach(sig => {
            const sigClass = sig.signal === 'BUY' ? 'color: #00ff88;' : (sig.signal === 'SELL' ? 'color: #ff3333;' : 'color: #94a3b8;');
            const statClass = sig.status === 'OPEN' ? 'color: #94a3b8;' : (sig.status === 'TP HIT' ? 'color: #00ff88;' : 'color: #ff3333;');
            const confColor = sig.confidence > 0.8 ? '#00ff88' : (sig.confidence > 0.5 ? '#ffaa00' : '#ff3333');
            
            // Format time ago dynamically
            const timeAgoStr = this.timeAgo(sig.time);
            const isRecent = timeAgoStr.includes('s') || timeAgoStr.includes('m ago');
            const rowAnim = isRecent ? 'animation: pulse-border 2s;' : '';
            
            const sigJsonStr = JSON.stringify(sig).replace(/'/g, "\\'");
            html += `
                <tr id="row-${sig.pair.replace('/', '-')}" style="border-bottom:1px solid #1a1f2e; ${rowAnim} transition: background 0.3s; cursor:pointer;" onmouseover="this.style.background='#0f141e'" onmouseout="this.style.background='transparent'" onclick='document.dispatchEvent(new CustomEvent("signalRowClicked", {detail: ${sigJsonStr}}))'>
                    <td style="padding:1rem; font-weight:bold; color:#fff;">${sig.pair}</td>
                    <td style="padding:1rem; color:var(--dim);">${sig.timeframe}</td>
                    <td style="padding:1rem; font-weight:bold; ${sigClass}">${sig.signal}</td>
                    <td style="padding:1rem; color:#fff;">${sig.entry ? sig.entry.toFixed(4) : '-'}</td>
                    <td style="padding:1rem; color:#ff3333;">${sig.sl ? sig.sl.toFixed(4) : '-'}</td>
                    <td style="padding:1rem; color:#00ff88;">${sig.tp ? sig.tp.toFixed(4) : '-'}</td>
                    <td style="padding:1rem;">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="color:${confColor}; width:35px;">${(sig.confidence * 100).toFixed(0)}%</span>
                            <div style="width:60px; height:6px; background:#1a1f2e; border-radius:3px; overflow:hidden;">
                                <div style="width:${sig.confidence*100}%; height:100%; background:${confColor}; transition: width 0.5s;"></div>
                            </div>
                        </div>
                    </td>
                    <td style="padding:1rem; color:#fff;">${sig.rr ? sig.rr.toFixed(2) : '-'}</td>
                    <td style="padding:1rem; ${statClass}">${sig.status}</td>
                    <td style="padding:1rem; color:var(--dim); font-size:0.85rem;">${timeAgoStr}</td>
                </tr>
            `;
        });
        
        if (!html) {
            html = `<tr><td colspan="10" style="text-align:center; padding: 2rem; color:var(--dim);">No signals matching criteria</td></tr>`;
        }
        
        tbody.innerHTML = html;
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