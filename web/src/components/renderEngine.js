import { AuthManager } from '../auth/authManager.js';

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
        const planType = AuthManager.getPlanType();
        const allTimeframes = ['1m','3m','5m','15m','30m','1h','4h','1d'];
        let allowedTimeframes = ['1m','3m','5m','15m','30m','1h','4h','1d'];
        
        // Use token-based hasAccess if available, else fallback to old logic
        const data = AuthManager.getUserData();
        if (data) {
            if (!AuthManager.hasAccess('extended_timeframes')) {
                allowedTimeframes = ['15m', '30m'];
                if (!allowedTimeframes.includes(this.signalStore.timeframe)) {
                    this.signalStore.timeframe = '15m';
                }
            }
        } else if (planType === 'free_trial' || planType === 'basic') {
            allowedTimeframes = ['15m', '30m'];
            // ensure current timeframe is valid
            if (!allowedTimeframes.includes(this.signalStore.timeframe)) {
                this.signalStore.timeframe = '15m';
            }
        }

        this.container.innerHTML = `
            <div class="flex flex-col gap-4 mb-6">
                <!-- Search Bar at the Top -->
                <div class="relative w-full">
                    <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"><i class="fas fa-search"></i></span>
                    <input type="text" id="signalSearch" placeholder="Search pairs... (e.g. BTC/USDT)" class="w-full bg-black/60 border border-white/10 rounded-xl pl-10 pr-4 py-3 min-h-[44px] text-white focus:outline-none focus:border-cyan focus:ring-1 focus:ring-cyan transition-all font-mono">
                </div>
                
                <!-- Timeframe Bar and Controls -->
                <div class="flex flex-wrap justify-between items-center gap-4 bg-black/40 p-2 rounded-xl border border-white/5">
                    <div class="flex gap-2 overflow-x-auto custom-scrollbar pb-1 w-full lg:w-auto" style="-webkit-overflow-scrolling: touch;">
                        ${allTimeframes.map(tf => {
                            const isAllowed = allowedTimeframes.includes(tf);
                            const isActive = this.signalStore.timeframe === tf;
                            const activeClasses = isActive ? 'bg-cyan text-black shadow-[0_0_10px_rgba(0,242,255,0.4)]' : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white';
                            const lockClasses = !isAllowed ? 'opacity-50 blur-[0.5px] pointer-events-none relative' : '';
                            const lockIcon = !isAllowed ? `<i class="fas fa-lock absolute -top-1 -right-1 text-[10px] text-orange"></i>` : '';
                            return `<button class="tf-btn px-4 py-1.5 min-h-[44px] rounded-lg text-sm font-bold transition-all whitespace-nowrap ${activeClasses} ${lockClasses}" data-tf="${tf}">${tf}${lockIcon}</button>`;
                        }).join('')}
                    </div>
                    <button id="exportCsvBtn" class="flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 px-4 py-1.5 min-h-[44px] rounded-lg text-sm font-bold transition-all whitespace-nowrap w-full lg:w-auto"><i class="fas fa-download"></i> Export CSV</button>
                </div>
            </div>
            
            <!-- Signal Grid -->
            <div id="signalCardsContainer" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"></div>
        `;

        // Event listeners
        const tfBtns = this.container.querySelectorAll('.tf-btn');
        tfBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                tfBtns.forEach(b => {
                    b.classList.remove('bg-cyan', 'text-black', 'shadow-[0_0_10px_rgba(0,242,255,0.4)]');
                    if (!b.classList.contains('pointer-events-none')) {
                        b.classList.add('bg-white/5', 'text-gray-400');
                    }
                });
                e.currentTarget.classList.remove('bg-white/5', 'text-gray-400');
                e.currentTarget.classList.add('bg-cyan', 'text-black', 'shadow-[0_0_10px_rgba(0,242,255,0.4)]');
                this.signalStore.setTimeframe(e.currentTarget.dataset.tf);
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
        
        const data = AuthManager.getUserData();
        let accessExpired = false;
        if (data) {
            accessExpired = !AuthManager.hasAccess('signals');
        } else {
            const subStatus = AuthManager.getSubscriptionStatus();
            accessExpired = subStatus === 'expired';
        }

        if (accessExpired) {
            container.innerHTML = `
                <div class="col-span-full flex flex-col items-center justify-center py-16 px-4 bg-red-900/10 border border-red-500/20 rounded-2xl">
                    <i class="fas fa-lock text-4xl text-red-500 mb-4"></i>
                    <h3 class="text-xl font-bold text-white mb-2">Access Expired</h3>
                    <p class="text-gray-400 text-center max-w-md">Your trial or subscription has expired. Please upgrade your plan to restore real-time fleet monitoring.</p>
                    <a href="/src/pages/pricing.html" class="mt-6 px-6 py-2 bg-gradient-to-r from-red-600 to-orange hover:from-red-500 hover:to-orange rounded font-bold text-white shadow-[0_0_15px_rgba(255,140,0,0.4)]">Upgrade Now</a>
                </div>
            `;
            return;
        }

        const planType = data ? data.plan_type : AuthManager.getPlanType();
        if (planType === 'free_trial' || planType === 'basic') {
            const allowed = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ARB/USDT', 'AAVE/USDT'];
            signals = signals.filter(s => allowed.includes(s.pair));
        }
        
        let html = '';
        signals.forEach(sig => {
            // Tailwind adapted styling for signal cards
            const sigClass = sig.signal.includes('BUY') ? 'bg-cyan/20 text-cyan border-cyan/50' : (sig.signal.includes('SELL') ? 'bg-red-500/20 text-red-400 border-red-500/50' : 'bg-gray-500/20 text-gray-400 border-gray-500/50');
            const confColor = sig.confidence > 0.8 ? 'text-green-400' : (sig.confidence > 0.5 ? 'text-orange' : 'text-red-400');
            const timeAgoStr = this.timeAgo(sig.time);
            const isRecent = timeAgoStr.includes('s') || timeAgoStr.includes('m ago');
            const rowAnim = isRecent ? 'animate-pulse shadow-[0_0_15px_rgba(255,255,255,0.1)]' : '';
            
            const sigJsonStr = JSON.stringify(sig).replace(/'/g, "\\'");
            
            html += `
                <div class="bg-black/60 border border-white/10 rounded-xl p-4 cursor-pointer hover:border-cyan/50 hover:bg-white/5 transition-all transform hover:-translate-y-1 overflow-hidden flex flex-col gap-3 ${rowAnim}" onclick='document.dispatchEvent(new CustomEvent("signalRowClicked", {detail: ${sigJsonStr}}))'>
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-lg font-mono text-white">${sig.pair}</span>
                        <span class="px-2 py-0.5 rounded text-xs font-bold border ${sigClass}">${sig.signal}</span>
                    </div>
                    
                    <div class="flex justify-between items-center text-sm">
                        <span class="text-gray-500">Entry</span>
                        <span class="font-mono text-white">${sig.entry ? sig.entry.toFixed(4) : '-'}</span>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-white/5">
                        <div class="flex flex-col">
                            <span class="text-gray-500">Stop Loss</span>
                            <span class="font-mono text-red-400">${sig.sl ? sig.sl.toFixed(4) : '-'}</span>
                        </div>
                        <div class="flex flex-col text-right">
                            <span class="text-gray-500">Take Profit</span>
                            <span class="font-mono text-green-400">${sig.tp ? sig.tp.toFixed(4) : '-'}</span>
                        </div>
                    </div>

                    <div class="flex justify-between items-end mt-1 pt-3 border-t border-white/5">
                        <div class="flex flex-col">
                            <span class="text-[10px] uppercase tracking-wider text-gray-500">AI Conviction</span>
                            <span class="font-bold font-mono ${confColor}">${(sig.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <span class="text-xs text-gray-500 font-mono">${timeAgoStr}</span>
                    </div>
                </div>
            `;
        });
        
        if (!html) {
            html = `<div class="col-span-full flex flex-col items-center justify-center py-12 px-4 bg-black/40 border border-white/5 rounded-xl">
                <i class="fas fa-search text-2xl text-gray-600 mb-3"></i>
                <span class="text-gray-400 text-sm">No active signals found matching your criteria.</span>
            </div>`;
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
