  let winnerProb = probs.HOLD || 0;
  let winnerColor = "text-gray-400";

  if ((probs.LONG || 0) > winnerProb) {
    winnerProb = probs.LONG;
    winnerColor = "text-emerald-400";
  }
  if ((probs.SHORT || 0) > winnerProb) {
    winnerProb = probs.SHORT;
    winnerColor = "text-rose-400";
  }
  let shapHTML = '';

  if (!shapList || shapList.length === 0) {
    shapHTML = `<div class="text-xs text-gray-500 italic py-2">Attribution map unavailable.</div>`;
    } else {
        // Max absolute impact for scaling
        const maxImpact = Math.max(...shapList.map(s => Math.abs(s.impact)), 0.01);
        
        shapHTML = shapList.map(s => {
            const isPositive = s.impact > 0;
            const barWidth = (Math.abs(s.impact) / maxImpact) * 100;
            const barColorClass = isPositive ? 'bg-emerald-500/20 border-emerald-500/30' : 'bg-rose-500/20 border-rose-500/30';
            const textColorClass = isPositive ? 'text-emerald-400' : 'text-rose-400';
            const sign = isPositive ? '+' : '-';
            
            return \`
                <div class="flex items-center gap-2 mb-2 w-full text-[10px] font-mono">
                    <div class="flex-1 text-right truncate text-gray-400 \${!isPositive ? textColorClass : ''}">\${!isPositive ? s.feature : ''}</div>
                    
                    <div class="w-1/3 flex items-center justify-center relative h-3 bg-black/30 rounded border border-white/5 overflow-hidden">
                        <div class="absolute h-full border-r \${!isPositive ? 'border-r-0 border-l' : ''} border-white/10 \${barColorClass} \${isPositive ? 'left-1/2' : 'right-1/2'}" style="width: \${barWidth / 2}%"></div>
                        <div class="absolute w-[1px] h-full bg-white/20 left-1/2"></div>
                    </div>
                    
                    <div class="flex-1 truncate text-gray-400 \${isPositive ? textColorClass : ''}">\${isPositive ? s.feature : ''}</div>
                </div>
            \`;
        }).join('');
    }

    container.innerHTML = \`
        <div class="mb-4">
            <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold flex items-center gap-2">
                <i class="fas fa-brain text-emerald-400"></i> Model Telemetry
            </h4>
        </div>
        
        <!-- Conviction Meter -->
        <div class="mb-4 bg-black/40 p-3 rounded-lg border border-white/5">
            <div class="flex justify-between items-center text-[10px] uppercase tracking-widest font-bold mb-2">
                <span class="text-gray-500">Conviction Vector</span>
                <span class="\${winnerColor}">\${winnerProb.toFixed(1)}% \${winner}</span>
            </div>
            <div class="w-full h-1.5 rounded-full overflow-hidden flex bg-black/50 shadow-inner">
                <div class="h-full bg-rose-500 transition-all duration-500" style="width: \${probs.SHORT || 0}%"></div>
                <div class="h-full bg-gray-500 transition-all duration-500" style="width: \${probs.HOLD || 0}%"></div>
                <div class="h-full bg-emerald-500 transition-all duration-500" style="width: \${probs.LONG || 0}%"></div>
            </div>
            <div class="flex justify-between text-[8px] mt-1 text-gray-500 font-mono">
                <span>SH \${(probs.SHORT || 0).toFixed(0)}%</span>
                <span>HD \${(probs.HOLD || 0).toFixed(0)}%</span>
                <span>LN \${(probs.LONG || 0).toFixed(0)}%</span>
            </div>
        </div>

        <!-- Live Logic Engine (Micro-SHAP) -->
        <div class="bg-black/40 p-3 rounded-lg border border-white/5 flex-1 flex flex-col justify-center">
            <div class="text-[9px] text-gray-500 uppercase tracking-widest font-bold mb-3 text-center border-b border-white/5 pb-2">
                Live Logic Engine
            </div>
            <div class="flex flex-col w-full justify-center">
                \${shapHTML}
            </div>
        </div>
    \`;
}

// ============================================================
// API PORTABILITY & DEVELOPER PORTAL
// ============================================================

window.regenerateApiKey = async function(symbol) {
    try {
        let token = localStorage.getItem('access_token') || localStorage.getItem('authToken');
        if (!token && typeof AuthManager !== 'undefined') {
            token = AuthManager.getToken();
        }
        
        if (!token) {
            alert('You must be logged in to regenerate API keys.');
            return;
        }

        const btn = document.getElementById('btn-regen-key');
        if (btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        const response = await fetch('/api/v1/developer/regenerate_key', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to generate key. Are you on the PRO tier?');
        }

        const data = await response.json();
        const apiKey = data.api_key;

        // Show it to the user
        const keyDisplay = document.getElementById('dev-api-key-display');
        if (keyDisplay) {
            keyDisplay.value = apiKey;
            keyDisplay.type = 'text';
        }

        alert(`Copy your API key now. This is the only time it will be shown:\n\n${apiKey}`);

        if (btn) btn.innerHTML = '<i class="fas fa-sync-alt"></i> Regenerate API Key';
    } catch (e) {
        console.error(e);
        alert(e.message);
        const btn = document.getElementById('btn-regen-key');
        if (btn) btn.innerHTML = '<i class="fas fa-sync-alt"></i> Regenerate API Key';
    }
};

window.copyEndpointUrl = function(symbol) {
    const url = `${window.location.origin}/api/v1/signals/fleet?symbol=${symbol}`;
    navigator.clipboard.writeText(url).then(() => {
        alert('Endpoint URL copied to clipboard!');
    });
};

function renderDeveloperPortal(signal) {
    const container = document.getElementById('sd-developer-portal');
    if (!container) return;

    const tier = getUserTier();

    if (tier !== 'PRO') {
        container.innerHTML = `
            <div class="relative w-full h-full min-h-[140px] rounded-xl overflow-hidden group cursor-pointer" onclick="window.location.href='/web/src/pages/pricing.html'">
                <!-- Blurred background metrics -->
                <div class="absolute inset-0 p-4 blur-[4px] opacity-40 bg-black/50 flex flex-col gap-3 pointer-events-none">
                    <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold flex items-center gap-2">
                        <i class="fas fa-code text-purple-400"></i> API Portability
                    </h4>
                    <input type="password" value="aegis_live_fakekey_xxxxxxxxxxxxxxxx" class="w-full bg-black/50 border border-white/10 rounded px-3 py-2 text-xs font-mono text-gray-500" disabled />
                    <div class="flex gap-2">
                        <button class="flex-1 bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded py-2 text-[10px] font-bold"><i class="fas fa-sync-alt"></i> REGENERATE</button>
                        <button class="flex-1 bg-white/5 text-gray-400 border border-white/10 rounded py-2 text-[10px] font-bold"><i class="fas fa-copy"></i> COPY ENDPOINT</button>
                    </div>
                </div>
                
                <!-- Lock Overlay -->
                <div class="absolute inset-0 flex flex-col items-center justify-center bg-black/40 bg-gradient-to-t from-black/90 to-transparent backdrop-blur-sm z-10 transition-all group-hover:bg-black/50">
                    <i class="fas fa-lock text-gray-300 text-2xl mb-2 group-hover:text-purple-400 transition-colors"></i>
                    <span class="text-[10px] text-purple-400 font-bold tracking-widest text-center px-4 leading-relaxed group-hover:text-purple-300">
                        Unlock Developer JSON Endpoints & Webhook Integrations with Pro Tier
                    </span>
                </div>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <h4 class="text-xs uppercase tracking-widest text-gray-400 font-bold mb-3 flex items-center gap-2">
            <i class="fas fa-code text-purple-400"></i> API Portability & Developer Access
        </h4>
        <div class="flex flex-col gap-3">
            <div class="relative">
                <input type="password" id="dev-api-key-display" value="aegis_live_••••••••••••••••"
                    class="w-full bg-black/50 border border-white/10 rounded px-3 py-2 text-xs font-mono text-gray-300 outline-none focus:border-purple-500/50 transition-colors" readonly />
            </div>
            <div class="flex gap-2">
                <button id="btn-regen-key" onclick="window.regenerateApiKey('${signal.symbol}')"
                    class="flex-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30 rounded py-2 text-[10px] font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-2">
                    <i class="fas fa-sync-alt"></i> Regenerate API Key
                </button>
                <button onclick="window.copyEndpointUrl('${signal.symbol}')"
                    class="flex-1 bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 rounded py-2 text-[10px] font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-2">
                    <i class="fas fa-copy"></i> Copy Endpoint URL
                </button>
            </div>
            <div class="text-[9px] text-gray-500 font-mono mt-1 text-center">
                Requires header: <span class="text-purple-400">X-API-Key</span>
            </div>
        </div>
    `;
}
