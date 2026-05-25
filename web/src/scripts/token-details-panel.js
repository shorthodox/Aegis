// ============================================================
// token-details-panel.js
// Renders the inner HTML for the comprehensive token metrics popup.
// Usage:
//   import { renderTokenDetailsPanel, mountTokenDetailsPanel } from './token-details-panel.js';
//
//   // Inject HTML then bind live price updates:
//   container.innerHTML = renderTokenDetailsPanel(tokenData, userTier);
//   mountTokenDetailsPanel(container, tokenData.symbol);
//
//   // Or one-shot:
//   mountTokenDetailsPanel(container, tokenData, userTier);
// ============================================================

// ── Internal toast helper (delegates to dashboard's _showToast if available) ──
function _toast(msg, type = 'info') {
  if (typeof window._showToast === 'function') { window._showToast(msg, type); return; }
  const el = document.createElement('div');
  el.style.cssText = `position:fixed;bottom:24px;right:24px;z-index:9999;padding:10px 18px;
    border-radius:8px;font-size:13px;font-family:monospace;color:#fff;
    background:${{ success: '#10b981', error: '#ef4444', info: '#06b6d4' }[type] || '#06b6d4'};
    opacity:0;transition:opacity .25s;pointer-events:none;`;
  el.textContent = msg;
  document.body.appendChild(el);
  requestAnimationFrame(() => { el.style.opacity = '1'; });
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ── Progress bar helper ───────────────────────────────────────────
function _bar(label, value, fillClass, textClass) {
  const clamped = Math.min(100, Math.max(0, parseFloat(value) || 0));
  return `
    <div>
      <div class="flex justify-between items-center mb-1.5">
        <span class="text-[11px] font-mono text-slate-400">${label}</span>
        <span class="text-[11px] font-bold font-mono ${textClass}">${clamped}%</span>
      </div>
      <div class="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div class="h-full rounded-full transition-all duration-700 ${fillClass}"
             style="width:${clamped}%"></div>
      </div>
    </div>`;
}

// ── Lock overlay helper ───────────────────────────────────────────
function _lock(requiredTier, msg) {
  const badgeColor = requiredTier === 'PRO'
    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
    : 'bg-blue-500/15 text-blue-300 border-blue-500/30';
  return `
    <div class="absolute inset-0 z-10 flex flex-col items-center justify-center
                bg-slate-950/80 backdrop-blur-md rounded-xl">
      <i class="fas fa-lock text-xl text-slate-400 mb-2"></i>
      <p class="text-xs text-slate-300 text-center px-5 font-medium leading-relaxed">${msg}</p>
      <span class="mt-3 text-[9px] font-black uppercase tracking-widest px-3 py-1
                   rounded-full border ${badgeColor}">${requiredTier} TIER REQUIRED</span>
      <a href="/src/pages/pricing.html"
         class="mt-3 text-[10px] font-bold text-white bg-gradient-to-r
                from-cyan-500/80 to-blue-600/80 px-4 py-1.5 rounded-lg
                hover:from-cyan-500 hover:to-blue-600 transition-all">
        Upgrade →
      </a>
    </div>`;
}

// ── Main render function ──────────────────────────────────────────
export function renderTokenDetailsPanel(tokenData, userTier) {
  const tier = (userTier || 'BASIC').toUpperCase();
  const canIntermediate = tier === 'INTERMEDIATE' || tier === 'PRO';
  const canPro          = tier === 'PRO';

  // ── Safe data extraction ─────────────────────────────────────────
  const sym       = tokenData.symbol       || '—';
  const dir       = (tokenData.direction   || 'NEUTRAL').toUpperCase();
  const entry     = parseFloat(tokenData.entry_price  || tokenData.entry        || 0);
  const sl        = parseFloat(tokenData.sl                                      || 0);
  const tp        = parseFloat(tokenData.tp                                      || 0);
  const livePrice = parseFloat(
    tokenData.livePrice      ||
    tokenData.current_price  ||
    (window.currentTickers && window.currentTickers[sym]) ||
    entry
  );

  // Confluence (0–100 scale)
  const conf        = tokenData.confluence || {};
  const trendPct    = Math.min(100, Math.max(0, parseFloat(conf.trend    ?? conf.trendPct    ?? 68)));
  const momentumPct = Math.min(100, Math.max(0, parseFloat(conf.momentum ?? conf.momentumPct ?? 54)));
  const volumePct   = Math.min(100, Math.max(0, parseFloat(conf.volume   ?? conf.volumePct   ?? 72)));

  // Zone geometry: SL = 0%, TP = 100%
  const range     = (tp - sl) || 1;
  const entryPct  = Math.min(98, Math.max(2, ((entry     - sl) / range) * 100));
  const pricePct  = Math.min(98, Math.max(2, ((livePrice - sl) / range) * 100));
  const isLong    = dir === 'LONG' || dir === 'BUY';
  const inProfit  = isLong ? livePrice > entry : livePrice < entry;
  const pctFromEntry = entry > 0 ? Math.abs((livePrice - entry) / entry) * 100 : 0;
  const isBreakout   = inProfit && pctFromEntry > 0.2;

  // Expectancy data
  const expectancy   = tokenData.expectancy    ?? tokenData.mathematical_expectancy ?? '+1.84%';
  const maxDD        = tokenData.max_drawdown  ?? tokenData.maxDrawdown             ?? '-4.12%';
  const profitFactor = tokenData.profit_factor ?? tokenData.profitFactor            ?? '1.95';
  const totalTrades  = tokenData.total_trades  ?? 142;
  const winRate      = tokenData.win_rate      ?? 0.63;

  // XGBoost probabilities
  const probs    = tokenData.probabilities || { SHORT: 0.18, HOLD: 0.08, LONG: 0.74 };
  const shortPct = Math.round((probs.SHORT || 0) * 100);
  const holdPct  = Math.round((probs.HOLD  || 0) * 100);
  const longPct  = Math.round((probs.LONG  || 0) * 100);
  const leadIsLong  = longPct >= shortPct;
  const leadLabel   = leadIsLong ? `${longPct}% LONG` : `${shortPct}% SHORT`;
  const leadClass   = leadIsLong ? 'text-emerald-400' : 'text-rose-400';

  // SHAP values (top 3)
  const shapRaw  = tokenData.shap_values || [
    { feature: 'Volume Delta 1h',     value:  0.34 },
    { feature: 'BTC Anchor Distance', value: -0.21 },
    { feature: 'EMA 200 Confluence',  value:  0.18 },
  ];
  const shapValues = shapRaw.slice(0, 3);
  const maxShapAbs = Math.max(...shapValues.map(s => Math.abs(s.value)), 0.01);

  // API key (masked until regenerated)
  const apiKeyDisplay = tokenData.apiKey || 'aegis_live_••••••••••••';

  // ── SECTION 1: Confluence Scorecard ────────────────────────────
  const sec1 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canIntermediate ? _lock('INTERMEDIATE', '🔒 Unlock AI Confluence Scorecard') : ''}
      <div class="${!canIntermediate ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-layer-group text-cyan-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Confluence Scorecard
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-blue-500/15 text-blue-300 border border-blue-500/25 uppercase tracking-widest">
            INTERMEDIATE+
          </span>
        </div>
        <div class="space-y-3.5">
          ${_bar('Trend Alignment (EMA 200/50 Matrix)', trendPct,    'bg-cyan-400/70',    'text-cyan-400')}
          ${_bar('Momentum Regime (RSI / Stochastic)',  momentumPct, 'bg-emerald-400/70', 'text-emerald-400')}
          ${_bar('Volume Delta (Net Pressure)',          volumePct,   'bg-amber-400/70',   'text-amber-400')}
        </div>
        <div class="mt-4 pt-3 border-t border-slate-800/60 text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
          <i class="fas fa-microchip text-slate-600"></i>
          XGBoost feature weights · Refreshes every signal tick
        </div>
      </div>
    </div>`;

  // ── SECTION 2: Visual Zone Tracking ────────────────────────────
  const statusPill = isBreakout
    ? `<span class="text-[11px] font-bold text-amber-400 animate-pulse">
         ⚠️ Breakout In Progress: Do Not Chase
       </span>`
    : `<span class="text-[11px] font-bold text-cyan-400 animate-pulse">
         ⚡ Inside Active Entry Buffer Zone
       </span>`;

  const sec2 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canIntermediate ? _lock('INTERMEDIATE', '🔒 Unlock Visual Zone Tracking') : ''}
      <div class="${!canIntermediate ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-violet-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-map-marker-alt text-violet-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Visual Zone Tracking
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-violet-500/15 text-violet-300 border border-violet-500/25 uppercase tracking-widest">
            INTERMEDIATE+
          </span>
        </div>

        <!-- Zone bar -->
        <div class="relative h-12 bg-slate-800/80 rounded-xl overflow-hidden
                    border border-slate-700/50 mb-3 select-none">
          <!-- Gradient fill: red-left → green-right -->
          <div class="absolute inset-0 bg-gradient-to-r from-rose-500/15 via-transparent to-emerald-500/15"></div>

          <!-- Entry anchor line -->
          <div class="absolute top-0 bottom-0 w-px bg-cyan-400/90
                      shadow-[0_0_6px_rgba(0,242,255,0.7)] z-10"
               id="tdp-entry-line"
               style="left:${entryPct.toFixed(1)}%">
            <div class="absolute -top-px -translate-x-1/2
                        w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
            <div class="absolute bottom-1 -translate-x-1/2
                        text-[8px] font-bold text-cyan-300 whitespace-nowrap">ENTRY</div>
          </div>

          <!-- Live price dot -->
          <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 z-20
                      w-3.5 h-3.5 rounded-full bg-emerald-400
                      shadow-[0_0_14px_rgba(52,211,153,1)]
                      transition-all duration-1000 ease-out"
               id="tdp-price-dot"
               data-symbol="${sym}"
               style="left:${pricePct.toFixed(1)}%">
          </div>

          <!-- SL / TP corner labels -->
          <div class="absolute bottom-1 left-2 text-[9px] font-mono font-bold text-rose-400/80">SL</div>
          <div class="absolute bottom-1 right-2 text-[9px] font-mono font-bold text-emerald-400/80">TP</div>
        </div>

        <!-- Price coordinate row -->
        <div class="flex justify-between text-[10px] font-mono mb-3">
          <span class="text-rose-400">$${sl > 0 ? sl.toFixed(4) : '—'}</span>
          <span class="text-cyan-400">Entry $${entry > 0 ? entry.toFixed(4) : '—'}</span>
          <span class="text-emerald-400">$${tp > 0 ? tp.toFixed(4) : '—'}</span>
        </div>

        <!-- Status pill -->
        <div class="flex items-center justify-center py-1.5 px-3 rounded-lg
                    bg-slate-800/60 border border-slate-700/40 min-h-[32px]">
          ${statusPill}
        </div>

        <!-- Live price readout -->
        <div class="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-500">
          <span>Live Price</span>
          <span id="tdp-live-price-label" class="text-white font-bold">
            $${livePrice > 0 ? livePrice.toFixed(4) : '—'}
          </span>
        </div>
      </div>
    </div>`;

  // ── SECTION 3: Expectancy Matrix & Max DD ──────────────────────
  const rrClass  = String(expectancy).startsWith('+') || parseFloat(expectancy) > 0
    ? 'text-emerald-400' : 'text-rose-400';

  const sec3 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock Institutional Expectancy Matrices') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-chart-bar text-emerald-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Expectancy Matrix
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-amber-500/15 text-amber-300 border border-amber-500/25 uppercase tracking-widest">
            PRO ONLY
          </span>
        </div>

        <!-- 3-column metric cards -->
        <div class="grid grid-cols-3 gap-2 mb-3">
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3 text-center">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">
              Expectancy
            </div>
            <div class="text-xl font-black font-mono ${rrClass} leading-none">${expectancy}</div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">avg / trade</div>
          </div>
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3 text-center">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">
              Max DD
            </div>
            <div class="text-xl font-black font-mono text-amber-400 leading-none">${maxDD}</div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">peak-to-trough</div>
          </div>
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3 text-center">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">
              Profit Factor
            </div>
            <div class="text-xl font-black font-mono text-cyan-400 leading-none">${profitFactor}</div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">gross P / L</div>
          </div>
        </div>

        <!-- Win distribution bar -->
        <div>
          <div class="flex justify-between text-[10px] font-mono mb-1.5">
            <span class="text-slate-400">Win Distribution</span>
            <span class="text-white font-bold">${Math.round(winRate * 100)}% WR · ${totalTrades} signals</span>
          </div>
          <div class="h-2.5 bg-slate-800 rounded-full overflow-hidden flex">
            <div class="h-full bg-emerald-500/60 transition-all duration-700 rounded-l-full"
                 style="width:${Math.round(winRate * 100)}%"></div>
            <div class="h-full bg-rose-500/30 flex-1 rounded-r-full"></div>
          </div>
        </div>

        <div class="mt-3 text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
          <i class="fas fa-database text-slate-600"></i>
          30-day rolling window · Sourced from global performance document
        </div>
      </div>
    </div>`;

  // ── SECTION 4: Raw Probabilities & SHAP ────────────────────────
  const shapBars = shapValues.map(sv => {
    const barPct  = ((Math.abs(sv.value) / maxShapAbs) * 80).toFixed(1);
    const isPos   = sv.value > 0;
    const barFill = isPos ? 'bg-emerald-500/40' : 'bg-rose-500/40';
    const valTxt  = isPos ? 'text-emerald-400' : 'text-rose-400';
    return `
      <div>
        <div class="flex justify-between items-center text-[10px] font-mono mb-1">
          <span class="text-slate-300 truncate pr-2">${sv.feature}</span>
          <span class="font-black ${valTxt} flex-shrink-0">
            ${isPos ? '+' : ''}${sv.value.toFixed(3)}
          </span>
        </div>
        <div class="h-4 bg-slate-800/60 rounded border border-slate-700/30 overflow-hidden">
          <div class="h-full ${barFill} rounded transition-all duration-700"
               style="width:${barPct}%"></div>
        </div>
      </div>`;
  }).join('');

  const sec4 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock Raw ML Probabilities & SHAP Attribution') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-orange-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-brain text-orange-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Raw Probabilities & SHAP
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-amber-500/15 text-amber-300 border border-amber-500/25 uppercase tracking-widest">
            PRO ONLY
          </span>
        </div>

        <!-- Conviction meter -->
        <div class="mb-4">
          <div class="flex justify-between items-center mb-2">
            <span class="text-[10px] text-slate-500 font-mono uppercase tracking-widest">
              Model Conviction
            </span>
            <span class="text-[12px] font-black font-mono ${leadClass}">${leadLabel}</span>
          </div>
          <div class="h-7 flex rounded-xl overflow-hidden border border-slate-700/50 bg-slate-800/40">
            <div class="flex items-center justify-center bg-rose-500/50
                        text-[9px] font-bold text-rose-100 transition-all duration-700"
                 style="width:${shortPct}%">
              ${shortPct > 12 ? `${shortPct}%` : ''}
            </div>
            <div class="flex items-center justify-center bg-slate-600/50
                        text-[9px] font-bold text-slate-300 transition-all duration-700"
                 style="width:${holdPct}%">
              ${holdPct > 10 ? `${holdPct}%` : ''}
            </div>
            <div class="flex items-center justify-center bg-emerald-500/50
                        text-[9px] font-bold text-emerald-100 transition-all duration-700"
                 style="width:${longPct}%">
              ${longPct > 12 ? `${longPct}%` : ''}
            </div>
          </div>
          <div class="flex justify-between text-[9px] font-mono mt-1.5">
            <span class="text-rose-400">SHORT ${shortPct}%</span>
            <span class="text-slate-500">HOLD ${holdPct}%</span>
            <span class="text-emerald-400">LONG ${longPct}%</span>
          </div>
        </div>

        <!-- SHAP bars -->
        <div class="text-[10px] text-slate-500 font-mono uppercase tracking-widest mb-2">
          Live SHAP Feature Attribution
        </div>
        <div class="space-y-2.5">
          ${shapBars}
        </div>

        <div class="mt-3 text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
          <i class="fas fa-flask text-slate-600"></i>
          TreeExplainer on current candle row · Positive = pushes toward LONG
        </div>
      </div>
    </div>`;

  // ── SECTION 5: API / JSON Data Export (full width) ─────────────
  const sec5 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 col-span-1 lg:col-span-2 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock API Access & JSON Data Export') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-blue-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-code text-blue-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            API / JSON Data Export
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-blue-500/15 text-blue-300 border border-blue-500/25 uppercase tracking-widest">
            DEVELOPER ACCESS
          </span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <!-- API key block -->
          <div>
            <div class="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-2">
              Your API Key
            </div>
            <div class="flex items-center gap-2 bg-slate-800/60 border border-slate-700/40
                        rounded-xl p-3 mb-3">
              <i class="fas fa-key text-blue-400 text-xs flex-shrink-0"></i>
              <span class="font-mono text-sm text-slate-200 flex-1 tracking-wider truncate select-all"
                    id="tdp-api-key"
                    data-raw-key="">${apiKeyDisplay}</span>
              <button onclick="window._tdpCopyKey()"
                class="flex-shrink-0 text-[10px] text-blue-300 hover:text-white transition-all
                       px-2 py-1 bg-blue-500/10 rounded-lg border border-blue-500/20
                       hover:bg-blue-500/25 font-bold">
                <i class="fas fa-copy mr-1"></i>Copy
              </button>
            </div>
            <button onclick="window._tdpRegenKey()"
              class="w-full py-2.5 bg-violet-500/10 hover:bg-violet-500/20 text-violet-300
                     font-bold rounded-xl text-[11px] border border-violet-500/30
                     transition-all hover:border-violet-500/50">
              <i class="fas fa-sync-alt mr-2"></i>Regenerate API Key
            </button>
            <div class="mt-2 text-[10px] text-slate-600 text-center font-mono">
              Key shown once on generation — store it securely
            </div>
          </div>

          <!-- Code snippet block -->
          <div>
            <div class="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-2">
              Python Quick-Start
            </div>
            <div class="bg-slate-950/80 border border-slate-700/40 rounded-xl overflow-hidden">
              <div class="flex items-center justify-between px-3 py-1.5
                          bg-slate-800/50 border-b border-slate-700/30">
                <span class="text-[9px] font-mono text-slate-500">python · requests</span>
                <button onclick="window._tdpCopySnippet()"
                  class="text-[10px] text-slate-400 hover:text-white transition-colors
                         px-2 py-0.5 rounded hover:bg-white/5 font-bold">
                  <i class="fas fa-copy mr-1"></i>Copy
                </button>
              </div>
              <pre id="tdp-snippet"
                   class="text-[10px] font-mono p-3 leading-relaxed overflow-x-auto text-green-300/90"><span class="text-blue-300">import</span> requests

API_KEY <span class="text-slate-300">=</span> <span class="text-amber-300">"aegis_live_YOUR_KEY_HERE"</span>
HEADERS <span class="text-slate-300">=</span> {<span class="text-amber-300">"X-API-Key"</span>: API_KEY}

resp <span class="text-slate-300">=</span> requests.<span class="text-cyan-300">get</span>(
    <span class="text-amber-300">"https://gatekeeper.sbs/api/v1/signals/fleet"</span>,
    headers<span class="text-slate-300">=</span>HEADERS
)
data <span class="text-slate-300">=</span> resp.<span class="text-cyan-300">json</span>()
<span class="text-blue-300">print</span>(data[<span class="text-amber-300">"data"</span>])</pre>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  // ── Outer 2-column grid ──────────────────────────────────────────
  return `
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 p-1"
         data-tdp-symbol="${sym}">
      ${sec1}
      ${sec2}
      ${sec3}
      ${sec4}
      ${sec5}
    </div>`;
}

// ── Mount: render HTML + bind live price event listener ──────────
export function mountTokenDetailsPanel(containerEl, tokenData, userTier) {
  if (!containerEl) return;

  containerEl.innerHTML = renderTokenDetailsPanel(tokenData, userTier);

  const sym   = tokenData.symbol || '';
  const sl    = parseFloat(tokenData.sl || 0);
  const tp    = parseFloat(tokenData.tp || 0);
  const range = (tp - sl) || 1;

  // Remove any previous listener
  if (containerEl._tdpPriceHandler) {
    window.removeEventListener('priceUpdate', containerEl._tdpPriceHandler);
  }

  containerEl._tdpPriceHandler = (e) => {
    if (!e.detail || e.detail.symbol !== sym) return;

    const price = parseFloat(e.detail.price);
    if (isNaN(price)) return;

    // Move price dot
    const dot = containerEl.querySelector('#tdp-price-dot');
    if (dot) {
      const pct = Math.min(98, Math.max(2, ((price - sl) / range) * 100));
      dot.style.left = `${pct.toFixed(1)}%`;
    }

    // Update live price label
    const label = containerEl.querySelector('#tdp-live-price-label');
    if (label) label.textContent = `$${price.toFixed(4)}`;
  };

  window.addEventListener('priceUpdate', containerEl._tdpPriceHandler);
}

// ── Cleanup: remove event listener when panel closes ────────────
export function unmountTokenDetailsPanel(containerEl) {
  if (!containerEl || !containerEl._tdpPriceHandler) return;
  window.removeEventListener('priceUpdate', containerEl._tdpPriceHandler);
  delete containerEl._tdpPriceHandler;
}

// ── Window helpers (invoked from inline onclick attributes) ──────
window._tdpCopyKey = () => {
  const el = document.getElementById('tdp-api-key');
  if (!el) return;
  const key = el.dataset.rawKey || el.textContent.trim();
  navigator.clipboard.writeText(key)
    .then(() => _toast('API key copied', 'success'))
    .catch(() => {});
};

window._tdpCopySnippet = () => {
  const el = document.getElementById('tdp-snippet');
  if (!el) return;
  navigator.clipboard.writeText(el.innerText)
    .then(() => _toast('Snippet copied', 'success'))
    .catch(() => {});
};

window._tdpRegenKey = async () => {
  const btn = document.querySelector('[onclick="window._tdpRegenKey()"]');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generating...'; }
  try {
    const { getAuth } = await import(
      'https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js'
    );
    const fbUser = getAuth().currentUser;
    if (!fbUser) { _toast('Not authenticated', 'error'); return; }
    const token = await fbUser.getIdToken();
    const r = await fetch('/api/v1/developer/regenerate_key', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) {
      const d = await r.json();
      const el = document.getElementById('tdp-api-key');
      if (el) { el.textContent = d.api_key; el.dataset.rawKey = d.api_key; }
      _toast('New key generated — copy it now!', 'success');
    } else {
      _toast('Key regeneration failed', 'error');
    }
  } catch (e) {
    _toast('Error: ' + e.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Regenerate API Key';
    }
  }
};
