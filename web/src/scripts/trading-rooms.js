// trading-rooms.js — owns Terminal, Analytics, and Signal History rooms
// Provides window globals consumed by gatekeeper.js:
//   window.renderTrades(trades)       — called on every WS tick with live trade data
//   window.addToSignalHistory(signal) — called when a signal card is selected
//   window.prefillFromSignal(signal)  — fills the terminal cockpit from a signal
//   window.closeTrade(tradeId)        — close a paper or live trade
//   window.trFromHistory(idx, mode)   — trigger trade from signal history row

import { AuthManager } from '../auth/authManager.js';

// ── State ────────────────────────────────────────────────────────────────────
let _paperTrades = [];        // local paper-only trades
let _liveTradesCache = [];    // last live trades received from WS
let _signalHistory = [];      // loaded from / saved to localStorage

// ── Auth helper ──────────────────────────────────────────────────────────────
async function _getToken() {
  try {
    const { getAuth } = await import('https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js');
    const fbUser = getAuth().currentUser;
    if (fbUser) {
      const tok = await fbUser.getIdToken();
      if (typeof AuthManager !== 'undefined') AuthManager.setToken(tok);
      return tok;
    }
  } catch (_) {}
  if (typeof AuthManager !== 'undefined') return AuthManager.getToken();
  return localStorage.getItem('access_token') || localStorage.getItem('authToken') || null;
}

// ── Toast ────────────────────────────────────────────────────────────────────
function _toast(msg, type = 'info') {
  const existing = document.getElementById('_tr-toast');
  if (existing) existing.remove();
  const c = { success: '#10b981', error: '#ef4444', info: '#06b6d4' };
  const ic = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
  const el = document.createElement('div');
  el.id = '_tr-toast';
  el.style.cssText = `position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;display:flex;align-items:center;gap:10px;padding:14px 20px;border-radius:12px;background:#111827;border:1px solid ${c[type]};color:${c[type]};font-size:.9rem;font-weight:600;box-shadow:0 0 20px ${c[type]}40;max-width:380px;`;
  el.innerHTML = `<i class="fas ${ic[type]}"></i><span>${msg}</span>`;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

// ── DOM helpers ──────────────────────────────────────────────────────────────
function _setText(id, val) { const e = document.getElementById(id); if (e) e.textContent = val; }
function _setVal(id, val) {
  const e = document.getElementById(id);
  if (!e) return;
  if (e.tagName === 'INPUT' || e.tagName === 'SELECT') e.value = val;
  else e.textContent = val;
}

// ── Position sizing ──────────────────────────────────────────────────────────
function _calcPos({ capital, riskPct, entry, sl, tp, leverage }) {
  const slDist = Math.abs(entry - sl);
  if (!slDist || !entry) return {};
  const riskAmt = capital * (riskPct / 100);
  const posUnits = riskAmt / slDist;
  const notional = posUnits * entry;
  const margin = notional / leverage;
  const direction = sl < entry ? 'LONG' : 'SHORT';
  const liqPrice = direction === 'LONG'
    ? entry - (entry / leverage) + (sl / leverage)
    : entry + (entry / leverage) - (sl / leverage);
  const rrRatio = tp ? Math.abs(tp - entry) / slDist : 0;
  return { posUnits, notional, margin, liqPrice, rrRatio, direction };
}

function _refreshCalc() {
  const entry    = parseFloat(document.getElementById('tr-entry')?.value);
  const sl       = parseFloat(document.getElementById('tr-sl')?.value);
  const tp       = parseFloat(document.getElementById('tr-tp')?.value);
  const capital  = parseFloat(document.getElementById('tr-capital')?.value || 10000);
  const riskPct  = parseFloat(document.getElementById('tr-risk')?.value || 2);
  const leverage = parseFloat(document.getElementById('tr-leverage')?.value || 1);

  if (!entry || !sl) {
    _setText('tr-pos-units', '—'); _setText('tr-notional', '—');
    _setText('tr-margin', '—'); _setText('tr-liqprice', '—');
    _setText('tr-rr', '—'); _setText('tr-suggested', '$0.00');
    return;
  }

  const { posUnits, notional, margin, liqPrice, rrRatio, direction } = _calcPos({ capital, riskPct, entry, sl, tp, leverage });

  _setText('tr-pos-units',  posUnits  ? posUnits.toFixed(4)   : '—');
  _setText('tr-notional',   notional  ? `$${notional.toFixed(2)}`  : '—');
  _setText('tr-margin',     margin    ? `$${margin.toFixed(2)}`    : '—');
  _setText('tr-liqprice',   liqPrice  ? `$${liqPrice.toFixed(4)}`  : '—');
  _setText('tr-rr',         rrRatio   ? `${rrRatio.toFixed(2)}:1`  : '—');
  _setText('tr-suggested',  notional  ? `$${notional.toFixed(2)}`  : '$0.00');

  const badge = document.getElementById('tr-direction-badge');
  if (badge && direction) {
    badge.textContent = direction;
    badge.className = direction === 'LONG'
      ? 'text-xs px-2 py-1 rounded font-bold bg-green-500/20 text-green-400 border border-green-500/30'
      : 'text-xs px-2 py-1 rounded font-bold bg-red-500/20 text-red-400 border border-red-500/30';
  }
}

// ── Prefill terminal from a signal ───────────────────────────────────────────
function prefillFromSignal(signal) {
  window._trActiveSignal = signal;

  // Populate symbol dropdown
  const symSel = document.getElementById('tr-symbol-select');
  if (symSel) {
    if (!Array.from(symSel.options).some(o => o.value === signal.symbol)) {
      const o = document.createElement('option');
      o.value = o.textContent = signal.symbol;
      symSel.appendChild(o);
    }
    symSel.value = signal.symbol;
  }

  _setVal('tr-entry', signal.entry_price != null ? signal.entry_price.toFixed(4) : '');
  _setVal('tr-sl',    signal.sl    != null ? signal.sl.toFixed(4)    : '');
  _setVal('tr-tp',    signal.tp    != null ? signal.tp.toFixed(4)    : '');
  _refreshCalc();

  // Show signal banner
  const banner = document.getElementById('tr-signal-banner');
  if (banner) {
    _setText('tr-banner-symbol', signal.symbol);
    _setText('tr-banner-conf',   signal.ai_prob != null ? `${(signal.ai_prob * 100).toFixed(1)}% AI confidence` : '');
    _setText('tr-banner-dir',    signal.direction || '');
    banner.classList.remove('hidden');
  }
}

// ── Real trade execution (broker API) ────────────────────────────────────────
async function _executeRealTrade() {
  const signal  = window._trActiveSignal;
  const symbol  = document.getElementById('tr-symbol-select')?.value || signal?.symbol;
  if (!symbol) { _toast('Select a signal first', 'error'); return; }

  const entry    = parseFloat(document.getElementById('tr-entry')?.value);
  const sl       = parseFloat(document.getElementById('tr-sl')?.value);
  const tp       = parseFloat(document.getElementById('tr-tp')?.value);
  const capital  = parseFloat(document.getElementById('tr-capital')?.value || 10000);
  const riskPct  = parseFloat(document.getElementById('tr-risk')?.value || 2);
  const leverage = parseFloat(document.getElementById('tr-leverage')?.value || 1);

  if (!entry || entry <= 0) { _toast('Enter a valid entry price', 'error'); return; }
  if (!sl    || sl    <= 0) { _toast('Enter a valid stop-loss',   'error'); return; }
  if (!tp    || tp    <= 0) { _toast('Enter a valid take-profit', 'error'); return; }

  const { posUnits, notional, direction } = _calcPos({ capital, riskPct, entry, sl, tp, leverage });

  const btn  = document.getElementById('tr-real-btn');
  const orig = btn?.innerHTML;
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Executing…'; }

  try {
    const token = await _getToken();
    if (!token) { _toast('Not authenticated — please sign in', 'error'); return; }

    const body = {
      symbol, side: direction, entryPrice: entry, stopLoss: sl, takeProfit: tp,
      riskPercent: riskPct, leverage, positionUnits: posUnits || 0,
      notionalValue: notional || 0, status: 'open', signalId: signal?.signal_id || null,
    };

    const r = await fetch('/api/trades/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    });

    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }

    const result = await r.json();
    _toast(`Real trade opened — ID ${result.trade_id}`, 'success');
    if (typeof window.switchRoom === 'function') window.switchRoom('analytics');
    setTimeout(() => { if (typeof window.forceTradesRefresh === 'function') window.forceTradesRefresh(); }, 200);
  } catch (err) {
    _toast(`Trade failed: ${err.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = orig; }
  }
}

// ── Paper trade (terminal cockpit simulation) ────────────────────────────────
function _executePaperTrade() {
  const signal = window._trActiveSignal;
  const symbol = document.getElementById('tr-symbol-select')?.value || signal?.symbol;
  if (!symbol) { _toast('Select a signal first', 'error'); return; }

  const entry = parseFloat(document.getElementById('tr-entry')?.value);
  const sl    = parseFloat(document.getElementById('tr-sl')?.value);
  const tp    = parseFloat(document.getElementById('tr-tp')?.value);
  if (!entry || !sl || !tp) { _toast('Fill in Entry, SL and TP', 'error'); return; }

  const { direction } = _calcPos({ capital: 10000, riskPct: 2, entry, sl, tp, leverage: 1 });

  const trade = {
    id: `paper-${Date.now()}`,
    type: 'paper',
    symbol,
    side: direction || (sl < entry ? 'LONG' : 'SHORT'),
    entryPrice: entry,
    stopLoss: sl,
    takeProfit: tp,
    positionUnits: parseFloat(document.getElementById('tr-pos-units')?.textContent || '0') || 1,
    openTime: new Date().toISOString(),
  };

  _paperTrades.unshift(trade);
  _renderAllTrades();
  _toast(`Paper trade opened for ${symbol}`, 'success');
}

// ── Trade rendering ──────────────────────────────────────────────────────────
// Called via window.renderTrades by gatekeeper.js on every WS tick
function renderTrades(liveTrades) {
  _liveTradesCache = (liveTrades || []).map(t => ({
    ...t,
    entryPrice:   parseFloat(t.entryPrice   || t.entry_price  || 0),
    stopLoss:     parseFloat(t.stopLoss     || t.stop_loss    || t.sl || 0),
    takeProfit:   parseFloat(t.takeProfit   || t.take_profit  || t.tp || 0),
    positionUnits:parseFloat(t.positionUnits|| t.position_size|| 1),
    side:         t.side || t.direction || 'LONG',
  }));
  _renderAllTrades();
}

function _renderAllTrades() {
  const all = [..._paperTrades, ..._liveTradesCache];
  _renderPositionCards(all);
  _renderTradesTable(all);
}

function _pnl(trade) {
  const cur = window.currentTickers?.[trade.symbol]
    ? parseFloat(window.currentTickers[trade.symbol])
    : trade.entryPrice;
  const units = trade.positionUnits || 1;
  return trade.side === 'LONG'
    ? (cur - trade.entryPrice) * units
    : (trade.entryPrice - cur) * units;
}

function _renderPositionCards(trades) {
  const el = document.getElementById('positionsContainer');
  if (!el) return;
  if (!trades.length) {
    el.innerHTML = `<p class="text-sm text-gray-500 py-6 text-center">No open positions. Select a signal to execute a trade.</p>`;
    return;
  }
  el.innerHTML = trades.map(t => {
    const pnl    = _pnl(t);
    const pnlPct = t.entryPrice > 0 ? (pnl / (t.entryPrice * (t.positionUnits || 1)) * 100) : 0;
    const isPaper = t.type === 'paper' || String(t.id).startsWith('paper-');
    const pnlClr  = pnl >= 0 ? 'text-green-400' : 'text-red-400';
    const sideCls = t.side === 'LONG'
      ? 'bg-green-500/20 text-green-400 border-green-500/30'
      : 'bg-red-500/20 text-red-400 border-red-500/30';
    return `<div class="bg-black/40 p-4 rounded-xl border border-white/10 flex flex-wrap items-center gap-4">
      <span class="text-xs px-2 py-1 rounded border font-bold ${sideCls}">${t.side}</span>
      ${isPaper ? '<span class="text-[10px] bg-yellow-500/20 text-yellow-300 border border-yellow-500/30 px-1.5 py-0.5 rounded font-bold">PAPER</span>' : '<span class="text-[10px] bg-blue-500/20 text-blue-300 border border-blue-500/30 px-1.5 py-0.5 rounded font-bold">LIVE</span>'}
      <div class="flex-1 min-w-0">
        <div class="font-bold text-white">${t.symbol}</div>
        <div class="text-xs text-gray-500 font-mono">Entry $${t.entryPrice.toFixed(4)}</div>
      </div>
      <div class="text-right">
        <div class="font-mono font-bold ${pnlClr}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</div>
        <div class="text-xs ${pnlClr} opacity-70">${pnl >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</div>
      </div>
      <div class="text-right font-mono text-xs text-gray-500 hidden sm:block">
        <div>SL <span class="text-red-400">$${(t.stopLoss||0).toFixed(4)}</span></div>
        <div>TP <span class="text-green-400">$${(t.takeProfit||0).toFixed(4)}</span></div>
      </div>
      <button onclick="window.closeTrade('${t.id}')"
        class="text-xs px-3 py-2 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/40 transition-colors font-bold whitespace-nowrap">
        <i class="fas fa-times mr-1"></i>Close
      </button>
    </div>`;
  }).join('');
}

function _renderTradesTable(trades) {
  const tbody = document.getElementById('trades-tbody');
  if (!tbody) return;
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="p-6 text-center text-gray-500">No active trades</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => {
    const pnl     = _pnl(t);
    const isPaper = t.type === 'paper' || String(t.id).startsWith('paper-');
    const pnlClr  = pnl >= 0 ? 'text-green-400' : 'text-red-400';
    const sideCls = t.side === 'LONG' ? 'text-green-400' : 'text-red-400';
    return `<tr class="hover:bg-white/5">
      <td class="p-3 font-bold text-white">${t.symbol}</td>
      <td class="p-3">
        <span class="text-[10px] px-1.5 py-0.5 rounded border font-bold ${isPaper ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' : 'bg-blue-500/20 text-blue-300 border-blue-500/30'}">${isPaper ? 'PAPER' : 'LIVE'}</span>
      </td>
      <td class="p-3 font-bold ${sideCls}">${t.side}</td>
      <td class="p-3 font-mono">$${t.entryPrice.toFixed(4)}</td>
      <td class="p-3 font-mono text-red-400">$${(t.stopLoss||0).toFixed(4)}</td>
      <td class="p-3 font-mono text-green-400">$${(t.takeProfit||0).toFixed(4)}</td>
      <td class="p-3 font-mono font-bold ${pnlClr}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</td>
      <td class="p-3">
        <button onclick="window.closeTrade('${t.id}')"
          class="text-xs px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/40 transition-colors font-bold">Close</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Close trade ──────────────────────────────────────────────────────────────
async function closeTrade(tradeId) {
  const isPaper = String(tradeId).startsWith('paper-');

  if (isPaper) {
    _paperTrades = _paperTrades.filter(t => t.id !== tradeId);
    _renderAllTrades();
    return;
  }

  // Optimistic remove
  _liveTradesCache = _liveTradesCache.filter(t => (t.id || t.tradeId) !== tradeId);
  _renderAllTrades();

  // Prune localStorage
  try {
    const lk = localStorage.getItem('lastKnownTrades');
    if (lk) localStorage.setItem('lastKnownTrades', JSON.stringify(JSON.parse(lk).filter(t => (t.id || t.tradeId) !== tradeId)));
  } catch (_) {}

  try {
    const token = await _getToken();
    if (!token) return;
    const r = await fetch(`/api/trades/${encodeURIComponent(tradeId)}/close`, {
      method: 'POST', headers: { Authorization: `Bearer ${token}` },
    });
    if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    _toast(`Close failed: ${e.message}`, 'error');
    if (typeof window.forceTradesRefresh === 'function') window.forceTradesRefresh();
  }
}

// ── Signal history ───────────────────────────────────────────────────────────
function addToSignalHistory(signal) {
  if (!signal || signal.signal === 'HOLD') return;
  if (_signalHistory.some(e => e.signal_id && e.signal_id === signal.signal_id)) return;

  _signalHistory.unshift({
    symbol:      signal.symbol,
    signal:      signal.signal,
    direction:   signal.direction || 'NEUTRAL',
    entry_price: signal.entry_price,
    sl:          signal.sl,
    tp:          signal.tp,
    ai_prob:     signal.ai_prob,
    timestamp:   new Date().toISOString(),
    signal_id:   signal.signal_id,
  });

  if (_signalHistory.length > 100) _signalHistory.length = 100;
  try { localStorage.setItem('tr_signalHistory', JSON.stringify(_signalHistory)); } catch (_) {}
  _renderSignalHistory();
}

function _renderSignalHistory() {
  const tbody = document.getElementById('signalHistoryTbody');
  if (!tbody) return;

  if (!_signalHistory.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="p-6 text-center text-gray-500">No signal history. Click a signal card to record it here.</td></tr>';
    return;
  }

  tbody.innerHTML = _signalHistory.map((e, i) => {
    const isLong  = e.direction === 'LONG'  || e.signal.includes('BUY');
    const isShort = e.direction === 'SHORT' || e.signal.includes('SELL');
    const sigCls  = isLong  ? 'bg-green-500/20 text-green-400 border-green-500/50'
                  : isShort ? 'bg-red-500/20 text-red-400 border-red-500/50'
                            : 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    const dirClr  = isLong ? 'text-green-400' : isShort ? 'text-red-400' : 'text-gray-400';
    const conf    = e.ai_prob != null ? `${(e.ai_prob * 100).toFixed(1)}%` : '—';
    const ts      = new Date(e.timestamp).toLocaleString();
    return `<tr class="hover:bg-white/5">
      <td class="p-3 font-bold text-white whitespace-nowrap">${e.symbol}</td>
      <td class="p-3"><span class="text-[10px] px-2 py-0.5 rounded border font-bold ${sigCls}">${e.signal}</span></td>
      <td class="p-3 font-bold ${dirClr} text-xs">${e.direction}</td>
      <td class="p-3 font-mono text-xs">${e.entry_price != null ? '$' + e.entry_price.toFixed(4) : '—'}</td>
      <td class="p-3 font-mono text-xs text-red-400">${e.sl  != null ? '$' + e.sl.toFixed(4)  : '—'}</td>
      <td class="p-3 font-mono text-xs text-green-400">${e.tp != null ? '$' + e.tp.toFixed(4) : '—'}</td>
      <td class="p-3 font-mono text-cyan text-xs">${conf}</td>
      <td class="p-3 text-xs text-gray-400 whitespace-nowrap">${ts}</td>
      <td class="p-3">
        <div class="flex gap-1.5 flex-nowrap">
          <button onclick="window.trFromHistory(${i},'real')"
            class="text-[10px] px-2 py-1.5 rounded bg-cyan/20 text-cyan border border-cyan/30 hover:bg-cyan/40 transition-colors font-bold whitespace-nowrap">
            <i class="fas fa-rocket mr-1"></i>Real
          </button>
          <button onclick="window.trFromHistory(${i},'paper')"
            class="text-[10px] px-2 py-1.5 rounded bg-yellow-500/20 text-yellow-300 border border-yellow-500/30 hover:bg-yellow-500/40 transition-colors font-bold whitespace-nowrap">
            <i class="fas fa-flask mr-1"></i>Paper
          </button>
        </div>
      </td>
    </tr>`;
  }).join('');
}

// ── Trigger trade from signal history row ─────────────────────────────────────
function _fromHistory(index, mode) {
  const e = _signalHistory[index];
  if (!e) return;

  prefillFromSignal({
    symbol:      e.symbol,
    entry_price: e.entry_price,
    sl:          e.sl,
    tp:          e.tp,
    ai_prob:     e.ai_prob,
    signal_id:   e.signal_id,
    direction:   e.direction,
    signal:      e.signal,
  });

  if (typeof window.switchRoom === 'function') window.switchRoom('terminal');

  if (mode === 'paper') {
    // Small delay so terminal has rendered before we fire paper trade
    setTimeout(_executePaperTrade, 80);
  }
}

// ── Event wiring ─────────────────────────────────────────────────────────────
function _wireEvents() {
  // Position calc — any change to these fields recalculates metrics
  ['tr-entry', 'tr-sl', 'tr-tp'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', _refreshCalc);
  });

  // Risk slider syncs with hidden input
  document.getElementById('tr-risk-slider')?.addEventListener('input', e => {
    _setText('tr-risk-display', e.target.value + '%');
    _setVal('tr-risk', e.target.value);
    _refreshCalc();
    // Sync analytics Risk badge
    const rd = document.getElementById('riskDisplay');
    if (rd) rd.textContent = e.target.value + '%';
  });

  // Leverage slider
  document.getElementById('tr-leverage')?.addEventListener('input', e => {
    _setText('tr-lev-display', e.target.value + 'x');
    _refreshCalc();
    // Highlight matching quick button
    document.querySelectorAll('.tr-lev-btn').forEach(b => {
      const active = b.dataset.val === e.target.value;
      b.className = `tr-lev-btn flex-1 py-1.5 text-xs font-bold rounded border transition-colors ${
        active ? 'bg-cyan/20 border-cyan text-white' : 'bg-black/40 border-white/10 text-gray-400 hover:border-cyan/50 hover:text-white'}`;
    });
  });

  // Leverage quick-select buttons
  document.querySelectorAll('.tr-lev-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const v = btn.dataset.val;
      const s = document.getElementById('tr-leverage');
      if (s) { s.value = v; s.dispatchEvent(new Event('input')); }
    });
  });

  // Capital input
  document.getElementById('tr-capital')?.addEventListener('input', e => {
    const capDisplay = document.getElementById('capitalDisplay');
    if (capDisplay) capDisplay.textContent = `$${parseFloat(e.target.value || 0).toLocaleString()}`;
    _refreshCalc();
  });

  // Execute buttons
  document.getElementById('tr-real-btn')?.addEventListener('click', _executeRealTrade);
  document.getElementById('tr-paper-btn')?.addEventListener('click', _executePaperTrade);

  // Clear signal history
  document.getElementById('clearHistoryBtn')?.addEventListener('click', () => {
    if (!confirm('Clear all signal history?')) return;
    _signalHistory = [];
    localStorage.removeItem('tr_signalHistory');
    _renderSignalHistory();
  });

  // Live price updates → refresh PnL on open positions
  window.addEventListener('priceUpdate', () => {
    if (_paperTrades.length + _liveTradesCache.length > 0) _renderAllTrades();
  });
}

// ── Public init ───────────────────────────────────────────────────────────────
function initTradingRooms() {
  // Load persisted signal history
  try {
    const stored = localStorage.getItem('tr_signalHistory');
    if (stored) _signalHistory = JSON.parse(stored);
  } catch (_) {}

  _renderSignalHistory();
  _wireEvents();

  // Expose globals so gatekeeper.js can delegate to us without a hard import
  window.renderTrades        = renderTrades;
  window.closeTrade          = closeTrade;
  window.addToSignalHistory  = addToSignalHistory;
  window.prefillFromSignal   = prefillFromSignal;
  window.trFromHistory       = _fromHistory;
}

// Auto-init when the module loads (same pattern as gatekeeper.js / dashboard.js)
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTradingRooms);
} else {
  initTradingRooms();
}

export { initTradingRooms };
