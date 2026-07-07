/**
 * track-record.js
 * Fetches /api/trader/track-record and renders the public signal log page.
 * No Firestore dependency — data comes entirely from the backend JSON.
 */

const API_URL = '/api/track-record';

// ── Filters state ─────────────────────────────────────────────────────────────
let _allRows = [];
let _activeFilter = { direction: 'ALL', outcome: 'ALL' };

// ── Fetch ─────────────────────────────────────────────────────────────────────
async function fetchTrackRecord() {
    const res = await fetch(API_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

// No normalisation needed — /api/track-record already returns a unified format
// for both live_engine signals and AEGIS trader signals.

// ── Format helpers ────────────────────────────────────────────────────────────
function fmtTs(ts) {
    if (!ts) return '—';
    try {
        const d = new Date(ts);
        if (isNaN(d)) return ts;
        return d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    } catch { return ts; }
}

function fmtPrice(v) {
    if (v == null || v === 0) return '—';
    const n = parseFloat(v);
    if (isNaN(n) || n === 0) return '—';
    if (n >= 10000) return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (n >= 100)   return n.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (n >= 1)     return n.toPrecision(6);
    return n.toPrecision(4);
}

function fmtPnl(pnl, outcome) {
    if (pnl == null) return '—';
    const n = parseFloat(pnl);
    if (isNaN(n)) return '—';
    const sign = n >= 0 ? '+' : '';
    // Open rows show live unrealized PnL, marked so it reads as provisional
    const liveTag = (outcome || '').toUpperCase() === 'OPEN' ? ' <span style="font-size:0.85em;opacity:0.6;">live</span>' : '';
    return `${sign}${n.toFixed(2)}%${liveTag}`;
}

function pnlColor(pnl, outcome) {
    if (pnl == null) return '';
    return parseFloat(pnl) >= 0 ? 'color:#00ff88;' : 'color:#ff5555;';
}

function dirLabel(dir) {
    const d = (dir || '').toUpperCase();
    if (d === 'LONG'  || d.includes('BUY'))  return 'LONG';
    if (d === 'SHORT' || d.includes('SELL')) return 'SHORT';
    return 'HOLD';
}

function dirClass(dir) {
    const label = dirLabel(dir);
    if (label === 'LONG')  return 'tr-dir-long';
    if (label === 'SHORT') return 'tr-dir-short';
    return 'tr-dir-hold';
}

function outcomeBadge(outcome) {
    switch ((outcome || '').toUpperCase()) {
        case 'WIN':  return '<span class="tr-badge tr-badge-win">WIN</span>';
        case 'LOSS': return '<span class="tr-badge tr-badge-loss">LOSS</span>';
        default:     return '<span class="tr-badge tr-badge-open">OPEN</span>';
    }
}

function signalChip(type) {
    const t = (type || '').toUpperCase();
    if (t === 'STRONG_BUY'  || t === 'BUY')  return '<span style="color:#00ff88;font-weight:700;">BUY</span>';
    if (t === 'STRONG_SELL' || t === 'SELL') return '<span style="color:#ff5555;font-weight:700;">SELL</span>';
    return `<span style="color:#6b7280;">${t}</span>`;
}

// Risk tier badge — set at entry by the engine's risk-tier classifier.
// RISKY  = fired while carrying advisory-gate warnings (disclosed, not hidden)
// STRONG = fired with zero objections and top-tier conviction
function tierBadge(strength) {
    const s = (strength || '').toUpperCase();
    if (s === 'RISKY') {
        return ' <span class="tr-badge" style="background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.4);color:#fbbf24;" title="Fired with advisory-gate warnings — higher risk entry, disclosed for transparency">RISKY!</span>';
    }
    if (s === 'STRONG') {
        return ' <span class="tr-badge" style="background:rgba(0,255,136,0.10);border:1px solid rgba(0,255,136,0.35);color:#00ff88;" title="Zero gate objections and top-tier model conviction at entry">STRONG SIGNAL!</span>';
    }
    return '';
}


// ── Stats strip ───────────────────────────────────────────────────────────────
function renderStats(summary) {
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val ?? '—';
    };

    set('trTotalVal', summary.total_signals ?? 0);
    set('trWinsVal',  summary.wins ?? 0);
    set('trLossVal',  summary.losses ?? 0);
    set('trOpenVal',  summary.open ?? 0);

    const rateEl   = document.getElementById('trRateVal');
    const rateNote = document.getElementById('trRateNote');
    const closed   = (summary.wins ?? 0) + (summary.losses ?? 0);

    if (summary.win_rate_pct != null) {
        if (rateEl)   { rateEl.textContent = summary.win_rate_pct + '%'; rateEl.className = 'tr-stat-value'; }
        if (rateNote) rateNote.textContent = `from ${closed} closed signal${closed !== 1 ? 's' : ''}`;
    } else {
        if (rateEl)   { rateEl.textContent = 'N/A'; rateEl.className = 'tr-stat-value tr-na'; }
        if (rateNote) rateNote.textContent = 'no closed signals yet';
    }
    const lowSampleEl = document.getElementById('trLowSampleNote');
    if (lowSampleEl) lowSampleEl.style.display = closed > 0 && closed < 30 ? 'block' : 'none';

    const since = summary.tracking_since;
    const sinceEl = document.getElementById('trSinceVal');
    if (sinceEl) {
        if (since) {
            try { sinceEl.textContent = 'since ' + new Date(since).toISOString().slice(0, 10); }
            catch { sinceEl.textContent = ''; }
        } else {
            sinceEl.textContent = 'recording now';
        }
    }

    // Avg PnL card
    const avgEl = document.getElementById('trAvgPnlVal');
    if (avgEl) {
        if (summary.avg_pnl_pct != null) {
            const n = parseFloat(summary.avg_pnl_pct);
            avgEl.textContent = (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
            avgEl.style.color = n >= 0 ? '#00ff88' : '#ff5555';
        } else {
            avgEl.textContent = '—';
        }
    }

    // Net Profit ($) + Capital cards — real dollar context for the % stats.
    const fmtUsd = (v) => {
        const n = Number(v);
        const sign = n < 0 ? '-' : (n > 0 ? '+' : '');
        return sign + '$' + Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 2 });
    };
    const netEl    = document.getElementById('trNetProfitVal');
    const netSubEl  = document.getElementById('trNetProfitSub');
    const capEl    = document.getElementById('trCapitalVal');
    const capSubEl  = document.getElementById('trCapitalSub');
    const pnlUsdt   = summary.total_pnl_usdt;
    const initCap   = summary.initial_capital;
    const balance   = summary.balance;
    if (netEl) {
        if (pnlUsdt != null) {
            const n = Number(pnlUsdt);
            netEl.textContent = fmtUsd(n);
            netEl.style.color = n >= 0 ? '#00ff88' : '#ff5555';
            if (netSubEl && initCap) {
                const pct = (n / Number(initCap)) * 100;
                netSubEl.textContent = `${(pct >= 0 ? '+' : '')}${pct.toFixed(2)}% on capital`;
            }
        } else {
            netEl.textContent = '—';
        }
    }
    if (capEl) {
        // Show CURRENT capital as the headline number (green when in profit),
        // with the starting capital as context underneath.
        const _cur = (balance != null) ? balance : initCap;
        if (_cur != null) {
            capEl.textContent = '$' + Number(_cur).toLocaleString('en-US', { maximumFractionDigits: 0 });
            if (initCap != null) {
                const c = Number(_cur), i = Number(initCap);
                capEl.style.color = c > i ? '#00ff88' : c < i ? '#ff5555' : '';
                if (capSubEl) capSubEl.textContent = `from $${i.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
            }
        } else {
            capEl.textContent = '—';
        }
    }

    // Win-rate attribution note
    const noteEl = document.getElementById('trWinrateNote');
    if (noteEl) {
        noteEl.style.display = 'block';
        if (summary.win_rate_pct != null) {
            const lowSampleTag = closed < 30 ? ` ⚠ Low sample (n=${closed}) — not statistically reliable yet.` : '';
            noteEl.textContent = `Win rate of ${summary.win_rate_pct}% is computed from ${closed} closed signal${closed !== 1 ? 's' : ''} since ${since ? since.slice(0, 10) : 'launch'}. Both wins and losses are included. PnL is gross — no fees or slippage are deducted. This is not a marketing claim.${lowSampleTag}`;
        } else {
            noteEl.textContent = `Win rate will be displayed once signals close. Currently showing ${summary.total_signals ?? 0} tracked signal${(summary.total_signals ?? 0) !== 1 ? 's' : ''}. Check back as signals close.`;
        }
    }
}

// ── Live signal teaser (shown above the table) ────────────────────────────────
function renderLiveTeaser(openCount) {
    const el = document.getElementById('trLiveTeaser');
    if (!el) return;
    if (openCount <= 0) { el.style.display = 'none'; return; }
    el.style.display = 'flex';
    el.innerHTML = `
        <div style="display:flex;align-items:center;gap:0.6rem;flex:1;">
            <span style="width:8px;height:8px;border-radius:50%;background:#00ff88;box-shadow:0 0 6px #00ff88;flex-shrink:0;animation:tr-pulse 1.5s ease-in-out infinite;"></span>
            <span><strong style="color:#00ff88;">${openCount} live signal${openCount !== 1 ? 's' : ''}</strong> active right now — entry price, TP &amp; SL visible on the dashboard.</span>
        </div>
        <a href="/pricing" style="
            padding:0.4rem 1rem;background:rgba(0,242,255,0.12);border:1px solid rgba(0,242,255,0.35);
            border-radius:20px;color:#00f2ff;font-size:0.8rem;font-weight:700;text-decoration:none;
            white-space:nowrap;transition:background 0.2s;flex-shrink:0;
        " onmouseover="this.style.background='rgba(0,242,255,0.22)'" onmouseout="this.style.background='rgba(0,242,255,0.12)'">
            Get Access →
        </a>
    `;
}

// ── Table — open AND closed signals are shown (full transparency) ─────────────
function applyFilters(rows) {
    return rows.filter(r => {
        const dir = _activeFilter.direction;
        const out = _activeFilter.outcome;
        const outcome = (r.outcome || '').toUpperCase();
        const matchDir = dir === 'ALL' || dirLabel(r.direction) === dir;
        const matchOut = out === 'ALL' ? true : outcome === out;
        return matchDir && matchOut;
    });
}

function renderTable(rows) {
    const tbody = document.getElementById('trTableBody');
    if (!tbody) return;
    const filtered = applyFilters(rows);

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="11" style="text-align:center;padding:2.5rem;color:#4b5563;">
            No signals match this filter yet. Open signals appear the moment they fire; closed ones when TP or SL is hit.
        </td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(r => {
        // Open positions arrive masked from the API (symbol "HIDDEN", null
        // prices) — the trade is the paid product.  Render the lock state
        // explicitly; direction, live PnL and the tier badge stay visible.
        const isOpen = (r.outcome || '').toUpperCase() === 'OPEN';
        const masked = isOpen;
        const tokenCell = masked
            ? '<span style="color:#64748b;letter-spacing:2px;" title="Live signal — token visible on the dashboard (subscribers)"><i class="fas fa-lock" style="font-size:0.75em;margin-right:5px;"></i>•••••</span>'
            : `<span style="color:var(--ae-text-1);font-weight:600;">${r.symbol || '—'}</span>`;
        const priceCell = v => masked ? '<span style="color:#475569;">•••</span>' : fmtPrice(v);
        return `
        <tr>
            <td>${fmtTs(r.close_time || r.entry_time)}</td>
            <td>${tokenCell}</td>
            <td>${r.timeframe || '—'}</td>
            <td class="${dirClass(r.direction)}">${dirLabel(r.direction)}</td>
            <td style="white-space:nowrap;">${signalChip(r.signal_type)}${tierBadge(r.signal_strength)}</td>
            <td>${priceCell(r.entry_price)}</td>
            <td style="color:rgba(0,255,136,0.7);">${priceCell(r.take_profit)}</td>
            <td style="color:rgba(255,85,85,0.7);">${priceCell(r.stop_loss)}</td>
            <td style="color:#94a3b8;white-space:nowrap;">${r.position_value != null ? '$' + Number(r.position_value).toLocaleString('en-US', { maximumFractionDigits: 0 }) : '—'}</td>
            <td style="${pnlColor(r.pnl_pct, r.outcome)}">${fmtPnl(r.pnl_pct, r.outcome)}</td>
            <td>${outcomeBadge(r.outcome)}</td>
        </tr>`;
    }).join('');
}

// ── Filter UI ─────────────────────────────────────────────────────────────────
function initFilters() {
    document.querySelectorAll('[data-tr-filter-dir]').forEach(btn => {
        btn.addEventListener('click', () => {
            _activeFilter.direction = btn.dataset.trFilterDir;
            document.querySelectorAll('[data-tr-filter-dir]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderTable(_allRows);
        });
    });

    document.querySelectorAll('[data-tr-filter-out]').forEach(btn => {
        btn.addEventListener('click', () => {
            _activeFilter.outcome = btn.dataset.trFilterOut;
            document.querySelectorAll('[data-tr-filter-out]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderTable(_allRows);
        });
    });

}

// ── Main render ───────────────────────────────────────────────────────────────
function render(data) {
    const loadEl  = document.getElementById('trLoading');
    const wrapEl  = document.getElementById('trTableWrap');
    const emptyEl = document.getElementById('trEmpty');

    if (loadEl) loadEl.style.display = 'none';

    const allSignals = data.signals || [];

    // Use API summary directly (already computed server-side)
    const summary = data.summary || {};
    renderStats(summary);
    renderLiveTeaser(summary.open || 0);

    const tcEl = document.getElementById('trTokenCount');
    if (tcEl && summary.token_count) tcEl.textContent = summary.token_count;

    _allRows = allSignals;

    if (allSignals.length === 0) {
        if (emptyEl) emptyEl.style.display = 'block';
        return;
    }

    if (emptyEl) emptyEl.style.display = 'none';
    if (wrapEl) wrapEl.style.display = 'block';
    renderTable(_allRows);
}

// ── Scanning label ─────────────────────────────────────────────────────────────
function updateScanLabel() {
    const el = document.getElementById('trScanLabel');
    if (!el) return;
    const now = new Date();
    const hh = String(now.getUTCHours()).padStart(2, '0');
    const mm = String(now.getUTCMinutes()).padStart(2, '0');
    el.textContent = `live · refreshed ${hh}:${mm} UTC`;
}

// ── Auto-refresh every 60 s ───────────────────────────────────────────────────
function startAutoRefresh() {
    updateScanLabel();
    setInterval(async () => {
        try {
            const data = await fetchTrackRecord();
            render(data);
            updateScanLabel();
        } catch { /* silently ignore refresh errors */ }
    }, 60_000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
    initFilters();
    try {
        const data = await fetchTrackRecord();
        render(data);
        startAutoRefresh();
    } catch (err) {
        const loadEl = document.getElementById('trLoading');
        if (loadEl) {
            loadEl.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="color:rgba(255,85,85,0.5);font-size:1.5rem;"></i>
                <p style="margin-top:1rem;font-size:0.82rem;color:#4b5563;">
                    Could not load track record. Check your connection and try refreshing.<br>
                    <small style="opacity:0.6;">${err.message || ''}</small>
                </p>`;
        }
        console.warn('[AegisTrackRecord] fetch error:', err);
    }
}

window.refreshAegisTrackRecord = async function() {
    try {
        const data = await fetchTrackRecord();
        render(data);
    } catch { /* silently ignore */ }
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
