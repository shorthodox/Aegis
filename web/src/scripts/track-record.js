/**
 * track-record.js
 * Fetches /api/track-record and renders the public signal log page.
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
    if (outcome === 'OPEN' || pnl == null) return '—';
    const n = parseFloat(pnl);
    if (isNaN(n)) return '—';
    const sign = n >= 0 ? '+' : '';
    return `${sign}${n.toFixed(2)}%`;
}

function pnlColor(pnl, outcome) {
    if (outcome === 'OPEN' || pnl == null) return '';
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
    if (t === 'STRONG_BUY')  return '<span style="color:#00ff88;font-weight:700;">STRONG BUY</span>';
    if (t === 'BUY')         return '<span style="color:#00ff88;">BUY</span>';
    if (t === 'STRONG_SELL') return '<span style="color:#ff5555;font-weight:700;">STRONG SELL</span>';
    if (t === 'SELL')        return '<span style="color:#ff5555;">SELL</span>';
    return `<span style="color:#6b7280;">${t}</span>`;
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

    // Win-rate attribution note
    const noteEl = document.getElementById('trWinrateNote');
    if (noteEl) {
        noteEl.style.display = 'block';
        if (summary.win_rate_pct != null) {
            noteEl.textContent = `Win rate of ${summary.win_rate_pct}% is computed from ${closed} closed signal${closed !== 1 ? 's' : ''} since ${since ? since.slice(0, 10) : 'launch'}. Both wins and losses are included. This is not a marketing claim.`;
        } else {
            noteEl.textContent = `Win rate will be displayed once signals close. Currently showing ${summary.total_signals ?? 0} tracked signal${(summary.total_signals ?? 0) !== 1 ? 's' : ''}. Check back as signals close.`;
        }
    }
}

// ── Table ─────────────────────────────────────────────────────────────────────
function applyFilters(rows) {
    return rows.filter(r => {
        const dir = _activeFilter.direction;
        const out = _activeFilter.outcome;
        const matchDir = dir === 'ALL' || dirLabel(r.direction) === dir;
        const matchOut = out === 'ALL' || (r.outcome || 'OPEN').toUpperCase() === out;
        return matchDir && matchOut;
    });
}

function renderTable(rows) {
    const tbody = document.getElementById('trTableBody');
    if (!tbody) return;
    const filtered = applyFilters(rows);

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:2rem;color:#4b5563;">No signals match the current filter.</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(r => `
        <tr>
            <td>${fmtTs(r.entry_time)}</td>
            <td style="color:#e2e8f0;font-weight:600;">${r.symbol || '—'}</td>
            <td>${r.timeframe || '—'}</td>
            <td class="${dirClass(r.direction)}">${dirLabel(r.direction)}</td>
            <td>${signalChip(r.signal_type)}</td>
            <td>${fmtPrice(r.entry_price)}</td>
            <td style="color:rgba(0,255,136,0.7);">${fmtPrice(r.take_profit)}</td>
            <td style="color:rgba(255,85,85,0.7);">${fmtPrice(r.stop_loss)}</td>
            <td style="${pnlColor(r.pnl_pct, r.outcome)}">${fmtPnl(r.pnl_pct, r.outcome)}</td>
            <td>${outcomeBadge(r.outcome)}</td>
        </tr>
    `).join('');
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

    renderStats(data.summary || {});

    _allRows = data.signals || [];

    if (_allRows.length === 0) {
        if (emptyEl) emptyEl.style.display = 'block';
        return;
    }

    if (wrapEl) wrapEl.style.display = 'block';
    renderTable(_allRows);
}

// ── Auto-refresh every 60 s ───────────────────────────────────────────────────
function startAutoRefresh() {
    setInterval(async () => {
        try {
            const data = await fetchTrackRecord();
            render(data);
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

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
