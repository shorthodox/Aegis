// ============================================================
// token-details-panel.js  — v3 (rebuilt for accuracy)
//
// All four panel sections now read from real signal fields:
//   confluence.{total,trend,momentum,volume,smart_money,candle}
//   p_buy / p_sell / p_hold  (not raw_probabilities.*)
//   bull_tp1/2/3, bear_tp1/2/3, suggested_sl
//   support, resistance, s1, s2, r1, r2
//   expected_move_pct, risk_reward  (added by live_engine v3)
//   meta_confidence, threshold
//   rsi, adx, macd_signal, supertrend, funding_bias, oi_trend
//
// Backward-compatible: detects old [-1,+1] confluence scale and
// converts it to [0,10] automatically.
// ============================================================

// ── Toast helper ──────────────────────────────────────────────────
function _toast(msg, type = 'info') {
  if (typeof window._showToast === 'function') { window._showToast(msg, type); return; }
  const el = document.createElement('div');
  el.style.cssText = `position:fixed;bottom:24px;right:24px;z-index:9999;padding:10px 18px;
    border-radius:8px;font-size:13px;font-family:monospace;color:#fff;
    background:${{ success:'#10b981',error:'#ef4444',info:'#06b6d4' }[type]||'#06b6d4'};
    opacity:0;transition:opacity .25s;pointer-events:none;`;
  el.textContent = msg;
  document.body.appendChild(el);
  requestAnimationFrame(() => { el.style.opacity = '1'; });
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ── Lock overlay ──────────────────────────────────────────────────
function _lock(requiredTier, msg) {
  const c = requiredTier === 'PRO'
    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
    : 'bg-blue-500/15 text-blue-300 border-blue-500/30';
  return `
    <div class="absolute inset-0 z-10 flex flex-col items-center justify-center
                bg-slate-950/80 backdrop-blur-md rounded-xl">
      <i class="fas fa-lock text-xl text-slate-400 mb-2"></i>
      <p class="text-xs text-slate-300 text-center px-5 font-medium leading-relaxed">${msg}</p>
      <span class="mt-3 text-[9px] font-black uppercase tracking-widest px-3 py-1
                   rounded-full border ${c}">${requiredTier} TIER</span>
      <a href="/web/src/pages/pricing.html"
         class="mt-3 text-[10px] font-bold text-white bg-gradient-to-r
                from-cyan-500/80 to-blue-600/80 px-4 py-1.5 rounded-lg
                hover:from-cyan-500 hover:to-blue-600 transition-all">
        Upgrade →
      </a>
    </div>`;
}

// ── Price formatter (handles PEPE/SHIB/BTC equally) ───────────────
function _px(v) {
  v = parseFloat(v) || 0;
  if (v <= 0)        return '—';
  if (v >= 10000)    return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (v >= 1000)     return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 3 });
  if (v >= 1)        return v.toFixed(4);
  if (v >= 0.1)      return v.toFixed(5);
  if (v >= 0.01)     return v.toFixed(5);
  if (v >= 0.001)    return v.toFixed(6);
  if (v >= 0.0001)   return v.toFixed(7);
  return v.toExponential(4);
}

// ── Confluence scale normaliser ───────────────────────────────────
// Old engine used [-1, +1]; new engine (after fix) uses [0, 10].
// Detect by checking if the absolute value exceeds 1.05.
function _to10(raw) {
  const v = parseFloat(raw);
  if (isNaN(v)) return 5.0;  // default = neutral
  if (Math.abs(v) <= 1.05)   // old [-1, +1] scale
    return parseFloat(((v + 1) / 2 * 10).toFixed(2));
  return Math.min(10, Math.max(0, v));
}

// ── Confluence category bar ───────────────────────────────────────
// val10: [0, 10] where 5 = neutral, >5 = bullish, <5 = bearish
function _confBar(label, val10, weight, tooltip) {
  const v     = parseFloat(val10) || 5;
  const pct   = (v / 10) * 100;
  const delta = v - 5;                  // negative = bearish, positive = bullish
  const strength = Math.abs(delta) / 5; // 0–1 strength from neutral

  const isBull  = delta >  0.3;
  const isBear  = delta < -0.3;
  const dirLabel = isBull ? 'BULL' : isBear ? 'BEAR' : 'NEUT';
  const dirColor = isBull ? 'text-emerald-400' : isBear ? 'text-rose-400' : 'text-slate-500';
  const fillColor = isBull
    ? 'bg-gradient-to-r from-emerald-500/30 to-emerald-400/70'
    : isBear
      ? 'bg-gradient-to-r from-rose-400/70 to-rose-500/30'
      : 'bg-slate-600/40';
  const scoreColor = isBull ? 'text-emerald-400' : isBear ? 'text-rose-400' : 'text-slate-500';

  // Weight chip color
  const wColor = weight >= 2 ? 'text-amber-400' : weight >= 1.5 ? 'text-slate-300' : 'text-slate-600';

  return `
    <div title="${tooltip || ''}">
      <div class="flex items-center justify-between mb-1">
        <div class="flex items-center gap-1.5 min-w-0">
          <span class="text-[10px] font-mono text-slate-300 truncate">${label}</span>
          <span class="text-[8px] font-bold ${wColor} flex-shrink-0">×${weight}</span>
        </div>
        <div class="flex items-center gap-1.5 flex-shrink-0 ml-2">
          <span class="text-[9px] font-black ${dirColor} uppercase w-8 text-right">${dirLabel}</span>
          <span class="text-[10px] font-black font-mono ${scoreColor} w-9 text-right">${v.toFixed(1)}/10</span>
        </div>
      </div>
      <div class="h-2 bg-slate-800 rounded-full overflow-hidden relative">
        <!-- Neutral center marker -->
        <div class="absolute top-0 bottom-0 w-px bg-slate-600/60" style="left:50%"></div>
        <!-- Fill bar -->
        ${isBull
          ? `<div class="absolute top-0 bottom-0 left-1/2 ${fillColor} rounded-r-full transition-all duration-700"
                  style="width:${(strength * 50).toFixed(1)}%"></div>`
          : isBear
            ? `<div class="absolute top-0 bottom-0 right-1/2 ${fillColor} rounded-l-full transition-all duration-700"
                    style="width:${(strength * 50).toFixed(1)}%"></div>`
            : `<div class="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-2 ${fillColor} rounded-full"></div>`
        }
      </div>
    </div>`;
}

// ── Indicator pill helper ─────────────────────────────────────────
function _pill(label, value, bullish) {
  const color = bullish === true  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
              : bullish === false ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
              :                    'bg-slate-700/40 text-slate-400 border-slate-600/30';
  return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[9px] font-bold ${color}">
    <span class="opacity-60">${label}</span> ${value}
  </span>`;
}

// ── Main render ───────────────────────────────────────────────────
export function renderTokenDetailsPanel(tokenData, userTier) {
  const tier          = (userTier || 'BASIC').toUpperCase();
  const canIntermediate = tier === 'INTERMEDIATE' || tier === 'PRO';
  const canPro          = tier === 'PRO';

  // ── Core price & direction ────────────────────────────────────────
  const sym      = tokenData.symbol  || '—';
  const rawDir   = (tokenData.direction || tokenData.signal || 'NEUTRAL').toUpperCase();
  const isLong   = rawDir === 'LONG' || rawDir === 'BUY';
  const isSell   = rawDir === 'SHORT' || rawDir === 'SELL';
  const fire     = Boolean(tokenData.fire);
  const price    = parseFloat(tokenData.price || tokenData.entry_price || 0);

  // ── SL / TP levels ────────────────────────────────────────────────
  const sl  = parseFloat(tokenData.suggested_sl || tokenData.stop_loss  || 0);
  const tp1 = parseFloat(
    (isLong ? tokenData.bull_tp1 : tokenData.bear_tp1) ||
    tokenData.suggested_tp || 0
  );
  const tp2 = parseFloat((isLong ? tokenData.bull_tp2 : tokenData.bear_tp2) || 0);
  const tp3 = parseFloat((isLong ? tokenData.bull_tp3 : tokenData.bear_tp3) || 0);

  // ── S&R levels ────────────────────────────────────────────────────
  const support    = parseFloat(tokenData.support    || tokenData.s1 || 0);
  const resistance = parseFloat(tokenData.resistance || tokenData.r1 || 0);
  const pivot      = parseFloat(tokenData.pivot      || 0);
  const s1         = parseFloat(tokenData.s1         || 0);
  const r1         = parseFloat(tokenData.r1         || 0);

  // ── AI probabilities ──────────────────────────────────────────────
  const pBuy      = parseFloat(tokenData.p_buy  || 0);
  const pSell     = parseFloat(tokenData.p_sell || 0);
  const pHold     = parseFloat(tokenData.p_hold || 0);
  const metaConf  = parseFloat(tokenData.meta_confidence || 0);
  const threshold = parseFloat(tokenData.threshold || 0.6);
  const confGap   = (metaConf - threshold) * 100;  // positive = above threshold

  // ── Expected move / R:R (from live_engine v3) ─────────────────────
  const expectedMove = parseFloat(tokenData.expected_move_pct || 0);
  const riskReward   = parseFloat(tokenData.risk_reward       || 0);
  const atrPct       = parseFloat(tokenData.atr_pct           || 0);
  const atr          = parseFloat(tokenData.atr               || 0);

  // ── Confluence (normalize to [0,10] regardless of engine version) ─
  const rawConf   = tokenData.confluence || {};
  const cTotal    = _to10(rawConf.total    ?? 5);
  const cTrend    = _to10(rawConf.trend    ?? 5);
  const cMom      = _to10(rawConf.momentum ?? 5);
  const cVol      = _to10(rawConf.volume   ?? 5);
  const cSmart    = _to10(rawConf.smart_money ?? 5);
  const cBands    = _to10(rawConf.bands    ?? rawConf.smart_money ?? 5);  // bands not always in dict
  const cCandle   = _to10(rawConf.candle   ?? 5);
  const confSummary = rawConf.summary || (cTotal >= 7 ? 'Strong Bullish' : cTotal >= 5.5 ? 'Moderate Bullish' : cTotal <= 3 ? 'Strong Bearish' : cTotal <= 4.5 ? 'Moderate Bearish' : 'Neutral');

  // How many categories agree with the signal
  const signalCategories = [cTrend, cMom, cVol, cSmart, cCandle].filter(
    v => isLong ? v >= 5.5 : isSell ? v <= 4.5 : false
  ).length;
  const totalCategories = 5;

  // ── Other indicators ──────────────────────────────────────────────
  const rsi         = parseFloat(tokenData.rsi    || 50);
  const adx         = parseFloat(tokenData.adx    || 20);
  const macd        = (tokenData.macd_signal    || 'NEUTRAL').toUpperCase();
  const supertrend  = (tokenData.supertrend     || 'NEUTRAL').toUpperCase();
  const fundingBias = (tokenData.funding_bias   || 'NEUTRAL').toUpperCase();
  const oiTrend     = (tokenData.oi_trend       || 'STABLE').toUpperCase();
  const volStrength = (tokenData.volume_strength|| 'AVERAGE').toUpperCase();
  const volZscore   = parseFloat(tokenData.volume_zscore || 0);
  const volRegime   = (tokenData.volatility_regime || 'MEDIUM').toUpperCase();
  const trendRegime = (tokenData.trend_regime      || 'RANGING').toUpperCase();
  const marketBias  = (tokenData.market_bias        || 'NEUTRAL').toUpperCase();

  // ── Live price (real-time, falls back to signal price) ────────────
  const livePx = parseFloat(
    (window.currentTickers && window.currentTickers[sym]) ||
    tokenData.livePrice || price
  );

  // ── Dist helpers ──────────────────────────────────────────────────
  const distPct = (a, b) => b > 0 && a > 0 ? ((a - b) / b * 100) : 0;
  const distToTp1 = distPct(tp1, livePx);
  const distToSl  = distPct(sl, livePx);
  const distToTp2 = distPct(tp2, livePx);

  // ── Overall confluence badge ──────────────────────────────────────
  const confBadgeColor = cTotal >= 7 ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25'
    : cTotal >= 5.5                  ? 'text-cyan-300    bg-cyan-500/10    border-cyan-500/25'
    : cTotal <= 3                    ? 'text-rose-300    bg-rose-500/10    border-rose-500/25'
    : cTotal <= 4.5                  ? 'text-orange-300  bg-orange-500/10  border-orange-500/25'
    :                                  'text-slate-400   bg-slate-700/30   border-slate-600/30';
  const confTierLabel = cTotal >= 7 ? 'Strong' : cTotal >= 5.5 ? 'Moderate' : cTotal <= 3 ? 'Weak Bear' : cTotal <= 4.5 ? 'Moderate Bear' : 'Neutral';

  // ── Beginner plain-English confluence note ────────────────────────
  const confPlain = signalCategories >= 4
    ? `${signalCategories}/5 indicator groups support this ${isLong ? 'BUY' : 'SELL'} signal — strong setup.`
    : signalCategories >= 3
      ? `${signalCategories}/5 groups agree — reasonable setup, watch for confirmation.`
      : signalCategories >= 2
        ? `Only ${signalCategories}/5 groups align — mixed signals, trade smaller size.`
        : `Most indicators are neutral or conflicting — avoid unless high conviction.`;

  // ══════════════════════════════════════════════════════════════════
  // SECTION 1 — Confluence Scorecard
  // ══════════════════════════════════════════════════════════════════
  const sec1 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canIntermediate ? _lock('INTERMEDIATE', '🔒 Unlock AI Confluence Scorecard') : ''}
      <div class="${!canIntermediate ? 'blur-sm pointer-events-none select-none' : ''}">

        <!-- Header -->
        <div class="flex items-center gap-2 mb-3">
          <div class="w-6 h-6 rounded-md bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-layer-group text-cyan-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Confluence Scorecard
          </span>
          <!-- Overall score badge -->
          <span class="ml-auto text-[12px] font-black font-mono px-2.5 py-0.5
                       rounded-full border ${confBadgeColor}">
            ${cTotal.toFixed(1)}/10
          </span>
        </div>

        <!-- Summary strip -->
        <div class="flex items-center justify-between mb-3 px-3 py-2
                    rounded-lg bg-slate-800/50 border border-slate-700/40">
          <div>
            <div class="text-[9px] uppercase tracking-widest text-slate-500 font-mono mb-0.5">Overall Reading</div>
            <div class="text-[11px] font-black ${confBadgeColor.split(' ')[0]}">${confSummary}</div>
          </div>
          <div class="text-right">
            <div class="text-[9px] uppercase tracking-widest text-slate-500 font-mono mb-0.5">Agreement</div>
            <div class="text-[11px] font-black text-white">${signalCategories}/${totalCategories} <span class="text-[9px] text-slate-500 font-normal">groups</span></div>
          </div>
        </div>

        <!-- Category bars (6 rows) -->
        <div class="space-y-3">
          ${_confBar('Trend',
            cTrend, 2.0,
            'EMA stack, macro trend, market structure. Higher = clearer uptrend.')}
          ${_confBar('Momentum',
            cMom,   1.5,
            'RSI, MACD, Stochastic, CCI, Awesome Oscillator. Higher = stronger buying pressure.')}
          ${_confBar('Volume / Flow',
            cVol,   1.5,
            'CMF, MFI, OBV delta, EOM. Higher = smart money flowing in.')}
          ${_confBar('Smart Money / S&R',
            cSmart, 1.5,
            'Break-of-Structure, Change-of-Character events, proximity to key support/resistance zones.')}
          ${_confBar('Price Position / Bands',
            cBands, 1.0,
            'Bollinger Band %, ATR band position, Donchian range. Measures where price sits in its recent range.')}
          ${_confBar('Candle Patterns',
            cCandle, 0.5,
            'Hammer, Engulfing, Morning/Evening Star patterns detected on the last bar.')}
        </div>

        <!-- Weights explanation -->
        <div class="mt-3 pt-3 border-t border-slate-800/60">
          <div class="text-[10px] text-slate-500 font-mono leading-relaxed">
            <span class="text-amber-400/80">×weight</span> = contribution to total score.
            Trend (×2) and Smart Money (×1.5) carry the most weight — candles (×0.5) the least.
          </div>
        </div>

        <!-- Beginner note -->
        <div class="mt-2 px-3 py-2 rounded-lg bg-slate-800/30 border border-slate-700/30">
          <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-0.5">For beginners</div>
          <div class="text-[10px] text-slate-400 leading-relaxed">${confPlain}</div>
        </div>

      </div>
    </div>`;

  // ══════════════════════════════════════════════════════════════════
  // SECTION 2 — Zone Tracker  (multi-level TP + S&R)
  // ══════════════════════════════════════════════════════════════════

  // Build range: SL on one extreme, TP3 on the other
  const hasLevels = sl > 0 && tp1 > 0;
  // For BUY: range [sl ... tp3]; for SELL: range [tp3 ... sl]
  let rangeLow, rangeHigh;
  if (isLong) {
    rangeLow  = sl   > 0 ? sl   : price * 0.94;
    rangeHigh = tp3  > 0 ? tp3  : tp2 > 0 ? tp2 : tp1 > 0 ? tp1 * 1.02 : price * 1.08;
  } else if (isSell) {
    // tp3 is below entry; sl is above entry
    const lowestTp = tp3 > 0 ? tp3 : tp2 > 0 ? tp2 : tp1 > 0 ? tp1 * 0.98 : price * 0.92;
    rangeLow  = lowestTp;
    rangeHigh = sl > 0 ? sl : price * 1.06;
  } else {
    // No signal — show ±5% range around price
    rangeLow  = price * 0.95;
    rangeHigh = price * 1.05;
  }

  const rangePx  = rangeHigh - rangeLow || 1;
  function toPct(lvl) {
    return Math.min(97, Math.max(3, ((lvl - rangeLow) / rangePx) * 100));
  }

  const pricePct = toPct(livePx || price);
  const entryPct = toPct(price);
  const slPct    = toPct(sl);
  const tp1Pct   = toPct(tp1);
  const tp2Pct   = tp2 > 0 ? toPct(tp2) : null;
  const tp3Pct   = tp3 > 0 ? toPct(tp3) : null;
  const suppPct  = support > 0 && support >= rangeLow && support <= rangeHigh ? toPct(support) : null;
  const resPct   = resistance > 0 && resistance >= rangeLow && resistance <= rangeHigh ? toPct(resistance) : null;

  // Trade progress: distance from entry to current price, expressed as % of TP1 distance
  const tp1Dist  = Math.abs(tp1 - price);
  const priceFromEntry = livePx > 0 ? (isLong ? livePx - price : price - livePx) : 0;
  const progressToTp1 = tp1Dist > 0 ? Math.min(100, Math.max(0, (priceFromEntry / tp1Dist) * 100)) : 0;

  // Status label
  let zoneStatus = '';
  if (!hasLevels) {
    zoneStatus = `<span class="text-[11px] text-slate-500">No open trade — no zone to track</span>`;
  } else if (isLong) {
    if (livePx >= tp1 && tp1 > 0) {
      zoneStatus = `<span class="text-[11px] font-bold text-emerald-400 animate-pulse">✅ TP1 Reached — consider taking profits</span>`;
    } else if (livePx <= sl && sl > 0) {
      zoneStatus = `<span class="text-[11px] font-bold text-rose-400 animate-pulse">⚠️ SL Zone — stop loss approaching</span>`;
    } else if (priceFromEntry >= 0) {
      const pctDone = tp1Dist > 0 ? ((priceFromEntry / tp1Dist) * 100).toFixed(0) : 0;
      zoneStatus = `<span class="text-[11px] font-bold text-cyan-400">📈 In profit — ${pctDone}% to TP1</span>`;
    } else {
      const slDist = Math.abs(price - sl);
      const slUsed = slDist > 0 ? (Math.abs(priceFromEntry) / slDist * 100).toFixed(0) : 0;
      zoneStatus = `<span class="text-[11px] font-bold text-amber-400">⏳ Below entry — ${slUsed}% toward SL</span>`;
    }
  } else if (isSell) {
    if (livePx <= tp1 && tp1 > 0) {
      zoneStatus = `<span class="text-[11px] font-bold text-emerald-400 animate-pulse">✅ TP1 Reached — consider taking profits</span>`;
    } else if (livePx >= sl && sl > 0) {
      zoneStatus = `<span class="text-[11px] font-bold text-rose-400 animate-pulse">⚠️ SL Zone — stop loss approaching</span>`;
    } else if (priceFromEntry >= 0) {
      const pctDone = tp1Dist > 0 ? ((priceFromEntry / tp1Dist) * 100).toFixed(0) : 0;
      zoneStatus = `<span class="text-[11px] font-bold text-cyan-400">📉 In profit — ${pctDone}% to TP1</span>`;
    } else {
      zoneStatus = `<span class="text-[11px] font-bold text-amber-400">⏳ Above entry — moving toward SL</span>`;
    }
  }

  // Beginner zone explanation
  const zonePlain = !hasLevels
    ? 'No active trade levels to display. Open a position to see the zone tracker.'
    : isLong
      ? `You entered LONG at $${_px(price)}. The green line shows TP1 — your first profit target. The red line (left) is your stop loss. The dot shows where the price is right now.`
      : `You entered SHORT at $${_px(price)}. Price moving DOWN is profit. The green marks on the left are your profit targets. Red on the right is your stop loss.`;

  const sec2 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canIntermediate ? _lock('INTERMEDIATE', '🔒 Unlock Visual Zone Tracking') : ''}
      <div class="${!canIntermediate ? 'blur-sm pointer-events-none select-none' : ''}">

        <!-- Header -->
        <div class="flex items-center gap-2 mb-3">
          <div class="w-6 h-6 rounded-md bg-violet-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-map-marker-alt text-violet-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">Zone Tracker</span>
          ${hasLevels ? `<span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       ${isLong ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25'
                                : 'bg-rose-500/15 text-rose-300 border-rose-500/25'} border uppercase">
            ${isLong ? '▲ LONG' : '▼ SHORT'}
          </span>` : ''}
        </div>

        <!-- Main zone bar -->
        <div class="relative h-14 bg-slate-800/80 rounded-xl overflow-hidden
                    border border-slate-700/50 mb-3 select-none">

          <!-- Background gradient: loss side → profit side -->
          ${isLong
            ? `<div class="absolute inset-0 bg-gradient-to-r from-rose-500/20 via-transparent to-emerald-500/20"></div>`
            : `<div class="absolute inset-0 bg-gradient-to-r from-emerald-500/20 via-transparent to-rose-500/20"></div>`
          }

          <!-- TP3 marker (faintest) -->
          ${tp3Pct !== null ? `
          <div class="absolute top-0 h-3/4 w-px bg-emerald-400/20"
               style="left:${tp3Pct.toFixed(1)}%">
            <div class="absolute -top-0 -translate-x-1/2 text-[7px] font-bold text-emerald-400/50 whitespace-nowrap">TP3</div>
          </div>` : ''}

          <!-- TP2 marker -->
          ${tp2Pct !== null ? `
          <div class="absolute top-0 h-3/4 w-px bg-emerald-400/45"
               style="left:${tp2Pct.toFixed(1)}%">
            <div class="absolute -top-0 -translate-x-1/2 text-[7px] font-bold text-emerald-400/60 whitespace-nowrap">TP2</div>
          </div>` : ''}

          <!-- TP1 marker (brightest) -->
          ${tp1 > 0 ? `
          <div class="absolute top-0 bottom-0 w-px bg-emerald-400/80
                      shadow-[0_0_6px_rgba(52,211,153,0.5)]"
               style="left:${tp1Pct.toFixed(1)}%">
            <div class="absolute -top-0 -translate-x-1/2 text-[8px] font-bold text-emerald-300 whitespace-nowrap">TP1</div>
          </div>` : ''}

          <!-- SL marker -->
          ${sl > 0 ? `
          <div class="absolute top-0 bottom-0 w-px bg-rose-500/80
                      shadow-[0_0_6px_rgba(239,68,68,0.5)]"
               style="left:${slPct.toFixed(1)}%">
            <div class="absolute -top-0 -translate-x-1/2 text-[8px] font-bold text-rose-400 whitespace-nowrap">SL</div>
          </div>` : ''}

          <!-- Support dashed line -->
          ${suppPct !== null ? `
          <div class="absolute top-1/4 bottom-0 w-px border-l border-dashed border-yellow-400/40"
               style="left:${suppPct.toFixed(1)}%">
            <div class="absolute bottom-1 -translate-x-1/2 text-[7px] text-yellow-400/60 whitespace-nowrap">SUP</div>
          </div>` : ''}

          <!-- Resistance dashed line -->
          ${resPct !== null ? `
          <div class="absolute top-1/4 bottom-0 w-px border-l border-dashed border-orange-400/40"
               style="left:${resPct.toFixed(1)}%">
            <div class="absolute bottom-1 -translate-x-1/2 text-[7px] text-orange-400/60 whitespace-nowrap">RES</div>
          </div>` : ''}

          <!-- Entry anchor -->
          <div class="absolute top-0 bottom-0 w-0.5 bg-cyan-400/90
                      shadow-[0_0_6px_rgba(0,242,255,0.6)] z-10"
               id="tdp-entry-line"
               style="left:${entryPct.toFixed(1)}%">
            <div class="absolute bottom-1 -translate-x-1/2
                        text-[8px] font-bold text-cyan-300 whitespace-nowrap">ENTRY</div>
          </div>

          <!-- Live price dot -->
          <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 z-20
                      w-4 h-4 rounded-full border-2 border-white
                      ${progressToTp1 > 80 ? 'bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,1)]'
                        : progressToTp1 > 40 ? 'bg-cyan-400 shadow-[0_0_14px_rgba(0,242,255,1)]'
                        : 'bg-amber-400 shadow-[0_0_14px_rgba(251,191,36,1)]'}
                      transition-all duration-1000 ease-out"
               id="tdp-price-dot"
               data-symbol="${sym}"
               data-range-low="${rangeLow}"
               data-range-high="${rangeHigh}"
               style="left:${pricePct.toFixed(1)}%">
          </div>

        </div><!-- /zone bar -->

        <!-- Metrics row -->
        <div class="grid grid-cols-3 gap-2 mb-3 text-center">
          <div class="bg-slate-800/50 rounded-lg py-1.5 border border-slate-700/30">
            <div class="text-[8px] text-slate-500 uppercase font-mono mb-0.5">To SL</div>
            <div class="text-[11px] font-black font-mono text-rose-400">
              ${sl > 0 ? (isLong ? distToSl : -distToSl).toFixed(2) + '%' : '—'}
            </div>
          </div>
          <div class="bg-slate-800/50 rounded-lg py-1.5 border border-slate-700/30">
            <div class="text-[8px] text-slate-500 uppercase font-mono mb-0.5">To TP1</div>
            <div class="text-[11px] font-black font-mono text-emerald-400">
              ${tp1 > 0 ? (isLong ? distToTp1 : -distToTp1).toFixed(2) + '%' : '—'}
            </div>
          </div>
          <div class="bg-slate-800/50 rounded-lg py-1.5 border border-slate-700/30">
            <div class="text-[8px] text-slate-500 uppercase font-mono mb-0.5">R:R</div>
            <div class="text-[11px] font-black font-mono text-cyan-400">
              ${riskReward > 0 ? '1:' + riskReward.toFixed(2) : tp1 > 0 && sl > 0
                ? '1:' + (Math.abs(tp1 - price) / Math.abs(sl - price)).toFixed(2) : '—'}
            </div>
          </div>
        </div>

        <!-- Price levels row -->
        <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono mb-3">
          <div class="flex justify-between">
            <span class="text-slate-500">SL</span>
            <span class="text-rose-400 font-bold">$${_px(sl)}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-500">TP1</span>
            <span class="text-emerald-300 font-bold">$${_px(tp1)}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-500">Entry</span>
            <span class="text-cyan-400 font-bold">$${_px(price)}</span>
          </div>
          ${tp2 > 0 ? `<div class="flex justify-between">
            <span class="text-slate-500">TP2</span>
            <span class="text-emerald-400/70 font-bold">$${_px(tp2)}</span>
          </div>` : ''}
          <div class="flex justify-between">
            <span class="text-slate-500">Live</span>
            <span id="tdp-live-price-label" class="text-white font-bold">$${_px(livePx)}</span>
          </div>
          ${tp3 > 0 ? `<div class="flex justify-between">
            <span class="text-slate-500">TP3</span>
            <span class="text-emerald-400/40 font-bold">$${_px(tp3)}</span>
          </div>` : ''}
        </div>

        <!-- Status pill -->
        <div class="flex items-center justify-center py-2 px-3 rounded-lg
                    bg-slate-800/50 border border-slate-700/40">
          ${zoneStatus || `<span class="text-[11px] text-slate-500">Monitoring — no active position</span>`}
        </div>

        <!-- ATR context -->
        ${atrPct > 0 ? `
        <div class="mt-2 flex items-center justify-between text-[9px] font-mono text-slate-600">
          <span>ATR = ${atr > 0 ? '$' + _px(atr) : '—'} (${atrPct.toFixed(2)}% of price)</span>
          <span class="${volRegime === 'HIGH' ? 'text-rose-400/70' : volRegime === 'LOW' ? 'text-blue-400/70' : 'text-slate-500'}">
            ${volRegime} volatility
          </span>
        </div>` : ''}

        <!-- Beginner note -->
        <div class="mt-2 px-3 py-2 rounded-lg bg-slate-800/30 border border-slate-700/30">
          <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-0.5">For beginners</div>
          <div class="text-[10px] text-slate-400 leading-relaxed">${zonePlain}</div>
        </div>

      </div>
    </div>`;

  // ══════════════════════════════════════════════════════════════════
  // SECTION 3 — Signal Quality & Expectancy
  // ══════════════════════════════════════════════════════════════════

  const confGapColor  = confGap >= 10 ? 'text-emerald-400' : confGap >= 0 ? 'text-cyan-400' : 'text-rose-400';
  const confGapLabel  = confGap >= 10 ? 'High Conviction' : confGap >= 0 ? 'Above Threshold' : 'Below Threshold';
  const metaConfPct   = Math.round(metaConf * 100);
  const threshPct     = Math.round(threshold * 100);
  const moveColor     = expectedMove >= 3 ? 'text-emerald-400' : expectedMove >= 1.5 ? 'text-cyan-400' : 'text-slate-400';

  const sec3 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock Signal Quality & Expectancy') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">

        <!-- Header -->
        <div class="flex items-center gap-2 mb-3">
          <div class="w-6 h-6 rounded-md bg-emerald-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-chart-bar text-emerald-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            Signal Quality
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-amber-500/15 text-amber-300 border border-amber-500/25 uppercase tracking-widest">
            PRO
          </span>
        </div>

        <!-- AI Confidence gauge -->
        <div class="mb-4 px-3 py-2.5 rounded-lg bg-slate-800/50 border border-slate-700/40">
          <div class="flex items-center justify-between mb-2">
            <span class="text-[10px] text-slate-400 font-mono uppercase tracking-widest">AI Confidence</span>
            <span class="text-[11px] font-black font-mono ${confGapColor}">${confGapLabel}</span>
          </div>
          <!-- Stacked gauge: threshold marker + confidence fill -->
          <div class="relative h-3 bg-slate-800 rounded-full overflow-hidden">
            <!-- Required threshold marker -->
            <div class="absolute top-0 bottom-0 w-0.5 bg-amber-400/60 z-10"
                 style="left:${threshPct}%"
                 title="Threshold: ${threshPct}%">
            </div>
            <!-- Confidence fill -->
            <div class="h-full rounded-full transition-all duration-700
                        ${metaConfPct >= threshPct ? 'bg-gradient-to-r from-cyan-500/60 to-emerald-500/70'
                                                   : 'bg-gradient-to-r from-rose-500/50 to-amber-500/40'}"
                 style="width:${metaConfPct}%">
            </div>
          </div>
          <div class="flex justify-between text-[9px] font-mono mt-1.5">
            <span class="text-slate-600">0%</span>
            <span class="text-amber-400/80">${threshPct}% ← required</span>
            <span class="${confGapColor} font-bold">${metaConfPct}% current</span>
          </div>
        </div>

        <!-- 4-metric cards -->
        <div class="grid grid-cols-2 gap-2 mb-3">
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">Expected Move</div>
            <div class="text-lg font-black font-mono ${moveColor} leading-none">
              ${expectedMove > 0 ? '~' + expectedMove.toFixed(1) + '%' : atrPct > 0 ? '~' + (atrPct * 1.5).toFixed(1) + '%' : '—'}
            </div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">AI projection</div>
          </div>
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">Risk / Reward</div>
            <div class="text-lg font-black font-mono text-cyan-400 leading-none">
              ${riskReward > 0 ? '1:' + riskReward.toFixed(2)
                : tp1 > 0 && sl > 0 ? '1:' + (Math.abs(tp1 - price) / Math.abs(sl - price)).toFixed(2) : '—'}
            </div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">per $ risked</div>
          </div>
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">Volatility</div>
            <div class="text-lg font-black font-mono
                        ${volRegime === 'HIGH' ? 'text-rose-400' : volRegime === 'LOW' ? 'text-blue-400' : 'text-amber-400'}
                        leading-none">${volRegime}</div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">ATR = ${atrPct > 0 ? atrPct.toFixed(2) + '%' : '—'}</div>
          </div>
          <div class="bg-slate-800/60 border border-slate-700/40 rounded-xl p-3">
            <div class="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 font-mono">Trend Regime</div>
            <div class="text-[13px] font-black font-mono text-slate-200 leading-none">
              ${trendRegime.replace('_', ' ')}
            </div>
            <div class="text-[9px] text-slate-500 mt-1 font-mono">ADX = ${adx.toFixed(1)}</div>
          </div>
        </div>

        <!-- Historical stats — loaded async -->
        <div id="tdp-hist-stats" class="px-3 py-2 rounded-lg bg-slate-800/30 border border-slate-700/30
                                        text-[10px] text-slate-500 font-mono">
          Loading historical stats…
        </div>

        <!-- Beginner explanation -->
        <div class="mt-2 px-3 py-2 rounded-lg bg-slate-800/30 border border-slate-700/30">
          <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-0.5">For beginners</div>
          <div class="text-[10px] text-slate-400 leading-relaxed">
            The AI needs ${threshPct}% confidence to fire — it currently shows ${metaConfPct}%.
            ${metaConfPct >= threshPct
              ? `This signal is <strong class="text-emerald-400">above the bar</strong> — the model has enough evidence to act.`
              : `This signal is <strong class="text-rose-400">below the bar</strong> — the model is watching but not yet certain.`}
            ${expectedMove > 0 ? ` Expect roughly ${expectedMove.toFixed(1)}% price movement if the setup plays out.` : ''}
          </div>
        </div>

      </div>
    </div>`;

  // ══════════════════════════════════════════════════════════════════
  // SECTION 4 — AI Model Analysis (real p_buy/p_sell/p_hold + key drivers)
  // ══════════════════════════════════════════════════════════════════

  const pBuyPct  = Math.round(pBuy  * 100);
  const pSellPct = Math.round(pSell * 100);
  const pHoldPct = Math.round(pHold * 100);
  const topIsLong = pBuyPct >= pSellPct;

  // Key drivers: 8 indicators from real signal data
  const drivers = [
    {
      label: 'RSI',
      value: rsi.toFixed(0),
      bull:  rsi < 40 ? true : rsi > 65 ? false : null,
      tip:   rsi < 30 ? 'Oversold — potential bounce' : rsi > 70 ? 'Overbought — potential pullback' : rsi < 45 ? 'Mild bearish pressure' : rsi > 55 ? 'Mild bullish pressure' : 'Neutral — no clear signal',
    },
    {
      label: 'MACD',
      value: macd,
      bull:  macd === 'BULLISH' ? true : macd === 'BEARISH' ? false : null,
      tip:   macd === 'BULLISH' ? 'MACD crossed up — buying momentum building' : macd === 'BEARISH' ? 'MACD crossed down — selling pressure' : 'MACD is neutral — no momentum crossover',
    },
    {
      label: 'Supertrend',
      value: supertrend === 'BULLISH' ? 'GREEN' : supertrend === 'BEARISH' ? 'RED' : 'FLAT',
      bull:  supertrend === 'BULLISH' ? true : supertrend === 'BEARISH' ? false : null,
      tip:   supertrend === 'BULLISH' ? 'Price above Supertrend line — uptrend confirmed' : supertrend === 'BEARISH' ? 'Price below Supertrend line — downtrend confirmed' : 'No clear trend direction',
    },
    {
      label: 'ADX',
      value: adx.toFixed(0),
      bull:  adx > 40 ? null : null,  // ADX is trend strength, not direction
      tip:   adx > 40 ? 'Very strong trend — high confidence in direction' : adx > 25 ? 'Solid trend in place' : adx > 15 ? 'Mild trend — could be choppy' : 'Weak trend — ranging market, signals less reliable',
    },
    {
      label: 'Volume',
      value: volStrength === 'ABOVE_AVERAGE' ? 'HIGH' : volStrength === 'BELOW_AVERAGE' ? 'LOW' : 'AVG',
      bull:  volStrength === 'ABOVE_AVERAGE' ? true : volStrength === 'BELOW_AVERAGE' ? false : null,
      tip:   volStrength === 'ABOVE_AVERAGE' ? `Volume z-score ${volZscore.toFixed(1)} — unusual buying/selling activity` : volStrength === 'BELOW_AVERAGE' ? 'Low volume — move may not sustain' : 'Average volume — normal market activity',
    },
    {
      label: 'Funding',
      value: fundingBias === 'LONGS_PAYING' ? 'OVER-LONG' : fundingBias === 'SHORTS_PAYING' ? 'OVER-SHORT' : 'NEUTRAL',
      bull:  fundingBias === 'SHORTS_PAYING' ? true : fundingBias === 'LONGS_PAYING' ? false : null,
      tip:   fundingBias === 'LONGS_PAYING' ? 'Too many longs — risk of long squeeze (price drop)' : fundingBias === 'SHORTS_PAYING' ? 'Too many shorts — risk of short squeeze (price rise)' : 'Balanced positioning — no squeeze risk',
    },
    {
      label: 'OI',
      value: oiTrend,
      bull:  oiTrend === 'INCREASING' && isLong ? true : oiTrend === 'DECREASING' && isSell ? true : null,
      tip:   oiTrend === 'INCREASING' ? 'Open interest growing — more money entering the trade' : oiTrend === 'DECREASING' ? 'Open interest falling — participants exiting' : 'Stable open interest',
    },
    {
      label: 'Market',
      value: marketBias,
      bull:  marketBias === 'BULLISH' ? true : marketBias === 'BEARISH' ? false : null,
      tip:   marketBias === 'BULLISH' ? 'Most indicators lean bullish overall' : marketBias === 'BEARISH' ? 'Most indicators lean bearish overall' : 'Mixed signals — no dominant direction',
    },
  ];

  const driverPills = drivers.map(d =>
    _pill(d.label, d.value, d.bull)
  ).join(' ');

  const driverDetail = drivers.map(d => `
    <div class="flex items-start gap-2 py-2 border-b border-slate-800/60 last:border-0">
      <div class="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0
                  ${d.bull === true ? 'bg-emerald-400' : d.bull === false ? 'bg-rose-400' : 'bg-slate-600'}">
      </div>
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-[10px] font-bold text-slate-300">${d.label}</span>
          <span class="text-[9px] font-mono
                       ${d.bull === true ? 'text-emerald-400' : d.bull === false ? 'text-rose-400' : 'text-slate-500'}">
            ${d.value}
          </span>
        </div>
        <div class="text-[9px] text-slate-500 leading-relaxed mt-0.5">${d.tip}</div>
      </div>
    </div>`
  ).join('');

  const sec4 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 flex flex-col gap-1 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock AI Model Analysis') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">

        <!-- Header -->
        <div class="flex items-center gap-2 mb-3">
          <div class="w-6 h-6 rounded-md bg-orange-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-brain text-orange-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">
            AI Model Analysis
          </span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-amber-500/15 text-amber-300 border border-amber-500/25 uppercase">
            PRO
          </span>
        </div>

        <!-- 3-class conviction bar -->
        <div class="mb-4">
          <div class="flex justify-between items-center mb-1.5">
            <span class="text-[10px] text-slate-500 font-mono uppercase tracking-widest">
              XGBoost Raw Probabilities
            </span>
            <span class="text-[11px] font-black font-mono ${topIsLong ? 'text-emerald-400' : 'text-rose-400'}">
              ${topIsLong ? `${pBuyPct}% LONG` : `${pSellPct}% SHORT`}
            </span>
          </div>
          <!-- Stacked bar: SELL | HOLD | BUY -->
          <div class="h-8 flex rounded-xl overflow-hidden border border-slate-700/50">
            <div class="flex items-center justify-center bg-rose-500/50 transition-all duration-700"
                 style="width:${pSellPct}%" title="SELL: ${pSellPct}%">
              ${pSellPct > 10 ? `<span class="text-[9px] font-bold text-rose-100">${pSellPct}%</span>` : ''}
            </div>
            <div class="flex items-center justify-center bg-slate-600/50 transition-all duration-700"
                 style="width:${pHoldPct}%" title="HOLD: ${pHoldPct}%">
              ${pHoldPct > 8 ? `<span class="text-[9px] font-bold text-slate-200">${pHoldPct}%</span>` : ''}
            </div>
            <div class="flex items-center justify-center bg-emerald-500/50 transition-all duration-700"
                 style="width:${pBuyPct}%" title="LONG: ${pBuyPct}%">
              ${pBuyPct > 10 ? `<span class="text-[9px] font-bold text-emerald-100">${pBuyPct}%</span>` : ''}
            </div>
          </div>
          <div class="flex justify-between text-[9px] font-mono mt-1">
            <span class="text-rose-400">SELL ${pSellPct}%</span>
            <span class="text-slate-500">HOLD ${pHoldPct}%</span>
            <span class="text-emerald-400">LONG ${pBuyPct}%</span>
          </div>
        </div>

        <!-- Meta confidence vs threshold -->
        <div class="mb-4 text-[10px] font-mono text-slate-400 flex items-center gap-2 flex-wrap">
          <span>Meta gate:</span>
          <span class="font-black ${metaConfPct >= threshPct ? 'text-emerald-400' : 'text-rose-400'}">
            ${metaConfPct}%
          </span>
          <span class="text-slate-600">vs required</span>
          <span class="text-amber-400 font-bold">${threshPct}%</span>
          <span class="${metaConfPct >= threshPct ? 'text-emerald-400/70' : 'text-rose-400/70'}">
            ${metaConfPct >= threshPct ? '→ FIRE ✓' : '→ NO FIRE ✗'}
          </span>
        </div>

        <!-- Indicator pills (quick glance) -->
        <div class="mb-3">
          <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-2">
            Key Drivers at-a-glance
          </div>
          <div class="flex flex-wrap gap-1.5">${driverPills}</div>
        </div>

        <!-- Detailed driver breakdown -->
        <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-1">
          Detailed Breakdown
        </div>
        <div class="bg-slate-800/30 rounded-xl border border-slate-700/30 px-3 py-1">
          ${driverDetail}
        </div>

        <!-- Beginner note -->
        <div class="mt-3 px-3 py-2 rounded-lg bg-slate-800/30 border border-slate-700/30">
          <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-0.5">For beginners</div>
          <div class="text-[10px] text-slate-400 leading-relaxed">
            The AI looks at 200+ indicators and gives three numbers: chance of going UP, chance of going DOWN, and chance of going SIDEWAYS.
            Right now it says ${pBuyPct}% UP · ${pSellPct}% DOWN · ${pHoldPct}% SIDEWAYS.
            The signal only fires when the combined score (${metaConfPct}%) exceeds the required confidence level (${threshPct}%).
          </div>
        </div>

      </div>
    </div>`;

  // ── SECTION 5: API Export (unchanged) ──────────────────────────
  const apiKeyDisplay = tokenData.apiKey || 'aegis_live_••••••••••••';
  const sec5 = `
    <div class="relative bg-slate-900/60 backdrop-blur-md border border-slate-800/80
                rounded-xl p-4 col-span-1 lg:col-span-2 overflow-hidden">
      ${!canPro ? _lock('PRO', '🔒 Unlock API Access & JSON Data Export') : ''}
      <div class="${!canPro ? 'blur-sm pointer-events-none select-none' : ''}">
        <div class="flex items-center gap-2 mb-4">
          <div class="w-6 h-6 rounded-md bg-blue-500/10 flex items-center justify-center flex-shrink-0">
            <i class="fas fa-code text-blue-400 text-[10px]"></i>
          </div>
          <span class="text-[11px] font-black uppercase tracking-widest text-slate-200">API / JSON Data Export</span>
          <span class="ml-auto text-[9px] font-bold px-2 py-0.5 rounded-full
                       bg-blue-500/15 text-blue-300 border border-blue-500/25 uppercase tracking-widest">
            DEVELOPER
          </span>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <div class="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-2">Your API Key</div>
            <div class="flex items-center gap-2 bg-slate-800/60 border border-slate-700/40 rounded-xl p-3 mb-3">
              <i class="fas fa-key text-blue-400 text-xs flex-shrink-0"></i>
              <span class="font-mono text-sm text-slate-200 flex-1 tracking-wider truncate select-all"
                    id="tdp-api-key" data-raw-key="">${apiKeyDisplay}</span>
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
              Key shown once — store it securely
            </div>
          </div>
          <div>
            <div class="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-2">Python Quick-Start</div>
            <div class="bg-slate-950/80 border border-slate-700/40 rounded-xl overflow-hidden">
              <div class="flex items-center justify-between px-3 py-1.5 bg-slate-800/50 border-b border-slate-700/30">
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

  return `
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 p-1" data-tdp-symbol="${sym}">
      ${sec1}
      ${sec2}
      ${sec3}
      ${sec4}
      ${sec5}
    </div>`;
}

// ── Mount: render + bind live price ──────────────────────────────
export function mountTokenDetailsPanel(containerEl, tokenData, userTier) {
  if (!containerEl) return;
  containerEl.innerHTML = renderTokenDetailsPanel(tokenData, userTier);
  const sym = tokenData.symbol || '';

  // Remove previous listener
  if (containerEl._tdpPriceHandler)
    window.removeEventListener('priceUpdate', containerEl._tdpPriceHandler);

  containerEl._tdpPriceHandler = (e) => {
    if (!e.detail || e.detail.symbol !== sym) return;
    const price = parseFloat(e.detail.price);
    if (isNaN(price) || price <= 0) return;

    // Move price dot using data-range-low / data-range-high stored on the dot
    const dot = containerEl.querySelector('#tdp-price-dot');
    if (dot) {
      const lo = parseFloat(dot.dataset.rangeLow)  || 0;
      const hi = parseFloat(dot.dataset.rangeHigh) || 1;
      const pct = Math.min(97, Math.max(3, ((price - lo) / (hi - lo)) * 100));
      dot.style.left = `${pct.toFixed(1)}%`;
    }

    // Update live price label
    const label = containerEl.querySelector('#tdp-live-price-label');
    if (label) label.textContent = `$${_px(price)}`;
  };
  window.addEventListener('priceUpdate', containerEl._tdpPriceHandler);

  // Async-load global historical stats for the Expectancy section
  _loadHistStats(containerEl);
}

// ── Async historical stats loader ────────────────────────────────
async function _loadHistStats(containerEl) {
  const el = containerEl.querySelector('#tdp-hist-stats');
  if (!el) return;
  try {
    const r = await fetch('/web/track_record.json', { cache: 'no-cache' });
    if (!r.ok) throw new Error('no data');
    const d = await r.json();
    const s = d.summary || {};
    const wr    = s.win_rate_pct != null ? s.win_rate_pct.toFixed(1) + '%' : '—';
    const total = s.total_signals ?? '—';
    const wins  = s.wins  ?? '—';
    const loss  = s.losses ?? '—';
    const avgPnl= s.avg_pnl_pct != null
      ? (s.avg_pnl_pct >= 0 ? '+' : '') + s.avg_pnl_pct.toFixed(2) + '%'
      : '—';
    const totalPnl = s.total_pnl_pct != null
      ? (s.total_pnl_pct >= 0 ? '+' : '') + s.total_pnl_pct.toFixed(2) + '%'
      : '—';
    el.innerHTML = `
      <div class="text-[9px] uppercase tracking-widest text-slate-600 font-mono mb-1.5">Live Track Record (AEGIS-1)</div>
      <div class="grid grid-cols-3 gap-x-4 gap-y-1">
        <div><span class="text-slate-500">Signals:</span> <span class="text-white font-bold">${total}</span></div>
        <div><span class="text-slate-500">Wins:</span>    <span class="text-emerald-400 font-bold">${wins}</span></div>
        <div><span class="text-slate-500">Losses:</span>  <span class="text-rose-400 font-bold">${loss}</span></div>
        <div><span class="text-slate-500">Win Rate:</span><span class="${parseFloat(wr) >= 50 ? 'text-emerald-400' : 'text-rose-400'} font-bold">${wr}</span></div>
        <div><span class="text-slate-500">Avg PnL:</span> <span class="${avgPnl.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'} font-bold">${avgPnl}</span></div>
        <div><span class="text-slate-500">Total:</span>   <span class="${totalPnl.startsWith('+') ? 'text-emerald-400' : 'text-rose-400'} font-bold">${totalPnl}</span></div>
      </div>`;
  } catch {
    el.innerHTML = `<span class="text-slate-600">Track record loading… refresh in a moment.</span>`;
  }
}

// ── Cleanup ───────────────────────────────────────────────────────
export function unmountTokenDetailsPanel(containerEl) {
  if (!containerEl || !containerEl._tdpPriceHandler) return;
  window.removeEventListener('priceUpdate', containerEl._tdpPriceHandler);
  delete containerEl._tdpPriceHandler;
}

// ── Window helpers (inline onclick) ──────────────────────────────
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
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generating…'; }
  try {
    const { getAuth } = await import('https://www.gstatic.com/firebasejs/12.12.1/firebase-auth.js');
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
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Regenerate API Key'; }
  }
};
