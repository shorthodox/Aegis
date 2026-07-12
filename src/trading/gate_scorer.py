"""
gate_scorer.py — Unified Weighted Gate Score (UWGS)

ONE importance-weighted composite score, computed per direction (BUY / SELL / HOLD)
on a 0-100 scale.  The three scores are shares of a 100-point budget split across the
13 gates, weighted by the user's importance ranking:

    5★ = 10   ·   4★ = 8   ·   3★ = 6   ·   2★ = 5      (4·10 + 4·8 + 3·6 + 2·5 = 100)

The highest of (score_buy, score_sell, score_hold) decides the direction, the signal
quality (= the winning score), and — together with the fire rule and the four hard
vetoes — whether the signal fires.  The ML model is just two of the inputs (Model Edge
+ Model Confidence = 14 pts); it no longer picks the side on its own.

Design (agreement-rewarding, each score on a 0-100 scale):
  Phase 1 — the 6 DIRECTIONAL gates (54 pts) each cast buy/sell support in [0,1] of
            their weight. buy_frac / sell_frac = the fraction of directional weight
            each side won, so gates that AGREE reinforce instead of diluting.
  Phase 2 — the 7 QUALITY gates (46 pts) give an average quality Q ∈ [0,1]. The
            quality budget flows to the LEADING direction — half unconditional, half
            scaled by conviction (|buy_frac − sell_frac|). Weak or split direction
            leaves most of it in HOLD.
  Score  — BUY  = buy_frac·54  + (Q·46·(0.5+0.5·conviction) if BUY leads)
           SELL = sell_frac·54 + (…                          if SELL leads)
           HOLD = (1−max_frac)·54 + (46 − qual_gain)

Four HARD VETOES force fire=False regardless of score (winner reported as HOLD for the
purpose of firing): critical model drift, dead/invalid market, no valid S/R setup, and
extreme volatility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Weights (stars → points, sum = 100) ───────────────────────────────────────
W_REGIME     = 10.0   # ★★★★★ Market regime
W_HTF        = 10.0   # ★★★★★ Higher-timeframe macro (weekly + daily)
W_SETUP      = 10.0   # ★★★★★ Direction / setup-type coherence
W_SR         = 10.0   # ★★★★★ Support / resistance validation (quality)
W_MODEL_EDGE = 8.0    # ★★★★  Model edge (p_buy / p_sell)
W_FAKEBREAK  = 8.0    # ★★★★  Fake-breakout detection
W_CONTEXT    = 8.0    # ★★★★  Context quality (score_signal)
W_ATR        = 8.0    # ★★★★  ATR / volatility floor
W_STABILITY  = 6.0    # ★★★   Signal stability
W_MODEL_CONF = 6.0    # ★★★   Model confidence (meta)
W_HMM        = 6.0    # ★★★   HMM stability
W_PORTFOLIO  = 5.0    # ★★    Portfolio (non-binding for a signal service)
W_DRIFT      = 5.0    # ★★    Model drift

# ── Fire rule + veto thresholds (first-pass, tune in one place) ───────────────
FIRE_THRESHOLD   = 45.0   # winning score must reach this (looser — steadier signal flow)
FIRE_MARGIN      = 3.0    # winner must beat HOLD by this much
DIR_MARGIN       = 3.0    # winner must beat the OTHER direction by this much
SR_MIN           = 0.35   # srQuality below this → no-valid-S/R veto (V3)
HOLD_DIR_FACTOR  = 0.7    # fraction of the un-won directional weight that pools in HOLD
                          # (<1 so a few neutral gates don't let HOLD veto a clear lead)
DEAD_ATR_PCT     = 0.15   # atr_pct below this (%) → truly flatlined → dead-market veto (V2)
ATR_QUALITY_FLOOR= 0.20   # ATR quality gate ramps from here; below = low (not zero) quality
EXTREME_ATR_PCT  = 6.0    # atr_pct at/above this (%) → extreme-vol veto (V4) [abs proxy]
EXTREME_ATR_MULT = 3.0    # …or atr_pct ≥ this × rolling-normal atr_pct, when supplied
LOW_VOL_Z        = -1.6   # volume z at/below this (+ weak rel-vol) → dead-market veto
SPREAD_MAX_PCT   = 0.15   # book spread (%) above this → dead-market veto (illiquid)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _f(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        v = float(v)
        return default if v != v else v          # NaN guard
    except (TypeError, ValueError):
        return default


def _sr_rr(result: Dict[str, Any]) -> float:
    """RR headroom from range geometry: far leg / near leg. 0 when unknown."""
    sup = _f(result, 'support')
    res = _f(result, 'resistance')
    px  = _f(result, 'price') or _f(result, 'entry_price')
    if 0 < sup < px < res:
        near_leg = min(px - sup, res - px)
        far_leg  = max(px - sup, res - px)
        return far_leg / near_leg if near_leg > 0 else 0.0
    return 0.0


class WeightedGateScorer:
    """Pure aggregation over already-computed signals. No API calls, no engine deps."""

    # -- S/R quality (also drives veto V3) -------------------------------------
    @staticmethod
    def sr_quality(result: Dict[str, Any]) -> float:
        """
        0..1 quality of the structural level the trade would lean on.  Built from
        what already exists on the result dict: proximity to a level (range_position),
        whether a recent break has been retest-confirmed vs left hanging, HTF
        confluence, and RR headroom.  Higher = fresher / confirmed / roomy.
        """
        rp = _f(result, 'range_position', 0.5)
        # Proximity to a real level: 1 near either extreme (rp≤0.15 or rp≥0.85),
        # ramping to 0 in the dead middle of the range.
        near = _clip((abs(rp - 0.5) - 0.15) / 0.35)

        q = 0.35 + 0.35 * near                          # base + proximity

        # Retest-confirmed structure adds quality; a broken-but-unretested level removes it.
        levels = result.get('sr_levels') or []
        if isinstance(levels, list) and levels:
            states = [str((lv or {}).get('state', '')).upper() for lv in levels if isinstance(lv, dict)]
            if any(s in ('CONFIRMED', 'WAITING_RETEST') for s in states):
                q += 0.15
            if any(s == 'FAILED' for s in states):
                q -= 0.15
        # A very recent break that has NOT been retested = chasing → weak.
        if bool(result.get('resistance_broken_recent')) or bool(result.get('support_broken_recent')):
            q -= 0.10

        # HTF confluence (4h/1d level nearby) firms it up.
        if result.get('htf_support') or result.get('htf_resistance'):
            q += 0.10

        # RR headroom from the range geometry (far leg / near leg) — a setup pinned
        # hard against the near level (little room to the target) is low quality.
        rr = _sr_rr(result)
        if rr >= 2.5:
            q += 0.05
        elif 0 < rr < 1.5:
            q -= 0.15

        return _clip(q)

    # -- main scoring ----------------------------------------------------------
    @classmethod
    def score(cls, result: Dict[str, Any], regime: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        result : predictor/engine signal dict (p_buy/p_sell/p_hold, edge_score,
                 meta_confidence, macro_weekly/daily, range_position, atr_pct, rsi,
                 volume_zscore, hmm_*, sr_levels, risk_reward, is_fake_breakout, …).
        regime : RegimeState (regime.regime, regime.confidence) or None.
        ctx    : engine-side extras — drift_blocked(bool), drift_severity(str),
                 stability_frac(0..1), quality_score(0..100), portfolio_ok(bool),
                 atr_normal_pct(float, optional).
        """
        reg = str(getattr(regime, 'regime', None) or result.get('hmm_regime') or 'UNKNOWN').upper()
        rp  = _f(result, 'range_position', 0.5)                       # 0 support … 1 resistance
        p_buy  = _f(result, 'p_buy')
        p_sell = _f(result, 'p_sell')
        p_hold = _f(result, 'p_hold')
        mw = _f(result, 'macro_weekly'); md = _f(result, 'macro_daily')
        atr_pct = _f(result, 'atr_pct', 1.5)
        vol_z   = _f(result, 'volume_zscore', 0.0)
        rel_vol = _f(result, 'relative_volume', 1.0)

        srq = cls.sr_quality(result)

        W_DIR  = W_REGIME + W_HTF + W_SETUP + W_SR + W_MODEL_EDGE + W_MODEL_CONF                 # 54
        W_QUAL = W_FAKEBREAK + W_CONTEXT + W_ATR + W_STABILITY + W_HMM + W_PORTFOLIO + W_DRIFT   # 46
        dir_buy = dir_sell = 0.0
        breakdown: Dict[str, Dict[str, float]] = {}

        def add_dir(name: str, w: float, lean: float) -> None:
            """Directional gate → buy/sell support (each a fraction of its weight)."""
            nonlocal dir_buy, dir_sell
            lean = _clip(lean, -1.0, 1.0)
            bs = max(0.0, lean)
            ss = max(0.0, -lean)
            dir_buy += w * bs
            dir_sell += w * ss
            breakdown[name] = {'weight': w, 'buy': round(w * bs, 2),
                               'sell': round(w * ss, 2), 'hold': round(w * (1 - bs - ss), 2)}

        # ── PHASE 1 · directional gates ───────────────────────────────────────
        loc_lean = _clip((0.5 - rp) * 2.0, -1.0, 1.0)                 # +1 at support, −1 at resistance
        trend_lean = 1.0 if reg == 'TRENDING_BULL' else -1.0 if reg == 'TRENDING_BEAR' else 0.0
        # Counter-trend-extreme strength: 1.0 when price sits at the level OPPOSING the
        # trend (bear trend at support / bull trend at resistance) — the reversal zone
        # where the trend is stretched and a bounce is likely.
        counter = max(0.0, -trend_lean * loc_lean)

        # 1 · Regime — trend-led, but REVERSAL-AWARE: as price reaches the counter-
        # trend extreme the trend vote fades and tilts toward the reversal, so a
        # bear-trend-at-support stops voting hard SELL (mirror for a bull top). This
        # is what turns a buy-at-support-in-a-downtrend from a conflicted tie into a
        # coherent signal — the reversal setups the engine exists to catch.
        if reg in ('TRENDING_BULL', 'TRENDING_BEAR'):
            add_dir('Regime', W_REGIME, _clip(trend_lean * (1 - counter) + loc_lean * counter * 0.8, -1.0, 1.0))
        elif reg in ('RANGING', 'ACCUMULATION', 'DISTRIBUTION', 'COMPRESSION', 'VOLATILE_EXPANSION'):
            add_dir('Regime', W_REGIME, loc_lean)                    # range → reversal at the extreme
        else:                                                        # CHOPPY / UNKNOWN → all HOLD
            add_dir('Regime', W_REGIME, 0.0)

        # 2 · HTF macro — an ALIGNED weekly+daily bias votes its full weight; an
        # OPPOSING bias penalises the trade; a NEUTRAL HTF (ranging, no bias) is not
        # against the trade, so it lends partial credit to the direction the local
        # structure/trend already favours instead of dumping its full weight to HOLD.
        _htf_bias    = _clip(mw + md, -1.0, 1.0)
        _htf_neutral = 1.0 - min(1.0, abs(_htf_bias) / 0.3)          # 1 when flat, 0 when biased
        _dir_proxy   = _clip(0.5 * trend_lean + 0.5 * loc_lean, -1.0, 1.0)
        add_dir('HTF Macro', W_HTF, _clip(_htf_bias + _htf_neutral * 0.5 * _dir_proxy, -1.0, 1.0))

        # 3 · Direction / setup coherence. In a TREND the coherent setup is a
        # continuation/pullback (trend-led, location-confirmed) — UNLESS price is at
        # the counter-trend extreme, where the coherent setup IS the reversal, so the
        # lean flips to location. In a RANGE the reversal at the extreme is the whole
        # thesis, so location leads outright.
        if reg in ('TRENDING_BULL', 'TRENDING_BEAR'):
            _cont = 0.6 * trend_lean + 0.4 * loc_lean               # continuation / pullback
            setup_lean = _clip(_cont * (1 - counter) + loc_lean * counter, -1.0, 1.0)
        else:
            setup_lean = loc_lean
        add_dir('Direction/Setup', W_SETUP, setup_lean)

        # 4 · S/R validation — location gated by structural quality
        add_dir('S/R', W_SR, _clip(loc_lean * srq, -1.0, 1.0))

        # 5 · Model edge — directional preference, damped by p_hold
        denom = p_buy + p_sell + 1e-9
        model_lean = _clip((p_buy - p_sell) / denom, -1.0, 1.0) * (1.0 - _clip(p_hold))
        add_dir('Model Edge', W_MODEL_EDGE, model_lean)

        # 10 · Model confidence — meta on the model's favoured side
        meta = _f(result, 'meta_confidence', _f(result, 'edge_score'))
        meta = meta / 100.0 if meta > 1.0 else meta
        conf_mag = _clip((meta - 0.5) * 2.0)                         # >50% → ramps in
        conf_dir = 1.0 if p_buy >= p_sell else -1.0
        add_dir('Model Confidence', W_MODEL_CONF, conf_dir * conf_mag)

        # ── PHASE 2 · quality gates ───────────────────────────────────────────
        # Directional support so far, as a fraction of the 54 directional points.
        buy_frac  = dir_buy  / W_DIR
        sell_frac = dir_sell / W_DIR
        pref_buy  = buy_frac / (buy_frac + sell_frac + 1e-9)
        pref_sell = 1.0 - pref_buy
        qual_sum  = 0.0

        def add_qual(name: str, w: float, q: float) -> None:
            """Quality gate → 0..1 quality; its weight flows to the leader below."""
            nonlocal qual_sum
            q = _clip(q)
            qual_sum += w * q
            breakdown[name] = {'weight': w, 'q': round(q, 2),
                               'buy': round(w * q * pref_buy, 2), 'sell': round(w * q * pref_sell, 2),
                               'hold': round(w * (1 - q), 2)}

        # 6 · Fake breakout — trap → 0 quality (weight to HOLD)
        add_qual('Fake Breakout', W_FAKEBREAK, 0.0 if bool(result.get('is_fake_breakout')) else 1.0)
        # 7 · Context quality — reuse SignalQualityFilter score (0..100)
        add_qual('Context Quality', W_CONTEXT, _f(ctx, 'quality_score', 50.0) / 100.0)
        # 8 · ATR floor — healthy band = full, extreme = 0. Low volatility is a soft
        # quality penalty (floor 0.3), NOT a zero — a calm-but-liquid market is still
        # tradeable; only a genuine ATR spike zeroes it (the veto handles the extremes).
        if atr_pct >= EXTREME_ATR_PCT:
            atr_q = 0.0
        else:
            atr_q = _clip((atr_pct - ATR_QUALITY_FLOOR) / (2.0 - ATR_QUALITY_FLOOR), 0.3, 1.0)
        add_qual('ATR Floor', W_ATR, atr_q)
        # 9 · Signal stability
        add_qual('Signal Stability', W_STABILITY, _f(ctx, 'stability_frac', 0.5))
        # 11 · HMM stability
        hmm_q = _f(result, 'hmm_stability', 0.6)
        if bool(result.get('hmm_transition_risk')):
            hmm_q = min(hmm_q, 0.3)
        add_qual('HMM Stability', W_HMM, hmm_q)
        # 12 · Portfolio (non-binding for a signal service)
        add_qual('Portfolio', W_PORTFOLIO, 1.0 if ctx.get('portfolio_ok', True) else 0.6)
        # 13 · Drift
        sev = str(ctx.get('drift_severity', 'OK')).upper()
        add_qual('Model Drift', W_DRIFT, 0.0 if sev == 'CRITICAL' else 0.5 if sev == 'WARNING' else 1.0)

        # Agreement-rewarding aggregation: directional fraction × 54, plus the
        # quality budget (46) flowing to the leading side — half unconditional,
        # half scaled by how decisive the direction is (conviction). Weak or split
        # direction leaves most of the quality budget in HOLD.
        Q          = qual_sum / W_QUAL
        conviction = abs(buy_frac - sell_frac)
        qual_gain  = Q * W_QUAL * (0.65 + 0.35 * conviction)
        leader_buy = buy_frac >= sell_frac
        score_buy  = round(_clip(buy_frac  * W_DIR + (qual_gain if leader_buy else 0.0), 0.0, 100.0), 1)
        score_sell = round(_clip(sell_frac * W_DIR + (0.0 if leader_buy else qual_gain), 0.0, 100.0), 1)
        score_hold = round(_clip((1.0 - max(buy_frac, sell_frac)) * W_DIR * HOLD_DIR_FACTOR
                                 + (W_QUAL - qual_gain), 0.0, 100.0), 1)

        # ── HARD VETOES ───────────────────────────────────────────────────────
        vetoes: List[str] = []
        if bool(ctx.get('drift_blocked')) or sev == 'CRITICAL':
            vetoes.append('MODEL_DRIFT_CRITICAL')
        # V2 dead / invalid market — LIQUIDITY, not low volatility: collapsed volume,
        # a wide book spread, or a genuinely flatlined ATR. A calm-but-liquid market
        # (e.g. 0.49% ATR with healthy volume) is tradeable and must NOT be vetoed.
        spread_pct = _f(ctx, 'spread_pct', 0.0)
        if (atr_pct < DEAD_ATR_PCT or (vol_z <= LOW_VOL_Z and rel_vol < 0.6)
                or spread_pct > SPREAD_MAX_PCT):
            vetoes.append('DEAD_MARKET')
        _rr = _sr_rr(result)
        if srq < SR_MIN or (0.42 < rp < 0.58) or (0 < _rr < 1.2):
            vetoes.append('NO_VALID_SR')
        # V4 extreme volatility / news lock — ATR spike OR a scheduled macro event.
        atr_norm = _f(ctx, 'atr_normal_pct', 0.0)
        if atr_pct >= EXTREME_ATR_PCT or (atr_norm > 0 and atr_pct >= EXTREME_ATR_MULT * atr_norm):
            vetoes.append('EXTREME_VOLATILITY')
        if ctx.get('news_locked'):
            vetoes.append(str(ctx.get('news_label') or 'NEWS_LOCK'))

        # ── Winner + fire decision ────────────────────────────────────────────
        scores = {'BUY': score_buy, 'SELL': score_sell, 'HOLD': score_hold}
        winner = max(scores, key=scores.get)
        win_score = scores[winner]
        other_dir = score_sell if winner == 'BUY' else score_buy

        fire = (
            not vetoes
            and winner in ('BUY', 'SELL')
            and win_score >= FIRE_THRESHOLD
            and (win_score - score_hold) >= FIRE_MARGIN
            and (win_score - other_dir) >= DIR_MARGIN
        )

        return {
            'score_buy':  score_buy,
            'score_sell': score_sell,
            'score_hold': score_hold,
            'sr_quality': round(srq, 2),
            'winner':     winner if fire else 'HOLD',
            'side':       winner if fire else 'FLAT',
            'fire':       fire,
            'quality_score': win_score,
            'vetoes':     vetoes,
            'breakdown':  breakdown,
        }
