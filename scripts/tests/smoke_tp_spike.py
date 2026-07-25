from types import MethodType
import time
from datetime import datetime, timezone

from scripts.live_engine import LiveEngine, Position, DynamicRiskEngine

class FakeWallet:
    def __init__(self, engine):
        self.engine = engine
        self.trade_history = []

    def partial_close_trade(self, symbol, px, reason, pct):
        pos = self.engine.open_positions.get(symbol)
        if not pos:
            return None
        # closed value approximate
        closed_val = pos.position_value * pct
        # compute pnl_pct for closed slice
        if pos.direction == 'LONG':
            pnl_pct = (px - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - px) / pos.entry_price * 100
        rec = type('R', (), {})()
        rec.signal_id = pos.signal_id
        rec.pnl_pct = round(pnl_pct, 3)
        rec.pnl_usdt = round(closed_val * pnl_pct / 100, 2)
        rec.position_value = round(closed_val, 2)
        rec.outcome = 'WIN' if rec.pnl_pct > 0 else 'LOSS'
        # append to history
        self.trade_history.append(rec)
        # reduce remaining pos value (simulate partial close)
        pos.position_value = max(0.0, pos.position_value - closed_val)
        return rec

    def close_trade(self, symbol, exit_px, reason):
        pos = self.engine.open_positions.get(symbol)
        if not pos:
            return None
        # close remaining
        close_val = pos.position_value
        if pos.direction == 'LONG':
            pnl_pct = (exit_px - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_px) / pos.entry_price * 100
        rec = type('R', (), {})()
        rec.signal_id = pos.signal_id
        rec.pnl_pct = round(pnl_pct, 3)
        rec.pnl_usdt = round(close_val * pnl_pct / 100, 2)
        rec.position_value = round(close_val, 2)
        rec.outcome = 'WIN' if rec.pnl_pct > 0 else 'LOSS'
        self.trade_history.append(rec)
        # remove position
        self.engine.open_positions.pop(symbol, None)
        return rec

class Stub:
    def record_outcome(self, **kwargs):
        print('perf_tracker.record_outcome called', kwargs)

class DriftStub:
    def record(self, symbol, outcome):
        pass
    def save_state(self):
        pass
    def severity(self, symbol):
        return 'OK'


# Build a fake engine with minimal state required by _manage_exit
class FakeEngine:
    pass

engine = FakeEngine()
engine.live_prices = {}
engine._open_time = {}
engine._tp1_hit = {}
engine._tp2_hit = {}
engine._tp3_hit = {}
engine._tp4_hit = {}
engine._peak_price = {}
engine.MAX_HOLD_SECONDS = 24 * 3600
engine.MIN_HOLD_SECONDS = 3600
engine.risk_engine = DynamicRiskEngine()
engine.wallet = FakeWallet(engine)
engine.perf_tracker = Stub()
engine.drift_monitor = DriftStub()
engine._last_close_time = {}
engine._last_close_side = {}
engine._last_close_reason = {}
engine._last_loss_time = {}
engine._save_track_record = lambda: None
engine.last_signals = {}
# open_positions used by our FakeWallet
engine.open_positions = {}

# Bind the LiveEngine._manage_exit function to our fake engine
_manage_exit = MethodType(LiveEngine._manage_exit, engine)

# Create a test position that has TP1 at +1.8%
symbol = 'TEST/USDT'
entry_price = 100.0
pos = Position(
    symbol=symbol,
    direction='LONG',
    side='BUY',
    entry_price=entry_price,
    position_value=1000.0,
    stop_loss=99.0,
    signal_id='test-signal-1',
    entry_time=datetime.now(timezone.utc).isoformat(),
    meta_confidence=0.0,
    atr_multiplier=1.8,
    atr=0.5,
    take_profit_1=round(entry_price * 1.018, 8),
    take_profit_2=round(entry_price * 1.03, 8),
    take_profit_3=round(entry_price * 1.05, 8),
    take_profit_4=round(entry_price * 1.08, 8),
    take_profit_5=round(entry_price * 1.12, 8),
)
engine.open_positions[symbol] = pos
engine._open_time[symbol] = time.time()
engine._tp1_hit[symbol] = False
engine._peak_price[symbol] = entry_price

# result dict used by _manage_exit
result = {'atr': pos.atr, 'side': 'BUY', 'fire': False, 'edge_score': 0.0}

# Simulate price series: below TP, spike through TP to a peak, then back below TP
prices = [100.2, 101.5, pos.take_profit_1 + 0.5, 100.5]

print('Starting spike-through TP smoke test')
for p in prices:
    engine.live_prices[symbol] = p
    print(f'-- price update {p}')
    _manage_exit(symbol, pos, result, p)
    # small sleep to simulate time passing
    time.sleep(0.1)

print('\nTrade history:')
for t in engine.wallet.trade_history:
    print(vars(t))

if symbol in engine.open_positions:
    print('\nPosition remains open:', engine.open_positions[symbol])
else:
    print('\nPosition closed')
