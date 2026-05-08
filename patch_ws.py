import re
import os

main_path = r'd:\Content\Animesh\bots\ai_signal_bot\main.py'
with open(main_path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'(response_data = \{\s*"tickers": \{k: v for k, v in LIVE_STATE\.data\["tickers"\]\.items\(\) if k in allowed_tokens\},\s*"signals": filtered_signals,\s*"open_trades": LIVE_STATE\.data\["open_trades"\],\s*"balance": LIVE_STATE\.data\["balance"\],\s*"alpha_mode": LIVE_STATE\.data\["alpha_mode"\] and \(get_user_plan\(current_user_email\) == "pro" if current_user_email else False\),\s*"warmup": LIVE_STATE\.data\["warmup_progress"\],\s*"trial_expired": trial_expired if current_user_email else True,\s*"plan": get_user_plan\(current_user_email\) if current_user_email else "trial"\s*\}.*?await websocket\.send_json\(clean_data\))', re.DOTALL)

replacement = """response_data = {
                "type": "dashboard_update",
                "tickers": {k: v for k, v in LIVE_STATE.data["tickers"].items() if k in allowed_tokens},
                "signals": filtered_signals,
                "open_trades": LIVE_STATE.data["open_trades"],
                "balance": LIVE_STATE.data["balance"],
                "alpha_mode": LIVE_STATE.data["alpha_mode"] and (get_user_plan(current_user_email) == "pro" if current_user_email else False),
                "warmup": LIVE_STATE.data["warmup_progress"],
                "trial_expired": trial_expired if current_user_email else True,
                "plan": get_user_plan(current_user_email) if current_user_email else "trial"
            }
            clean_data = jsonable_encoder(response_data)
            await websocket.send_json(clean_data)
            
            # Unified signal_update schema
            for sym, sig in LIVE_STATE.data["signals"].items():
                if sym in allowed_tokens:
                    signal_payload = {
                        "pair": sym,
                        "signal": sig.get("signal", "WAITING"),
                        "entry": sig.get("entry", 0),
                        "sl": sig.get("sl", 0),
                        "tp": sig.get("tp", 0),
                        "status": sig.get("status", "OPEN"),
                        "time": sig.get("time", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
                        "timeframe": sig.get("timeframe", "15m"),
                        "confidence": sig.get("ai_prob", sig.get("confidence", 0)),
                        "rr": sig.get("rr", 0),
                        "atr": sig.get("atr", 0)
                    }
                    await websocket.send_json({
                        "type": "signal_update",
                        "data": jsonable_encoder(signal_payload)
                    })"""

if pattern.search(content):
    content = pattern.sub(replacement, content)
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated successfully')
else:
    print('Pattern not found')
