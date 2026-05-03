#!/usr/bin/env python3
import inspect
import ccxt.async_support as ccxt_async

print('ccxt_async module loaded')
exchange_cls = getattr(ccxt_async, 'binance', None)
print('exchange_cls:', exchange_cls)
if exchange_cls:
    try:
        print('init signature:', inspect.signature(exchange_cls.__init__))
    except Exception as e:
        print('Could not get signature:', e)
    attrs = [a for a in dir(exchange_cls) if 'session' in a.lower() or 'aiohttp' in a.lower()]
    print('session-related attrs:', attrs)
    instance_attrs = [a for a in dir(exchange_cls)[:200]]
    print('first 40 attrs of class:', instance_attrs[:40])
