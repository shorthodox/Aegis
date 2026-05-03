import asyncio
import json

try:
    import websockets
except Exception as e:
    print('websockets not installed:', e)
    raise

async def main():
    uri = 'ws://127.0.0.1:8000/ws/dashboard'
    print('Connecting to', uri)
    try:
        async with websockets.connect(uri) as ws:
            for i in range(3):
                msg = await ws.recv()
                print('MSG', i+1, ':', msg)
    except Exception as e:
        print('WS client error:', e)

if __name__ == '__main__':
    asyncio.run(main())
