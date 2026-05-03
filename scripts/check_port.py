import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('127.0.0.1', 8000))
    print('TCP connect succeeded')
except Exception as e:
    print('TCP connect failed:', e)
    sys.exit(2)
finally:
    try:
        s.close()
    except:
        pass
