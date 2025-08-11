import sys
import os
import time
import websockets.sync.client
from lerobot.common.utils.msgpack_utils import unpackb

def get_try_round(default=60):
    try:
        return int(os.environ.get('TRY_ROUND', default))
    except Exception:
        return default

host = os.environ.get('INFER_SERVER_HOST')
port = os.environ.get('INFER_SERVER_PORT')
if not host or not port:
    print('[ERROR] INFER_SERVER_HOST or INFER_SERVER_PORT not set')
    sys.exit(1)

ws_url = f'ws://{host}:{port}/'
try_round = get_try_round()

for i in range(try_round):
    try:
        print(f'[INFO] Attempt {i+1}: Connecting to {ws_url}')
        ws = websockets.sync.client.connect(ws_url, compression=None, max_size=None)
        try:
            msg = unpackb(ws.recv())
            print(f'[SUCCESS] Infer-server is ready. Received message: {msg}')
            ws.close()
            sys.exit(0)
        except Exception as e:
            print(f'[ERROR] Failed to receive or unpack message: {e}')
    except Exception as e:
        print(f'[WARN] Connection attempt failed: {e}')
    time.sleep(2)

print(f'[ERROR] Infer-server not ready after {try_round} attempts.')
sys.exit(1)

