import json
import time

heartbeat_path = "/tmp/frontend_heartbeat"
heartbeat_data = {
    "time": time.time(),
    "metrics": {
        "status": "MOCKED_ACTIVE",
        "health": "ACTIVE"
    }
}

try:
    with open(heartbeat_path, "w") as f:
        f.write(json.dumps(heartbeat_data))
    print(f" Mocked frontend heartbeat written to {heartbeat_path}")
except Exception as e:
    print(f" Failed to write mock frontend heartbeat: {e}")
