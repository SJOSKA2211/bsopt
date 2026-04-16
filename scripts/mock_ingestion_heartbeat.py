import json
import time

heartbeat_path = "/tmp/ingestion_heartbeat"
heartbeat_data = {
    "time": time.time(),
    "metrics": {
        "status": "MOCKED_ACTIVE",
        "processed_count": 0,
        "health": "ACTIVE"
    }
}

try:
    with open(heartbeat_path, "w") as f:
        f.write(json.dumps(heartbeat_data))
    print(f" Mocked heartbeat written to {heartbeat_path}")
except Exception as e:
    print(f" Failed to write mock heartbeat: {e}")
