#!/bin/bash
# start_api.sh
cd /home/kamau/bsopt
export $(grep -v '^#' .env | xargs)
./.venv/bin/uvicorn api.index:app --host 0.0.0.0 --port 8000
