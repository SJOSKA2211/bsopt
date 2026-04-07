#!/bin/bash
export REDIS_PORT=6380
export PYTHONPATH=.
/home/kamau/bsopt/.venv/bin/python -m uvicorn api.index:app --host 0.0.0.0 --port 8000
