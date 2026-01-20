#!/bin/bash
set -euo pipefail

PROTO_DIR="protos"
PYTHON_OUT="src/protos"
TYPESCRIPT_OUT="frontend/src/types/protos"

echo "Building Protocol Buffers..."

# Clean old generated files
rm -rf "$PYTHON_OUT"/*_pb2.py "$PYTHON_OUT"/*_pb2.pyi

# Create output directories
mkdir -p "$PYTHON_OUT" "$TYPESCRIPT_OUT"

# Generate Python code
# We use the python from the virtual environment if available
PYTHON_CMD="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
fi

$PYTHON_CMD -m grpc_tools.protoc \
    -I="$PROTO_DIR" \
    --python_out="$PYTHON_OUT" \
    --pyi_out="$PYTHON_OUT" \
    --grpc_python_out="$PYTHON_OUT" \
    "$PROTO_DIR"/*.proto

# Generate TypeScript definitions (for frontend)
# Note: Requires protoc-gen-ts to be installed
if [ -f "node_modules/.bin/protoc-gen-ts" ]; then
    protoc \
        -I="$PROTO_DIR" \
        --plugin=protoc-gen-ts=./node_modules/.bin/protoc-gen-ts \
        --ts_out="$TYPESCRIPT_OUT" \
        "$PROTO_DIR"/*.proto
else
    echo "Warning: protoc-gen-ts not found. Skipping TypeScript generation."
fi

# Create __init__.py
cat > "$PYTHON_OUT/__init__.py" <<EOF
"""
Generated Protocol Buffer definitions.
Auto-generated - do not edit manually.
"""
from .market_data_pb2 import *
EOF

echo "✓ Protocol Buffers compiled successfully"
