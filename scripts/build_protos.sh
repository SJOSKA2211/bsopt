#!/bin/bash
# scripts/build_protos.sh - Protocol Generation Factory
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GEN_DIR="src/shared/protos"
FBS_DIR="src/shared/fbs"
TS_DIR="src/frontend/src/generated"

echo "🧬 Executing Protocol Generation..."

# 1. Python gRPC
echo "🐍 Generating Python gRPC code..."
mkdir -p "$GEN_DIR"
uv run python -m grpc_tools.protoc -I./protos --python_out="$GEN_DIR" --grpc_python_out="$GEN_DIR" ./protos/*.proto
touch "$GEN_DIR/__init__.py"
# Fix absolute imports for all generated protos
sed -i 's/import \([^ ]*\)_pb2/from . import \1_pb2/g' "$GEN_DIR"/*_pb2*.py 2>/dev/null || true


# 2. TypeScript gRPC
echo "🧪 Generating TypeScript gRPC definitions..."
mkdir -p "$TS_DIR"
if command -v protoc &> /dev/null && [ -f "./protos/market_data.proto" ]; then
    npx protoc --proto_path=./protos --ts_out="$TS_DIR" ./protos/*.proto 2>/dev/null || echo "⚠️ protoc-gen-ts execution failed."
else
    echo "⚠️ protoc not found, skipping TS generation."
fi

# 3. FlatBuffers
echo "📦 Generating FlatBuffers code..."
mkdir -p "$FBS_DIR"
if command -v flatc &> /dev/null; then
    flatc --python -o "$FBS_DIR" protos/market_tick.fbs
else
    echo "⚠️ flatc not found, skipping FlatBuffers generation."
fi
touch "$FBS_DIR/__init__.py"

echo "✅ Production Protocol Synchronized."
