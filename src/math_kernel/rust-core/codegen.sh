#!/bin/bash
set -e

# Canonical schema location
SCHEMA="../../../shared/schemas/market_tick.fbs"

echo "Generating FlatBuffer code for Rust..."
flatc --rust -o src/generated "$SCHEMA"

echo "Generating FlatBuffers code for Python..."
flatc --python -o ../../ingestion/generated "$SCHEMA"

echo "Codegen complete. Best practice established."
