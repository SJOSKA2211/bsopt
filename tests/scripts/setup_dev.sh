#!/bin/bash
set -e

echo "Setting up development environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements_api.txt -r requirements-auth.txt

# Compile Protobuf
echo "Compiling Protobuf schemas..."
python -m grpc_tools.protoc -I. --python_out=. src/utils/schemas.proto

echo "Setup complete! Run 'source venv/bin/activate' to use the environment."
