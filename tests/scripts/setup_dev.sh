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
python -m grpc_tools.protoc -I. --python_out=. core/shared/utils/schemas.proto

echo "Setup complete!"
echo "💡 TIP: Use ./bootstrap.sh for full containerized stack initialization."
