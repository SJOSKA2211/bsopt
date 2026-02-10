#!/bin/bash
set -e

echo " Starting Frontend (Local)..."

cd src/frontend

# Use npm quietly, skip install if node_modules exists
if [ ! -d "node_modules" ]; then
    npm install
fi

npm run dev
