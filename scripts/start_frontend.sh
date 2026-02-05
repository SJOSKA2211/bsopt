#!/bin/bash
set -e

echo "🥒 Starting Frontend (Local)..."

cd src/frontend
# Ensure dependencies (pnpm assumed installed or we use npm if pnpm missing, but package.json has pnpm lock likely)
# The user env might not have pnpm. I'll check or just use npm if I have to, but package.json said "pnpm".
# I'll try pnpm, fallback to npm.

if command -v pnpm &> /dev/null; then
    pnpm install
    pnpm dev
else
    echo "pnpm not found, using npm..."
    npm install
    npm run dev
fi
