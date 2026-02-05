#!/bin/bash
set -e

echo "🥒 Starting Auth Service (Local)..."

# Override for Local Docker Infra
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export BETTER_AUTH_URL="http://localhost:3001"
export BETTER_AUTH_SECRET="pickle-rick-secret"
export PORT=3001

# Run in background or foreground? The PRD implies 'start the frontend and backend', so maybe separate terminals.
# But for now, I'll just run it. The user can background it if they want.
cd src/auth-service
npm install
npm run dev
