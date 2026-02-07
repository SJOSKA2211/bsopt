#!/bin/bash
set -e

echo "🥒 Starting Auth Service (Local)..."

# Override for Local Docker Infra
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export BETTER_AUTH_BASE_URL="http://localhost:3001"
export BETTER_AUTH_URL="http://localhost:3001"
export BETTER_AUTH_SECRET="development_secret_pickle_rick_is_the_best_scientist_in_the_multiverse_32_char"
export PORT=3001

cd src/auth-service
# Use npm install only if node_modules is missing to speed up startup
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev
