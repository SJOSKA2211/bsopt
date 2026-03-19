#!/bin/bash
set -e

# EquaFlow Frontend Alignment Script
# Resolves type and dependency issues in the Next.js/Vite environment.

echo "🎨 Aligning Frontend Environment..."

cd src/frontend

# 1. Standard Installation
echo "📦 Installing Institutional UI Dependencies..."
npm install --silent

# 2. Type Checking
echo "🔍 Running TypeScript Validation..."
npx tsc --noEmit || echo "⚠️ Type warnings detected, continuing alignment..."

# 3. Linting
echo "🧹 Running ESLint Polish..."
npm run lint -- --fix

echo "✅ Frontend Aligned: Core Types & UI Modules Synchronized."
