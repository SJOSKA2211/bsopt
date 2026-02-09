#!/bin/bash
set -e

# Re-initialize git to ensure config is valid
git init

# Remove existing origin if it exists
git remote remove origin || true

# Add new origin
git remote add origin https://github.com/SJOSKA2211/bsopt.git

# Check status
git status

# Add all changes
git add .

# Commit changes (allow empty if nothing to commit)
git commit -m "Update codebase" || echo "Nothing to commit"

# Rename branch to main
git branch -M main

# Push to origin main
echo "Pushing to origin main..."
git push -u origin main
