---
id: res001
title: "Research: OAuth2 Consolidation Strategy"
status: Triage
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: /home/kamau/bsopt/.gemini/pickle-rick/tickets/linear_ticket_parent.md
    title: Parent Ticket
labels: [research, auth]
assignee: Pickle Rick
---

# Description

## Problem to solve
The current auth logic is scattered across `src/auth/better_auth.py`, `src/auth/security.py`, and various middleware. We need a clear map of how to consolidate this into a unified Auth Server, Resource Server, and Client App architecture.

## Solution
1. Audit all existing auth entry points.
2. Design the OAuth2 flow (Authorization Code with PKCE).
3. Identify dependencies for the new Auth Server (e.g., Authlib, FastAPI).
4. Define the token storage and validation strategy using Neon.
