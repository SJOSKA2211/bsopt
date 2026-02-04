---
id: auth002
title: "Implementation: Resource Server & Client App Refactor"
status: Done
priority: High
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: /home/kamau/bsopt/.gemini/pickle-rick/tickets/linear_ticket_parent.md
    title: Parent Ticket
labels: [implementation, auth]
assignee: Morty
---

# Description

## Problem to solve
The Pricing API and other endpoints act as a "Resource Server" but lack a unified way to validate the new JWTs from our Auth Server. The frontend (Client App) also needs to be updated to use the new Authorization Code flow.

## Solution
1. Implement a JWT verification middleware in src/api/middleware/security.py that validates tokens against our new Auth Server's JWKS.
2. Update src/frontend/src/lib/auth-client.ts to support the OAuth2 Authorization Code flow with PKCE.
3. Remove legacy session cookie checks.
