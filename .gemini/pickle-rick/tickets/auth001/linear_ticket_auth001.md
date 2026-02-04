---
id: auth001
title: "Implementation: Unified Auth Server"
status: Triage
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
We lack a centralized authentication and authorization hub. Client applications are hard-coding security logic or using inconsistent providers.

## Solution
Implement a unified Auth Server using FastAPI and Authlib. 
1. Support OAuth2 Authorization Code flow.
2. Implement User management and registration.
3. Integrate with Neon for persistent user and token storage.
4. Export JWT tokens with custom claims for the Pricing Resource Server.
