---
id: oauth001
title: "Unified OAuth2/OIDC System Implementation"
status: Triage
priority: High
project: bsopt
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [auth, security]
assignee: Pickle Rick
---

# Description
## Problem to solve
The current system has hardcoded Keycloak integration and lacks support for external providers (Google/GitHub).

## Solution
Implement a multi-provider OAuth system using Authlib, supporting internal and external OIDC flows.
