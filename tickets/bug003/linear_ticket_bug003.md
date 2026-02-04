---
id: bug003
title: "Resolve auth-service test environment dependency issues"
status: Triage
priority: High
project: project
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../../linear_ticket_parent.md
    title: Parent Ticket
labels: bug, testing, python, auth-service
assignee: Pickle Rick
---

# Description

## Problem to solve
Tests `TC003` and `TC004` fail because `psycopg2` and `jsonschema` are missing from the test execution environment. This blocks validation of DB connectivity and OpenAPI compliance.

## Solution
Identify the correct `requirements.txt` or environment setup script for the `auth-service` tests. Add `psycopg2-binary` and `jsonschema`. Verify that the test runner can import these modules.
