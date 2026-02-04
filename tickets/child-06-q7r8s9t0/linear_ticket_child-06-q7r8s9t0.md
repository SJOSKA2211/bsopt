---
id: child-06-q7r8s9t0
title: 'Configure Frontend Debugging (Source Maps, Remote)'
status: Done
priority: High
project: project
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent-b17f8a9e.md
    title: Parent Ticket
labels: [docker, frontend, debugging, source-maps, remote-debug]
assignee: Pickle Rick
---

# Description

## Problem to solve
Debugging complex frontend applications running in Docker containers is challenging without proper tools like source maps and remote debugging capabilities, leading to extended debug times and developer frustration.

## Solution
Implement and configure source map generation for the `src/frontend` application within its Docker container, allowing developers to debug original source code. Additionally, configure the container and development environment to enable remote debugging from IDEs (e.g., VS Code) to the running frontend service.

# Discussion/Comments

- 2026-02-03 Pickle Rick: Child ticket created for configuring frontend debugging (source maps and remote debugging) as per the PRD.
