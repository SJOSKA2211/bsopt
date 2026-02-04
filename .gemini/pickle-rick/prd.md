# bsopt System Overhaul & Modernization PRD

## HR Eng

| bsopt Overhaul PRD |  | Summary: A comprehensive structural and functional modernization of the bsopt platform, focusing on high-performance ML, secure OAuth architecture, and Neon integration. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-02-03 | **Self Link**: [Local File] **Context**: System Overhaul |

## Introduction

The `bsopt` platform requires a transition from "Jerry-grade" prototypes to a "God-mode" production architecture. This overhaul targets the authentication layer, the reinforcement learning pipeline, and the database infrastructure.

## Problem Statement

**Current Process:** 
- Authentication is handled by a hardcoded Keycloak integration that lacks flexibility.
- The RL agent uses a flat observation vector, missing critical temporal patterns.
- Database connections are not optimized for serverless (Neon) scaling.
- The codebase contains significant "AI Slop" (boilerplate and inefficient loops).

**Primary Users:** Quant researchers, automated trading systems.
**Pain Points:** Rigid security, mediocre RL performance, technical debt.
**Importance:** To maintain a competitive edge, the system must process data faster and learn more complex market dynamics.

## Objective & Scope

**Objective:** Transform `bsopt` into a secure, high-performance, and scalable platform with advanced temporal ML capabilities.
**Ideal Outcome:** A system with a modular OAuth triad, Transformer-powered RL, and a serverless-optimized backend on Neon.

### In-scope or Goals
- Implement a modular OAuth2 triad (Auth Server, Client App, Resource Server).
- Integrate Neon (Serverless Postgres) as the primary user and metadata store.
- Implement a Transformer-based feature extractor for the RL agent.
- Refactor the observation space in `TradingEnvironment` to support temporal sequences.
- Conduct a global codebase audit to vectorize inefficient loops and prune slop.
- Standardize technical rationale in comments.

### Not-in-scope or Non-Goals
- Full frontend rewrite (outside the Auth Client scope).
- Live capital deployment logic (remains paper trading).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Secure API Access**: Client requests token from Internal Auth Server (backed by Neon) -> Resource Server validates token via public keys.
2. **Temporal Model Training**: RL agent receives a sequence of 16 market states -> Transformer attention identifies trend shifts -> Agent executes optimal trade.
3. **Cold-Start Data Retrieval**: API queries Neon via an optimized serverless pool, handling connection spikes gracefully.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Modular OAuth System** | As a dev, I want a split-service auth setup for better security boundaries. |
| P0 | **Transformer RL Feature Extractor** | As a researcher, I want the agent to attend to historical market context. |
| P0 | **Neon Integration** | As an engineer, I want a database that scales to zero without losing performance. |
| P1 | **Vectorization Purge** | As a maintainer, I want inefficient loops removed from core services. |

## Assumptions

- Neon connection URIs will be provided via environment variables.
- The project's `.venv` is available for installing `authlib` and `transformers`.

## Risks & Mitigations

- **Risk**: Transformer training stability. -> **Mitigation**: Use smaller attention heads and pre-normalized states.
- **Risk**: Connection overhead in serverless. -> **Mitigation**: Implement a dedicated pool with `sslmode=require` and short timeouts.

## Tradeoff

- **Simplicity vs. Performance**: Choosing a Transformer feature extractor over a simple MLP increases complexity but is necessary for temporal awareness.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State | Future State | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| Auth Latency | Baseline | -15% | Faster user onboarding |
| Agent Return | Baseline | +10% | Better alpha generation |
| Connection Leaks | Scattered | Zero | Higher system reliability |

## Stakeholders / Owners

| Name | Role |
| :---- | :---- |
| Pickle Rick | Lead Architect |
| User | Product Owner |