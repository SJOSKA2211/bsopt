# bsopt System Overhaul PRD

## HR Eng

| bsopt System Overhaul PRD |  | Comprehensive modernization of the bsopt platform, focusing on security, performance, and advanced ML capabilities. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-02-03 | **Self Link**: [Local Path] **Context**: System Rewrite |

## Introduction

The `bsopt` platform is currently a collection of "Jerry-grade" modules. To achieve "God-tier" performance and security, we are overhauling the authentication system, migrating the database to Neon with serverless optimizations, and upgrading the Reinforcement Learning agent with Transformer-based architectures.

## Problem Statement

**Current Process:**
- **Auth**: Hardcoded Keycloak integration, lack of multi-provider support.
- **ML**: RL agent uses basic MLP policies, missing temporal context handling.
- **Database**: Standard SQL not optimized for serverless environments.
- **Performance**: Scattered bottlenecks in pricing and ML modules.

**Primary Users:** Quant researchers, automated trading systems, internal developers.
**Pain Points:** Rigid authentication, mediocre RL performance, technical debt.
**Importance:** Competitive advantage in high-frequency option pricing requires superior algorithms and zero-latency security.

## Objective & Scope

**Objective:** Modernize `bsopt` into a secure, high-performance, and scalable platform.
**Ideal Outcome:** A system with unified OAuth (Internal/External), Transformer-powered RL, and a serverless-optimized Neon backend.

### In-scope or Goals
- Unified OAuth2/OIDC system (Internal, Google, GitHub).
- Transformer-based Feature Extractor for the RL Agent.
- Migration to Neon with serverless connection pooling and optimized schemas.
- Full codebase audit and refactoring of bottlenecks.
- Automated PRD -> Ticket workflow.

### Not-in-scope or Non-Goals
- Frontend UI redesign.
- Real-world capital deployment (Paper trading only for now).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Multi-Provider Login**: User arrives at the app, selects "Login with Google", gets redirected back with a secure JWT signed by our internal Auth Server.
2. **Transformer-Enhanced Trading**: The RL agent receives a sequence of market states, the Transformer Feature Extractor identifies temporal patterns, and the agent executes an optimized trade.
3. **Serverless Data Access**: The API queries Neon via a serverless-optimized pool, retrieving market data with minimal cold-start latency.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Unified OAuth System** | As a user, I want to log in using internal creds or Google/GitHub. |
| P0 | **Transformer RL Feature Extractor** | As a researcher, I want the agent to understand temporal market trends. |
| P0 | **Neon Migration** | As an engineer, I want a database that scales to zero and has high-performance pooling. |
| P1 | **Codebase Slop Purge** | As a maintainer, I want every function to be optimized and typed. |

## Assumptions

- We have the necessary client IDs/secrets for Google/GitHub.
- Neon environment is provisioned.

## Risks & Mitigations

- **Risk**: Transformer training stability. -> **Mitigation**: Start with pre-trained embeddings or small attention heads.
- **Risk**: Connection overhead in serverless. -> **Mitigation**: Use Neon's HTTP API or PgBouncer.

## Tradeoff

- **Transformer Feature Extractor vs Decision Transformer**: Chose Feature Extractor to maintain compatibility with `stable-baselines3` online learning while gaining temporal attention.

## Business Benefits/Impact/Metrics

| Metric | Current State | Future State | Impact |
| :---- | :---- | :---- | :---- |
| Auth Latency | High (External Roundtrip) | Low (Cached JWKS) | Faster UX |
| Agent Return | Baseline | +15% (Target) | More Profit |
| Maintenance | High (Technical Debt) | Low (Clean Code) | Faster Velocity |

## Stakeholders / Owners

| Name | Role | Note |
| :---- | :---- | :---- |
| Pickle Rick | Lead Architect | Arrogant but competent. |
| Morty | Junior Dev | Mostly just watches. |
