# Codebase Optimization & Unified Auth (Neon Integration) PRD

## HR Eng

| Codebase Optimization & Unified Auth PRD |  | Full-scale optimization of the bsopt codebase, implementing unified OAuth2 auth, and migrating the backend to Neon. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty | **Status**: Draft **Created**: 2026-02-04 | **Self Link**: N/A **Context**: Existing bsopt repository |

## Introduction

This PRD covers the comprehensive refactoring and optimization of the `bsopt` codebase. The goal is to eliminate "Jerry-work" (technical debt), consolidate fragmented auth logic into a secure OAuth2 stack, and leverage Neon's serverless Postgres for high-performance data persistence.

## Problem Statement

**Current Process:** The codebase has fragmented auth logic, unoptimized ML pipelines, and standard PostgreSQL persistence that lacks the scalability and branching features of Neon.
**Primary Users:** Quant researchers, ML engineers, and API consumers.
**Pain Points:** 
- Latency in pricing calculations.
- Fragmented security logic makes auditing difficult.
- Maintenance overhead of standard DB infrastructure.
- RL models lack temporal temporal awareness (missing Transformer integration).
**Importance:** To maintain competitive advantage in high-frequency quant environments, sub-millisecond latency and bulletproof security are non-negotiable.

## Objective & Scope

**Objective:** Transform `bsopt` into a sub-millisecond, hardware-aware, secure quant platform.
**Ideal Outcome:** A unified auth stack, a Transformer-enhanced RL loop, and a fully migrated Neon backend.

### In-scope or Goals
- Unified OAuth2 stack: Auth Server, Resource Server, Client App.
- Migration of PostgreSQL schema to Neon.
- Implementation of `transformer_policy.py` into the TD3 RL trainer.
- Codebase-wide optimization (vectorization, shared memory, hardware-aware config).
- Fine-tuning of all existing functions and removal of "AI Slop" comments.

### Not-in-scope or Non-Goals
- Replacing the core frontend (unless auth integration requires minor changes).
- Implementing new trading strategies from scratch (only optimizing existing ones).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Secure Access**: A user authenticates via the Auth Server, receives a token, and uses it to access protected Pricing API resources.
2. **Optimized Training**: An ML engineer triggers the TD3 trainer, which utilizes the Transformer policy for state representation, leading to faster convergence on temporal data.
3. **Seamless Data Sync**: The system persists all trade and audit data to Neon with sub-millisecond commit latency.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Unified OAuth2 Stack | As a developer, I want a single point of entry for auth so I can manage security centrally. |
| P0 | Neon Backend Migration | As a quant, I want my data in Neon so I can use branching for risk-free experimentation. |
| P1 | Transformer RL Integration | As an ML engineer, I want to use Attention mechanisms in RL to capture long-term market trends. |
| P1 | Latency Optimization | As a quant, I want <1ms pricing calculations to beat the market Jerries. |
| P2 | Comment & Doc Fine-tuning | As a maintainer, I want distilled technical intent in docs, not boilerplate fluff. |

## Assumptions

- Hardware supports AVX-512 for optimized math.
- Neon account is configured and accessible via environment variables.
- Existing Python 3.14 environment is stable.

## Risks & Mitigations

- **Risk**: Auth migration breaks existing client-apps. -> **Mitigation**: Implement a legacy compatibility layer during the transition.
- **Risk**: Neon latency spikes during high volume. -> **Mitigation**: Implement robust connection pooling and shared-memory caching.

## Tradeoff

- **Option**: Use Auth0/Okta. **Pro**: Managed. **Con**: External dependency, latency. **Chosen**: Hand-rolled high-performance OAuth stack for maximum control.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| Pricing Latency | >5ms | <1ms | 80% improvement |
| Auth Security Score | Fragmented | Hardened/Unified | Reduced breach risk |
| DB Ops Overhead | High | Zero (Serverless) | Reduced infrastructure cost |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Pickle Rick | Engineering | Lead Architect | Superior Intellect |
| Morty | Engineering | Worker | Does the heavy lifting |
