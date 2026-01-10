# Specification: GraphQL Federation for Unified API Gateway

## Overview
Implement a unified, high-performance API layer using **Apollo GraphQL Federation**. This track will transform the BS-Opt architecture from siloed services into a cohesive "Supergraph," allowing clients (Web, Mobile, CLI) to query complex data structures—such as a portfolio containing options with real-time Greeks and ML predictions—in a single network request.

## Functional Requirements
1.  **Unified API Gateway**: Deploy an Apollo Gateway (Node.js/Express) to serve as the single entry point for all frontend and external clients.
2.  **Five Federated Subgraphs**:
    *   **Options Subgraph**: Provides core contract metadata (strikes, expiries, symbols).
    *   **Pricing Subgraph**: Provides real-time Black-Scholes calculations and Greeks.
    *   **ML Subgraph**: Provides fair value predictions and direction signals.
    *   **Portfolio Subgraph**: Manages user accounts, balances, and positions.
    *   **Market Data Subgraph**: Connects to Kafka streams to provide current prices and historical trends.
3.  **Federated Schema Design**: 
    *   The `Option` type will be the primary entity, with fields contributed by multiple subgraphs (e.g., `Option.strike` from Options, `Option.delta` from Pricing, `Option.predicted_price` from ML).
4.  **Real-Time Subscriptions**: Implement WebSocket-based subscriptions in the subgraphs (using Strawberry) and configure the Gateway to route them correctly for real-time market data and portfolio value updates.
5.  **Authentication/Authorization**: Propagate JWT headers from the Gateway to all subgraphs to ensure consistent access control.

## Non-Functional Requirements
1.  **Performance**: The Gateway introspection and composition should not add more than 10ms of overhead (p95) to internal service calls.
2.  **Scalability**: Subgraphs and the Gateway must support horizontal scaling via Docker Compose/Kubernetes.
3.  **Type Safety**: All subgraphs must use Python `strawberry` with Pydantic for strict schema enforcement.

## Acceptance Criteria
1.  The Apollo Gateway successfully composes a supergraph from all five subgraphs.
2.  A single GraphQL query can return a `Portfolio` with its `Positions`, including each option's `strike`, `last_price`, and `predicted_fair_value`.
3.  GraphQL Subscriptions successfully stream market updates via WebSockets through the Gateway.
4.  The Gateway provides a GraphiQL/Apollo Sandbox interface for developers to explore the schema.

## Out of Scope
1.  Migrating existing internal gRPC or REST communication between services (this track focuses on the public/external API).
2.  Implementation of complex rate-limiting or persisted queries (deferred to AIOps/Security tracks).
