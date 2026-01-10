# Plan: GraphQL Federation for Unified API Gateway (Track: graphql_federation_20260110)

## Phase 1: Subgraph Core Implementation (TDD) [checkpoint: 8bbcc34]
- [x] Task: Create `Options` subgraph with `strawberry` and `FastAPI` (wrapping `src/api`) 435fbc9
- [x] Task: Create `Pricing` subgraph (wrapping existing pricing logic) f326f60
- [x] Task: Create `ML` subgraph (wrapping `src/ml` inference logic) d7e3fd6
- [x] Task: Implement `Portfolio` subgraph (New service for user positions) 3317d60
- [x] Task: Implement `MarketData` subgraph (Kafka consumer integration) caacc23
- [x] Task: Verify all subgraphs expose `/graphql` and valid SDL 8168763
- [x] Task: Conductor - User Manual Verification 'Phase 1: Subgraph Core Implementation' (Protocol in workflow.md) 8bbcc34

## Phase 2: Schema Federation & Entity Mapping [checkpoint: f33341d]
- [x] Task: Define the `Option` federated entity with shared keys (`id`, `symbol`)
- [x] Task: Implement cross-service field resolvers (e.g., ML subgraph resolving predicted price for an Option)
- [x] Task: Implement `Portfolio` -> `Option` entity relationship
- [x] Task: Verify entity resolution across subgraphs using a local composer
- [x] Task: Conductor - User Manual Verification 'Phase 2: Schema Federation' (Protocol in workflow.md) f33341d

## Phase 3: Apollo Gateway & Infrastructure [checkpoint: 35e8b04]
- [x] Task: Implement Apollo Gateway using Node.js and `@apollo/gateway` f98d893
- [x] Task: Configure Docker Compose for all subgraphs and the Gateway 65164be
- [x] Task: Implement JWT propagation from Gateway to all subgraphs 5494787
- [x] Task: Verify unified schema composition and query plan execution
- [x] Task: Conductor - User Manual Verification 'Phase 3: Apollo Gateway & Infrastructure' (Protocol in workflow.md) 35e8b04

## Phase 4: Real-Time Subscriptions (TDD) [checkpoint: 7f550d8]
- [x] Task: Implement WebSocket support in `MarketData` subgraph for tick data 8e18a1e
- [x] Task: Implement WebSocket support in `Portfolio` subgraph for P&L updates 1b4d8a7
- [x] Task: Configure Apollo Gateway to support federated subscriptions ba7b644
- [x] Task: Verify end-to-end subscription streaming from Kafka to GraphQL client 17848d3
- [x] Task: Conductor - User Manual Verification 'Phase 4: Real-Time Subscriptions' (Protocol in workflow.md) 7f550d8

## Phase 5: Production Hardening & Documentation
- [x] Task: Implement response caching in Apollo Gateway 9b4cd10
- [-] Task: Configure OpenTelemetry instrumentation for distributed tracing across the supergraph
- [-] Task: Document the GraphQL API (queries, mutations, and subscription examples)
- [-] Task: Verify SLA targets (sub-10ms Gateway overhead) with benchmarks
- [~] Task: Conductor - User Manual Verification 'Phase 5: Production Hardening' (Protocol in workflow.md)
