# Plan: GraphQL Federation for Unified API Gateway (Track: graphql_federation_20260110)

## Phase 1: Subgraph Core Implementation (TDD)
- [x] Task: Create `Options` subgraph with `strawberry` and `FastAPI` (wrapping `src/api`) 435fbc9
- [x] Task: Create `Pricing` subgraph (wrapping existing pricing logic) f326f60
- [x] Task: Create `ML` subgraph (wrapping `src/ml` inference logic) d7e3fd6
- [x] Task: Implement `Portfolio` subgraph (New service for user positions) 3317d60
- [x] Task: Implement `MarketData` subgraph (Kafka consumer integration) caacc23
- [ ] Task: Verify all subgraphs expose `/graphql` and valid SDL
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Subgraph Core Implementation' (Protocol in workflow.md)

## Phase 2: Schema Federation & Entity Mapping
- [ ] Task: Define the `Option` federated entity with shared keys (`id`, `symbol`)
- [ ] Task: Implement cross-service field resolvers (e.g., ML subgraph resolving predicted price for an Option)
- [ ] Task: Implement `Portfolio` -> `Option` entity relationship
- [ ] Task: Verify entity resolution across subgraphs using a local composer
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Schema Federation' (Protocol in workflow.md)

## Phase 3: Apollo Gateway & Infrastructure
- [ ] Task: Implement Apollo Gateway using Node.js and `@apollo/gateway`
- [ ] Task: Configure Docker Compose for all subgraphs and the Gateway
- [ ] Task: Implement JWT propagation from Gateway to all subgraphs
- [ ] Task: Verify unified schema composition and query plan execution
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Apollo Gateway & Infrastructure' (Protocol in workflow.md)

## Phase 4: Real-Time Subscriptions (TDD)
- [ ] Task: Implement WebSocket support in `MarketData` subgraph for tick data
- [ ] Task: Implement WebSocket support in `Portfolio` subgraph for P&L updates
- [ ] Task: Configure Apollo Gateway to support federated subscriptions
- [ ] Task: Verify end-to-end subscription streaming from Kafka to GraphQL client
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Real-Time Subscriptions' (Protocol in workflow.md)

## Phase 5: Production Hardening & Documentation
- [ ] Task: Implement response caching in Apollo Gateway
- [ ] Task: Configure OpenTelemetry instrumentation for distributed tracing across the supergraph
- [ ] Task: Document the GraphQL API (queries, mutations, and subscription examples)
- [ ] Task: Verify SLA targets (sub-10ms Gateway overhead) with benchmarks
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Production Hardening' (Protocol in workflow.md)
