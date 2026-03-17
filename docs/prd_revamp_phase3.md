# BS-OPT Revamp: Phase 3 PRD (Blockchain Integration)

## Introduction
Phase 3 focuses on hardening and optimizing the DeFi options protocol layer. The current implementation relies on basic JSON-RPC calls and lacks the high-performance features required for competitive on-chain trading (e.g., meta-transactions, off-chain oracles, and MEV protection).

## Problem Statement
- **RPC Latency**: Relying on standard `eth_call` for every price check is too slow for real-time risk management.
- **Gas Costs**: Frequent on-chain transactions for rebalancing are expensive.
- **Transaction Reliability**: Nonce management and gas price spikes cause transaction failures.

## Objective
Implement a high-throughput, resilient blockchain integration layer.

## Scope
- **Off-Chain Price Oracle (Speed-v1)**: Implement a hybrid oracle that combines on-chain state with high-frequency WebSocket feeds from DEXs.
- **EIP-712 Meta-Transactions**: Add support for gasless signing and permit-based approvals to reduce friction and costs.
- **Mempool Monitoring & SOR**: Implement a basic Smart Order Router (SOR) that monitors the mempool to avoid being front-run.
- **Nonce Orchestrator**: Implement a stateful, persistent nonce manager to handle high-concurrency transaction submission.

## Technical Requirements
- Use `web3.py`'s `AsyncWeb3` for all interactions.
- Implement a `NonceManager` using Redis for cross-process synchronization.
- Integrate with `MarketMesh` for off-chain price synchronization.
