# BS-OPT Revamp: Phase 2 PRD (ML/RL Models)

## Introduction
Phase 2 focuses on a "God-tier" overhaul of the Reinforcement Learning and Decision Transformer layers. The current implementation is functional but lacks the high-performance features (Flash Attention, Fused Kernels, and Scalable Training) required for sub-millisecond execution and large-scale offline optimization.

## Problem Statement
- **Training Bottlenecks**: The current Transformer implementation uses standard `nn.TransformerEncoder`, which is suboptimal for long sequences and high-frequency data.
- **State Space Sparsity**: The 100-dimensional state vector is built using basic concatenation; it lacks advanced feature engineering (e.g., Fourier transforms, wavelets) for capturing market micro-structure.
- **Offline RL Stability**: The Decision Transformer lacks modern stabilization techniques like Conservative Q-Learning (CQL) or IQL integration.

## Objective
Rewrite the ML/RL core to use advanced architectural patterns and performance optimizations.

## Scope
- **Advanced Decision Transformer (DT-v2)**: Implement a custom DT with Flash Attention support and learned positional embeddings.
- **Fourier State Kernel**: Refactor `kernels.py` to include spectral features in the state representation.
- **High-Performance Training**: Implement a training loop that leverages `torch.compile` and automatic mixed precision (AMP).
- **Observability**: Add weight distribution monitoring and gradient flow analysis to MLflow.

## Technical Requirements
- Use `torch.nn.functional.scaled_dot_product_attention` for Flash Attention.
- Ensure 100% backward compatibility with existing `TradingEnvironment`.
- Target 2x speedup in training throughput.
