# BS-OPT Documentation Index

## 🏛️ System Architecture
Detailed technical specifications for the core platform components:
- [Market Mesh (Shared Memory)](architecture/MARKET_MESH.md): Zero-copy market data distribution.
- [Hybrid Distributed Workers](architecture/HYBRID_WORKERS.md): Scalable math compute via Celery and Ray.
- [Vectorized Risk Management](architecture/VECTORIZED_RISK.md): Silicon-level pre-trade risk validation using Numba JIT.
- [Trading Engine Flow](architecture/TRADING_ENGINE.md): Low-latency order execution from gateway to blockchain.

## 🔒 Security & Operations
Protocols for protecting and maintaining the BS-OPT manifold:
- [Security & Hardening Protocol](SECURITY_PROTOCOL.md): Zero-trust, mTLS, and encryption specifications.
- [Anti-Freeze Guide (Build Optimization)](mlops/anti-freeze.md): Resource management for local development.

## 🚀 Getting Started
Refer to the root [README.md](../README.md) for quick-start commands and toolchain requirements.
