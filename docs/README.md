# BS-OPT Documentation Index 📚

## 🏛️ System Architecture
Detailed technical specifications for the core platform components:
- [Market Mesh (Shared Memory)](architecture/MARKET_MESH.md): Zero-copy market data distribution with msgspec/orjson.
- [Hybrid Distributed Workers](architecture/HYBRID_WORKERS.md): Scalable math compute via Celery, Ray, and persistent gRPC.
- [Vectorized Risk Management](architecture/VECTORIZED_RISK.md): Silicon-level pre-trade risk validation (< 300ns).
- [Trading Engine Flow](architecture/TRADING_ENGINE.md): Low-latency order execution with zero-allocation hot loops.

## 🔒 Security & Operations
Protocols for protecting and maintaining the BS-OPT manifold:
- [Security & Hardening Protocol](SECURITY_PROTOCOL.md): Zero-trust, mTLS, and optimized PII masking.
- [Anti-Freeze Guide (Build Optimization)](mlops/anti-freeze.md): Multi-stage Docker and binary stripping.

##  Getting Started
Refer to the root [README.md](../README.md) for quick-start commands and toolchain requirements.
