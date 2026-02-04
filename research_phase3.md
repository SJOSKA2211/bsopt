# Research: Singularity Phase 3 (Streaming & AIOps)

**Date**: 2026-02-04

## 1. Executive Summary
Audit of the streaming and AIOps modules reveals functional high-throughput designs that suffer from orchestration "slop". The Kafka producer has redundant serialization paths, and the self-healing orchestrator lacks a targeted remediation strategy, potentially over-remediating issues.

## 2. Technical Context
- **Kafka Producer**: `src/streaming/kafka_producer.py:15` uses `confluent-kafka` with Avro serialization.
- **AIOps Orchestrator**: `src/aiops/self_healing_orchestrator.py:8` implements a closed-loop control system.
- **Remediation**: `src/aiops/remediation_strategies.py` (implied) contains the logic for fixing anomalies.

## 3. Findings & Analysis
- **Serialization Overhead**: `MarketDataProducer.produce` (`kafka_producer.py:61`) serializes every message even in batches. We can use `msgspec` for faster internal buffers before hitting the Avro layer for Kafka.
- **Remediation Slop**: `SelfHealingOrchestrator.run_cycle` (`self_healing_orchestrator.py:32`) loops through *all* remediators for *every* anomaly. It should use a routing key or anomaly type filter.
- **Async Efficiency**: The orchestrator correctly uses `asyncio.gather`, but `asyncio.to_thread` (`self_healing_orchestrator.py:25`) for CPU-bound detectors might be slower than using a dedicated process pool for heavy inference (Autoencoders/Isolation Forest).

## 4. Technical Constraints
- Kafka messages must adhere to the Avro schema defined in `src/streaming/schemas/market_data.avsc`.
- Remediation actions must be idempotent to prevent oscillation.

## 5. Architecture Documentation
- **Streaming**: Follows a Producer/Consumer pattern with Schema Registry support.
- **AIOps**: Implements a "Observe-Orient-Decide-Act" (OODA) loop.
EOF
