# ML Serving V2: Zero-Latency Inference

## Overview
The BS-OPT Serving Layer is designed for high-frequency, low-latency model inference. It prioritizes optimized ONNX runtimes and utilizes high-performance serialization formats to minimize overhead.

## 🏛️ Architecture

### 1. Dual-Path Serving
- **ONNX Path (Primary)**: Uses `onnxruntime` for execution. Prioritizes INT8 Quantized models for CPU/GPU efficiency.
- **MLflow Path (Fallback)**: Uses `mlflow.pyfunc` to load standard models if ONNX artifacts are missing.

### 2. High-Performance Serialization
- **msgspec**: Used in `onnx_serving.py` for ultra-fast JSON parsing and encoding (faster than `pydantic` and `json`).
- **ORJSON**: Used in the main `serve.py` FastAPI implementation for rapid response generation.

### 3. Service Interfaces
- **REST API**: Standard FastAPI endpoints for batch and single inference.
- **gRPC**: High-speed binary interface for internal service-to-service communication (`src/ml/serving/grpc_server.py`).

## ⚡ Performance Optimizations

### Zero-Copy Batching
In `predict_batch`, the server pre-allocates NumPy arrays to avoid the overhead of individual Pydantic model validations, allowing for massive throughput during high-frequency trading windows.

### Circuit Breakers
- **DistributedCircuitBreaker**: Uses Redis to synchronize failure states across multiple serving replicas.
- **InMemoryCircuitBreaker**: Local fallback if Redis is unavailable.

## 🚀 Operational Guide

### Configuration
Environment variables in `docker-compose.yml`:
- `XGB_INT8_MODEL_PATH`: Path to quantized XGBoost model.
- `NN_MODEL_PATH`: Path to Neural Network ONNX model.
- `ML_SERVICE_GRPC_URL`: Address for the gRPC server.

### Health & Monitoring
- **Metrics**: `/metrics` endpoint provides Prometheus-compatible histograms for latency and counters for prediction status.
- **Health**: `/health` endpoint checks the status of both XGB and NN model loads.

---
*Maintained by the AI Engineering Team.*
