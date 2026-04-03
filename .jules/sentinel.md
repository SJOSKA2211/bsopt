## 2026-04-03 - [Security] Fix insecure deserialization in ML pipeline
**Vulnerability:** The ML training scripts (`offline_train.py` and `distributed_training.py`) used `pickle.load` for loading trajectory data. Pickle deserialization is inherently insecure as it can execute arbitrary code during deserialization if the input data is manipulated.
**Learning:** `pickle` is often chosen out of convenience for serializing nested ML objects, but it shouldn't be used when safer formats like Parquet or JSON are viable.
**Prevention:** Always use safe serialization formats like Parquet (`pandas.read_parquet`), JSON (`json.load`), or `safetensors` for ML data loading, and avoid `pickle`.
## 2026-04-03 - [Security] CI Fixes for Bandit and GitHub Actions
**Vulnerability:** Not a direct vulnerability in codebase, but security CI checks were failing. First due to a syntax issue where GitHub actions uses directives (`uses:`) were indented inside a `run:` block, breaking CI for institutional workflows. Second, Bandit flagged `host="0.0.0.0"` in `uvicorn` and `serve.py` as a B104 (hardcoded bind all interfaces).
**Learning:** Containerized ML/Auth services running in Docker intentionally bind to 0.0.0.0 to receive traffic outside the container network namespace. This is safe so long as network policies at the cluster level restrict ingress.
**Prevention:** Use `# nosec B104` to silence the specific Bandit warning for intended `0.0.0.0` binds while avoiding wholesale silencing of Bandit.
