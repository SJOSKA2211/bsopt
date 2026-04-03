## 2026-04-03 - [Security] Fix insecure deserialization in ML pipeline
**Vulnerability:** The ML training scripts (`offline_train.py` and `distributed_training.py`) used `pickle.load` for loading trajectory data. Pickle deserialization is inherently insecure as it can execute arbitrary code during deserialization if the input data is manipulated.
**Learning:** `pickle` is often chosen out of convenience for serializing nested ML objects, but it shouldn't be used when safer formats like Parquet or JSON are viable.
**Prevention:** Always use safe serialization formats like Parquet (`pandas.read_parquet`), JSON (`json.load`), or `safetensors` for ML data loading, and avoid `pickle`.
