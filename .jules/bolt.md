## 2026-04-23 - Database Aggregation for One-To-Many Relationships
**Learning:** When computing sums or aggregations over large one-to-many relationship rows (e.g., trades in a portfolio), offload the aggregation directly to the database using SQLAlchemy's func.sum() and conditional logic like case() instead of fetching all records with .all() and iterating in Python memory.
**Action:** Use database aggregation (e.g. func.sum) for future one-to-many numerical calculations to save memory and reduce latency.
