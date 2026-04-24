## 2024-04-24 - Database Aggregation
**Learning:** Offloading aggregations to the database using SQLAlchemy's func.sum() and case() is significantly faster than fetching all records into memory.
**Action:** Always prefer database-side aggregations for large datasets instead of iterating in Python.
