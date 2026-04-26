## 2024-04-26 - Offload Database Aggregations

**Learning:** When computing sums or aggregations over large one-to-many relationship rows (like trades in a portfolio), it's highly inefficient to fetch all records into Python memory and iterate using `result.scalars().all()`.

**Action:** Offload the aggregation directly to the database using SQLAlchemy's `func.sum()` and conditional logic like `case()` to calculate the aggregate value on the database side. This significantly reduces data transfer and memory usage.