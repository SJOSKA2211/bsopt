## 2024-04-25 - Offloading aggregations to the DB
**Learning:** For large portfolios, `result.scalars().all()` on trades fetches every trade record into Python memory. This suffers from N+1 query effects in memory mapping and memory scaling issues for very large lists.
**Action:** When computing sums or aggregations over large one-to-many relationship rows (like trades in a portfolio), offload the aggregation directly to the database using SQLAlchemy's `func.sum()` and conditional logic like `case()` instead of iterating in Python memory.
