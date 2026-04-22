## 2024-04-22 - Database Aggregation for Portfolio Valuation
**Learning:** Computing sums over large one-to-many relationship rows (like trades in a portfolio) using python iteration and `.all()` causes massive memory and computation overhead.
**Action:** Always offload large one-to-many aggregations directly to the database using SQLAlchemy's `func.sum()` and `case()` logic instead of iterating in Python.
