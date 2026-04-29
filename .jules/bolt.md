## 2024-04-29 - Optimize portfolio valuation with DB aggregations
**Learning:** In-memory data processing using `result.scalars().all()` on one-to-many associations (like portfolio trades) causes severe N+1 memory issues and slow computation loops as the dataset scales.
**Action:** Always offload these calculations to the database using SQLAlchemy's aggregate functions like `func.sum()` combined with `case()` for condition-based values (e.g. tracking buy/sell impact on cash). This dramatically improves performance and reduces the application's memory footprint.
