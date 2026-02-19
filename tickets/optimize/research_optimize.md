# Research: Codebase Optimization

## Objectives
- Identify bottlenecks.
- Refactor hot paths.
- Remove slop.

## Findings
- `src/api/routes/ml.py` had inline imports and unoptimized dependencies.
- `src/workers/math_worker.py` was over-complicated with mixed concurrency models.

## Strategy
- Refactor `ml.py` to use top-level imports and async/await properly.
- Simplify `math_worker.py` to use Ray ActorPool efficiently.

