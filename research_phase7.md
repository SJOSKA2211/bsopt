# Research: Singularity Phase 7 (Scrapers & Exotics)

**Date**: 2026-02-04

## 1. Executive Summary
Audit of the scrapers and exotic pricing kernels identifies sub-optimal lookup logic and incomplete analytical models. The NSE scraper uses sequential keyword matching for symbol mapping, and the barrier option pricer uses placeholders for several critical barrier configurations.

## 2. Technical Context
- **Scraper Engine**: `src/scrapers/engine.py:115` implements proxy rotation and batched cleaning.
- **Barrier Pricer**: `src/pricing/exotic.py:155` implements analytical barrier pricing but lacks the full Reiner-Rubinstein components.
- **Symbol Mapping**: `src/scrapers/engine.py:245` uses a loop over `_symbol_map.items()`.

## 3. Findings & Analysis
- **Lookup Slop**: `_map_name_to_symbol` is called for every item in every refresh cycle. As the symbol list grows, the sequential substring match becomes a bottleneck. We should use a trie or a pre-computed exact-match map where possible.
- **Mathematical Slop**: `price_barrier_analytical` only has components A, B, and C. It needs D, E, and F to support "Up" barriers and "In" options correctly. Current implementation returns placeholders (vanilla * 0.8) for several types.
- **Async Efficiency**: The scraper correctly uses `asyncio.gather` for sectors, but the single-flight pattern (`_refresh_future`) could be more robust against timed-out futures.

## 4. Technical Constraints
- Analytical solutions must remain differentiable for future Greek support.
- Scrapers must handle dirty HTML fragments from legacy WordPress endpoints.

## 5. Architecture Documentation
- **Pattern**: Strategy Pattern for pricing types; Protocol-based scrapers.
- **Singularity Status**: Incomplete in the exotic manifold.
EOF
