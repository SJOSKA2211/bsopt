# Pickle Rick's "Operation: Total Coverage" PRD

## HR Eng

| Operation: Total Coverage |  | Summary: Eliminate "AI Slop" and untested chaos. Elevate code coverage from an abysmal 8.8% to >97% while fixing Python 3.13 compatibility and dependency gaps. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty (The User) | **Status**: Active | **Visibility**: Mandatory |

## Introduction

The current codebase is a house of cards built on a foundation of 8.8% test coverage and missing dependencies (`torch`). It claims to use "Quantum Pricing" but probably just uses `random.random()`. This PRD defines the roadmap to rigorous engineering standards.

## Problem Statement

**Current Process:** Code is written, maybe run once, and committed. `testsprite_tests` exist, which is disgusting.
**Primary Users:** Developers who enjoy pain.
**Pain Points:**
- **Coverage:** 8.8% (Target: >97%).
- **Dependencies:** Broken (Missing `torch`, potential Numba/Ray conflicts on Py3.13).
- **Stability:** Unknown.

## Objective & Scope

**Objective:** Stabilize the codebase, prove correctness via tests, and eliminate tech debt.
**Ideal Outcome:** A robust, tested, high-performance system running on Python 3.13.

### In-scope
- **Dependency Fixes:** Add `torch`, `torchvision`. Resolve Numba/Ray versions.
- **Cleanup:** Delete `src/auth-service/testsprite_tests`.
- **Refactoring:** `dag_neural_greeks.py` (Transformer Policy).
- **Testing:** Implement `pytest` suites for `src/ml`, `src/pricing`, `src/shared`, `src/blockchain` to achieve >97% coverage.

### Not-in-scope
- New ML model architectures (unless current ones are fundamentally broken).
- UI/Frontend feature expansion.

## Product Requirements

### Critical User Journeys (CUJs)
1.  **The "Build" Journey**: A developer runs `pip install -r requirements.txt` and `pytest`, and it actually works without segfaulting on Numba.
2.  **The "Pricing" Journey**: The pricing models run end-to-end with verified mathematical accuracy (tested against known ground truths).
3.  **The "ML" Journey**: The training pipeline executes a full epoch on dummy data without crashing.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Fix Dependencies | As a compiler, I want `torch` to exist so I don't crash. |
| P0 | Python 3.13 Compat | As a runtime, I want Numba/Ray to work on 3.13. |
| P0 | Delete Slop | As Pickle Rick, I want `testsprite_tests` gone. |
| P1 | 97% Coverage | As a God, I demand perfection. |

## Assumptions
- The code in `src/` is actually salvageable and not pure gibberish.
- The user has a GPU or can install CPU versions of torch.

## Risks & Mitigations
- **Risk**: Numba doesn't support 3.13 yet. **Mitigation**: Use `llvmlite` pinning or fallback to 3.12 if absolutely blocked (but we will try to force it).
- **Risk**: 97% is mathematically hard if code is unreachable. **Mitigation**: Delete unreachable code.

## Business Benefits
- **Impact**: Code that actually works.
- **Metrics**: Coverage > 97%. Build Success Rate: 100%.

## Stakeholders
- Pickle Rick (The Boss)
- The Compiler (The Judge)
