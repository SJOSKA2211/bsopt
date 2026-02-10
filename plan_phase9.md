# Optimized Phase 9 Implementation Plan (Core Audit)

## Overview
Sanitization and optimization of the system's entry points and core data ingestion layers. Goal is to remove "AI Slop" (hyperbolic comments, redundant code) and consolidate system utilities.

## Current State Analysis
- `src/streaming/ingestion_worker.py`: Duplicate `stop()` method at lines 163 and 170.
- `bs_cli.py` & `bs_hft_launch.py`: Contains "Advanced" comments and boilerplate affinity setting code.
- `src/shared/shm_mesh.py`: Inefficient read operations.

## Implementation Approach
1.  **Consolidate**: Centralize system-level utilities (affinity).
2.  **Sanitize**: Remove duplicate code and unprofessional comments.
3.  **Optimize**: Clean up shared memory buffer logic.

## Phase 1: Core Hygiene & Consolidation
### Overview
Remove defects and centralize thread management.

### Changes Required:
#### 1. `src/shared/system_utils.py` (Create)
**Changes**: Implement `set_thread_affinity(core_id)` with robust error handling.

#### 2. `src/streaming/ingestion_worker.py`
**Changes**:
- Remove duplicate `stop()` method.
- Import and use `set_thread_affinity`.

#### 3. `bs_hft_launch.py`
**Changes**:
- Import and use `set_thread_affinity`.
- Remove "Optimized" branding/comments.

## Phase 2: CLI Sanitization
### Overview
Professionalize the CLI interface.

### Changes Required:
#### 1. `bs_cli.py`
**Changes**:
- Remove "Advanced" comments.
- Replace placeholder strings with calls to `SharedMemoryRingBuffer.get_stats()` (or similar) if available, or structured TODOs.

## Phase 3: Buffer Optimization
### Overview
Improve `SharedMemoryRingBuffer` read performance.

### Changes Required:
#### 1. `src/shared/shm_mesh.py`
**Changes**:
- Refactor `read()` method.
- Ensure efficient wrapping logic.

### Success Criteria:
#### Automated:
- [ ] `pytest tests/` (Ensure no regressions).
- [ ] Manual launch of `bs_cli.py` works without errors.
