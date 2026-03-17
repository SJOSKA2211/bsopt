# Plan Review: Fix Syntax & Core Imports (Ticket c001)

**Status**:  APPROVED
**Reviewed**: 2026-02-20

## 1. Structural Integrity
- [x] **Atomic Phases**: Phases are cleanly separated by domain (API vs. Pricing).
- [x] **Worktree Safe**: Assumes a standard environment; no complex dependency on uncommitted state.

*Architect Comments*: The phasing is logical. Fixing the API syntax is the highest priority as it blocks everything else.

## 2. Specificity & Clarity
- [x] **File-Level Detail**: Targets `src/api/main.py` and `src/pricing/quant_utils.py` specifically.
- [x] **No "Magic"**: Clear steps for conflict removal and function implementation.

*Architect Comments*: The implementation of scalar functions is specifically mentioned to avoid the current batch-only limitation.

## 3. Verification & Safety
- [x] **Automated Tests**: Every phase uses `py_compile` or `pytest` for verification.
- [x] **Manual Steps**: Verification steps are clear and reproducible.
- [x] **Rollback/Safety**: No database migrations involved; low risk of side effects.

*Architect Comments*: Using `py_compile` is a smart way to verify syntax before moving to full test collection.

## 4. Architectural Risks
- Low risk. The changes are restorative (fixing broken code) rather than additive.

## 5. Recommendations
- Proceed to implementation. Ensure `scalar_bs_price_jit` handles edge cases like `T=0` consistently with the batch version.
