---
id: quant01
title: Phase 4: Quantum Realization & Verification
status: Done
priority: High
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [quantum, qiskit, phase4]
assignee: Pickle Rick
---

# Description

## Problem to solve
Phase 4 requires actual Quantum Amplitude Estimation, not just simulations or math fallbacks. The current `quantum_pricing.py` has the logic but relies heavily on fallbacks and needs verification against Qiskit 1.0+ standards.

## Solution
1.  Install `qiskit`, `qiskit-aer`, and `qiskit-algorithms`.
2.  Verify `QISKIT_AVAILABLE` becomes True.
3.  Implement/Verify `AmplitudeEstimation` logic using `qiskit_algorithms`.
4.  Write a dedicated test `tests/unit/test_pricing_quantum_real.py` that asserts `backend != "analytical_fallback"`.
5.  Optimize circuit depth if possible.
