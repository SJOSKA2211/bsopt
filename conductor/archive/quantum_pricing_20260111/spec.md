# Specification: Quantum-Ready Option Pricing (IBM Qiskit Integration)

## Overview
Implement a quantum-accelerated option pricing engine using IBM Qiskit. This system will provide a significant performance advantage for complex Monte Carlo simulations by utilizing Quantum Amplitude Estimation (QAE). The engine will operate in a hybrid mode, automatically switching between classical and quantum methods based on the problem dimensionality and required precision.

## Functional Requirements
1.  **Quantum Option Pricer**:
    *   Implement `QuantumOptionPricer` using `qiskit` and `qiskit-aer`.
    *   Support European call option pricing using Quantum Amplitude Estimation.
    *   Generate log-normal stock price distributions using quantum amplitude encoding.
2.  **Hybrid Adaptive Logic**:
    *   Implement `HybridQuantumClassicalPricer` to intelligently route requests.
    *   **Decision Logic**: Use Quantum when dimensionality > 3 or requested accuracy < 1%.
3.  **Circuit Optimization**:
    *   Implement `QuantumCircuitOptimizer` to minimize circuit depth and gate count for efficient execution.
4.  **Hardware & Simulator Support**:
    *   Configurable backend: Support both local `AerSimulator` and remote IBM Quantum hardware (via `IBMProvider`).
    *   Manage IBM Quantum tokens and backend selection via environment variables.
5.  **Exposed Analytics**:
    *   Calculate and expose the **Speedup Factor** compared to classical Monte Carlo.
    *   Provide **Confidence Intervals** for every quantum-derived price.
    *   Expose **Circuit Statistics** (depth, gate count) for technical transparency.

## Non-Functional Requirements
1.  **Performance**: Quantum Amplitude Estimation should achieve a quadratic speedup ($O(\sqrt{M})$ vs $O(M)$) for error convergence.
2.  **Scalability**: The infrastructure should support up to 5-qubit simulations locally and be ready for 20+ qubit hardware execution.
3.  **Observability**: Integrate with MLflow to track quantum parameters, metrics, and circuit metadata.

## Acceptance Criteria
1.  `QuantumOptionPricer` successfully calculates option prices within 1% of the Black-Scholes analytical solution for standard parameters.
2.  The `HybridQuantumClassicalPricer` correctly selects the Quantum method when 4+ underlying assets are provided.
3.  Circuit optimization reduces gate count by at least 10% compared to raw generated circuits.
4.  The speedup factor and confidence intervals are visible in the pricing output/logs.

## Out of Scope
1.  Implementation of complex exotic options (e.g., Barrier, Asian) in this initial track.
2.  Real-time streaming of quantum results (batch/request-response only).