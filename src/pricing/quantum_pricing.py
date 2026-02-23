import os
import warnings
from typing import Any

import numpy as np
import structlog
from scipy.stats import norm

from src.pricing.models import BSParameters
from src.pricing.quantum_backend import QuantumBackendManager

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="qiskit_algorithms.*"
)

try:
    import mlflow
except ImportError:

    class MlflowMock:
        def start_run(self, *args, **kwargs):
            class RunMock:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return RunMock()

        def log_params(self, *args, **kwargs):
            pass

        def log_metrics(self, *args, **kwargs):
            pass

    mlflow = MlflowMock()

# Global check for Qiskit availability
QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

    QISKIT_AVAILABLE = True
except ImportError:
    # Use None for missing types to avoid misleading mocks
    QuantumCircuit = None
    QuantumRegister = None
    transpile = None
    StatePreparation = None
    StatevectorSampler = None
    EstimationProblem = None
    IterativeAmplitudeEstimation = None


logger = structlog.get_logger()


class QuantumOptionPricer:
    """
    Quantum-accelerated option pricing using quantum amplitude estimation.
    Uses Iterative Amplitude Estimation (IAE) for quadratic speedup.
    """

    def __init__(
        self, use_real_quantum: bool = False, backend_name: str = "aer_simulator"
    ):
        self.use_real_quantum = use_real_quantum
        self.backend_manager = QuantumBackendManager()
        self.backend = None

        if QISKIT_AVAILABLE:
            try:
                from qiskit_aer import AerSimulator

                if use_real_quantum:
                    if backend_name == "aer_simulator":
                        backend_name = os.getenv(
                            "QUANTUM_BACKEND", "ibmq_qasm_simulator"
                        )
                    self.backend = self.backend_manager.get_backend(backend_name)
                else:
                    self.backend = AerSimulator()
            except Exception as e:
                logger.warning("backend_init_failed_absolute_fallback", error=str(e))
                try:
                    from qiskit_aer import AerSimulator

                    self.backend = AerSimulator()
                except ImportError:
                    self.backend = None
        else:
            self.backend = None

        self.optimizer = QuantumCircuitOptimizer(backend=self.backend)

    def create_stock_price_distribution(
        self, S0: float, mu: float, sigma: float, T: float, num_qubits: int = 5
    ) -> tuple[Any, np.ndarray]:
        """Prepares the probability distribution of stock prices."""
        if not QISKIT_AVAILABLE:
            return None, np.array([])

        qr = QuantumRegister(num_qubits, "price")
        qc = QuantumCircuit(qr)

        N = 2**num_qubits
        prices = np.linspace(S0 * 0.5, S0 * 1.5, N)
        log_returns = np.log(prices / S0)

        sigma_sqrt_T = max(sigma * np.sqrt(T), 1e-6)
        pdf = (1 / (sigma_sqrt_T * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((log_returns - (mu - 0.5 * sigma**2) * T) / sigma_sqrt_T) ** 2
        )

        probabilities = pdf / pdf.sum()
        amplitudes = np.sqrt(probabilities)

        state_prep = StatePreparation(amplitudes.real.astype(float))
        qc.compose(state_prep, qr, inplace=True)

        return qc, prices

    def add_payoff_operator(
        self, qc: Any, prices: np.ndarray, K: float, S0: float
    ) -> None:
        """Encodes the option payoff into an objective qubit."""
        if not QISKIT_AVAILABLE:
            return

        payoff_qubits = None
        for reg in qc.qregs:
            if reg.name == "payoff":
                payoff_qubits = reg
                break

        if payoff_qubits is None:
            payoff_qubits = QuantumRegister(1, "payoff")
            qc.add_register(payoff_qubits)

        price_qubits = None
        for reg in qc.qregs:
            if reg.name == "price":
                price_qubits = reg
                break

        if price_qubits is None:
            if len(qc.qregs) > 0 and qc.qregs[0].name != "payoff":
                price_qubits = qc.qregs[0]
            else:
                raise ValueError("Price register not found in circuit")

        num_qubits = price_qubits.size
        # High-Fidelity encoding: map discrete prices to rotation angles
        for i, price in enumerate(prices):
            if price > K:
                payoff = price - K
                normalized_payoff = min(payoff / (S0 * 2.0), 1.0)
                angle = 2 * np.arcsin(np.sqrt(normalized_payoff))

                binary_state = format(i, f"0{num_qubits}b")
                x_indices = [
                    j for j, bit in enumerate(reversed(binary_state)) if bit == "0"
                ]

                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])

                qc.mcry(angle, price_qubits, payoff_qubits[0])

                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])

    def price_european_call_quantum(
        self, S0: float, K: float, T: float, r: float, sigma: float, num_qubits: int = 5
    ) -> dict[str, Any]:
        """Run Iterative Amplitude Estimation to calculate option price."""
        if not QISKIT_AVAILABLE:
            return self._math_fallback(S0, K, T, r, sigma)

        qc, prices = self.create_stock_price_distribution(S0, r, sigma, T, num_qubits)
        self.add_payoff_operator(qc, prices, K, S0)

        payoff_qubit_index = 0
        current_idx = 0
        found = False
        for reg in qc.qregs:
            if reg.name == "payoff":
                payoff_qubit_index = current_idx
                found = True
                break
            current_idx += reg.size

        if not found:
            payoff_qubit_index = num_qubits

        try:
            problem = EstimationProblem(
                state_preparation=qc, objective_qubits=[payoff_qubit_index]
            )
            sampler = StatevectorSampler()
            iae = IterativeAmplitudeEstimation(
                epsilon_target=0.01, alpha=0.05, sampler=sampler
            )
            result = iae.estimate(problem)

            # The result.estimation represents the probability of the objective qubit being |1>
            # which maps to the expected payoff normalized by (S0 * 2.0)
            expected_payoff = result.estimation * (S0 * 2.0)
            option_price = np.exp(-r * T) * expected_payoff

            # Quadratic speedup factor compared to standard Monte Carlo (O(1/epsilon^2) vs O(1/epsilon))
            speedup = float(max(1.0, 1.0 / (0.01**2) / result.num_oracle_queries))

            return {
                "price": float(option_price),
                "confidence_interval": [
                    float(v * S0 * 2.0) for v in result.confidence_interval
                ],
                "num_queries": result.num_oracle_queries,
                "speedup_factor": speedup,
                "backend": str(self.backend) if self.backend else "simulated",
            }
        except Exception as e:
            return self._math_fallback(S0, K, T, r, sigma, error=str(e))

    def _math_fallback(self, S0, K, T, r, sigma, error=None):
        """Analytical fallback when quantum simulation is unavailable."""
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        res = {
            "price": float(price),
            "confidence_interval": [float(price * 0.98), float(price * 1.02)],
            "speedup_factor": 1.0,
            "backend": "analytical_fallback",
        }
        if error:
            res["error"] = error
        return res


class QuantumCircuitOptimizer:
    def __init__(self, backend: Any = None):
        self.backend = backend

    def optimize_circuit(self, qc: Any) -> Any:
        """Optimize circuit for specific hardware or depth."""
        if not QISKIT_AVAILABLE or qc is None:
            return qc
        try:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import (
                CommutativeCancellation,
                Optimize1qGatesDecomposition,
            )

            pm = PassManager(
                [Optimize1qGatesDecomposition(), CommutativeCancellation()]
            )

            if self.backend:
                optimized_qc = transpile(qc, backend=self.backend, optimization_level=3)
            else:
                optimized_qc = transpile(qc, optimization_level=3)

            return pm.run(optimized_qc)
        except Exception:
            return qc


class HybridQuantumClassicalPricer:
    """Intelligently routes pricing requests between classical and quantum engines."""

    def __init__(self):
        self.quantum_pricer = QuantumOptionPricer(use_real_quantum=False)
        from src.pricing.monte_carlo import MonteCarloEngine

        self.classical_pricer = MonteCarloEngine()

    def price_option_adaptive(self, **params) -> dict[str, Any]:
        """Adaptive routing based on complexity and accuracy requirements."""
        num_underlyings = params.get("num_underlyings", 1)
        accuracy_required = params.get("accuracy", 0.01)

        clean_params = params.copy()
        clean_params.pop("num_underlyings", None)
        clean_params.pop("accuracy", None)

        if num_underlyings > 3 or accuracy_required < 0.01:
            return self.quantum_pricer.price_european_call_quantum(**clean_params)
        bs_params = BSParameters(
            spot=float(params.get("S0", 100.0)),
            strike=float(params.get("K", 100.0)),
            maturity=float(params.get("T", 1.0)),
            volatility=float(params.get("sigma", 0.2)),
            rate=float(params.get("r", 0.05)),
            dividend=float(params.get("q", 0.0)),
        )
        res = self.classical_pricer.price_european(bs_params)
        if isinstance(res, tuple):
            return {"price": res[0]}
        return res
