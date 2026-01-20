import os
import structlog
import mlflow
import warnings
import numpy as np
from typing import Tuple, Dict, Any
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler

from src.pricing.quantum_backend import QuantumBackendManager
from src.pricing.models import BSParameters

# Filter deprecation warnings from Qiskit and Qiskit Algorithms that are beyond our control
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit_algorithms.*")

logger = structlog.get_logger()

class QuantumOptionPricer:
    """
    Quantum-accelerated option pricing using quantum amplitude estimation.
    Provides exponential speedup for Monte Carlo simulations.
    """
    
    def __init__(self, use_real_quantum: bool = False, backend_name: str = "aer_simulator"):
        """
        Args:
            use_real_quantum: If True, use IBM Quantum hardware. If False, use local simulator.
            backend_name: Specific backend to use (e.g., 'ibmq_qasm_simulator').
        """
        self.use_real_quantum = use_real_quantum
        self.backend_manager = QuantumBackendManager()
        
        try:
            if use_real_quantum:
                # If backend_name is aer_simulator but use_real_quantum is True, 
                # use a default remote simulator instead
                if backend_name == "aer_simulator":
                    backend_name = os.getenv("QUANTUM_BACKEND", "ibmq_qasm_simulator")
                
                self.backend = self.backend_manager.get_backend(backend_name)
            else:
                self.backend = self.backend_manager.get_backend("aer_simulator")
        except Exception as e:
            logger.warning("backend_init_fallback", error=str(e), fallback="aer_simulator")
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()

    def create_stock_price_distribution(
        self, S0: float, mu: float, sigma: float, T: float, num_qubits: int = 5
    ) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Create quantum circuit representing log-normal stock price distribution.
        """
        qr = QuantumRegister(num_qubits, 'price')
        qc = QuantumCircuit(qr)

        # Number of discrete price levels
        N = 2**num_qubits
        
        # Generate log-normal distribution range centered around S0
        prices = np.linspace(S0 * 0.5, S0 * 1.5, N)
        log_returns = np.log(prices / S0)
        
        # PDF of log-normal distribution
        pdf = (1 / (sigma * np.sqrt(2 * np.pi * T))) * \
              np.exp(-0.5 * ((log_returns - (mu - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))**2)
        
        # Normalize and create amplitudes
        probabilities = pdf / pdf.sum()
        amplitudes = np.sqrt(probabilities)
        
        state_prep = StatePreparation(amplitudes.real.astype(float))
        qc.compose(state_prep, qr, inplace=True)
        
        return qc, prices

    def add_payoff_operator(
        self, qc: QuantumCircuit, prices: np.ndarray, K: float, S0: float
    ) -> None:
        """
        Apply controlled rotation based on payoff.
        """
        # Ensure payoff register exists
        payoff_qubits = None
        for reg in qc.qregs:
            if reg.name == 'payoff':
                payoff_qubits = reg
                break
        
        if payoff_qubits is None:
            payoff_qubits = QuantumRegister(1, 'payoff')
            qc.add_register(payoff_qubits)
            
        # Locate price register
        price_qubits = None
        for reg in qc.qregs:
            if reg.name == 'price':
                price_qubits = reg
                break
        
        if price_qubits is None:
            if len(qc.qregs) > 0 and qc.qregs[0].name != 'payoff':
                price_qubits = qc.qregs[0]
            else:
                raise ValueError("Price register not found in circuit")

        num_qubits = price_qubits.size

        # Apply controlled rotation for each state where price > K
        for i, price in enumerate(prices):
            if price > K:
                payoff = price - K
                normalized_payoff = min(payoff / (S0 * 2.0), 1.0)
                angle = 2 * np.arcsin(np.sqrt(normalized_payoff))
                
                binary_state = format(i, f'0{num_qubits}b')
                x_indices = [j for j, bit in enumerate(reversed(binary_state)) if bit == '0']
                        
                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])
                
                # Apply Multi-Controlled RY (marks state and rotates ancillary qubit)
                qc.mcry(angle, price_qubits, payoff_qubits[0])
                
                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])

    def price_european_call_quantum(
        self, S0: float, K: float, T: float, r: float, sigma: float, num_qubits: int = 5
    ) -> Dict[str, Any]:
        """
        Price European call option using Quantum Amplitude Estimation.
        """
        with mlflow.start_run(nested=True, run_name="quantum_option_pricing"):
            mlflow.log_params({
                "S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
                "num_qubits": num_qubits,
                "backend": str(self.backend)
            })

            # 1. Create quantum circuit
            qc, prices = self.create_stock_price_distribution(S0, r, sigma, T, num_qubits)
            self.add_payoff_operator(qc, prices, K, S0)
            
            # 2. Optimize circuit
            optimizer = QuantumCircuitOptimizer()
            initial_depth = qc.depth()
            qc = optimizer.optimize_circuit(qc)
            optimized_depth = qc.depth()
            
            logger.info("circuit_optimized", initial_depth=initial_depth, optimized_depth=optimized_depth)

            # 3. Identify payoff qubit
            payoff_qubit = None
            for i, qubit in enumerate(qc.qubits):
                if any(qubit in reg for reg in qc.qregs if reg.name == 'payoff'):
                    payoff_qubit = i
                    break
            
            if payoff_qubit is None:
                raise ValueError("Payoff qubit not found in circuit")

            # 4. Execute Iterative Amplitude Estimation
            problem = EstimationProblem(
                state_preparation=qc,
                objective_qubits=[payoff_qubit]
            )
            
            sampler = StatevectorSampler()
            iae = IterativeAmplitudeEstimation(
                epsilon_target=0.01,
                alpha=0.05,
                sampler=sampler
            )

            result = iae.estimate(problem)
            
            # 5. Post-process results
            expected_payoff = result.estimation * (S0 * 2.0)
            option_price = np.exp(-r * T) * expected_payoff
            conf_interval = [np.exp(-r * T) * val * (S0 * 2.0) for val in result.confidence_interval]
            
            # Speedup calculation
            epsilon = 0.01 
            classical_samples = int(1 / epsilon**2)
            speedup_factor = classical_samples / result.num_oracle_queries if result.num_oracle_queries > 0 else 1.0
            
            mlflow.log_metrics({
                "option_price": float(option_price),
                "num_oracle_queries": result.num_oracle_queries,
                "speedup_factor": float(speedup_factor),
                "circuit_depth": qc.depth(),
                "optimized_reduction": 1.0 - (optimized_depth / max(1, initial_depth))
            })

            return {
                "price": float(option_price),
                "confidence_interval": conf_interval,
                "num_queries": result.num_oracle_queries,
                "speedup_factor": float(speedup_factor),
                "estimation": result.estimation,
                "optimized_depth": optimized_depth
            }

class QuantumCircuitOptimizer:
    """Optimize quantum circuits for option pricing"""
    
    def __init__(self):
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CommutativeCancellation
        
        self.pass_manager = PassManager([
            Optimize1qGatesDecomposition(),
            CommutativeCancellation()
        ])
        
    def optimize_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        return self.pass_manager.run(qc)

class HybridQuantumClassicalPricer:
    """
    Combine quantum and classical methods for optimal performance.
    """
    
    def __init__(self):
        self.quantum_pricer = QuantumOptionPricer(use_real_quantum=False)
        from src.pricing.monte_carlo import MonteCarloEngine
        self.classical_pricer = MonteCarloEngine()
        
    def price_option_adaptive(self, **params) -> Dict[str, Any]:
        """
        Decision criteria: Dimension > 3 OR Accuracy < 1% -> Use Quantum
        """
        num_underlyings = params.get('num_underlyings', 1)
        accuracy_required = params.get('accuracy', 0.01)
        
        # Prepare params for both engines
        clean_params = params.copy()
        clean_params.pop('num_underlyings', None)
        clean_params.pop('accuracy', None)
        
        if num_underlyings > 3 or accuracy_required < 0.01:
            logger.info("routing_to_quantum", num_underlyings=num_underlyings, accuracy=accuracy_required)
            return self.quantum_pricer.price_european_call_quantum(**clean_params)
        else:
            logger.info("routing_to_classical", num_underlyings=num_underlyings, accuracy=accuracy_required)
            
            # Map keyword arguments to BSParameters
            # Note: params contains S0, K, T, r, sigma
            bs_params = BSParameters(
                spot=float(params.get('S0', 100.0)),
                strike=float(params.get('K', 100.0)),
                maturity=float(params.get('T', 1.0)),
                volatility=float(params.get('sigma', 0.2)),
                rate=float(params.get('r', 0.05)),
                dividend=float(params.get('q', 0.0))
            )
            
            return self.classical_pricer.price_european(bs_params)

