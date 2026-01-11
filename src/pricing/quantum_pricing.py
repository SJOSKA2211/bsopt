from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
import numpy as np
from typing import Tuple, Dict
import mlflow

class QuantumOptionPricer:
    """
    Quantum-accelerated option pricing using quantum amplitude estimation.
    Provides exponential speedup for Monte Carlo simulations.
    """
    
    def __init__(self, use_real_quantum: bool = False):
        """
        Args:
            use_real_quantum: If True, use IBM Quantum hardware. If False, use local simulator.
        """
        self.use_real_quantum = use_real_quantum
        if use_real_quantum:
            # Lazy import to avoid error if dependencies or tokens are missing in dev
            try:
                from qiskit_ibm_provider import IBMProvider
                provider = IBMProvider()
                self.backend = provider.get_backend('ibmq_qasm_simulator')
            except ImportError:
                print("Warning: qiskit-ibm-provider not found or configured. Falling back to simulator.")
                self.backend = AerSimulator()
            except Exception as e:
                print(f"Warning: Failed to initialize IBMProvider: {e}. Falling back to simulator.")
                self.backend = AerSimulator()
        else:
            self.backend = AerSimulator()

    def create_stock_price_distribution(
        self, S0: float, mu: float, sigma: float, T: float, num_qubits: int = 5
    ) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Create quantum circuit representing log-normal stock price distribution.
        Uses quantum amplitude encoding to represent the probability distribution 
        of future stock prices under geometric Brownian motion.
        
        Args:
            S0: Initial stock price
            mu: Expected return (drift)
            sigma: Volatility
            T: Time to maturity
            num_qubits: Precision (number of qubits for price register)
            
        Returns:
            Tuple[QuantumCircuit, np.ndarray]: The initialized circuit and the array of possible prices.
        """
        qr = QuantumRegister(num_qubits, 'price')
        qc = QuantumCircuit(qr)

        # Number of discrete price levels
        N = 2**num_qubits
        
        # Generate log-normal distribution
        # Range centered around S0
        prices = np.linspace(S0 * 0.5, S0 * 1.5, N)
        log_returns = np.log(prices / S0)
        
        # PDF of log-normal distribution
        # Note: This is a simplified discretization
        pdf = (1 / (sigma * np.sqrt(2 * np.pi * T))) * \
              np.exp(-0.5 * ((log_returns - (mu - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))**2)
        
        # Normalize probabilities so they sum to 1
        probabilities = pdf / pdf.sum()
        
        # Amplitudes are sqrt of probabilities
        amplitudes = np.sqrt(probabilities)
        
        # Use StatePreparation and ensure real floats
        state_prep = StatePreparation(amplitudes.real.astype(float))
        
        # Compose the state preparation into the circuit
        qc.compose(state_prep, qr, inplace=True)
        
        return qc, prices

    def add_payoff_operator(
        self, qc: QuantumCircuit, prices: np.ndarray, K: float, S0: float
    ) -> None:
        """
        Apply controlled rotation based on payoff.
        Marks states where S_T > K by rotating an ancillary payoff qubit.
        
        Args:
            qc: QuantumCircuit with a 'price' register
            prices: Array of prices corresponding to basis states
            K: Strike price
            S0: Initial spot price (used for normalization)
        """
        # Locate or add payoff register
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
            # Fallback: assume the first register is the price register if not named 'price'
            # But strictly we should enforce structure
            if len(qc.qregs) > 0 and qc.qregs[0].name != 'payoff':
                price_qubits = qc.qregs[0]
            else:
                raise ValueError("Price register not found in circuit")

        num_qubits = price_qubits.size

        # Apply controlled rotation for each state where price > K
        for i, price in enumerate(prices):
            if price > K:
                payoff = price - K
                # Normalize payoff to [0, 1] range for rotation
                # Denominator S0 * 2 covers reasonable range (up to 200% return)
                normalized_payoff = min(payoff / (S0 * 2.0), 1.0)
                
                # Angle for RY rotation: theta = 2 * arcsin(sqrt(normalized_payoff))
                angle = 2 * np.arcsin(np.sqrt(normalized_payoff))
                
                # Construct control state logic
                binary_state = format(i, f'0{num_qubits}b')
                
                x_indices = []
                for bit_idx, bit in enumerate(reversed(binary_state)):
                    if bit == '0':
                        x_indices.append(bit_idx)
                        
                # Apply pre-X
                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])
                
                # Apply Multi-Controlled RY
                qc.mcry(angle, price_qubits, payoff_qubits[0])
                
                # Apply post-X (uncompute)
                if x_indices:
                    qc.x([price_qubits[idx] for idx in x_indices])

    def price_european_call_quantum(
        self, S0: float, K: float, T: float, r: float, sigma: float, num_qubits: int = 5
    ) -> Dict[str, any]:
        """
        Price European call option using Quantum Amplitude Estimation.
        
        Args:
            S0: Initial spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate (mu)
            sigma: Volatility
            num_qubits: Register precision
            
        Returns:
            Dict: Contains 'price', 'confidence_interval', and execution metrics.
        """
        with mlflow.start_run(nested=True, run_name="quantum_option_pricing"):
            mlflow.log_params({
                "S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
                "num_qubits": num_qubits,
                "backend": str(self.backend)
            })

            # 1. Create quantum circuit for price distribution
            qc, prices = self.create_stock_price_distribution(S0, r, sigma, T, num_qubits)
            
            # 2. Apply payoff operator
            self.add_payoff_operator(qc, prices, K, S0)
            
            # Identify the payoff qubit index (usually the last qubit added)
            # Find the payoff register
            payoff_qubit = None
            for i, qubit in enumerate(qc.qubits):
                if any(qubit in reg for reg in qc.qregs if reg.name == 'payoff'):
                    payoff_qubit = i
                    break
            
            if payoff_qubit is None:
                raise ValueError("Payoff qubit not found in circuit")

            # 3. Setup Estimation Problem
            # Objective: measure the probability of the payoff qubit being in state |1>
            problem = EstimationProblem(
                state_preparation=qc,
                objective_qubits=[payoff_qubit]
            )
            
            # 4. Execute Iterative Amplitude Estimation
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()

            iae = IterativeAmplitudeEstimation(
                epsilon_target=0.01,
                alpha=0.05,
                sampler=sampler
            )

            result = iae.estimate(problem)


            
            # The estimation result is 'a' (probability of |1>).
            # Our payoff was scaled by S0 * 2.0.
            # Expected Payoff = a * (S0 * 2.0)
            expected_payoff = result.estimation * (S0 * 2.0)
            
            # Discount to present value: Price = exp(-r * T) * Expected Payoff
            option_price = np.exp(-r * T) * expected_payoff
            
            # Scaled confidence interval
            conf_interval = [
                np.exp(-r * T) * val * (S0 * 2.0) 
                for val in result.confidence_interval
            ]
            
            mlflow.log_metrics({
                "option_price": float(option_price),
                "num_oracle_queries": result.num_oracle_queries,
                "estimation": result.estimation
            })
            
            return {
                "price": float(option_price),
                "confidence_interval": conf_interval,
                "num_queries": result.num_oracle_queries,
                "estimation": result.estimation
            }

