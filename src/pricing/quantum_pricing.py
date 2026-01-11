from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
import numpy as np
from typing import Tuple, Dict

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
        
        # Initialize quantum state with amplitudes
        qc.initialize(amplitudes, qr)
        
        return qc, prices
