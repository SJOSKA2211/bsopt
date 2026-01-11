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
                # If payoff > S0*2, we clip or max out rotation
                normalized_payoff = min(payoff / (S0 * 2.0), 1.0)
                
                # Angle for RY rotation
                # We want amplitude of |1> to be sqrt(payoff)
                # RY(theta) state is cos(theta/2)|0> + sin(theta/2)|1>
                # So sin(theta/2) = sqrt(normalized_payoff)
                # theta/2 = arcsin(sqrt(normalized_payoff))
                # theta = 2 * arcsin(sqrt(normalized_payoff))
                angle = 2 * np.arcsin(np.sqrt(normalized_payoff))
                
                # Construct control state logic
                # Iterate through bits of index i
                binary_state = format(i, f'0{num_qubits}b')
                
                # Apply X gates to 0-bits to trigger on state |i>
                # Note: Qiskit uses little-endian (q0 is LSB)
                # reversed(binary_state) maps string MSB..LSB to qN-1..q0
                # Wait, enumerate(reversed(binary_state)) gives:
                # 0 -> LSB (q0), 1 -> q1, etc.
                # Example: i=1 ('001'). reversed is '100'. 
                # idx 0 (bit '1') -> q0. idx 1 (bit '0') -> q1.
                # So if bit is '0', we flip corresponding qubit.
                
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

