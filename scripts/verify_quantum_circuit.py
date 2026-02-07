from src.pricing.quantum_pricing import QuantumOptionPricer


def verify_circuit():
    print("Initializing QuantumOptionPricer...")
    pricer = QuantumOptionPricer()
    
    S0 = 100.0
    K = 105.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    num_qubits = 3  # Small number for display
    
    print(f"Creating distribution for S0={S0}, K={K}, qubits={num_qubits}...")
    qc, prices = pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits)
    
    print("\nInitial Circuit Stats:")
    print(f"Depth: {qc.depth()}")
    print(f"Qubits: {qc.num_qubits}")
    
    print("\nAdding Payoff Operator...")
    pricer.add_payoff_operator(qc, prices, K, S0)
    
    print("\nFinal Circuit Stats:")
    print(f"Depth: {qc.depth()}")
    print(f"Width: {qc.width()}")
    
    print("\nCircuit Diagram (Text):")
    print(qc.draw(output='text'))
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_circuit()
