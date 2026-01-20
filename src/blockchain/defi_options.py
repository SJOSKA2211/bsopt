from web3 import Web3
from typing import Optional
import structlog

logger = structlog.get_logger()

class DeFiOptionsProtocol:
    """
    Interface for interacting with decentralized options protocols (e.g., Opyn, Lyra) 
    on Ethereum/Polygon using Web3.py.
    """
    def __init__(self, rpc_url: str = "https://polygon-rpc.com", private_key: str = None, w3=None):
        """
        Initialize the protocol interface.
        
        Args:
            rpc_url: RPC endpoint for the blockchain (Polygon PoS)
            private_key: Optional private key for signing transactions.
                         CRITICAL: This should come from a secure secret management system, not directly as a string.
            w3: Optional Web3 instance for testing
        """
        self.rpc_url = rpc_url
        if w3 is not None:
            self.w3 = w3
        else:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key
        
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

    def get_option_price(self, contract_address: str) -> float:
        """Fetch the current price of an option from its smart contract."""
        # Simple ERC20/Contract interface for demonstration
        # In production, this would use specific ABIs for Opyn/Lyra
        abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "get_price",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
        price_wei = contract.functions.get_price().call()
        return float(Web3.from_wei(price_wei, 'ether'))

    def buy_option(self, contract_address: str, amount: int) -> str:
        """Execute a purchase of an option contract."""
        if not self.private_key:
            raise ValueError("Private key required for transactions.")

        abi = [
            {
                "constant": False,
                "inputs": [{"name": "_amount", "type": "uint256"}],
                "name": "buy",
                "outputs": [],
                "type": "function"
            }
        ]

        contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
        
        # Build transaction
        transaction = contract.functions.buy(amount).build_transaction({
            'from': self.address,
            'nonce': self.w3.eth.get_transaction_count(self.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt['status'] == 1:
            logger.info("option_purchase_success", tx_hash=tx_hash.hex(), address=contract_address)
            return tx_hash.hex()
        else:
            logger.error("option_purchase_failed", tx_hash=tx_hash.hex())
            raise Exception("Transaction failed")

if __name__ == "__main__":
    # Example usage
    protocol = DeFiOptionsProtocol()
    print("Web3 Connected:", protocol.w3.is_connected())
