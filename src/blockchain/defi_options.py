import asyncio
import time
from web3 import AsyncWeb3, Web3
from web3.providers import AsyncHTTPProvider
from typing import Optional, Dict, List
import structlog

logger = structlog.get_logger()

class DeFiOptionsProtocol:
    """
    High-performance asynchronous interface for DeFi options protocols.
    Includes caching, batching, and circuit-breaking.
    """
    def __init__(
        self, 
        rpc_url: str = "https://polygon-rpc.com",
        private_key: Optional[str] = None,
        cache_ttl: int = 10
    ):
        self.rpc_url = rpc_url
        self.w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
        self.private_key = private_key
        self.cache_ttl = cache_ttl
        self._price_cache: Dict[str, Dict] = {}
        
        # Circuit Breaker state
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False
        
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

    async def _check_circuit(self):
        if self._circuit_open:
            if time.time() - self._last_failure_time > 30: # 30s timeout
                self._circuit_open = False
                self._failure_count = 0
                logger.info("blockchain_circuit_half_open")
            else:
                raise Exception("Blockchain RPC circuit open")

    async def get_option_price(self, contract_address: str) -> float:
        """Fetch option price with caching and circuit breaking."""
        now = time.time()
        if contract_address in self._price_cache:
            cache_entry = self._price_cache[contract_address]
            if now - cache_entry["time"] < self.cache_ttl:
                return cache_entry["price"]

        await self._check_circuit()
        
        abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "get_price",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        try:
            contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
            price_wei = await contract.functions.get_price().call()
            price = float(Web3.from_wei(price_wei, 'ether'))
            
            self._price_cache[contract_address] = {"price": price, "time": now}
            self._failure_count = 0
            return price
        except Exception as e:
            self._failure_count += 1
            if self._failure_count > 5:
                self._circuit_open = True
                self._last_failure_time = now
                logger.error("blockchain_circuit_opened", error=str(e))
            raise

    async def get_option_prices_batch(self, contract_addresses: List[str]) -> Dict[str, float]:
        """Fetch multiple option prices in parallel."""
        tasks = [self.get_option_price(addr) for addr in contract_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for addr, res in zip(contract_addresses, results):
            if isinstance(res, float):
                output[addr] = res
            else:
                logger.error("batch_price_fetch_error", address=addr, error=str(res))
        return output

    async def buy_option(self, contract_address: str, amount: int) -> str:
        """Execute a purchase transaction asynchronously."""
        if not self.private_key:
            raise ValueError("Private key required for transactions.")

        await self._check_circuit()

        abi = [
            {
                "constant": False,
                "inputs": [{"name": "_amount", "type": "uint256"}],
                "name": "buy",
                "outputs": [],
                "type": "function"
            }
        ]

        try:
            contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
            nonce = await self.w3.eth.get_transaction_count(self.address)
            gas_price = await self.w3.eth.gas_price
            
            transaction = await contract.functions.buy(amount).build_transaction({
                'from': self.address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': gas_price
            })

            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info("option_purchase_success", tx_hash=tx_hash.hex(), address=contract_address)
                return tx_hash.hex()
            else:
                raise Exception(f"Transaction failed: {tx_hash.hex()}")
        except Exception as e:
            logger.error("blockchain_tx_error", error=str(e))
            raise


if __name__ == "__main__":
    # Example usage
    protocol = DeFiOptionsProtocol()
    print("Web3 Connected:", protocol.w3.is_connected())
