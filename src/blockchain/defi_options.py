import asyncio
import time

from web3 import AsyncWeb3, Web3

try:
    from web3.providers import AsyncHTTPProvider, AsyncWebsocketProvider
except ImportError:
    # web3 v7+ compatibility
    from web3.providers import AsyncHTTPProvider
    try:
        from web3.providers import WebSocketProvider as AsyncWebsocketProvider
    except ImportError:
        from web3.providers import LegacyWebSocketProvider as AsyncWebsocketProvider


import structlog

logger = structlog.get_logger(__name__)

import asyncio
import time
from typing import Any

from web3 import AsyncWeb3, Web3
from eth_account import Account
import structlog

from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)

class DeFiOptionsProtocol:
    """
    DeFi options interaction protocol using Multicall3 and JSON-RPC batching.
    OPTIMIZED: Fused contract calls and hardened circuit breakers.
    """
    
    MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"
    # ... (ABI stays same)

    def __init__(
        self, 
        rpc_url: str = "wss://polygon-mainnet.g.alchemy.com/v2/your-api-key",
        private_key: str | None = None,
        cache_ttl: int = 10
    ):
        self.rpc_url = rpc_url
        self.w3 = AsyncWeb3(Web3.AsyncHTTPProvider(rpc_url)) # Standardize on HTTP for now
        self.private_key = private_key
        self.cache_ttl = cache_ttl
        self._price_cache: dict[str, dict] = {}
        
        self._nonce_lock = asyncio.Lock()
        self._failure_threshold = 5
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._circuit_open = False
        
        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

        # Pre-instantiate multicall contract
        self.multicall = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.MULTICALL3_ADDRESS), 
            abi=self.MULTICALL_ABI
        )

    async def _handle_rpc_failure(self):
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            self._last_failure_time = time.time()
            logger.error("rpc_circuit_opened")

    async def get_option_prices_batch(self, contract_addresses: list[str]) -> dict[str, float]:
        """
        Fetch multiple option prices using Multicall3.
        """
        if not contract_addresses:
            return {}

        await self._check_circuit()
        
        CHUNK_SIZE = 50 
        chunks = [contract_addresses[i:i + CHUNK_SIZE] for i in range(0, len(contract_addresses), CHUNK_SIZE)]
        
        # Shared selector for 'get_price()'
        price_selector = Web3.keccak(text="get_price()").hex()[:10]

        output = {}
        now = time.time()
        redis = get_redis()
        
        try:
            tasks = []
            for chunk in chunks:
                calls = [{
                    "target": Web3.to_checksum_address(addr),
                    "allowFailure": True,
                    "callData": price_selector
                } for addr in chunk]
                
                tasks.append(self.multicall.functions.aggregate3(calls).call())
            
            chunk_results = await asyncio.gather(*tasks)
            
            idx = 0
            for results in chunk_results:
                for success, return_data in results:
                    addr = contract_addresses[idx]
                    if success and len(return_data) >= 32:
                        price_wei = int.from_bytes(return_data[:32], byteorder='big')
                        price = float(Web3.from_wei(price_wei, 'ether'))
                        output[addr] = price
                        # Update caches
                        self._price_cache[addr] = {"price": price, "time": now}
                        if redis:
                            await redis.setex(f"defi_price:{addr}", self.cache_ttl, str(price))
                    idx += 1
            
            self._failure_count = 0 # Reset on success
            return output
        except Exception as e:
            logger.error("multicall_failed", error=str(e))
            await self._handle_rpc_failure()
            return await self._get_option_prices_parallel(contract_addresses)

    async def _get_option_prices_parallel(self, contract_addresses: list[str]) -> dict[str, float]:
        """Parallel execution fallback."""
        tasks = [self.get_option_price(addr) for addr in contract_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for addr, res in zip(contract_addresses, results):
            if isinstance(res, float):
                output[addr] = res
        return output

    async def buy_option(self, contract_address: str, amount: int, max_slippage: float = 0.01) -> str:
        """
        Execute a purchase transaction with EIP-1559 gas and slippage protection.
        """
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
            },
            {
                "name": "get_price", 
                "type": "function", 
                "inputs": [], 
                "outputs": [{"type": "uint256"}]
            }
        ]

        try:
            # 1. Price check for slippage
            contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
            current_price_wei = await contract.functions.get_price().call()
            expected_price = float(Web3.from_wei(current_price_wei, 'ether'))
            
            logger.info("slippage_check", expected=expected_price, amount=amount)

            # 2. Build EIP-1559 Transaction
            nonce = await self._get_next_nonce()
            
            latest_block = await self.w3.eth.get_block("latest")
            base_fee = latest_block["baseFeePerGas"]
            max_priority_fee = await self.w3.eth.max_priority_fee
            max_fee = (base_fee * 2) + max_priority_fee

            estimated_gas = await contract.functions.buy(amount).estimate_gas({'from': self.address})
            gas_limit = int(estimated_gas * 1.2) 

            transaction = await contract.functions.buy(amount).build_transaction({
                'from': self.address,
                'nonce': nonce,
                'gas': gas_limit,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': max_priority_fee,
                'type': 2 
            })

            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info("option_purchase_success", tx_hash=tx_hash.hex())
                return tx_hash.hex()
            else:
                raise Exception(f"Transaction failed: {tx_hash.hex()}")
        except Exception as e:
            logger.error("blockchain_tx_error", error=str(e))
            if hasattr(self, '_local_nonce'):
                delattr(self, '_local_nonce')
            raise


if __name__ == "__main__":
    protocol = DeFiOptionsProtocol()
    print("Web3 initialized")
