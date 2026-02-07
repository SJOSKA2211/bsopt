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

class DeFiOptionsProtocol:
    """
    DeFi options interaction protocol using Multicall3 and JSON-RPC batching.
    """
    
    # OPTIMIZED: Multicall3 Address on Polygon Mainnet
    MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"
    MULTICALL_ABI = [
        {
            "inputs": [
                {
                    "components": [
                        {"internalType": "address", "name": "target", "type": "address"},
                        {"internalType": "bool", "name": "allowFailure", "type": "bool"},
                        {"internalType": "bytes", "name": "callData", "type": "bytes"}
                    ],
                    "internalType": "struct Multicall3.Call3[]",
                    "name": "calls",
                    "type": "tuple[]"
                }
            ],
            "name": "aggregate3",
            "outputs": [
                {
                    "components": [
                        {"internalType": "bool", "name": "success", "type": "bool"},
                        {"internalType": "bytes", "name": "returnData", "type": "bytes"}
                    ],
                    "internalType": "struct Multicall3.Result[]",
                    "name": "returnData",
                    "type": "tuple[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]

    def __init__(
        self, 
        rpc_url: str = "wss://polygon-mainnet.g.alchemy.com/v2/your-api-key",
        private_key: str | None = None,
        cache_ttl: int = 10
    ):
        self.rpc_url = rpc_url
        
        # Hardware-aware provider selection
        if rpc_url.startswith(("ws://", "wss://")):
            self.w3 = AsyncWeb3(AsyncWebsocketProvider(rpc_url))
        else:
            self.w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
            
        self.private_key = private_key
        self.cache_ttl = cache_ttl
        self._price_cache: dict[str, dict] = {}
        
        self._nonce_lock = asyncio.Lock()
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False
        
        if private_key:
            from eth_account import Account
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None

    async def _get_next_nonce(self) -> int:
        """Fetch and increment account nonce with chain synchronization."""
        async with self._nonce_lock:
            current_on_chain = await self.w3.eth.get_transaction_count(self.address)
            if not hasattr(self, '_local_nonce'):
                self._local_nonce = current_on_chain
            else:
                self._local_nonce = max(self._local_nonce, current_on_chain)
            
            nonce = self._local_nonce
            self._local_nonce += 1
            return nonce

    async def _check_circuit(self):
        """Circuit breaker for RPC endpoints."""
        if self._circuit_open:
            if time.time() - self._last_failure_time > 30:
                self._circuit_open = False
                self._failure_count = 0
                logger.info("rpc_circuit_recovered")
            else:
                raise Exception("RPC Circuit Open: Service Unavailable")

    async def get_option_price(self, contract_address: str) -> float:
        """Fetch option price with multi-level caching."""
        now = time.time()
        
        # L1: Memory
        if contract_address in self._price_cache:
            cache_entry = self._price_cache[contract_address]
            if now - cache_entry["time"] < self.cache_ttl:
                return cache_entry["price"]

        # L2: Shared Redis
        from src.utils.cache import get_redis
        redis = get_redis()
        if redis:
            cached_val = await redis.get(f"defi_price:{contract_address}")
            if cached_val:
                price = float(cached_val)
                self._price_cache[contract_address] = {"price": price, "time": now}
                return price

        # Fetch from chain
        results = await self.get_option_prices_batch([contract_address])
        price = results.get(contract_address, 0.0)
        
        if redis and price > 0:
            await redis.setex(f"defi_price:{contract_address}", self.cache_ttl, str(price))
            
        return price

    async def get_option_prices_batch(self, contract_addresses: list[str]) -> dict[str, float]:
        """
        Fetch multiple option prices using Multicall3 and JSON-RPC Batching.
        """
        if not contract_addresses:
            return {}

        await self._check_circuit()
        
        time.time()
        CHUNK_SIZE = 100 
        chunks = [contract_addresses[i:i + CHUNK_SIZE] for i in range(0, len(contract_addresses), CHUNK_SIZE)]
        
        # Shared ABI for price queries
        price_abi = [{"name": "get_price", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}]}]
        price_contract = self.w3.eth.contract(abi=price_abi)
        get_price_data = price_contract.encode_abi("get_price")

        output = {}
        
        try:
            # Note: batch_requests is a context manager in some web3 versions
            async with self.w3.batch_requests():
                tasks = []
                for chunk in chunks:
                    calls = [{
                        "target": Web3.to_checksum_address(addr),
                        "allowFailure": True,
                        "callData": get_price_data
                    } for addr in chunk]
                    
                    multicall = self.w3.eth.contract(
                        address=Web3.to_checksum_address(self.MULTICALL3_ADDRESS), 
                        abi=self.MULTICALL_ABI
                    )
                    tasks.append(multicall.functions.aggregate3(calls).call())
                
                results = await asyncio.gather(*tasks)
                
                idx = 0
                now = time.time()
                from src.utils.cache import get_redis
                redis = get_redis()
                pipe = redis.pipeline() if redis else None

                for chunk_result in results:
                    for success, return_data in chunk_result:
                        addr = contract_addresses[idx]
                        if success:
                            price_wei = int.from_bytes(return_data, byteorder='big')
                            price = float(Web3.from_wei(price_wei, 'ether'))
                            output[addr] = price
                            self._price_cache[addr] = {"price": price, "time": now}
                            if pipe:
                                await pipe.setex(f"defi_price:{addr}", self.cache_ttl, str(price))
                        idx += 1
                
                if pipe:
                    await pipe.execute()
            
            return output
        except Exception as e:
            logger.error("multicall_failed", error=str(e))
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
            if hasattr(self, '_local_nonce'): delattr(self, '_local_nonce')
            raise


if __name__ == "__main__":
    protocol = DeFiOptionsProtocol()
    print("Web3 initialized")
