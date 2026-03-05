import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from src.blockchain.nonce_manager import NonceManager
from src.blockchain.oracle import OracleManager
from src.blockchain.defi_options import DeFiOptionsProtocol

class TestRevampPhase3:
    @pytest.mark.asyncio
    async def test_nonce_manager_redis_flow(self):
        address = "0x1234567890123456789012345678901234567890"
        chain_id = 137
        
        with patch("src.blockchain.nonce_manager.get_redis") as mock_get_redis:
            mock_redis = AsyncMock()
            mock_get_redis.return_value = mock_redis
            
            # 1. Initial state (Redis empty)
            mock_redis.get.return_value = None
            manager = NonceManager(address, chain_id)
            
            mock_w3_func = AsyncMock(return_value=10)
            nonce = await manager.get_next_nonce(mock_w3_func)
            
            assert nonce == 10
            mock_redis.set.assert_called_with(manager.redis_key, 10)
            
            # 2. Sequential call (Redis has value)
            mock_redis.get.return_value = b"10"
            mock_redis.incr.return_value = 11
            nonce2 = await manager.get_next_nonce(mock_w3_func)
            assert nonce2 == 10
            
            # 3. Reset
            await manager.reset(mock_w3_func)
            mock_redis.set.assert_called_with(manager.redis_key, 10)

    @pytest.mark.asyncio
    async def test_oracle_manager_layered_discovery(self):
        with patch("src.blockchain.oracle.get_redis") as mock_get_redis:
            mock_redis = AsyncMock()
            mock_get_redis.return_value = mock_redis
            
            oracle = OracleManager()
            mock_w3_func = AsyncMock(return_value=105.0)
            
            # 1. Local override
            oracle.update_price_feed("AAPL", 110.0)
            price = await oracle.get_price("AAPL", "0xaddr", mock_w3_func)
            assert price == 110.0
            
            # 2. Redis cache
            mock_redis.get.return_value = b"102.0"
            price2 = await oracle.get_price("MSFT", "0xaddr2", mock_w3_func)
            assert price2 == 102.0
            
            # 3. On-chain fallback
            mock_redis.get.return_value = None
            price3 = await oracle.get_price("GOOG", "0xaddr3", mock_w3_func)
            assert price3 == 105.0

    @pytest.mark.asyncio
    async def test_defi_protocol_nonce_integration(self):
        rpc_url = "http://localhost:8545"
        with patch("src.blockchain.defi_options.AsyncWeb3"), \
             patch("src.blockchain.defi_options.NonceManager") as mock_nonce_manager_cls, \
             patch("src.blockchain.defi_options.OracleManager") as mock_oracle_manager_cls:
            
            mock_nonce_manager = mock_nonce_manager_cls.return_value
            mock_nonce_manager.get_next_nonce = AsyncMock(return_value=42)
            mock_nonce_manager.reset = AsyncMock()
            
            mock_oracle = mock_oracle_manager_cls.return_value
            mock_oracle.get_price = AsyncMock(return_value=1.0)
            
            protocol = DeFiOptionsProtocol(rpc_url=rpc_url, private_key="0x" + "1"*64)
            
            # Mock transaction construction
            protocol.w3.eth.get_block = AsyncMock(return_value={"baseFeePerGas": 100})
            type(protocol.w3.eth).max_priority_fee = PropertyMock(side_effect=AsyncMock(return_value=10))
            
            # Mock contract
            mock_contract = MagicMock()
            mock_contract.functions.get_price().call = AsyncMock(return_value=10**18)
            mock_contract.functions.buy().estimate_gas = AsyncMock(return_value=21000)
            mock_contract.functions.buy().build_transaction = AsyncMock(return_value={})
            
            protocol.w3.eth.contract = MagicMock(return_value=mock_contract)
            
            # Mock sign/send
            protocol.w3.eth.account.sign_transaction = MagicMock()
            mock_tx_res = MagicMock()
            mock_tx_res.hex.return_value = "0xhash"
            protocol.w3.eth.send_raw_transaction = AsyncMock(return_value=mock_tx_res)
            protocol.w3.eth.wait_for_transaction_receipt = AsyncMock(return_value={"status": 1})
            protocol.w3.eth.get_transaction_count = AsyncMock(return_value=10)
            
            tx_hash = await protocol.buy_option("0xcontract", 1, params={"symbol": "AAPL"})
            
            assert tx_hash == "0xhash"
            mock_nonce_manager.get_next_nonce.assert_called_once()
            mock_oracle.get_price.assert_called_once()
