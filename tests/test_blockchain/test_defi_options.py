import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.blockchain.defi_options import DeFiOptionsProtocol


@pytest.fixture
def mock_web3():
    with patch("services.blockchain.defi_options.AsyncWeb3") as MockW3:
        mock_instance = AsyncMock()
        # Fix: eth.contract should be synchronous (MagicMock)
        mock_instance.eth.contract = MagicMock()
        # Fix: eth.account should be synchronous (MagicMock)
        mock_instance.eth.account = MagicMock()
        # Fix: eth.send_raw_transaction should be Async (default is AsyncMock, but explicitly setting ensures it)
        mock_instance.eth.send_raw_transaction = AsyncMock()

        MockW3.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def protocol(mock_web3):
    return DeFiOptionsProtocol(rpc_url="http://mock-rpc", private_key="0x" + "01" * 32)


@pytest.mark.asyncio
async def test_initialization(protocol):
    assert protocol.rpc_url == "http://mock-rpc"
    assert protocol.address is not None


@pytest.mark.asyncio
async def test_get_option_price_cached(protocol):
    protocol._price_cache["0x0000000000000000000000000000000000000001"] = {
        "price": 100.0,
        "time": 9999999999,
    }
    price = await protocol.get_option_price("0x0000000000000000000000000000000000000001")
    assert price == 100.0


@pytest.mark.asyncio
async def test_get_option_price_fetch(protocol, mock_web3):
    addr = "0x0000000000000000000000000000000000000001"

    # Mock fallback parallel call logic
    with patch.object(protocol, "_get_option_prices_parallel", return_value={addr: 50.0}):
        # Mock batch_requests to fail
        mock_web3.batch_requests.side_effect = Exception("Batch failed")

        # Setup encode_abi for the synchronous contract call
        contract_instance = MagicMock()
        contract_instance.encode_abi.return_value = b"data"
        mock_web3.eth.contract.return_value = contract_instance

        price = await protocol.get_option_prices_batch([addr])
        assert price[addr] == 50.0


@pytest.mark.asyncio
async def test_buy_option_slippage(protocol, mock_web3):
    addr = "0x0000000000000000000000000000000000000001"

    # Mock get_price call
    contract_mock = MagicMock()
    # Contract functions return an AsyncMock for the call()
    contract_mock.functions.get_price.return_value.call = AsyncMock(return_value=10**18)  # 1 ETH
    mock_web3.eth.contract.return_value = contract_mock

    # Mock Gas
    mock_web3.eth.get_block.return_value = {"baseFeePerGas": 100}

    # FIX: max_priority_fee is awaited as a property, so it must be a Future/Awaitable
    f = asyncio.Future()
    f.set_result(10)
    mock_web3.eth.max_priority_fee = f

    # Mock Transaction
    contract_mock.functions.buy.return_value.estimate_gas = AsyncMock(return_value=21000)
    contract_mock.functions.buy.return_value.build_transaction = AsyncMock(
        return_value={"data": "0x"}
    )

    # Fix: account.sign_transaction is synchronous
    mock_web3.eth.account.sign_transaction.return_value = MagicMock(raw_transaction=b"signed")

    mock_web3.eth.send_raw_transaction = AsyncMock(return_value=b"hash")
    mock_web3.eth.wait_for_transaction_receipt = AsyncMock(return_value={"status": 1})

    tx = await protocol.buy_option(addr, 1)
    assert tx == "68617368"  # 'hash'.hex()
