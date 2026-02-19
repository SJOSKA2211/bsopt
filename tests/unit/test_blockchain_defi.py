import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.blockchain.defi_options import DeFiOptionsProtocol


@pytest.fixture
def blockchain_config():
    return {"rpc_url": "http://mock-eth-node:8545", "private_key": "0x" + "a" * 64}


@pytest.mark.asyncio
@patch("src.blockchain.defi_options.AsyncWeb3")
async def test_defi_options_protocol_initialization(mock_web3_class, blockchain_config):
    mock_web3 = mock_web3_class.return_value
    # Async methods need to be mocked appropriately
    mock_web3.is_connected = AsyncMock(return_value=True)

    protocol = DeFiOptionsProtocol(
        rpc_url=blockchain_config["rpc_url"],
        private_key=blockchain_config["private_key"],
    )
    assert await protocol.w3.is_connected()
    assert protocol.address is not None


@pytest.mark.asyncio
@patch("src.blockchain.defi_options.AsyncWeb3")
@patch("src.blockchain.defi_options.Web3")
async def test_defi_options_buy_logic(
    mock_web3_sync, mock_web3_class, blockchain_config
):
    mock_web3 = mock_web3_class.return_value
    mock_web3.eth.get_transaction_count = AsyncMock(return_value=10)
    mock_web3.eth.get_block = AsyncMock(return_value={"baseFeePerGas": 1000000000})

    # Properly mock max_priority_fee as an awaitable property
    mock_web3.eth.max_priority_fee = asyncio.Future()
    mock_web3.eth.max_priority_fee.set_result(2000000000)

    mock_web3.eth.send_raw_transaction = AsyncMock(
        return_value=MagicMock(hex=lambda: "tx_hash")
    )
    mock_web3.eth.wait_for_transaction_receipt = AsyncMock(return_value={"status": 1})

    # Mock account and signing
    mock_web3.eth.account.sign_transaction = MagicMock(
        return_value=MagicMock(raw_transaction=b"signed_tx")
    )

    protocol = DeFiOptionsProtocol(
        rpc_url=blockchain_config["rpc_url"],
        private_key=blockchain_config["private_key"],
    )

    # Mock the contract
    mock_contract = MagicMock()
    mock_contract.functions.get_price.return_value.call = AsyncMock(
        return_value=10**18
    )  # 1 ETH
    mock_contract.functions.buy.return_value.estimate_gas = AsyncMock(
        return_value=100000
    )
    mock_contract.functions.buy.return_value.build_transaction = AsyncMock(
        return_value={"data": "0x123"}
    )

    protocol.w3.eth.contract = MagicMock(return_value=mock_contract)

    # Run the buy logic
    tx_hash = await protocol.buy_option("0x1234567890123456789012345678901234567890", 1)

    assert tx_hash is not None
    assert mock_contract.functions.buy.called
