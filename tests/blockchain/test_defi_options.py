from unittest.mock import MagicMock, patch

import pytest

from src.blockchain.defi_options import DeFiOptionsProtocol


@pytest.fixture
def mock_web3():
    with patch("src.blockchain.defi_options.Web3") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

def test_protocol_initialization(mock_web3):
    """Verify that the protocol initializes with a Web3 provider."""
    rpc_url = "https://polygon-rpc.com"
    protocol = DeFiOptionsProtocol(rpc_url=rpc_url)
    assert protocol.rpc_url == rpc_url
    assert protocol.w3 == mock_web3

def test_get_option_price(mock_web3):
    """Verify that the protocol can fetch option prices from a contract."""
    # Mock contract behavior
    mock_contract = MagicMock()
    mock_web3.eth.contract.return_value = mock_contract
    mock_contract.functions.get_price.return_value.call.return_value = 1500000000000000000 # 1.5 ETH in wei
    
    # Mock Web3.from_wei (it's a static method, often called on the class or instance)
    with patch("src.blockchain.defi_options.Web3.from_wei", return_value=1.5):
        protocol = DeFiOptionsProtocol()
        price = protocol.get_option_price("0x0000000000000000000000000000000000000000")
    
    assert price == 1.5

def test_buy_option_success(mock_web3):
    """Verify that the protocol can execute an option purchase."""
    mock_contract = MagicMock()
    mock_web3.eth.contract.return_value = mock_contract
    
    # Mock transaction flow
    mock_web3.eth.get_transaction_count.return_value = 10
    mock_web3.eth.account.sign_transaction.return_value.raw_transaction = b"signed_tx"
    
    # Return a real bytes object for tx_hash
    tx_hash_bytes = b"tx_hash_bytes_32"
    mock_web3.eth.send_raw_transaction.return_value = tx_hash_bytes
    mock_web3.eth.wait_for_transaction_receipt.return_value = {"status": 1}
    
    # Mock private key initialization
    mock_web3.eth.account.from_key.return_value.address = "0xAddress"
    
    protocol = DeFiOptionsProtocol(private_key="0x" + "a"*64)
    tx_hash_hex = protocol.buy_option("0x0000000000000000000000000000000000000000", amount=1)
    
    assert tx_hash_hex == tx_hash_bytes.hex()
    mock_web3.eth.send_raw_transaction.assert_called_once()
