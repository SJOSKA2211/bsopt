import pytest
from unittest.mock import MagicMock, patch
from src.blockchain.defi_options import DeFiOptionsProtocol
from web3 import Web3

# Valid checksum address for testing
VALID_ADDRESS = "0x" + "0" * 40

@pytest.fixture
def mock_w3():
    mock_w3 = MagicMock(spec=Web3)
    # Mock eth component
    mock_w3.eth = MagicMock()
    # Mock account
    mock_w3.eth.account = MagicMock()
    return mock_w3

def test_protocol_init(mock_w3):
    protocol = DeFiOptionsProtocol(w3=mock_w3)
    assert protocol.w3 == mock_w3
    assert protocol.private_key is None

def test_protocol_init_with_key(mock_w3):
    private_key = "0x" + "1" * 64
    mock_account = MagicMock()
    mock_account.address = VALID_ADDRESS
    mock_w3.eth.account.from_key.return_value = mock_account
    
    protocol = DeFiOptionsProtocol(private_key=private_key, w3=mock_w3)
    assert protocol.address == VALID_ADDRESS
    mock_w3.eth.account.from_key.assert_called_with(private_key)

def test_get_option_price(mock_w3):
    protocol = DeFiOptionsProtocol(w3=mock_w3)
    mock_contract = mock_w3.eth.contract.return_value
    mock_contract.functions.get_price.return_value.call.return_value = Web3.to_wei(10.5, 'ether')
    
    price = protocol.get_option_price(VALID_ADDRESS)
    assert price == 10.5
    mock_w3.eth.contract.assert_called()

def test_buy_option_success(mock_w3):
    private_key = "0x" + "1" * 64
    mock_account = MagicMock()
    mock_account.address = VALID_ADDRESS
    mock_w3.eth.account.from_key.return_value = mock_account
    
    protocol = DeFiOptionsProtocol(private_key=private_key, w3=mock_w3)
    
    mock_contract = mock_w3.eth.contract.return_value
    mock_tx = {"nonce": 0}
    mock_contract.functions.buy.return_value.build_transaction.return_value = mock_tx
    
    mock_signed = MagicMock()
    mock_signed.raw_transaction = b"signed_tx"
    mock_w3.eth.account.sign_transaction.return_value = mock_signed
    
    tx_hash = MagicMock()
    tx_hash.hex.return_value = "0xHash"
    mock_w3.eth.send_raw_transaction.return_value = tx_hash
    
    mock_receipt = {"status": 1}
    mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
    
    result = protocol.buy_option(VALID_ADDRESS, 100)
    assert result == "0xHash"

def test_buy_option_failure(mock_w3):
    private_key = "0x" + "1" * 64
    mock_account = MagicMock()
    mock_w3.eth.account.from_key.return_value = mock_account
    mock_account.address = VALID_ADDRESS
    protocol = DeFiOptionsProtocol(private_key=private_key, w3=mock_w3)
    
    tx_hash = MagicMock()
    mock_w3.eth.send_raw_transaction.return_value = tx_hash
    
    mock_w3.eth.wait_for_transaction_receipt.return_value = {"status": 0}
    
    with pytest.raises(Exception, match="Transaction failed"):
        protocol.buy_option(VALID_ADDRESS, 100)

def test_buy_option_no_key(mock_w3):
    protocol = DeFiOptionsProtocol(w3=mock_w3)
    with pytest.raises(ValueError, match="Private key required"):
        protocol.buy_option(VALID_ADDRESS, 100)
