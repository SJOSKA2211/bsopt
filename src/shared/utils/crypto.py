import base64
import os
import struct
import threading

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AES256GCM:
    """
    AES-256-GCM authenticated encryption.
    OPTIMIZED: High-speed nonce generation and raw byte paths.
    """

    def __init__(self, key_base64: str):
        self.key = base64.urlsafe_b64decode(key_base64)
        if len(self.key) != 32:
            import hashlib

            self.key = hashlib.sha256(self.key).digest()
        self.aesgcm = AESGCM(self.key)

        # High-speed Nonce State
        self._nonce_prefix = os.urandom(8)
        self._nonce_counter = 0
        self._nonce_lock = threading.Lock()

    def _get_next_nonce(self) -> bytes:
        """Atomic fast nonce generator (Prefix + Counter)."""
        with self._nonce_lock:
            self._nonce_counter += 1
            return self._nonce_prefix + struct.pack("!I", self._nonce_counter % 0xFFFFFFFF)

    def encrypt_raw(self, data: bytes) -> bytes:
        """High-performance encryption returning raw bytes (Nonce + Tag + Cipher)."""
        nonce = self._get_next_nonce()
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def encrypt(self, data: bytes) -> str:
        """Standard encryption returning URL-safe Base64 string."""
        return base64.urlsafe_b64encode(self.encrypt_raw(data)).decode("utf-8")

    def decrypt_raw(self, data: bytes) -> bytes:
        """High-performance decryption from raw bytes."""
        nonce = data[:12]
        ciphertext = data[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)

    def decrypt(self, token_base64: str) -> bytes:
        """Standard decryption from URL-safe Base64 string."""
        return self.decrypt_raw(base64.urlsafe_b64decode(token_base64))

class EIP712Signer:
    """
    Institutional EIP-712 Message Signer.
    Ensures secure, structured data signing for DeFi settlement.
    """

    @staticmethod
    def sign_settlement(private_key: str, trade_id: str, amount: int, recipient: str) -> bytes:
        from eth_account import Account

        account = Account.from_key(private_key)

        structured_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Settlement": [
                    {"name": "tradeId", "type": "bytes32"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "recipient", "type": "address"},
                ],
            },
            "primaryType": "Settlement",
            "domain": {
                "name": "EquaFlowSettlement",
                "version": "1",
                "chainId": 1,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "message": {
                "tradeId": trade_id,
                "amount": amount,
                "recipient": recipient,
            },
        }

        signed = account.sign_typed_data(full_message=structured_data)
        return signed.signature
