import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AES256GCM:
    """
    AES-256-GCM authenticated encryption for sensitive data.
    Provides better security and performance than Fernet (AES-128-CBC).
    """

    def __init__(self, key_base64: str):
        # Key must be 32 bytes for AES-256
        self.key = base64.urlsafe_b64decode(key_base64)
        if len(self.key) != 32:
            # Fallback/Derive if key is wrong size (e.g. from legacy Fernet keys)
            import hashlib

            self.key = hashlib.sha256(self.key).digest()
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, data: bytes) -> str:
        nonce = os.urandom(12)  # Standard GCM nonce size
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("utf-8")

    def decrypt(self, token_base64: str) -> bytes:
        data = base64.urlsafe_b64decode(token_base64)
        nonce = data[:12]
        ciphertext = data[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)
