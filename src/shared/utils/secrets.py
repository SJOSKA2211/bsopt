import base64
import os

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = structlog.get_logger(__name__)

class InstitutionalSecretManager:
    """
    Secure Secret Management for Institutional Credentials.
    Uses PBKDF2 for key derivation and Fernet for symmetric encryption.
    """

    def __init__(self, master_key_env: str = "BSOPT_MASTER_KEY"):
        password = os.getenv(master_key_env, "dev-default-secure-password").encode()
        salt = b"bsopt_institutional_salt"  # In prod, use a unique salt from env
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.fernet = Fernet(key)

    def encrypt_secret(self, plain_text: str) -> str:
        """Encrypt a plain text secret."""
        return self.fernet.encrypt(plain_text.encode()).decode()

    def decrypt_secret(self, encrypted_text: str) -> str:
        """Decrypt an encrypted secret."""
        try:
            return self.fernet.decrypt(encrypted_text.encode()).decode()
        except Exception as e:
            logger.error("secret_decryption_failed", error=str(e))
            raise

secret_manager = InstitutionalSecretManager()
