from cryptography.fernet import Fernet
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class EncryptionService:
    def __init__(self):
        try:
            self.fernet = Fernet(settings.FIELD_ENCRYPTION_KEY)
        except Exception as e:
            logger.critical(f"Failed to initialize encryption service: {e}")
            raise

    def encrypt(self, data: str) -> str:
        """Encrypts a string."""
        if not data:
            return data
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypts a token."""
        if not token:
            return token
        return self.fernet.decrypt(token.encode()).decode()

encryption_service = EncryptionService()
