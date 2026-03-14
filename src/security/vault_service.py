"""
HashiCorp Vault Secret Management Service

Handles rotational secret fetching, JWT key management, and 
secure configuration injection for EquaFlow.
"""

import os
from typing import Any

import hvac
import structlog
from src.config import settings

logger = structlog.get_logger(__name__)

class VaultService:
    """
    Secure client for HashiCorp Vault.
    """

    def __init__(self):
        self.url = os.getenv("VAULT_ADDR", "http://vault:8200")
        self.token = os.getenv("VAULT_TOKEN")
        self.client = hvac.Client(url=self.url, token=self.token)

    def is_authenticated(self) -> bool:
        """Checks if the client is authenticated with Vault."""
        try:
            return self.client.is_authenticated()
        except Exception as e:
            logger.error("vault_auth_check_failed", error=str(e))
            return False

    def get_secret(self, path: str, mount_point: str = "secret") -> dict[str, Any]:
        """
        Fetches a secret from Vault KV V2 engine.
        """
        try:
            read_response = self.client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=mount_point
            )
            return read_response["data"]["data"]
        except Exception as e:
            logger.error("vault_secret_fetch_failed", path=path, error=str(e))
            return {}

    def get_jwt_keys(self) -> dict[str, str]:
        """
        Fetches JWT RSA/ECC keys from Vault.
        """
        keys = self.get_secret("jwt/keys")
        return {
            "RSA_PRIVATE": keys.get("RSA_PRIVATE_KEY", ""),
            "RSA_PUBLIC": keys.get("RSA_PUBLIC_KEY", ""),
            "ECC_PRIVATE": keys.get("ECC_PRIVATE_KEY", ""),
            "ECC_PUBLIC": keys.get("ECC_PUBLIC_KEY", ""),
        }

vault_service = VaultService()
