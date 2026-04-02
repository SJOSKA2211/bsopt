"""
HashiCorp Vault Secret Management Service

Handles rotational secret fetching, JWT key management, and
secure configuration injection for Manifold.
Enhanced with automatic token renewal and AppRole support.
"""

import os
import threading
import time
from typing import Any

import hvac
import structlog

logger = structlog.get_logger(__name__)

class VaultService:
    """
    Secure client for HashiCorp Vault.
    Optimized for high-availability with automatic token renewal.
    """

    def __init__(self):
        self.url = os.getenv("VAULT_ADDR", "http://vault:8200")
        self.token = os.getenv("VAULT_TOKEN")
        self.role_id = os.getenv("VAULT_ROLE_ID")
        self.secret_id = os.getenv("VAULT_SECRET_ID")
        
        self.client = hvac.Client(url=self.url, token=self.token)
        
        # 1. Prefer AppRole for production identity
        if self.role_id and self.secret_id:
            self._authenticate_approle()
        
        # 2. Start token renewal background loop if authenticated
        self._stop_event = threading.Event()
        if self.is_authenticated():
            self._renewal_thread = threading.Thread(target=self._token_renewal_loop, daemon=True)
            self._renewal_thread.start()
            logger.info("vault_renewal_loop_started")

    def _authenticate_approle(self):
        """Authenticates using AppRole credentials."""
        try:
            response = self.client.auth.approle.login(
                role_id=self.role_id,
                secret_id=self.secret_id,
            )
            self.client.token = response['auth']['client_token']
            logger.info("vault_approle_auth_success")
        except Exception as e:
            logger.error("vault_approle_auth_failed", error=str(e))

    def _token_renewal_loop(self):
        """Background thread to keep the Vault token alive."""
        while not self._stop_event.is_set():
            try:
                # Only renew if we have a token and it's actually renewable
                auth_info = self.client.lookup_token()
                if auth_info['data']['renewable']:
                    self.client.auth.token.renew_self()
                    logger.debug("vault_token_renewed")
            except Exception as e:
                # If lookup fails, try to re-authenticate if AppRole is used
                if self.role_id and self.secret_id:
                    logger.warning("vault_token_stale_reauthenticating", error=str(e))
                    self._authenticate_approle()
                else:
                    logger.warning("vault_token_renewal_failed", error=str(e))
            
            # Check every 5 minutes
            time.sleep(300)

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
        Returns empty dict on failure to prevent application crash.
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

    def close(self):
        """Stop background tasks."""
        self._stop_event.set()

vault_service = VaultService()
