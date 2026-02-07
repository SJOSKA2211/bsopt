"""
CLI Authentication Manager
==========================

Handles user login, logout, and token management for the CLI.
Stores tokens securely in the user's home directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)


import click


@click.group(name="auth")
def auth_group():
    """Authentication and session management."""
    pass


@auth_group.command(name="login")
@click.option("--client-id", required=True, help="OAuth2 Client ID")
@click.option(
    "--client-secret",
    required=True,
    prompt=True,
    hide_input=True,
    help="OAuth2 Client Secret",
)
def login_command(client_id: str, client_secret: str):
    """Log in to the BSOPT API."""
    manager = AuthManager()
    try:
        manager.login(client_id, client_secret)
        click.secho("Login successful!", fg="green")
    except AuthenticationError as e:
        click.secho(str(e), fg="red")


@auth_group.command(name="logout")
def logout_command():
    """Log out from the BSOPT API."""
    manager = AuthManager()
    if manager.logout():
        click.secho("Logged out successfully.", fg="green")
    else:
        click.secho("Already logged out.", fg="yellow")


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""

    pass


class AuthManager:
    """Manages CLI authentication state and API interaction."""

    def __init__(self, api_base_url: str | None = None):
        self.api_base_url = api_base_url or "http://localhost:8000"
        self.token_file = Path.home() / ".bsopt" / "token.json"
        self._ensure_token_dir()

    def _ensure_token_dir(self):
        """Ensure the directory for storing tokens exists."""
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

    def login(self, client_id: str, client_secret: str) -> dict[str, Any]:
        """
        Authenticate with the API via OAuth2 client_credentials flow.
        """
        try:
            with httpx.Client() as client:
                # 🚀 OPTIMIZATION: Real OAuth2 Token endpoint
                response = client.post(
                    f"{self.api_base_url}/api/v1/auth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                )
                response.raise_for_status()
                data = cast(dict[str, Any], response.json())

            # 🚀 SECURITY: Set restricted permissions (600)
            self.token_file.touch(mode=0o600)
            with open(self.token_file, "w") as f:
                json.dump(data, f)

            logger.info("cli_auth_successful", client_id=client_id)
            return data
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except (json.JSONDecodeError, AttributeError):
                error_detail = str(e)
            logger.error(f"Authentication failed: {error_detail}")
            raise AuthenticationError(f"Login failed: {error_detail}")
        except (httpx.RequestError, OSError) as e:
            logger.error(f"Connection error during login: {e}")
            raise AuthenticationError(
                f"Could not connect to authentication server: {str(e)}"
            )

    def logout(self) -> bool:
        """Clear stored authentication tokens."""
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                return True
        except OSError as e:
            logger.error(f"Failed to delete token file: {e}")
        return False

    def get_current_user(self) -> dict[str, Any] | None:
        """Get information about the currently logged-in user."""
        if not self.token_file.exists():
            return None

        try:
            with open(self.token_file) as f:
                data = json.load(f)
                return cast(dict[str, Any] | None, data.get("user"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read user data from token file: {e}")
            return None

    def get_token(self) -> str | None:
        """Get the stored access token."""
        if not self.token_file.exists():
            return None

        try:
            with open(self.token_file) as f:
                data = json.load(f)
                return cast(str | None, data.get("access_token"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read access token from token file: {e}")
            return None


def require_auth(func):
    """Decorator to ensure user is authenticated before running a command."""
    from functools import wraps

    import click

    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        cli_ctx = ctx.obj.get("cli_ctx")
        if not cli_ctx or not cli_ctx.auth.get_token():
            click.secho("Error: Authentication required. Please login first.", fg="red")
            ctx.exit(1)
        return func(*args, **kwargs)

    return wrapper
