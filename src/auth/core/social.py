"""
Social OAuth2 Substrate.
Supports Google/GitHub with Account Merging logic.
"""

import logging
from authlib.integrations.starlette_client import OAuth
from src.shared.config import settings

logger = logging.getLogger(__name__)

oauth = OAuth()

# Registration of OAuth providers
if settings.GOOGLE_CLIENT_ID:
    oauth.register(
        name="google",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if settings.GITHUB_CLIENT_ID:
    oauth.register(
        name="github",
        client_id=settings.GITHUB_CLIENT_ID,
        client_secret=settings.GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        access_token_params=None,
        authorize_url="https://github.com/login/oauth/authorize",
        authorize_params=None,
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "user:email"},
    )

class SocialAuthService:
    """
    Social OAuth2 orchestration and Account Merging.
    """
    async def get_user_info(self, client_name: str, token: dict):
        """Fetch user info from the OAuth provider."""
        client = oauth.create_client(client_name)
        if client_name == "google":
            return await client.userinfo(token=token)
        elif client_name == "github":
            resp = await client.get("user", token=token)
            user_info = resp.json()
            # GitHub might not have email in root if private
            if not user_info.get("email"):
                emails = await client.get("user/emails", token=token)
                user_info["email"] = next((e["email"] for e in emails.json() if e["primary"]), None)
            return user_info
        return None

    def should_merge_accounts(self, existing_user, social_email: str):
        """
        Logic to determine if a social account should be merged with an existing local account.
        Rule: If email matches and is verified, merge is allowed.
        """
        if not existing_user:
            return False
        if existing_user.email == social_email and existing_user.is_verified:
            return True
        return False

# Global instance for easy access
social_service = SocialAuthService()
