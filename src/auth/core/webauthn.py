"""
WebAuthn (Passkey) Substrate.
Supports multi-device registration and authentication.
"""

import logging

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.structs import (
    AuthenticationCredential,
    AuthenticatorSelectionCriteria,
    RegistrationCredential,
    UserVerificationRequirement,
)

from src.shared.config import settings

logger = logging.getLogger(__name__)


class WebAuthnService:
    """
    WebAuthn/Passkey registration and authentication orchestration.
    Deterministic implementation: No mocks allowed.
    """

    def __init__(self):
        self.rp_id = settings.WEBAUTHN_RP_ID
        self.rp_name = settings.WEBAUTHN_RP_NAME
        self.origin = settings.WEBAUTHN_ORIGIN

    def get_registration_options(
        self, user_id: str, email: str, existing_credentials: list[bytes] = []
    ):
        """Generate options for a new passkey registration."""
        # Purged: No longer returns mocked status if library is missing.
        options = generate_registration_options(
            rp_id=self.rp_id,
            rp_name=self.rp_name,
            user_id=user_id,
            user_name=email,
            exclude_credentials=existing_credentials,
            authenticator_selection=AuthenticatorSelectionCriteria(
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
        )
        return options_to_json(options)

    def verify_registration(self, registration_response: dict, expected_challenge: str):
        """Verify the response from the authenticator during registration."""
        # Purged: No longer checks HAS_WEBAUTHN.
        verification = verify_registration_response(
            credential=RegistrationCredential.parse_obj(registration_response),
            expected_challenge=expected_challenge,
            expected_origin=self.origin,
            expected_rp_id=self.rp_id,
        )
        return verification

    def get_authentication_options(self, allow_credentials: list[bytes] = []):
        """Generate options for passkey login."""
        # Purged mocked responses.
        options = generate_authentication_options(
            rp_id=self.rp_id,
            allow_credentials=allow_credentials,
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        return options_to_json(options)

    def verify_authentication(
        self,
        authentication_response: dict,
        expected_challenge: str,
        credential_public_key: bytes,
        credential_current_sign_count: int,
    ):
        """Verify the response from the authenticator during login."""
        verification = verify_authentication_response(
            credential=AuthenticationCredential.parse_obj(authentication_response),
            expected_challenge=expected_challenge,
            expected_origin=self.origin,
            expected_rp_id=self.rp_id,
            credential_public_key=credential_public_key,
            credential_current_sign_count=credential_current_sign_count,
        )
        return verification


# Global instance for easy access
webauthn_service = WebAuthnService()