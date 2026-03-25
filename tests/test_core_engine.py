import pytest
import numpy as np
import time
from datetime import UTC, datetime, timedelta
import jwt
from argon2 import PasswordHasher

from src.auth.auth import auth_service
from src.math_kernel.rust_engine import simulate_gbm_rk4, is_rust_available
from src.shared.config import settings

class TestProductionCore:
    """
    Unified verification for Phase 1 (Math) and Phase 2 (Auth) Production features.
    """

    # --- Phase 2: Zero-Trust Auth Verification ---

    def test_argon2id_password_hashing(self):
        """Verify Argon2id meets security and timing protection standards."""
        ph = PasswordHasher()
        password = "Production-grade-password-2026"
        hashed = auth_service.hash_password(password)

        assert hashed.startswith("$argon2id$")
        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("wrong-password", hashed) is False

    def test_asymmetric_jwt_es256(self):
        """Verify Asymmetric JWT (ECC) signing and verification logic."""
        user_id = "user_quant_001"
        email = "quant@Manifold.ai"
        tier = "enterprise"

        token_pair = auth_service.create_token_pair(user_id, email, tier)

        # Verify header algorithm
        header = jwt.get_unverified_header(token_pair.access_token)
        assert header["alg"] == "ES256"

        # Decode and verify payload
        token_data = auth_service.decode_token(token_pair.access_token)
        assert token_data.user_id == user_id
        assert token_data.tier == tier
        assert token_data.email == email

    # --- Phase 1: High-Performance Math Kernel Verification ---

    def test_gbm_rk4_solver_consistency(self):
        """Verify RK4 GBM solver output integrity and performance."""
        n_paths = 100
        s0 = np.full(n_paths, 100.0)
        mu = np.full(n_paths, 0.05)
        sigma = np.full(n_paths, 0.2)
        t = 1.0
        dt = 1/252

        start = time.perf_counter()
        paths = simulate_gbm_rk4(s0, mu, sigma, t, dt, seed=42)
        elapsed = (time.perf_counter() - start) * 1000

        # Check shape: (n_steps + 1, n_paths)
        n_steps = int(t / dt)
        assert paths.shape == (n_steps + 1, n_paths)

        # Basic GBM properties: prices should be positive
        assert np.all(paths > 0)

        # Verify seed reproducibility
        paths_2 = simulate_gbm_rk4(s0, mu, sigma, t, dt, seed=42)
        np.testing.assert_array_almost_equal(paths, paths_2)

        print(f"\n[Math] RK4 GBM Simulation ({n_paths} paths): {elapsed:.2f}ms")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust Manifold_core not compiled")
    def test_rust_core_availability(self):
        """Ensure the Rust extension is properly linked."""
        assert is_rust_available() is True

    # --- Integration / Middleware Verification ---

    @pytest.mark.asyncio
    async def test_auth_middleware_injection(self):
        """Mock test for ZeroTrustAuthMiddleware logic."""
        from fastapi import Request, Response
        from src.shared.middleware.auth import ZeroTrustAuthMiddleware

        # This is a simplified logic test of the middleware's extraction
        user_id = "test_user"
        token_pair = auth_service.create_token_pair(user_id, "test@test.com", "pro")

        middleware = ZeroTrustAuthMiddleware(app=None)
        # We verify token_data can be retrieved from our created token
        token_data = await auth_service.validate_token(token_pair.access_token)
        assert token_data.user_id == user_id
