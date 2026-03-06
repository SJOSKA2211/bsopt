import numpy as np
import pytest
import torch

from src.ml.reinforcement_learning.kernels import _fused_state_kernel
from src.ml.reinforcement_learning.transformer_policy import CausalSelfAttention


class TestRevampPhase2:
    def test_fused_state_kernel_spectral(self):
        balance = 100000.0
        initial_balance = 100000.0
        positions = np.zeros(10, dtype=np.float32)
        prices = np.ones(10, dtype=np.float32) * 105.0
        strikes = np.ones(10, dtype=np.float32) * 100.0
        greeks = np.zeros(50, dtype=np.float32)
        indicators = np.zeros(20, dtype=np.float32)
        window_size = 5
        window_buffer = np.zeros((window_size, 100), dtype=np.float32)

        # Test step 0
        obs = _fused_state_kernel(
            balance,
            initial_balance,
            positions,
            prices,
            strikes,
            greeks,
            indicators,
            window_buffer,
            0,
            window_size,
        )

        assert obs.shape == (window_size, 100)
        # Latest observation (index 4) should have the values
        latest = obs[-1]
        assert latest[0] == pytest.approx(1.0)  # balance/initial
        assert latest[11] == pytest.approx(np.log(105.0 / 100.0))  # log-moneyness

        # Spectral features (Sine/Cosine) at indices 61-70
        assert latest[61] == pytest.approx(np.sin(105.0 / 100.0))
        assert latest[66] == pytest.approx(np.cos(105.0 / 100.0))

    def test_flash_attention_forward(self):
        n_embd = 128
        n_head = 4
        n_positions = 20
        attn = CausalSelfAttention(n_embd, n_head, n_positions, 0.1, 0.1)

        x = torch.randn(2, 10, n_embd)  # Batch=2, Seq=10
        output = attn(x)

        assert output.shape == (2, 10, n_embd)
        assert not torch.isnan(output).any()
