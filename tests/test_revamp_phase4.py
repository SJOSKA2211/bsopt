from unittest.mock import MagicMock, patch

import pytest

from src.ml.distributed_training import BSOptDistributedTrainer
from src.math_kernel.quantum_pricing import QISKIT_AVAILABLE, QuantumOptionPricer


class TestRevampPhase4:
    @pytest.mark.asyncio
    async def test_quantum_pricer_speedup_calculation(self):
        # Even if Qiskit is not available, we can test the fallback or mock the AE
        pricer = QuantumOptionPricer(use_real_quantum=False)

        # Test math fallback if QISKIT_AVAILABLE is False
        if not QISKIT_AVAILABLE:
            res = await pricer.price_european_call_quantum(100, 100, 1.0, 0.05, 0.2)
            assert res["backend"] == "analytical_fallback"
            assert res["speedup_factor"] == 1.0
        else:
            # If Qiskit is available, test real AE execution (Simulation)
            res = await pricer.price_european_call_quantum(100, 100, 1.0, 0.05, 0.2, num_qubits=3)
            assert "price" in res
            assert res["speedup_factor"] > 1.0  # Should show quadratic speedup logic
            assert "num_queries" in res

    def test_distributed_trainer_config(self):
        with (
            patch("src.ml.distributed_training.TorchTrainer") as mock_trainer_cls,
            patch("src.ml.distributed_training.ScalingConfig") as mock_scaling_cls,
        ):
            # Setup ScalingConfig mock to return a known object
            mock_scaling_instance = MagicMock()
            mock_scaling_instance.num_workers = 4
            mock_scaling_cls.return_value = mock_scaling_instance

            # Refactored for pure CPU execution
            trainer = BSOptDistributedTrainer(num_workers=4, use_gpu=False)
            config = {"lr": 1e-3, "epochs": 5}

            trainer.run(config)

            # Verify ScalingConfig construction - Force CPU only
            mock_scaling_cls.assert_called_once_with(
                num_workers=4, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0}
            )

            # Verify TorchTrainer call
            args, kwargs = mock_trainer_cls.call_args
            assert kwargs["scaling_config"] == mock_scaling_instance
            assert kwargs["train_loop_config"] == config

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
    def test_quantum_circuit_construction(self):
        pricer = QuantumOptionPricer()
        qc, prices = pricer.create_stock_price_distribution(100, 0.05, 0.2, 1.0, num_qubits=4)

        assert qc.num_qubits == 4
        assert len(prices) == 16

        pricer.add_payoff_operator(qc, prices, 100, 100)
        # 4 price qubits + 1 payoff qubit
        assert qc.num_qubits == 5
