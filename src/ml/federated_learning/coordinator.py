from typing import Any

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    fl = None  # type: ignore

import structlog

logger = structlog.get_logger()

class FederatedLearningCoordinator:
    """
    Coordinator for federated learning using the Flower framework.
    Manages the central server and aggregation strategies.
    """

    def __init__(self, server_address: str = "0.0.0.0:8080", strategy_name: str = "FedAvg") -> None:
        self.server_address = server_address
        self.strategy_name = strategy_name
        self.strategy = self._get_strategy() if FLWR_AVAILABLE else None

    def _get_strategy(self) -> Any:
        """Initialize the requested aggregation strategy."""
        if not FLWR_AVAILABLE:
            logger.warning("flwr_not_installed")
            return None

        if self.strategy_name == "FedAvg":
            return fl.server.strategy.FedAvg()
        logger.warning("unknown_strategy", strategy=self.strategy_name)
        return fl.server.strategy.FedAvg()

    def start(self, num_rounds: int = 3) -> None:
        """Start the Flower server."""
        if not FLWR_AVAILABLE:
            logger.warning("cannot_start_flwr_not_installed")
            return

        logger.info(
            "starting_fl_server",
            address=self.server_address,
            rounds=num_rounds,
            strategy=self.strategy_name,
        )

        # Configure the server
        config = fl.server.ServerConfig(num_rounds=num_rounds)

        # Start server
        fl.server.start_server(
            server_address=self.server_address,
            config=config,
            strategy=self.strategy,
        )
        logger.info("fl_server_stopped")

if __name__ == "__main__":
    import argparse

    import mlflow

    parser = argparse.ArgumentParser(description="Run Federated Learning Coordinator")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--strategy", type=str, default="FedAvg")
    parser.add_argument("--study_name", type=str, default="federated_v1")
    parser.add_argument("--tracking_uri", type=str, default=None)

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    with mlflow.start_run(run_name=args.study_name):
        mlflow.log_params(
            {"rounds": args.rounds, "strategy": args.strategy, "address": args.address}
        )
        coordinator = FederatedLearningCoordinator(
            server_address=args.address, strategy_name=args.strategy
        )
        coordinator.start(num_rounds=args.rounds)
