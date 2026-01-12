import flwr as fl
import structlog

logger = structlog.get_logger()

class FederatedLearningCoordinator:
    """
    Coordinator for federated learning using the Flower framework.
    Manages the central server and aggregation strategies.
    """
    def __init__(
        self, 
        server_address: str = "0.0.0.0:8080",
        strategy_name: str = "FedAvg"
    ):
        self.server_address = server_address
        self.strategy_name = strategy_name
        self.strategy = self._get_strategy()

    def _get_strategy(self) -> fl.server.strategy.Strategy:
        """Initialize the requested aggregation strategy."""
        if self.strategy_name == "FedAvg":
            return fl.server.strategy.FedAvg()
        else:
            logger.warning("unknown_strategy", strategy=self.strategy_name)
            return fl.server.strategy.FedAvg()

    def start(self, num_rounds: int = 3):
        """Start the Flower server."""
        logger.info("starting_fl_server", 
                    address=self.server_address, 
                    rounds=num_rounds, 
                    strategy=self.strategy_name)
        
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
    coordinator = FederatedLearningCoordinator()
    coordinator.start()
