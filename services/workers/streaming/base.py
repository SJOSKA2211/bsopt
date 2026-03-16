from abc import ABC, abstractmethod
from typing import Any


class Producer(ABC):
    """
    Abstract base class for all streaming producers.
    Enforces a unified interface for data ingestion.
    """

    @abstractmethod
    async def produce(self, data: dict[str, Any], **kwargs: Any) -> None:
        """
        Produce a message to the stream.

        Args:
            data: The data payload.
            **kwargs: Backend-specific arguments (e.g., topic, key).
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Flush pending messages.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the producer connection.
        """
        pass
