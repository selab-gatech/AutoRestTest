from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def initialize_q_table(self) -> None:
        """Populate the agent's Q-table before use."""

    @abstractmethod
    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        """Return the next action given the current state information."""

    @abstractmethod
    def get_best_action(self, *args: Any, **kwargs: Any) -> Any:
        """Return the highest-value action for the provided state."""

    @abstractmethod
    def get_random_action(self, *args: Any, **kwargs: Any) -> Any:
        """Return a random action for exploration."""

    @abstractmethod
    def get_Q_next(self, *args: Any, **kwargs: Any) -> Any:
        """Return the next-step Q-value(s) used in TD updates."""

    @abstractmethod
    def get_Q_curr(self, *args: Any, **kwargs: Any) -> Any:
        """Return the current Q-value(s) for the provided state/action."""

    @abstractmethod
    def update_Q_item(self, *args: Any, **kwargs: Any) -> None:
        """Apply a TD-error update to part of the Q-table."""
