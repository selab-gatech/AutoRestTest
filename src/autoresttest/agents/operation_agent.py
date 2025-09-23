import random
from typing import Dict

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph


class OperationAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.operation_graph = operation_graph
        self.q_table: Dict[str, float] = {}

    def initialize_q_table(self) -> None:
        operation_ids = self.operation_graph.operation_nodes.keys()
        self.q_table = {operation_id: 0 for operation_id in operation_ids}

    def get_action(self) -> str:
        if random.random() < self.epsilon:
            return self.get_random_action()
        return self.get_best_action()

    def get_best_action(self) -> str:
        return max(self.q_table.items(), key=lambda x: x[1])[0]

    def get_random_action(self) -> str:
        return random.choice(list(self.q_table.keys()))

    def update_q_table(self, operation_id: str, reward: float) -> None:
        current_q = self.q_table[operation_id]
        best_next_q = self.get_Q_next(operation_id)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id] = new_q

    def get_Q_next(self, operation_id: str) -> float:
        return max(self.q_table.values()) if self.q_table else 0.0

    def get_Q_curr(self, operation_id: str) -> float:
        return self.q_table.get(operation_id, 0.0)

    def update_Q_item(self, operation_id: str, td_error: float) -> None:
        self.q_table[operation_id] = (
            self.q_table.get(operation_id, 0.0) + self.alpha * td_error
        )
