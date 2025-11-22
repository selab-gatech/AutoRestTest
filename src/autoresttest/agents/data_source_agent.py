import random
from typing import Dict, List

from .base_agent import BaseAgent

from autoresttest.graph.generate_graph import OperationGraph


class DataSourceAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.available_data_sources: List[str] = (
            ["LLM", "DEPENDENCY", "DEFAULT"]
            if operation_graph.operation_edges
            else ["LLM", "DEFAULT"]
        )

    def initialize_q_table(self) -> None:
        for operation_id in self.operation_graph.operation_nodes.keys():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {
                    data_source: 0 for data_source in self.available_data_sources
                }

    def initialize_dependency_source(self) -> None:
        if "DEPENDENCY" not in self.available_data_sources:
            for data_sources in self.q_table.values():
                data_sources["DEPENDENCY"] = 0
            self.available_data_sources.append("DEPENDENCY")

    def get_action(self, operation_id: str) -> str:
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id: str) -> str:
        return max(self.q_table[operation_id].items(), key=lambda x: x[1])[0]

    def get_random_action(self, operation_id: str) -> str:
        return random.choice(list(self.q_table[operation_id].keys()))

    def update_q_table(self, operation_id: str, action: str, reward: float) -> None:
        current_q = self.q_table[operation_id][action]
        best_next_q = self.get_Q_next(operation_id)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id][action] = new_q

    def get_Q_next(self, operation_id: str) -> float:
        return max(self.q_table[operation_id].values())

    def get_Q_curr(self, operation_id: str, action: str) -> float:
        return self.q_table[operation_id][action]

    def update_Q_item(self, operation_id: str, action: str, td_error: float) -> None:
        self.q_table[operation_id][action] += self.alpha * td_error

    def number_of_zeros(self, operation_id: str) -> int:
        zeros = 0
        for value in self.q_table[operation_id].values():
            if value == 0:
                zeros += 1
        return zeros
