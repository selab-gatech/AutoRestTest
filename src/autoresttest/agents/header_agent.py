import random
from typing import Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.utils import construct_basic_token


class HeaderAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.q_table: Dict[str, List[List]] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self) -> None:
        request_generator = self.operation_graph.request_generator
        token_list: List = []
        for operation_node in self.operation_graph.operation_nodes.values():
            token_info = request_generator.get_auth_info(operation_node, 5)
            for token in token_info:
                token_list.append(construct_basic_token(token))
        for operation_id in self.operation_graph.operation_nodes.keys():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = []
            random.shuffle(token_list)
            for i in range(min(9, len(token_list))):
                self.q_table[operation_id].append([token_list[i], 0])
            self.q_table[operation_id].append([None, 0])

    def get_action(self, operation_id: str) -> Optional[str]:
        if operation_id not in self.q_table:
            raise ValueError(f"Operation '{operation_id}' not found in the Q-table for HeaderAgent.")
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id: str) -> Optional[str]:
        if not self.q_table.get(operation_id):
            return None
        return max(self.q_table[operation_id], key=lambda x: x[1])[0]

    def get_random_action(self, operation_id: str) -> Optional[str]:
        return random.choice(self.q_table.get(operation_id, [[None]]))[0]

    def update_q_table(
        self, operation_id: str, action: Optional[str], reward: float
    ) -> None:
        current_q = 0.0
        best_next_q = -np.inf
        for mapping in self.q_table[operation_id]:
            best_next_q = max(best_next_q, mapping[1])
            if mapping[0] == action:
                current_q = mapping[1]
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        for mapping in self.q_table[operation_id]:
            if mapping[0] == action:
                mapping[1] = new_q

    def get_Q_next(self, operation_id: str) -> float:
        best_next_q = -np.inf
        for mapping in self.q_table[operation_id]:
            best_next_q = max(best_next_q, mapping[1])
        return best_next_q

    def get_Q_curr(self, operation_id: str, token: Optional[str]) -> float:
        current_q = 0.0
        for mapping in self.q_table[operation_id]:
            if mapping[0] == token:
                current_q = mapping[1]
        return current_q

    def update_Q_item(
        self, operation_id: str, token: Optional[str], td_error: float
    ) -> None:
        for mapping in self.q_table[operation_id]:
            if mapping[0] == token:
                mapping[1] += self.alpha * td_error

    def number_of_zeros(self, operation_id: str) -> int:
        zeros = 0
        for mapping in self.q_table.get(operation_id, []):
            if mapping[1] == 0:
                zeros += 1
        return zeros
