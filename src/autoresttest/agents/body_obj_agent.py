import random
from typing import Dict, Optional

import numpy as np

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.utils import get_combinations, get_required_body_params


class BodyObjAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.q_table: Dict[str, Dict[str, Dict[Optional[str], float]]] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self) -> None:
        for (
            operation_id,
            operation_node,
        ) in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {}
            if operation_node.operation_properties.request_body:
                for (
                    mime,
                    body_properties,
                ) in operation_node.operation_properties.request_body.items():
                    if body_properties.type == "object" and body_properties.properties:
                        body_obj_combinations = get_combinations(
                            body_properties.properties.keys()
                        )
                        self.q_table[operation_id][mime] = {
                            body_obj: 0 for body_obj in body_obj_combinations
                        }
                        self.q_table[operation_id][mime]["None"] = 0

    def get_action(self, operation_id: str, mime: str):
        if operation_id not in self.q_table:
            raise ValueError(f"Operation '{operation_id}' not found in the Q-table for BodyObjAgent.")
        if mime not in self.q_table.get(operation_id, {}):
            raise ValueError(f"MIME type '{mime}' not found for operation '{operation_id}' in the Q-table for BodyObjAgent.")
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id, mime)
        return self.get_best_action(operation_id, mime)

    def get_best_action(self, operation_id: str, mime: str):
        required_obj_params = get_required_body_params(
            self.operation_graph.operation_nodes[
                operation_id
            ].operation_properties.request_body[mime]
        )
        best_action = (None, -np.inf)
        if required_obj_params:
            for body_obj, value in self.q_table[operation_id][mime].items():
                if value > best_action[1] and required_obj_params.issubset(
                    set(body_obj)
                ):
                    best_action = (body_obj, value)
            best_action = best_action[0]  # Extract just the body_obj key
        else:
            best_action = (
                max(self.q_table[operation_id][mime].items(), key=lambda x: x[1])[0]
                if self.q_table[operation_id][mime]
                else None
            )
        if best_action == "None":
            best_action = None
        return best_action

    def get_random_action(self, operation_id: str, mime: str):
        action = (
            random.choice(list(self.q_table[operation_id][mime].keys()))
            if self.q_table[operation_id][mime]
            else None
        )
        if action == "None":
            action = None
        return action

    def update_q_table(
        self, operation_id: str, mime: str, action, reward: float
    ) -> None:
        if action is None:
            action = "None"
        current_q = (
            self.q_table[operation_id][mime][action]
            if self.q_table[operation_id][mime]
            else 0.0
        )
        best_next_q = (
            max(self.q_table[operation_id][mime].values())
            if self.q_table[operation_id][mime]
            else 0.0
        )
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id][mime][action] = new_q

    def get_Q_next(self, operation_id: str, mime: str) -> float:
        return (
            max(self.q_table[operation_id][mime].values())
            if self.q_table[operation_id][mime]
            else 0.0
        )

    def get_Q_curr(self, operation_id: str, mime: str, action) -> float:
        if action is None:
            action = "None"
        return (
            self.q_table[operation_id][mime].get(action, 0.0)
            if self.q_table[operation_id][mime]
            else 0.0
        )

    def update_Q_item(
        self, operation_id: str, mime: str, action, td_error: float
    ) -> None:
        if action is None:
            action = "None"
        self.q_table[operation_id][mime][action] += self.alpha * td_error

    def number_of_zeros(self, operation_id: str) -> int:
        zeros = 0
        for body_obj_mappings in self.q_table.get(operation_id, {}).values():
            for value in body_obj_mappings.values():
                if value == 0:
                    zeros += 1
        return zeros
