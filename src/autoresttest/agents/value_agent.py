import random
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.models import ValueAction


class ValueAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.q_table: Dict[str, Dict[str, Dict[str, list]]] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self) -> None:
        responses = defaultdict(list)
        visited = set()
        request_generator = self.operation_graph.request_generator
        for (
            operation_id,
            operation_node,
        ) in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                request_generator.value_depth_traversal(
                    operation_node, self.q_table, responses, visited
                )

    def get_action(self, operation_id: str) -> ValueAction:
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id: str) -> ValueAction:
        param_mappings = (
            {
                param: max(param_mappings, key=lambda pm: pm[1])[0]
                for param, param_mappings in self.q_table[operation_id][
                    "params"
                ].items()
            }
            if self.q_table[operation_id].get("params")
            else None
        )
        body_mappings = (
            {
                mime: max(body_mappings, key=lambda bm: bm[1])[0]
                for mime, body_mappings in self.q_table[operation_id]["body"].items()
            }
            if self.q_table[operation_id].get("body")
            else None
        )
        return ValueAction(param_mappings, body_mappings)

    def get_random_action(self, operation_id: str) -> ValueAction:
        param_mappings = (
            {
                param: random.choice(param_mappings)[0]
                for param, param_mappings in self.q_table[operation_id][
                    "params"
                ].items()
            }
            if self.q_table[operation_id].get("params")
            else None
        )
        body_mappings = (
            {
                mime: random.choice(body_mappings)[0]
                for mime, body_mappings in self.q_table[operation_id]["body"].items()
            }
            if self.q_table[operation_id].get("body")
            else None
        )
        return ValueAction(param_mappings, body_mappings)

    def update_q_table(
        self, operation_id: str, filtered_action: ValueAction, reward: float
    ) -> None:
        if filtered_action.param_mappings:
            for param, value in filtered_action.param_mappings.items():
                current_q = 0.0
                best_next_q = -np.inf
                for mapping in self.q_table[operation_id]["params"][param]:
                    best_next_q = max(best_next_q, mapping[1])
                    if mapping[0] == value:
                        current_q = mapping[1]
                new_q = current_q + self.alpha * (
                    reward + self.gamma * best_next_q - current_q
                )
                for mapping in self.q_table[operation_id]["params"][param]:
                    if mapping[0] == value:
                        mapping[1] = new_q
        if filtered_action.body_mappings:
            for mime, body in filtered_action.body_mappings.items():
                current_q = 0.0
                best_next_q = -np.inf
                for mapping in self.q_table[operation_id]["body"][mime]:
                    best_next_q = max(best_next_q, mapping[1])
                    if mapping[0] == body:
                        current_q = mapping[1]
                new_q = current_q + self.alpha * (
                    reward + self.gamma * best_next_q - current_q
                )
                for mapping in self.q_table[operation_id]["body"][mime]:
                    if mapping[0] == body:
                        mapping[1] = new_q

    def get_Q_next(self, operation_id: str, filtered_action: ValueAction):
        best_Q_next_params = []
        best_Q_next_body = []
        if filtered_action.param_mappings:
            best_next_q = -np.inf
            for param, value in filtered_action.param_mappings.items():
                for mapping in self.q_table[operation_id]["params"][param]:
                    best_next_q = max(best_next_q, mapping[1])
                best_Q_next_params.append(best_next_q)
        if filtered_action.body_mappings:
            best_next_q = -np.inf
            for mime, body in filtered_action.body_mappings.items():
                for mapping in self.q_table[operation_id]["body"][mime]:
                    best_next_q = max(best_next_q, mapping[1])
                best_Q_next_body.append(best_next_q)
        return best_Q_next_params, best_Q_next_body

    def get_Q_curr(self, operation_id: str, filtered_action: ValueAction):
        current_Q_params = []
        current_Q_body = []
        if filtered_action.param_mappings:
            for param, value in filtered_action.param_mappings.items():
                current_q = 0.0
                for mapping in self.q_table[operation_id]["params"][param]:
                    if mapping[0] == value:
                        current_q = mapping[1]
                current_Q_params.append(current_q)
        if filtered_action.body_mappings:
            for mime, body in filtered_action.body_mappings.items():
                current_q = 0.0
                for mapping in self.q_table[operation_id]["body"][mime]:
                    if mapping[0] == body:
                        current_q = mapping[1]
                current_Q_body.append(current_q)
        return current_Q_params, current_Q_body

    def update_Q_item(
        self, operation_id: str, action: ValueAction, td_error: float
    ) -> None:
        if action.param_mappings:
            for param, value in action.param_mappings.items():
                for mapping in self.q_table[operation_id]["params"][param]:
                    if mapping[0] == value:
                        mapping[1] += self.alpha * td_error
        if action.body_mappings:
            for mime, body in action.body_mappings.items():
                for mapping in self.q_table[operation_id]["body"][mime]:
                    if mapping[0] == body:
                        mapping[1] += self.alpha * td_error

    def number_of_zeros(self, operation_id: str) -> int:
        zeros = 0
        for param_mappings in self.q_table[operation_id].get("params", {}).values():
            for mapping in param_mappings:
                if mapping[1] == 0:
                    zeros += 1
        for body_mappings in self.q_table[operation_id].get("body", {}).values():
            for mapping in body_mappings:
                if mapping[1] == 0:
                    zeros += 1
        return zeros
