import random
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cosine

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.utils import get_body_params

if TYPE_CHECKING:
    from autoresttest.marl import QLearning


class DependencyAgent(BaseAgent):
    def __init__(
        self, operation_graph: OperationGraph, alpha=0.1, gamma=0.9, epsilon=0.1
    ):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    @staticmethod
    def _bucket_location(location: str, is_source: bool = False) -> str:
        """
        Normalize various parameter locations into the q_table buckets.
        Source locations default to 'params' unless explicitly 'body'.
        Destination locations can be 'params', 'body', or 'response'.
        """
        if location == "body":
            return "body"
        if not is_source and location == "response":
            return "response"
        return "params"

    @staticmethod
    def _param_label(param_key):
        return param_key[0] if isinstance(param_key, tuple) else param_key

    def initialize_q_table(self):
        # NOTE: Could flip dependent operation id and curr parameter name around to group by operation if need-be
        for (
            operation_id,
            operation_node,
        ) in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {"params": {}, "body": {}}

            for edge in operation_node.outgoing_edges:
                for parameter, similarities in edge.similar_parameters.items():
                    for similarity in similarities:
                        processed_in_val = similarity.in_value.split(" to ")
                        src_loc = processed_in_val[0] if processed_in_val else "params"
                        dest_loc = (
                            processed_in_val[1]
                            if len(processed_in_val) > 1
                            else "params"
                        )
                        dependent_parameter = similarity.dependent_val
                        destination = edge.destination.operation_id

                        # Note: Body should be nested

                        source_bucket = self._bucket_location(src_loc, is_source=True)
                        dest_bucket = self._bucket_location(dest_loc)

                        if (
                            source_bucket == "params"
                            and parameter not in self.q_table[operation_id]["params"]
                        ):
                            self.q_table[operation_id]["params"][parameter] = {}
                        elif (
                            source_bucket == "body"
                            and parameter not in self.q_table[operation_id]["body"]
                        ):
                            self.q_table[operation_id]["body"][parameter] = {}

                        if (
                            source_bucket == "params"
                            and destination
                            not in self.q_table[operation_id]["params"][parameter]
                        ):
                            self.q_table[operation_id]["params"][parameter][
                                destination
                            ] = {"params": {}, "body": {}, "response": {}}
                        elif (
                            source_bucket == "body"
                            and destination
                            not in self.q_table[operation_id]["body"][parameter]
                        ):
                            self.q_table[operation_id]["body"][parameter][
                                destination
                            ] = {"params": {}, "body": {}, "response": {}}

                        if source_bucket == "params":
                            self.q_table[operation_id]["params"][parameter][
                                destination
                            ][dest_bucket][dependent_parameter] = 0
                        elif source_bucket == "body":
                            self.q_table[operation_id]["body"][parameter][
                                destination
                            ][dest_bucket][dependent_parameter] = 0

    def get_action(self, operation_id, qlearning):
        if operation_id not in self.q_table:
            raise ValueError(f"Operation '{operation_id}' not found in the Q-table for DependencyAgent.")
        has_success = any(
            status_code // 100 == 2
            for status_codes in qlearning.operation_response_counter.values()
            for status_code in status_codes
        )

        if random.random() < self.epsilon:
            if has_success and random.random() < 0.3:
                return self.assign_random_dependency_from_successful(
                    operation_id, qlearning
                )
            return self.get_random_action(operation_id, qlearning)

        return self.get_best_action(operation_id, qlearning)

    def get_best_action(self, operation_id, qlearning):
        successful_responses = qlearning.successful_responses
        successful_params = qlearning.successful_parameters
        successful_body = qlearning.successful_bodies

        best_params = {}
        for param, dependent_ops in self.q_table[operation_id]["params"].items():
            best_dependent = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": -np.inf,
                "in_value": None,
            }
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if (
                            value > best_dependent["value"]
                            and location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
            best_params[param] = best_dependent

        best_body = {}
        for param, dependent_ops in self.q_table[operation_id]["body"].items():
            best_dependent = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": -np.inf,
                "in_value": None,
            }
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if (
                            value > best_dependent["value"]
                            and location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
            best_body[param] = best_dependent

        return "BEST", best_params, best_body

    def get_random_action(self, operation_id, qlearning):

        successful_responses = qlearning.successful_responses
        successful_params = qlearning.successful_parameters
        successful_body = qlearning.successful_bodies

        random_params = {}
        for param, dependent_ops in self.q_table[operation_id]["params"].items():
            random_dependencies = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if (
                            location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
            random_params[param] = (
                random.choice(random_dependencies)
                if random_dependencies
                else {
                    "dependent_val": None,
                    "dependent_operation": None,
                    "value": 0,
                    "in_value": None,
                }
            )

        random_body = {}
        for param, dependent_ops in self.q_table[operation_id]["body"].items():
            random_dependencies = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if (
                            location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
            random_body[param] = (
                random.choice(random_dependencies)
                if random_dependencies
                else {
                    "dependent_val": None,
                    "dependent_operation": None,
                    "value": 0,
                    "in_value": None,
                }
            )

        return "EXPLORE", random_params, random_body

    def update_q_table(self, operation_id, dependent_params, dependent_body, reward):
        if operation_id not in self.q_table:
            return
        if dependent_params:
            for param, dependent in dependent_params.items():
                current_q = 0
                best_next_q = -np.inf
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("params", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["params"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["params"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                        best_next_q = max(best_next_q, value)
                new_q = current_q + self.alpha * (
                    reward + self.gamma * best_next_q - current_q
                )
                for location, dependent_params in self.q_table[operation_id]["params"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]["params"][param][
                                dependent["dependent_operation"]
                            ][location][dependent_param] = new_q

        if dependent_body:
            for param, dependent in dependent_body.items():
                current_q = 0
                best_next_q = -np.inf
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("body", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["body"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["body"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                        best_next_q = max(best_next_q, value)
                new_q = current_q + self.alpha * (
                    reward + self.gamma * best_next_q - current_q
                )
                for location, dependent_params in self.q_table[operation_id]["body"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]["body"][param][
                                dependent["dependent_operation"]
                            ][location][dependent_param] = new_q

    def get_Q_next(self, operation_id, dependent_params, dependent_body):
        best_next_q_params = []
        best_next_q_body = []

        if operation_id not in self.q_table:
            return best_next_q_params, best_next_q_body

        if dependent_params:
            for param, dependent in dependent_params.items():
                best_next_q = -np.inf
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("params", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["params"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["params"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        best_next_q = max(best_next_q, value)
                best_next_q_params.append(best_next_q)

        if dependent_body:
            for param, dependent in dependent_body.items():
                best_next_q = -np.inf
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("body", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["body"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["body"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        best_next_q = max(best_next_q, value)
                best_next_q_body.append(best_next_q)

        return best_next_q_params, best_next_q_body

    def get_Q_curr(self, operation_id, dependent_params, dependent_body):
        current_Q_params = []
        current_Q_body = []

        if operation_id not in self.q_table:
            return current_Q_params, current_Q_body

        if dependent_params:
            for param, dependent in dependent_params.items():
                current_q = 0
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("params", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["params"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["params"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                current_Q_params.append(current_q)

        if dependent_body:
            for param, dependent in dependent_body.items():
                current_q = 0
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("body", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["body"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["body"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                current_Q_body.append(current_q)

        return current_Q_params, current_Q_body

    def update_Q_item(self, operation_id, dependent_params, dependent_body, td_error):
        if operation_id not in self.q_table:
            return
        if dependent_params:
            for param, dependent in dependent_params.items():
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("params", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["params"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["params"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]["params"][param][
                                dependent["dependent_operation"]
                            ][location][dependent_param] += (self.alpha * td_error)

        if dependent_body:
            for param, dependent in dependent_body.items():
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("body", {}):
                    continue
                if dependent["dependent_operation"] not in self.q_table[operation_id]["body"][param]:
                    continue
                for location, dependent_params in self.q_table[operation_id]["body"][
                    param
                ][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]["body"][param][
                                dependent["dependent_operation"]
                            ][location][dependent_param] += (self.alpha * td_error)

    def add_undocumented_responses(self, new_operation_response_id, new_property):
        updated_tables = False
        dependency_comparator = self.operation_graph.dependency_comparator
        embedding_model = self.operation_graph.embedding_model
        for operation_id, operation_props in self.q_table.items():
            for location, param_values in operation_props.items():
                for param, dependent_values in param_values.items():
                    processed_param = embedding_model.handle_word_cases(
                        self._param_label(param)
                    )
                    processed_response = embedding_model.handle_word_cases(new_property)
                    param_embedding = embedding_model.encode_sentence_or_word(
                        processed_param
                    )
                    response_embedding = embedding_model.encode_sentence_or_word(
                        processed_response
                    )
                    if param_embedding is not None and response_embedding is not None:
                        similarity = 1 - cosine(param_embedding, response_embedding)
                        if similarity > dependency_comparator.threshold:
                            if new_operation_response_id not in dependent_values:
                                dependent_values[new_operation_response_id] = {}
                            if (
                                "response"
                                not in dependent_values[new_operation_response_id]
                            ):
                                dependent_values[new_operation_response_id][
                                    "response"
                                ] = {}
                            dependent_values[new_operation_response_id]["response"][
                                new_property
                            ] = 0
                            updated_tables = True
                            print(
                                "New dependency discovered between operation {} and operation {} with parameter {} and response {}".format(
                                    operation_id,
                                    new_operation_response_id,
                                    param,
                                    new_property,
                                )
                            )
        return updated_tables

    def add_new_dependency(
        self,
        operation_id,
        param_location,
        operation_param,
        dependent_operation_id,
        dependent_location,
        dependent_param,
    ):
        if operation_param not in self.q_table[operation_id][param_location]:
            self.q_table[operation_id][param_location][operation_param] = {}
        if (
            dependent_operation_id
            not in self.q_table[operation_id][param_location][operation_param]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ] = {"params": {}, "body": {}, "response": {}}
        
        # Ensure dependent_location is valid
        if dependent_location not in ["params", "body", "response"]:
            print(f"Warning: Invalid dependent_location '{dependent_location}'. Skipping dependency.")
            return
        
        # Ensure the dependent_location key exists in the structure
        if (
            dependent_location
            not in self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location] = {}
        
        if (
            dependent_param
            not in self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location][dependent_param] = 0
        print(
            "New dependency discovered between operation {} and operation {} with operation parameter {} and dependent parameter {}".format(
                operation_id, dependent_operation_id, operation_param, dependent_param
            )
        )

    # Get a random value from the successful operations to test dependencies
    def assign_random_dependency_from_successful(
        self, operation_id, qlearning: "QLearning"
    ):
        possible_options = []

        for (
            operation_idx,
            operation_parameters,
        ) in qlearning.successful_parameters.items():
            if operation_idx == operation_id:
                continue
            for parameter_name, parameter_values in operation_parameters.items():
                for parameter_value in parameter_values:
                    possible_options.append(
                        {
                            "dependent_val": parameter_name,
                            "dependent_operation": operation_idx,
                            "value": parameter_value,
                            "in_value": "params",
                        }
                    )

        for operation_idx, operation_body_parms in qlearning.successful_bodies.items():
            if operation_idx == operation_id:
                continue
            for body_name, body_values in operation_body_parms.items():
                for body_value in body_values:
                    possible_options.append(
                        {
                            "dependent_val": body_name,
                            "dependent_operation": operation_idx,
                            "value": body_value,
                            "in_value": "body",
                        }
                    )

        for (
            operation_idx,
            operation_responses,
        ) in qlearning.successful_responses.items():
            if operation_idx == operation_id:
                continue
            for response_name, response_values in operation_responses.items():
                for response_value in response_values:
                    possible_options.append(
                        {
                            "dependent_val": response_name,
                            "dependent_operation": operation_idx,
                            "value": response_value,
                            "in_value": "response",
                        }
                    )

        if not possible_options:
            return "RANDOM", {}, {}

        parameter_dependency_assignment = {}
        if qlearning.operation_graph.operation_nodes[
            operation_id
        ].operation_properties.parameters:
            for (
                parameter_name,
                parameter_properties,
            ) in qlearning.operation_graph.operation_nodes[
                operation_id
            ].operation_properties.parameters.items():
                if parameter_properties.schema:
                    parameter_dependency_assignment[parameter_name] = random.choice(
                        possible_options
                    )

        body_dependency_assignment = {}
        if qlearning.operation_graph.operation_nodes[
            operation_id
        ].operation_properties.request_body:
            for mime, body_properties in qlearning.operation_graph.operation_nodes[
                operation_id
            ].operation_properties.request_body.items():
                possible_body_params = get_body_params(body_properties)
                for prop in possible_body_params:
                    body_dependency_assignment[prop] = random.choice(possible_options)

        return "RANDOM", parameter_dependency_assignment, body_dependency_assignment

    def number_of_zeros(self, operation_id):
        if operation_id not in self.q_table:
            return 0
        zeros = 0
        for location, param_values in self.q_table[operation_id].items():
            for param, dependent_values in param_values.items():
                for dependent_op, dependent_props in dependent_values.items():
                    for dependent_location, dependent_params in dependent_props.items():
                        for dependent_param, value in dependent_params.items():
                            if value == 0:
                                zeros += 1
        return zeros
