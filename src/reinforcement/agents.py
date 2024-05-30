from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import random

from src.generate_graph import OperationGraph
from src.utils import get_combinations, get_param_combinations, \
    construct_basic_token, get_required_params, get_required_body_params


class OperationAgent:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.operation_graph = operation_graph
        self.q_table = {}
        # Note: State is 1 dimensional (constant) and action is operation selection

    def initialize_q_table(self):
        print("Initiating Operation Agent Q-Table")
        operation_ids = self.operation_graph.operation_nodes.keys()
        self.q_table = {operation_id: 0 for operation_id in operation_ids}
        print("Initiated Operation Agent Q-Table")

    def get_action(self):
        """
        Get the next action based on the Q-table
        :return: operation_id for next action
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action()
        return self.get_best_action()

    def get_best_action(self):
        """
        Get the best action based on the Q-table where errors are given a higher score
        :return: operation_id for next action
        """
        best_action = max(self.q_table.items(), key=lambda x: x[1])[0]
        return best_action

    def get_random_action(self):
        return random.choice(list(self.q_table.keys()))

    def update_q_table(self, operation_id, reward):
        """
        Update the Q-table based on the reward received
        :param operation_id: operation_id of the action taken
        :param reward: reward received for the action
        """
        current_q = self.q_table[operation_id]
        best_next_q = max(self.q_table.values())
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id] = new_q
        # TODO: Check if we want to simplify the Q-table update equation since the state is constant

class HeaderAgent:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph: OperationGraph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        request_generator = self.operation_graph.request_generator
        token_list = []
        print("Initiating Header Agent Q-Table")
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            token_info = request_generator.get_auth_info(operation_node, 5)
            for token in token_info:
                token_list.append(construct_basic_token(token))
        print("Token list has been constructed with the following number of tokens: ", len(token_list))
        for operation_id in self.operation_graph.operation_nodes.keys():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = []
            random.shuffle(token_list)
            for i in range(min(9, len(token_list))):
                self.q_table[operation_id].append([token_list[i],0])
            self.q_table[operation_id].append([None,0])
        print(self.q_table)
        print("Initiated Header Agent Q-Table")

    def get_action(self, operation_id):
        """
        Get the next action based on the Q-table
        :return: operation_id for next action
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id):
        """
        Get the best action based on the Q-table where errors are given a higher score
        :return: operation_id for next action
        """
        best_action = max(self.q_table[operation_id], key=lambda x: x[1])[0] if self.q_table[operation_id] else None
        return best_action

    def get_random_action(self, operation_id):
        return random.choice(self.q_table[operation_id])[0] if self.q_table[operation_id] else None

    def update_q_table(self, operation_id, action, reward):
        """
        Update the Q-table based on the reward received
        :param operation_id: operation_id of the action taken
        :param action: action taken
        :param reward: reward received for the action
        """
        current_q = 0
        best_next_q = -np.inf
        for mapping in self.q_table[operation_id]:
            best_next_q = max(best_next_q, mapping[1])
            if mapping[0] == action:
                current_q = mapping[1]
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        for mapping in self.q_table[operation_id]:
            if mapping[0] == action:
                mapping[1] = new_q

@dataclass
class ParameterAction:
    req_params: List
    mime_type: str

class ParameterAgent:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph: OperationGraph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        print("Initiating Parameter Agent Q-Table")
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {'params': {}, 'body': {}}
            params = get_param_combinations(operation_node.operation_properties.parameters)
            self.q_table[operation_id]['params'] = {param: 0 for param in params}
            #mimes = get_combinations(list(operation_node.operation_properties.request_body.keys())) if operation_node.operation_properties.request_body else []
            mimes = list(operation_node.operation_properties.request_body.keys()) if operation_node.operation_properties.request_body else []
            self.q_table[operation_id]['body'] = {mime: 0 for mime in mimes}

            self.q_table[operation_id]['params']["None"] = 0
            self.q_table[operation_id]['body']["None"] = 0
            # NOTE: Moved function of request body parameter agent to select combinations of mime types
        print("Initiated Parameter Agent Q-Table")

    def get_action(self, operation_id) -> ParameterAction:
        """
        Get the next action based on the Q-table
        :param operation_id:
        :return:
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_random_action(self, operation_id) -> ParameterAction:
        """
        Get a random action based on the Q-table
        :param operation_id:
        :return:
        """
        random_params = random.choice(list(self.q_table[operation_id]['params'].keys())) if self.q_table[operation_id]['params'] else None
        random_mime = random.choice(list(self.q_table[operation_id]['body'].keys())) if self.q_table[operation_id]['body'] else None
        if random_params == "None":
            random_params = None
        if random_mime == "None":
            random_mime = None
        return ParameterAction(req_params=random_params, mime_type=random_mime)

    def get_best_action(self, operation_id) -> ParameterAction:
        """
        Get the best param + body combination based on the Q-table. Note that we can return None for body or params
        depending on whether they are assigned for the operation
        :param operation_id:
        :return:
        """
        required_params = get_required_params(self.operation_graph.operation_nodes[operation_id].operation_properties.parameters)
        best_params = (None, -np.inf)
        if self.q_table[operation_id]['params'] and required_params:
            for params, score in self.q_table[operation_id]['params'].items():
                if params and required_params.issubset(set(params)) and score > best_params[1]:
                    best_params = (params, score)
            best_params = best_params[0]
        else:
            best_params = max(self.q_table[operation_id]['params'].items(), key=lambda x: x[1])[0] if self.q_table[operation_id]['params'] else None

        best_body = (None, -np.inf)
        if self.q_table[operation_id]['body']:
            for mime, score in self.q_table[operation_id]['body'].items():
                if mime != "None" and score > best_body[1]:
                    best_body = (mime, score)
        best_body = best_body[0]

        if best_params == "None":
            best_params = None
        if best_body == "None":
            best_body = None

        return ParameterAction(req_params=best_params, mime_type=best_body)

        #best_body: Tuple = max(((mime, param, value) for mime, mime_params in self.q_table[operation_id]['body'].items()
        #                        for param, value in mime_params.items()), key=lambda x: x[2]) if self.q_table[operation_id]['body'] else (None, None, 0)
        #return ParameterAction(req_params=best_params[0], req_body=best_body[1], mime_type=best_body[0])

    def update_q_table(self, operation_id, action, reward):
        """
        Update the Q-table based on the reward received
        Note that we use combined Q-values for the parameter + body combination in the case of requiring both
        :param operation_id:
        :param action:
        :param reward:
        :return:
        """
        if action.req_params is None:
            action.req_params = "None"
        if action.mime_type is None:
            action.mime_type = "None"

        current_q_params = self.q_table[operation_id]['params'][action.req_params] if self.q_table[operation_id]['params'] else 0
        current_q_body = self.q_table[operation_id]['body'][action.mime_type] if self.q_table[operation_id]['body'] else 0
        # Note: State is constant "active", so next state = current state
        best_next_q_params = max(self.q_table[operation_id]['params'].values()) if self.q_table[operation_id]['params'] else 0
        best_next_q_body = max(self.q_table[operation_id]['body'].values()) if self.q_table[operation_id]['body'] else 0

        new_q_params = current_q_params + self.alpha * (reward + self.gamma * best_next_q_params - current_q_params)
        new_q_body = current_q_body + self.alpha * (reward + self.gamma * best_next_q_body - current_q_body)

        if self.q_table[operation_id]['params']:
            self.q_table[operation_id]['params'][action.req_params] = new_q_params
        if self.q_table[operation_id]['body']:
            self.q_table[operation_id]['body'][action.mime_type] = new_q_body

@dataclass
class ValueAction:
    param_mappings: Dict[str, Any]
    body_mappings: Dict[str, Any] # maps mime_type to body values

class ValueAgent:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        responses = defaultdict(list)
        visited = set()
        request_generator = self.operation_graph.request_generator
        print("Initiating Value Agent Q-Table")
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                request_generator.value_depth_traversal(operation_node, self.q_table, responses, visited)
        print("Initiated Value Agent Q-Table")

    def get_action(self, operation_id):
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id):
        """
        Get the best action based on the Q-table where errors are given a higher score
        :return: operation_id for next action
        """
        param_mappings = {
            param: max(param_mappings, key=lambda pm: pm[1])[0]
            for param, param_mappings in self.q_table[operation_id]['params'].items()
        } if self.q_table[operation_id]['params'] else None

        body_mappings = {
            mime: max(body_mappings, key=lambda bm: bm[1])[0]
            for mime, body_mappings in self.q_table[operation_id]['body'].items()
        } if self.q_table[operation_id]['body'] else None

        return ValueAction(param_mappings, body_mappings)

    def get_random_action(self, operation_id):
        param_mappings = {
            param: random.choice(param_mappings)[0]
            for param, param_mappings in self.q_table[operation_id]['params'].items()
        } if self.q_table[operation_id]['params'] else None

        body_mappings = {
            mime: random.choice(body_mappings)[0]
            for mime, body_mappings in self.q_table[operation_id]['body'].items()
        } if self.q_table[operation_id]['body'] else None

        return ValueAction(param_mappings, body_mappings)

    def update_q_table(self, operation_id, filtered_action, reward):
        """
        Update the Q-table based on the reward received
        :param operation_id: operation_id of the action taken
        :param filtered_action: action taken considering only the used parameters
        :param reward: reward received for the action
        """

        if filtered_action.param_mappings:
            for param, value in filtered_action.param_mappings.items():
                current_q = 0
                best_next_q = -np.inf
                for mapping in self.q_table[operation_id]['params'][param]:
                    best_next_q = max(best_next_q, mapping[1])
                    if mapping[0] == value: current_q = mapping[1]
                new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
                for mapping in self.q_table[operation_id]['params'][param]:
                    if mapping[0] == value:
                        mapping[1] = new_q

        if filtered_action.body_mappings:
            for mime, body in filtered_action.body_mappings.items():
                current_q = 0
                best_next_q = -np.inf
                for mapping in self.q_table[operation_id]['body'][mime]:
                    best_next_q = max(best_next_q, mapping[1])
                    if mapping[0] == body: current_q = mapping[1]
                new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
                for mapping in self.q_table[operation_id]['body'][mime]:
                    if mapping[0] == body:
                        mapping[1] = new_q


class BodyObjAgent:
    def __init__(self, operation_graph: OperationGraph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {}
            if operation_node.operation_properties.request_body:
                for mime, body_properties in operation_node.operation_properties.request_body.items():
                    if body_properties.type == "object":
                        # TODO: Check if this requires a change to account for object list containers
                        body_obj_combinations = get_combinations(body_properties.properties.keys())
                        self.q_table[operation_id][mime] = {body_obj: 0 for body_obj in body_obj_combinations}
                        self.q_table[operation_id][mime]["None"] = 0

    def get_action(self, operation_id, mime):
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id, mime)
        return self.get_best_action(operation_id, mime)

    def get_best_action(self, operation_id, mime):
        required_obj_params = get_required_body_params(
            self.operation_graph.operation_nodes[operation_id].operation_properties.request_body[mime])
        best_action = (None, -np.inf)
        if required_obj_params:
            for body_obj, value in self.q_table[operation_id][mime].items():
                if value > best_action[1] and required_obj_params.issubset(set(body_obj)):
                    best_action = (body_obj, value)
        else:
            best_action = max(self.q_table[operation_id][mime].items(), key=lambda x: x[1])[0] if self.q_table[operation_id][mime] else None
        if best_action == "None":
            best_action = None
        return best_action

    def get_random_action(self, operation_id, mime):
        action = random.choice(list(self.q_table[operation_id][mime].keys())) if self.q_table[operation_id][mime] else None
        if action == "None":
            action = None
        return action

    def update_q_table(self, operation_id, mime, action, reward):
        if action is None:
            action = "None"
        current_q = self.q_table[operation_id][mime][action] if self.q_table[operation_id][mime] else 0
        best_next_q = max(self.q_table[operation_id][mime].values()) if self.q_table[operation_id][mime] else 0
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id][mime][action] = new_q

class DataSourceAgent:
    def __init__(self, operation_graph: OperationGraph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {"LLM": 0, "DEPENDENCY": 0, "DEFAULT": 0}

    def get_action(self, operation_id):
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id):
        best_action = max(self.q_table[operation_id].items(), key=lambda x: x[1])[0]
        return best_action

    def get_random_action(self, operation_id):
        return random.choice(list(self.q_table[operation_id].keys()))

    def update_q_table(self, operation_id, action, reward):
        current_q = self.q_table[operation_id][action]
        best_next_q = max(self.q_table[operation_id].values())
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id][action] = new_q

class DependencyAgent:
    def __init__(self, operation_graph: OperationGraph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self):
        # NOTE: Could flip dependent operation id and curr parameter name around to group by operation if need-be
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {"params": {}, "body": {}}

            for edge in operation_node.outgoing_edges:
                for parameter, similarity_value in edge.similar_parameters.items():
                    processed_in_val = similarity_value.in_value.split(" to ")
                    dependent_parameter = similarity_value.dependent_val
                    destination = edge.destination.operation_id

                    if processed_in_val[0] == "query" and parameter not in self.q_table[operation_id]['params']:
                        self.q_table[operation_id]['params'][parameter] = {}
                    elif processed_in_val[0] == "body" and parameter not in self.q_table[operation_id]['body']:
                        self.q_table[operation_id]['body'][parameter] = {}

                    if processed_in_val[0] == "query" and destination not in self.q_table[operation_id]['params'][parameter]:
                        self.q_table[operation_id]['params'][parameter][destination] = {"params": {}, "body": {}, "response": {}}
                    elif processed_in_val[0] == "body" and destination not in self.q_table[operation_id]['body'][parameter]:
                        self.q_table[operation_id]['body'][parameter][destination] = {"params": {}, "body": {}, "response": {}}

                    if processed_in_val[0] == "query":
                        if processed_in_val[1] == "query":
                            self.q_table[operation_id]['params'][parameter][destination]["params"][dependent_parameter] = 0
                        elif processed_in_val[1] == "body":
                            self.q_table[operation_id]['params'][parameter][destination]["body"][dependent_parameter] = 0
                        elif processed_in_val[1] == "response":
                            self.q_table[operation_id]['params'][parameter][destination]["response"][dependent_parameter] = 0

                    elif processed_in_val[0] == "body":
                        if processed_in_val[1] == "query":
                            self.q_table[operation_id]['body'][parameter][destination]["params"][dependent_parameter] = 0
                        elif processed_in_val[1] == "body":
                            self.q_table[operation_id]['body'][parameter][destination]["body"][dependent_parameter] = 0
                        elif processed_in_val[1] == "response":
                            self.q_table[operation_id]['body'][parameter][destination]["response"][dependent_parameter] = 0

    def get_action(self, operation_id) -> Tuple[Dict, Dict]:
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_best_action(self, operation_id):
        best_params = {}
        for param, dependent_ops in self.q_table[operation_id]['params'].items():
            best_dependent = {"dependent_val": None, "dependent_operation": None, "value": -np.inf, "in_value": None}
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if value > best_dependent["value"]:
                            best_dependent = {"dependent_val": dependent_param, "dependent_operation": dependent_op, "value": value, "in_value": location}
            best_params[param] = best_dependent

        best_body = {}
        for param, dependent_ops in self.q_table[operation_id]['body'].items():
            best_dependent = {"dependent_val": None, "dependent_operation": None, "value": -np.inf, "in_value": None}
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        if value > best_dependent["value"]:
                            best_dependent = {"dependent_val": dependent_param, "dependent_operation": dependent_op, "value": value, "in_value": location}
            best_body[param] = best_dependent

        return best_params, best_body

    def get_random_action(self, operation_id):
        random_params = {}
        for param, dependent_ops in self.q_table[operation_id]['params'].items():
            random_dependencies = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        random_dependencies.append({"dependent_val": dependent_param, "dependent_operation": dependent_op, "value": value, "in_value": location})
            random_params[param] = random.choice(random_dependencies)

        random_body = {}
        for param, dependent_ops in self.q_table[operation_id]['body'].items():
            random_dependencies = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, dependent_params in value_dict.items():
                    for dependent_param, value in dependent_params.items():
                        random_dependencies.append({"dependent_val": dependent_param, "dependent_operation": dependent_op, "value": value, "in_value": location})
            random_body[param] = random.choice(random_dependencies)

        return random_params, random_body

    def update_q_table(self, operation_id, dependent_params, dependent_body, reward):
        if dependent_params:
            for param, dependent in dependent_params.items():
                current_q = 0
                best_next_q = -np.inf
                for location, dependent_params in self.q_table[operation_id]['params'][param][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                        best_next_q = max(best_next_q, value)
                new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
                for location, dependent_params in self.q_table[operation_id]['params'][param][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]['params'][param][dependent["dependent_operation"]][location][dependent_param] = new_q

        if dependent_body:
            for param, dependent in dependent_body.items():
                current_q = 0
                best_next_q = -np.inf
                for location, dependent_params in self.q_table[operation_id]['body'][param][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            current_q = value
                        best_next_q = max(best_next_q, value)
                new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
                for location, dependent_params in self.q_table[operation_id]['body'][param][dependent["dependent_operation"]].items():
                    for dependent_param, value in dependent_params.items():
                        if dependent_param == dependent["dependent_val"]:
                            self.q_table[operation_id]['body'][param][dependent["dependent_operation"]][location][dependent_param] = new_q










