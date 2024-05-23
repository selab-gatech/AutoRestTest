import random
import time
from collections import defaultdict

import requests

from src.graph.specification_parser import OperationProperties
from src.reinforcement.agents import OperationAgent, HeaderAgent, ParameterAgent, ValueAgent, ValueAction


class QLearning:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.3, time_duration=300):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.api_url = operation_graph.request_generator.api_url
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.operation_agent = OperationAgent(operation_graph, alpha, gamma, epsilon)
        self.header_agent = HeaderAgent(operation_graph, alpha, gamma, epsilon)
        self.parameter_agent = ParameterAgent(operation_graph, alpha, gamma, epsilon)
        self.value_agent = ValueAgent(operation_graph, alpha, gamma, epsilon)
        self.time_duration = time_duration
        self.responses = defaultdict(int)

    def initialize_agents(self):
        self.operation_agent.initialize_q_table()
        self.header_agent.initialize_q_table()
        self.parameter_agent.initialize_q_table()
        self.value_agent.initialize_q_table()

    def get_mapping(self, select_params, select_values):
        mapping = {}
        for i in range(len(select_params)):
            mapping[select_params[i]] = select_values[select_params[i]]
        return mapping

    def send_operation(self, operation_properties: OperationProperties, parameters, body, header):
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method.lower()
        if body:
            first_key = next(iter(body))
            first_value = body[first_key]
            body = {first_key: first_value}

        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                path_value = parameters[parameter_name]
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(path_value))

        try:
            select_method = getattr(requests, http_method)
            full_url = self.api_url + endpoint_path
            #print(f"Sending {http_method} request to {full_url}"
            #      f" with parameters: {parameters}, body: {body}, header: {header}")
            if body:
                for mime_type, body_content in body.items():
                    if not header:
                        header = {}
                    if mime_type == "application/json":
                        header["Content-Type"] = "application/json"
                        response = select_method(full_url, json=body_content, headers=header)
                    else: # mime_type == "application/x-www-form-urlencoded":
                        header["Content-Type"] = "application/x-www-form-urlencoded"
                        response = select_method(full_url, data=body_content, headers=header)
                    return response
            response = select_method(full_url, params=parameters, headers=header)
            return response
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")
            return None

    def determine_reward(self, response):
        if response.status_code // 200 == 2:
            return 2
        elif response.status_code // 500 == 1:
            return 5
        elif response.status_code // 400 == 1:
            return -1
        else:
            return -2

    def execute_operations(self):
        start_time = time.time()
        while time.time() - start_time < self.time_duration:
            operation_id = self.operation_agent.get_action()
            select_params = self.parameter_agent.get_action(operation_id)
            select_header = self.header_agent.get_action(operation_id)
            select_values = self.value_agent.get_action(operation_id)

            parameters = self.get_mapping(select_params.req_params, select_values.param_mappings) if select_params.req_params else None
            body = self.get_mapping(select_params.mime_type, select_values.body_mappings) if select_params.mime_type else None
            header = {"Authorization": select_header} if select_header else None

            response = self.send_operation(self.operation_graph.operation_nodes[operation_id].operation_properties, parameters, body, header)
            if response is not None:
                reward = self.determine_reward(response)

                impacted_operation = random.choice([self.parameter_agent, self.header_agent, self.value_agent])
                if impacted_operation == self.parameter_agent:
                    self.parameter_agent.update_q_table(operation_id, select_params, reward)
                elif impacted_operation == self.header_agent:
                    self.header_agent.update_q_table(operation_id, select_header, reward)
                else:
                    processed_value_action = ValueAction(param_mappings=parameters, body_mappings=body)
                    self.value_agent.update_q_table(operation_id, processed_value_action, reward)

                self.responses[response.status_code] += 1

                self.operation_agent.update_q_table(operation_id, reward)

            print(f"Responses: {self.responses}")


    def run(self):
        self.initialize_agents()
        self.execute_operations()
        print("COLLECTED RESPONSES: ", self.responses)
