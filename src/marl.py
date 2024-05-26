import json
import os
import random
import time
from collections import defaultdict

import requests
import shelve

from src.graph.specification_parser import OperationProperties
from src.reinforcement.agents import OperationAgent, HeaderAgent, ParameterAgent, ValueAgent, ValueAction
from src.utils import _construct_db_dir, construct_basic_token, get_nested_obj_mappings
from src.value_generator import identify_generator, randomize_string, random_generator, randomize_object


class QLearning:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.3, time_duration=600, mutation_rate=0.3):
        self.q_table = {}
        self.operation_graph = operation_graph
        self.api_url = operation_graph.request_generator.api_url
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
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

    def print_q_tables(self):
        print("OPERATION Q-TABLE: ", self.operation_agent.q_table)
        print("HEADER Q-TABLE: ", self.header_agent.q_table)
        print("PARAMETER Q-TABLE: ", self.parameter_agent.q_table)
        print("VALUE Q-TABLE: ", self.value_agent.q_table)

    def get_mapping(self, select_params, select_values):
        mapping = {}
        for i in range(len(select_params)):
            if select_params[i] in select_values:
                mapping[select_params[i]] = select_values[select_params[i]]
        return mapping

    def get_mutated_value(self, param_type):
        if not param_type:
            return None
        avail_types = ["integer", "number", "string", "boolean", "array", "object"]
        avail_types.remove(param_type)
        select_type = random.choice(avail_types)
        return identify_generator(select_type)()

    def mutate_values(self, operation_properties: OperationProperties, parameters, body, header):

        avail_medias = ["application/json", "application/x-www-form-urlencoded", "multipart/form-data", "text/plain"]
        avail_methods = ["get", "post", "put", "delete", "patch"]

        individual_mutation_rate = 0.4
        method_mutate_rate = 0.1
        media_mutate_rate = 0.1
        parameter_selection_mutate_rate = 0.3

        specific_method = None
        if operation_properties.http_method and random.random() < method_mutate_rate and operation_properties.http_method.lower() in avail_methods:
            avail_methods.remove(operation_properties.http_method.lower())
            specific_method = random.choice(avail_methods)

        if random.random() < individual_mutation_rate:
            random_token_params = {"username": randomize_string(), "password": randomize_string()}
            header = {"Authorization": "Basic " + construct_basic_token(random_token_params)}

        if random.random() < parameter_selection_mutate_rate:
            parameters = {randomize_string(): random_generator()() for _ in range(random.randint(1,5))}

        if operation_properties.parameters and parameters:
            for parameter_name, parameter_properties in operation_properties.parameters.items():
                if parameter_name in parameters:
                    if parameter_properties.schema and random.random() < individual_mutation_rate:
                        parameters[parameter_name] = self.get_mutated_value(parameter_properties.schema.type)
                    if parameters[parameter_name] is None:
                        parameters.pop(parameter_name, None)

        if operation_properties.request_body and body:
            for mime, body_properties in operation_properties.request_body.items():
                if mime in body and random.random() < individual_mutation_rate:
                    if random.random() < 0.5:
                        body[mime] = self.get_mutated_value(body_properties.type)
                    else:
                        body[mime] = randomize_object()
                if body[mime] is None:
                    body.pop(mime, None)

        if random.random() < media_mutate_rate and body:
            for media in body.keys():
                avail_medias.remove(media)
            if avail_medias and random.random() < individual_mutation_rate:
                new_body = {random.choice(avail_medias): body.popitem()[1]}
                body = new_body

        return parameters, body, header, specific_method

    def send_operation(self, operation_properties: OperationProperties, parameters, body, header, specific_method=None):
        endpoint_path = operation_properties.endpoint_path
        http_method = specific_method if specific_method else operation_properties.http_method.lower()

        if parameters:
            for parameter_name, parameter_properties in operation_properties.parameters.items():
                if parameter_properties.in_value == "path" and parameter_name in parameters:
                    path_value = parameters[parameter_name]
                    endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(path_value))

        try:
            select_method = getattr(requests, http_method)
            full_url = self.api_url + endpoint_path
            #print(f"Sending {http_method} request to {full_url}"
            #      f" with parameters: {parameters}, body: {body}, header: {header}")
            if body:
                if not header:
                    header = {}
                if "application/json" in body:
                    header["Content-Type"] = "application/json"
                    response = select_method(full_url, json=body["application/json"], headers=header)
                elif "application/x-www-form-urlencoded" in body: # mime_type == "application/x-www-form-urlencoded":
                    header["Content-Type"] = "application/x-www-form-urlencoded"
                    body_data = get_nested_obj_mappings(body["application/x-www-form-urlencoded"])
                    if not body_data or not isinstance(body_data, dict):
                        body_data = {"data": body["application/x-www-form-urlencoded"]}
                    body["application/x-www-form-urlencoded"] = body_data
                    response = select_method(full_url, data=body["application/x-www-form-urlencoded"], headers=header)
                elif "multipart/form-data" in body:
                    header["Content-Type"] = "multipart/form-data"
                    file = {"file": ("file.txt", json.dumps(body["multipart/form-data"]).encode('utf-8'), "application/json"),
                            "metadata": (None, "metadata")}
                    body["multipart/form-data"] = file
                    response = select_method(full_url, files=body["multipart/form-data"], headers=header)
                else: # mime_type == "text/plain":
                    header["Content-Type"] = "text/plain"
                    if not isinstance(body["text/plain"], str):
                        body["text/plain"] = str(body["text/plain"])
                    response = select_method(full_url, data=body["text/plain"], headers=header)
                return response
            response = select_method(full_url, params=parameters, headers=header)
            return response
        except requests.exceptions.RequestException as err:
            print(f"Error with operation {operation_properties.operation_id}: {err}")
            return None
        except Exception as err:
            print(f"Unexpected error with operation {operation_properties.operation_id}: {err}")
            print("Parameters: ", parameters)
            print("Body: ", body)
            return None

    def determine_header_reward(self, response):
        if response is None:
            return 0
        status_code = response.status_code
        if status_code == 401:
            return -2
        elif status_code // 100 == 4:
            return 0
        elif status_code // 100 == 5:
            return 1
        elif status_code // 100 == 2:
            return 2
        else:
            return -3

    def determine_good_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code // 100 == 2:
            return 2
        elif status_code // 100 == 4:
            return -2
        elif status_code // 100 == 5:
            return -1
        else:
            return -5

    def determine_bad_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code == 405:
            return -3
        elif status_code == 401:
            return -1
        if status_code // 100 == 4:
            return 1
        elif status_code // 100 == 5:
            return 1
        elif status_code // 100 == 2:
            return -1
        else:
            return 0

    def execute_operations(self):
        start_time = time.time()
        while time.time() - start_time < self.time_duration:
            print(f"Responses: {self.responses}")
            print("TIME REMAINING: ", self.time_duration - (time.time() - start_time))
            mutate_operation = random.random() < self.mutation_rate
            operation_id = self.operation_agent.get_action()
            select_params = self.parameter_agent.get_action(operation_id)
            select_header = self.header_agent.get_action(operation_id)
            select_values = self.value_agent.get_action(operation_id)

            parameters = self.get_mapping(select_params.req_params, select_values.param_mappings) if select_params.req_params else None
            body = self.get_mapping(select_params.mime_type, select_values.body_mappings) if select_params.mime_type else None
            header = {"Authorization": select_header} if select_header else None

            if mutate_operation:
                parameters, body, header, specific_method = self.mutate_values(self.operation_graph.operation_nodes[operation_id].operation_properties, parameters, body, header)
                response = self.send_operation(self.operation_graph.operation_nodes[operation_id].operation_properties, parameters, body, header, specific_method)
            else:
                response = self.send_operation(self.operation_graph.operation_nodes[operation_id].operation_properties, parameters, body, header)

            #reward = self.determine_reward(response)

            if not mutate_operation:
                #impacted_operation = random.choice([self.parameter_agent, self.header_agent, self.value_agent])
                #if impacted_operation == self.parameter_agent:
                #    self.parameter_agent.update_q_table(operation_id, select_params, reward)
                #elif impacted_operation == self.header_agent:
                #    self.header_agent.update_q_table(operation_id, select_header, reward)
                #else:

                self.parameter_agent.update_q_table(operation_id, select_params, self.determine_good_response_reward(response))
                self.header_agent.update_q_table(operation_id, select_header, self.determine_header_reward(response))
                processed_value_action = ValueAction(param_mappings=parameters, body_mappings=body)
                self.value_agent.update_q_table(operation_id, processed_value_action, self.determine_good_response_reward(response))
                #    self.value_agent.update_q_table(operation_id, processed_value_action, reward)
                self.operation_agent.update_q_table(operation_id, self.determine_bad_response_reward(response))

            if response is not None: self.responses[response.status_code] += 1

    def run(self):
        self.execute_operations()
        #self.print_q_tables()
        print("COLLECTED RESPONSES: ", self.responses)
