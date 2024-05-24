from collections import defaultdict
from dataclasses import dataclass, asdict, field
from json import JSONDecodeError
from typing import Dict, Set, Any, List, Optional, TYPE_CHECKING

import itertools

from src.graph.handle_response import ResponseHandler
from src.utils import remove_nulls, OpenAILanguageModel, get_params, get_request_body_params, get_nested_obj_mappings

from src.graph.specification_parser import OperationProperties, ParameterProperties, SchemaProperties
import requests
import pickle
import os

from .value_generator import NaiveValueGenerator, SmartValueGenerator

if TYPE_CHECKING:
    from .generate_graph import OperationGraph, OperationNode, OperationEdge

@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: Dict[str, Any] # dict of parameter name to value
    request_body: Dict[str, Any] # dict of mime type to request body
    operation_properties: OperationProperties
    requirements: 'RequestRequirements' = None

@dataclass
class RequestRequirements:
    edge: 'OperationEdge'
    parameter_requirements: Dict[str, Any] = field(default_factory=dict)
    request_body_requirements: Dict[str, Any] = field(default_factory=dict)

    def generate_combinations(self) -> List['RequestRequirements']:
        combinations = []

        param_combinations = [] if not self.parameter_requirements else []
        for i in range(1, len(self.parameter_requirements) + 1):
            for subset in itertools.combinations(self.parameter_requirements.keys(), i):
                param_combinations.append({key: self.parameter_requirements[key] for key in subset})

        body_combinations = []
        for i in range(1, len(self.request_body_requirements) + 1):
            for subset in itertools.combinations(self.request_body_requirements.keys(), i):
                body_combinations.append({key: self.request_body_requirements[key] for key in subset})

        if not param_combinations and not body_combinations:
            return [self]
        elif not param_combinations:
            for body_comb in body_combinations:
                combinations.append(RequestRequirements(
                    edge=self.edge,
                    parameter_requirements={},
                    request_body_requirements=body_comb
                ))
        elif not body_combinations:
            for param_comb in param_combinations:
                combinations.append(RequestRequirements(
                    edge=self.edge,
                    parameter_requirements=param_comb,
                    request_body_requirements={}
                ))
        else:
            for param_comb in param_combinations:
                for body_comb in body_combinations:
                    combinations.append(RequestRequirements(
                        edge=self.edge,
                        parameter_requirements=param_comb,
                        request_body_requirements=body_comb
                    ))

        combinations.sort(key=lambda x: len(x.parameter_requirements) + len(x.request_body_requirements), reverse=True)
        return combinations

@dataclass
class RequestResponse:
    request: 'RequestData'
    response: requests.Response
    response_text: str

@dataclass
class StatusCode:
    status_code: int
    count: int
    requests_and_responses: List[RequestResponse]

class RequestGenerator:
    def __init__(self, operation_graph: 'OperationGraph', api_url: str, is_naive=True):
        self.operation_graph: 'OperationGraph' = operation_graph
        self.api_url = api_url  
        self.status_codes: Dict[int: StatusCode] = {} # dictionary to track status code occurrences
        self.requests_generated = 0  # Initialize the request count
        self.successful_query_data = [] # List to store successful query data
        self.response_handler = ResponseHandler()
        self.is_naive = is_naive
        self.responses: Dict[str, RequestResponse] = {} # Dictionary to store responses for each operation_id
        self.allowed_retries = 1

    @staticmethod
    def generate_naive_values(parameters: Dict[str, ParameterProperties], request_body: Dict[str, SchemaProperties]):
        value_generator = NaiveValueGenerator(parameters=parameters, request_body=request_body)
        return value_generator.generate_parameters() if parameters else None, value_generator.generate_request_body() if request_body else None

    @staticmethod
    def generate_smart_values(operation_properties: OperationProperties, requirements: RequestRequirements = None):
        """
        Generate smart values for parameters and request body using LLMs
        :param operation_properties: Dictionary mapping of operation properties
        :param requirements: RequestRequirements object that contains any parameters or request body requirements
        :return: a tuple of the generated parameters and request body
        """
        value_generator = SmartValueGenerator(operation_properties=operation_properties, requirements=requirements)
        return value_generator.generate_parameters(), value_generator.generate_request_body()

    def process_response(self, request_response: RequestResponse, operation_node: 'OperationNode'):
        """
        Process the response from the API.
        """
        if request_response is None:
            return
        request_data = request_response.request
        response = request_response.response

        self.requests_generated += 1
        if response.status_code not in self.status_codes:
            self.status_codes[response.status_code] = StatusCode(
                status_code=response.status_code,
                count=1,
                requests_and_responses=[request_response],
            )
        else:
            self.status_codes[response.status_code].count += 1
            self.status_codes[response.status_code].requests_and_responses.append(request_response)
        if response.ok:
            self.successful_query_data.append(request_data)
        else:  # For non-2xx responses
            self.response_handler.handle_error(response, operation_node, request_data, self)
        
        #print(f"Request {request_data.operation_properties.operation_id} completed with response text {response.text} and status code {response.status_code}")

    def make_request_data(self, operation_properties: OperationProperties, requirements: RequestRequirements = None) -> RequestData:
        '''
        Process the operation properties by preparing request data for queries with mapping values to parameters and request body
        '''
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method.lower()

        if self.is_naive:
            parameters, request_body = self.generate_naive_values(
                operation_properties.parameters, operation_properties.request_body
            )
        else:
            parameters, request_body = self.generate_smart_values(
                operation_properties=operation_properties,
                requirements=requirements
            )

        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                path_value = parameters[parameter_name]
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(path_value))

        return RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=parameters,
            request_body=request_body,
            operation_properties=operation_properties,
            requirements=requirements
        )

    def make_request_retry_data(self, request_data: RequestData, response: requests.Response) -> RequestData:
        if self.is_naive:
            return request_data
        else:
            requirements = request_data.requirements
            value_generator = SmartValueGenerator(operation_properties=request_data.operation_properties, requirements=requirements, engine="gpt-4o")
            parameters, request_body = value_generator.generate_retry_parameters(request_data, response), value_generator.generate_retry_request_body(request_data, response)
            return RequestData(
                endpoint_path=request_data.endpoint_path,
                http_method=request_data.http_method,
                parameters=parameters,
                request_body=request_body,
                operation_properties=request_data.operation_properties,
                requirements=requirements
            )

    def _handle_retry(self, request_data: RequestData, response: requests.Response, curr_retry: int) -> Optional[RequestResponse]:
        retry_data = self.make_request_retry_data(request_data, response)
        response = self.send_operation_request(retry_data, retry_nums=curr_retry + 1, allow_retry=False)
        return response

    def _determine_auth_mappings(self, request_val: Any, auth_parameters: Dict, auth_mappings):
        if type(request_val) == dict:
            for item, properties in request_val.items():
                if auth_parameters.get("username") == item:
                    auth_mappings["username"] = item
                elif auth_parameters.get("password") == item:
                    auth_mappings["password"] = item
                else:
                    self._determine_auth_mappings(properties, auth_parameters, auth_mappings)
        elif type(request_val) == list:
            for item in request_val:
                self._determine_auth_mappings(item, auth_parameters, auth_mappings)

    def _determine_if_valid_auth(self, param_auth):
        if not param_auth or type(param_auth) != dict:
            return False
        return param_auth.get("username") and param_auth.get("username") != "None" and param_auth.get("password") and param_auth.get("password") != "None"

    def get_auth_info(self, operation_node: 'OperationNode', auth_attempts: int = 3):
        operation_properties = operation_node.operation_properties
        value_generator = SmartValueGenerator(operation_properties=operation_properties, engine="gpt-3.5-turbo-0125")

        auth_parameters = value_generator.determine_auth_params()
        if not auth_parameters:
            return []
        query_param_auth = auth_parameters.get("query_parameters")
        body_param_auth = auth_parameters.get("body_parameters")

        token_info = []
        if query_param_auth and self._determine_if_valid_auth(query_param_auth):
            print("Query Param Auth: ", query_param_auth)
            for i in range(auth_attempts):
                self.find_query_auth_mappings(operation_node, query_param_auth, token_info)
        if body_param_auth and self._determine_if_valid_auth(body_param_auth):
            print("Body Param Auth: ", body_param_auth)
            for i in range(auth_attempts):
                self.find_query_auth_mappings(operation_node, body_param_auth, token_info)

        return token_info

    def find_query_auth_mappings(self, operation_node, query_param_auth, token_info):
        response = self.create_and_send_request(operation_node, allow_retry=True)
        if response and response.response.ok:
            auth_mappings = {}
            self._determine_auth_mappings(response.request.parameters, query_param_auth, auth_mappings)
            if len(auth_mappings) == 2: token_info.append(auth_mappings) # check for both username and password

    def send_operation_request(self, request_data: RequestData, retry_nums: int = 0, allow_retry: bool=False) -> Optional[RequestResponse]:
        """
        Send the operation request to the API
        :param request_data:
        :param retry_nums:
        :param allow_retry:
        :return:
        """
        endpoint_path = request_data.endpoint_path
        http_method = request_data.http_method
        parameters = request_data.parameters
        request_body = request_data.request_body
        try:
            select_method = getattr(requests, http_method) # selects correct http method
            full_url = f"{self.api_url}{endpoint_path}"
            # TODO: Determine if should do a request_body conditional check instead of http_method hardcoded
            if request_body:
                response = self._send_mime_type(select_method, full_url, parameters, request_body)
            else:
                response = select_method(full_url, params=parameters)
            if response is not None:
                if not response.ok and retry_nums < self.allowed_retries and allow_retry:
                    print("Attempting retry due to failed response...")
                    return self._handle_retry(request_data, response, retry_nums)
                else:
                    return RequestResponse(
                        request=request_data,
                        response=response,
                        response_text=response.text
                    )
            return None
        except requests.exceptions.RequestException as err:
            print(f"Request exception due to error: {err}")
            print(f"Endpoint Path: {endpoint_path}")
            print(f"Params: {parameters}")
            print(f"Request Body: {request_body}")
            return None
        #except Exception as err:
        #    print(f"Unexpected error due to: {err}")
        #    print(f"Endpoint Path: {endpoint_path}")
        #    print(f"Params: {parameters}")
        #    print(f"Request Body: {request_body}")
        #    return None

    def _send_mime_type(self, select_method, full_url, parameters, request_body):
        params = parameters if parameters else {}
        if "application/json" in request_body:
            req_body = request_body["application/json"]
            response = select_method(full_url, params=params, json=req_body)
        elif "application/x-www-form-urlencoded" in request_body:
            req_body = request_body["application/x-www-form-urlencoded"]
            response = select_method(full_url, params=params, data=req_body)
        elif "multipart/form-data" in request_body:
            req_body = request_body["multipart/form-data"]
            response = select_method(full_url, params=params, files=req_body)
        elif "text/plain" in request_body:
            req_body = request_body["text/plain"]
            headers = {"Content-Type": "text/plain"}
            response = select_method(full_url, params=params, headers=headers, data=req_body)
        else:
            # should not reach here
            raise ValueError("Request body mime type not supported")
        return response

    def determine_requirement(self, dependent_response: RequestResponse, edge: 'OperationEdge') -> Optional[RequestRequirements]:
        """
        Determine the requirement values and mappings for the given request using the responses of its dependent requests
        :param dependent_response: Dependent response object
        :param edge: Contains the operation edge information to dictate what is required
        :return: A RequestRequirements object containing the requirements
        """
        if dependent_response is None or not dependent_response.response.ok or dependent_response.response is None:
            return None
        try:
            response_mappings = get_nested_obj_mappings(dependent_response.response.json())
        except JSONDecodeError as err:
            print("FAILED TO PARSE JSON RESPONSE ", dependent_response.response.text)
            response_mappings = {}

        if not response_mappings:
            return None
        request_body_matchings = {}
        parameter_matchings = {}

        for parameter, similarity_value in edge.similar_parameters.items():
            if similarity_value.response_val in response_mappings:
                if similarity_value.in_value == "query":
                    parameter_matchings[parameter] = response_mappings[similarity_value.response_val]
                elif similarity_value.in_value == "request body":
                    request_body_matchings[parameter] = response_mappings[similarity_value.response_val]

        request_requirement = RequestRequirements(
            edge=edge,
            parameter_requirements=parameter_matchings,
            request_body_requirements=request_body_matchings
        )
        return request_requirement if parameter_matchings or request_body_matchings else None

    @staticmethod
    def _determine_best_response(responses: List[RequestResponse]) -> Optional[RequestResponse]:
        if not responses:
            return None
        for response in responses:
            if response and response.response and response.response.ok:
                return response
        return responses[0]

    def depth_traversal(self, curr_node: 'OperationNode', visited: Set):
        """
        Generate low-level requests (with no dependencies and hence high depth) first
        :param curr_node: Current operation node
        :param visited: Set of visited operation nodes
        :return: RequestResponse object to allow for requirements parsing
        """
        visited.add(curr_node.operation_id)
        dependent_responses: Dict['OperationEdge', RequestResponse] = {}
        print("Building dependencies for operation: ", curr_node.operation_id)
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.depth_traversal(edge.destination, visited)
            dependent_response = self.responses[edge.destination.operation_id] if edge.destination.operation_id in self.responses else None
            if dependent_response is not None and dependent_response.response.ok:
                dependent_responses[edge] = dependent_response
        respones = self.handle_request_and_dependencies(curr_node, dependent_responses)
        best_response = self._determine_best_response(respones)
        self.responses[curr_node.operation_id] = best_response
        print("Completed building dependencies for operation: ", curr_node.operation_id)

    def _validate_value_mappings(self, curr_node: 'OperationNode', parameter_mappings: Dict, req_param_mappings: Dict[str, List[Any]], req_body_mappings: Dict[str, List[Any]], occurrences: Dict):
        operation_id = curr_node.operation_id
        if operation_id not in parameter_mappings:
            parameter_mappings[operation_id] = {'params': {}, 'body': {}}

        operation_params = get_params(curr_node.operation_properties.parameters)

        #operation_body_params = get_request_body_params(curr_node.operation_properties.request_body)

        # take the reverse of mappings since API outputs least likely values first
        req_param_mappings = {k: v for k, v in reversed(list(req_param_mappings.items()))} if req_param_mappings else {}
        req_body_mappings = {k: v for k, v in reversed(list(req_body_mappings.items()))} if req_body_mappings else {}

        if req_param_mappings:
            for parameter, values in req_param_mappings.items():
                for value in values:
                    if parameter not in parameter_mappings[operation_id]['params']:
                        parameter_mappings[operation_id]['params'][parameter] = []
                    if parameter in operation_params and len(parameter_mappings[operation_id]['params'][parameter]) < 10:
                        parameter_mappings[operation_id]['params'][parameter].append([value, 0])
                        occurrences[parameter] = occurrences.get(parameter, 0) + 1

        if req_body_mappings:
            for mime, mime_params in req_body_mappings.items():
                if mime not in parameter_mappings[operation_id]['body']:
                    parameter_mappings[operation_id]['body'][mime] = []
                for mime_param in mime_params:
                    num_bodies = len(parameter_mappings[operation_id]['body'][mime])
                    if num_bodies < 10:
                        parameter_mappings[operation_id]['body'][mime].append([mime_param, 0])
                        occurrences[mime] = occurrences.get(mime, 0) + 1
                #body_params = get_nested_obj_mappings(mime_params) # handler works for getting first obj values same as get_request_body_params
                #for body_param, value in body_params.items():
                #    if body_param not in parameter_mappings[operation_id]['body'][mime]:
                #        parameter_mappings[operation_id]['body'][mime][body_param] = {}
                #    if body_param in operation_body_params[mime] and len(parameter_mappings[operation_id]['body'][mime][body_param]) < 10:
                #        parameter_mappings[operation_id]['body'][mime][body_param][value] = 0
                #        occurrences[body_param] = occurrences.get(body_param, 0) + 1

    def value_depth_traversal(self, curr_node: 'OperationNode', parameter_mappings: Dict, responses: Dict[str, List], visited: Set):
        visited.add(curr_node.operation_id)
        dependent_responses: Dict['OperationEdge', List[RequestResponse]] = defaultdict(list)

        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.value_depth_traversal(edge.destination, parameter_mappings, responses, visited)
            possible_dependent_responses = responses[edge.destination.operation_id]
            for dependent_response in possible_dependent_responses:
                if dependent_response and dependent_response.response and dependent_response.response.ok:
                    dependent_responses[edge].append(dependent_response)

        occurrences = {}
        self.handle_dependent_values(curr_node, dependent_responses, occurrences, parameter_mappings, responses)

        desired_size = 10
        lowest_occurrences = min(occurrences.values()) if occurrences else 0
        if lowest_occurrences < desired_size:
            value_generator = SmartValueGenerator(operation_properties=curr_node.operation_properties, temperature=0.7)
            parameters, request_body = value_generator.generate_value_agent_params(num_values=desired_size - lowest_occurrences), value_generator.generate_value_agent_body(num_values=desired_size - lowest_occurrences)
            self._validate_value_mappings(curr_node, parameter_mappings, parameters, request_body, occurrences)

        if not responses[curr_node.operation_id]:
            response = self.create_and_send_request(curr_node, allow_retry=True)
            if response and response.response and response.response.ok:
                responses[curr_node.operation_id].append(response)

        print("Completed value table generation for operation: ", curr_node.operation_id)

    def _embed_obj_val_list(self, obj: Optional[Dict[Any, List]]):
        if not obj:
            return {}
        for key, value in obj.items():
            obj[key] = [value]
        return obj

    def handle_dependent_values(self, curr_node, dependent_responses, occurrences, parameter_mappings, responses):
        if not dependent_responses:
            return
        for edge, response_returns in dependent_responses.items():
            for dependent_response in response_returns:
                if occurrences and max(occurrences.values()) == 7:  # limit number of dependency created values
                    break
                requirements = self.determine_requirement(dependent_response, edge)
                #response = self.create_and_send_request(curr_node, requirements, allow_retry=True)
                self._validate_value_mappings(curr_node, parameter_mappings, self._embed_obj_val_list(requirements.parameter_requirements),
                                              self._embed_obj_val_list(requirements.request_body_requirements), occurrences)
                #if response and response.response and response.response.ok:
                #    responses[curr_node.operation_id].append(response)

    def handle_request_and_dependencies(self, curr_node: 'OperationNode', dependent_responses: Dict['OperationEdge', RequestResponse] = None) -> Optional[List[RequestResponse]]:
        if not dependent_responses:
            return [self._send_req_handle_respo(curr_node)]
        else:
            responses: List[RequestResponse] = []
            for edge, dependent_response in dependent_responses.items():
                requirement: RequestRequirements = self.determine_requirement(dependent_response, edge)
                req_combos: List[RequestRequirements] = requirement.generate_combinations() if requirement else []
                for requirement in req_combos:
                    response = self._send_req_handle_respo(curr_node, requirement)
                    if response and response.response and response.response.ok:
                        responses.append(response)
                        self.handle_edge_adjustment(edge, requirement)
                        return
                self.handle_edge_adjustment(edge)
            return responses

    def _send_req_handle_respo(self, curr_node, requirement=None) -> RequestResponse:
        # currently only used during initial depth traversal
        response = self.create_and_send_request(curr_node, requirement, allow_retry=True)
        if response is not None:
            self.process_response(response, curr_node)
        return response

    def handle_edge_adjustment(self, edge: 'OperationEdge', requirement: RequestRequirements=None):
        if not requirement:
            self.operation_graph.remove_edge(edge.source.operation_id, edge.destination.operation_id)
            return

        adjusted_similar_parameters = {
            parameter: similarity_value for parameter, similarity_value in edge.similar_parameters.items()
            if parameter in requirement.parameter_requirements or parameter in requirement.request_body_requirements
        }

        if adjusted_similar_parameters:
            edge.similar_parameters = adjusted_similar_parameters
        else:
            self.operation_graph.remove_edge(edge.source.operation_id, edge.destination.operation_id)

    def handle_tentative_dependency(self, tentative_edge, failed_operation_node) -> bool:
        dependent_response = self.create_and_send_request(tentative_edge.destination, allow_retry=True)
        if dependent_response and dependent_response.response and dependent_response.response.ok:
            requirement = self.determine_requirement(dependent_response, tentative_edge)
            response = self.create_and_send_request(failed_operation_node, requirement)
            if response and response.response and response.response.ok:
                self.operation_graph.add_operation_edge(tentative_edge.source.operation_id, tentative_edge.destination.operation_id, tentative_edge.similar_parameters)
                self.operation_graph.remove_tentative_edge(tentative_edge.source.operation_id, tentative_edge.destination.operation_id)
                return True
            elif response and response.response:
                self.operation_graph.remove_tentative_edge(tentative_edge.source.operation_id, tentative_edge.destination.operation_id)
        return False

    def create_and_send_request(self, curr_node: 'OperationNode', requirement: RequestRequirements=None, allow_retry:bool=False) -> RequestResponse:
        request_data: RequestData = self.make_request_data(curr_node.operation_properties, requirement)
        response: RequestResponse = self.send_operation_request(request_data, allow_retry=allow_retry)
        return response

    def perform_all_requests(self):
        '''
        Generate requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.depth_traversal(operation_node, visited)

if __name__ == "__main__":
    pass