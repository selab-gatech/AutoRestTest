import json
from dataclasses import dataclass, asdict
from typing import Dict, Set, Any, List, Optional, TYPE_CHECKING

from requests import Response

from src.handle_response import ResponseHandler
from src.utils import remove_nulls

from .specification_parser import OperationProperties, ParameterProperties, SchemaProperties
import requests
import pickle
import os 

from .value_generator import NaiveValueGenerator, identify_generator, SmartValueGenerator

if TYPE_CHECKING:
    from .generate_graph import OperationGraph, OperationNode, OperationEdge

@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: Dict[str, Any] # dict of parameter name to value
    request_body: Dict[str, Any]
    operation_properties: OperationProperties

@dataclass
class RequestRequirements:
    edge: 'OperationEdge'
    parameter_requirements: Dict[str, Any]
    request_body_requirements: Dict[str, Any]

@dataclass
class RequestResponse:
    request: RequestData
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

    @staticmethod
    def generate_naive_values(parameters: Dict[str, ParameterProperties], request_body: Dict[str, SchemaProperties]):
        value_generator = NaiveValueGenerator(parameters=parameters, request_body=request_body)
        return value_generator.generate_parameters() if parameters else None, value_generator.generate_request_body() if request_body else None

    @staticmethod
    def generate_smart_values(operation_properties: Dict, requirements: RequestRequirements = None):
        """
        Generate smart values for parameters and request body using LLMs
        :param operation_properties: Dictionary mapping of operation properties
        :param requirements: RequestRequirements object that contains any parameters or request body requirements
        :return: a tuple of the generated parameters and request body
        """
        value_generator = SmartValueGenerator(operation_properties=operation_properties)
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
        if response.status_code // 100 == 2:
            self.successful_query_data.append(request_data)
        else:  # For non-2xx responses
            self.response_handler.handle_error(response, operation_node, request_data, self)
        
        print(f"Request {request_data.operation_properties.operation_id} completed with response text {response.text} and status code {response.status_code}")

    def make_request_data(self, operation_properties: OperationProperties, requirements: RequestRequirements = None) -> RequestData:
        '''
        Process the operation properties by preparing request data for queries with mapping values to parameters and request body
        '''
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method.lower()

        print("=====================================")
        print(f"Processing operation {operation_properties.operation_id} with method {http_method} and path {endpoint_path}")

        # Generate values for parameters and request body
        if self.is_naive:
            parameters, request_body = self.generate_naive_values(
                operation_properties.parameters, operation_properties.request_body
            )
        else:
            parsed_operation = remove_nulls(asdict(operation_properties))
            parameters, request_body = self.generate_smart_values(
                operation_properties=parsed_operation,
            )

        print(f"Parameters: {parameters}")
        print(f"Request Body: {request_body}")

        # Replace path parameters in the endpoint path
        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                path_value = parameters[parameter_name]
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(path_value))

        # Create RequestData object
        return RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=parameters,
            request_body=request_body,
            operation_properties=operation_properties,
        )

    def send_operation_request(self, request_data: RequestData) -> Optional[RequestResponse]:
        '''
        Generate naive requests based on the default values and types
        '''

        '''
        Send the request to the API using the request data.
        '''
        endpoint_path = request_data.endpoint_path
        http_method = request_data.http_method
        parameters = request_data.parameters
        request_body = request_data.request_body

        try:
            select_method = getattr(requests, http_method) # selects correct http method
            full_url = f"{self.api_url}{endpoint_path}"
            if http_method in {"put", "post", "patch"}:
                response = self._send_mime_type(select_method, full_url, parameters, request_body)
            else:
                response = select_method(full_url, params=parameters)
            if response is not None:
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
            return None
        except Exception as err:
            print(f"Unexpected error due to: {err}")
            print(f"Error type: {type(err)}")
            print(f"Endpoint Path: {endpoint_path}")
            print(f"Params: {parameters}")
            return None

    def _send_mime_type(self, select_method, full_url, parameters, request_body):
        params = parameters if parameters else {}
        if "application/json" in request_body:
            req_body = request_body["application/json"]
            print(
                f"Sending request JSON to {full_url} with parameters {parameters} and request body {req_body}")
            response = select_method(full_url, params=params, json=req_body)
        elif "application/x-www-form-urlencoded" in request_body:
            req_body = request_body["application/x-www-form-urlencoded"]
            print(
                f"Sending request ENCODED to {full_url} with parameters {parameters} and request body {req_body}")
            response = select_method(full_url, params=params, data=req_body)
        else:
            # should not reach here
            print(f"Sending request to {full_url} with parameters {parameters}")
            response = select_method(full_url, params=params)
        return response

    def get_response_mappings(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Get the response mappings for the given response
        :param response: The requests response object
        :return:
        """
        if not response:
            return None
        mappings = {}
        if type(response) == dict:
            for key, value in response.items():
                mappings[key] = value
        elif type(response) == list and len(response) > 0:
            mappings = self.get_response_mappings(response[0])
        return mappings

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
            response_mappings = self.get_response_mappings(dependent_response.response.json())
        except:
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
            if response.response.status_code // 100 == 2:
                return response
        return responses[0]

    def depth_traversal(self, curr_node: 'OperationNode', visited: Set) -> Optional[RequestResponse]:
        """
        Generate low-level requests (with no dependencies and hence high depth) first
        :param curr_node: Current operation node
        :param visited: Set of visited operation nodes
        :return: RequestResponse object to allow for requirements parsing
        """
        visited.add(curr_node.operation_id)
        dependent_responses: List[RequestResponse] = []
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                dependent_response = self.depth_traversal(edge.destination, visited)
                if dependent_response is not None and dependent_response.response.ok:
                    dependent_responses.append(dependent_response)
        best_response = self.handle_request_and_dependencies(curr_node, dependent_responses)
        return best_response

    def handle_request_and_dependencies(self, curr_node: 'OperationNode', dependent_responses: List[RequestResponse] = None) -> Optional[RequestResponse]:
        if not dependent_responses:
            response = self.create_and_send_request(curr_node)
            if response is not None:
                self.process_response(response, curr_node)
            return response
        else:
            print(f"Handling dependencies for operation {curr_node.operation_id}")
            responses: List[RequestResponse] = []
            for dependent_response in dependent_responses:
                requirement: RequestRequirements = self.determine_requirement(dependent_response)
                response = self.create_and_send_request(curr_node, requirement)
                responses.append(response)
                if response is not None:
                    self.process_response(response, curr_node)
            return self._determine_best_response(responses)

    def create_and_send_request(self, curr_node: 'OperationNode', requirement: RequestRequirements=None):
        request_data: RequestData = self.make_request_data(curr_node.operation_properties, requirement)
        response: RequestResponse = self.send_operation_request(request_data)
        return response

    def perform_all_requests(self):
        '''
        Generate requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.depth_traversal(operation_node, visited)

    def test_tentative_edge(self, failed_operation_node: 'OperationNode', tentative_edge: 'OperationEdge'):
        '''
        Test the tentative edge by generating a request for the failed operation node
        '''
        request_data = self.make_request_data(failed_operation_node.operation_properties)
        dependent_parameter_requirements = []
        for parameter, dependent_parameter in tentative_edge.similar_parameters.items():
            dependent_parameter_requirements.append(dependent_parameter)

        response: RequestResponse = self.send_operation_request(request_data)
        if response is not None:
            self.process_response(response, failed_operation_node)

#I am guessing this is only used for testing, and that we will do it in a more organized way in the future
def setup_request_generation(api_url, spec_path, spec_name, cached_graph=False):
    # Create and populate the operation graph -- useful for development
    if cached_graph:
        if os.path.exists("operation_graph_" + spec_name + ".pkl"):
            with open("operation_graph_" + spec_name + ".pkl", "rb") as f:
                operation_graph = pickle.load(f)
        else:
            operation_graph = OperationGraph(spec_path, spec_name=spec_name, initialize_graph=False)
            operation_graph.create_graph()  # Generate the graph
            if cached_graph:
                with open("operation_graph_" + spec_name + ".pkl", "wb") as f:
                    pickle.dump(operation_graph, f)
    else:
        operation_graph = OperationGraph(spec_path, spec_name=spec_name, initialize_graph=False)
        operation_graph.create_graph()

    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print("=====================================")
        print(f"Operation: {operation_id}")
        for edge in operation_node.outgoing_edges:
            print(f"Edge: {edge.source.operation_id} -> {edge.destination.operation_id} with parameters: {edge.similar_parameters}")
        for tentative_edge in operation_node.tentative_edges:
            print(f"Tentative Edge: {tentative_edge.source.operation_id} -> {tentative_edge.destination.operation_id} with parameters: {tentative_edge.similar_parameters}")
        print()
        print()

    # Create the request generator with the populated graph
    request_generator = RequestGenerator(operation_graph, api_url)
    return request_generator

if __name__ == "__main__":
    api_url = "http://0.0.0.0:9003"  # API URL for genome-nexus
    spec_path = "specs/original/oas/language-tool.yaml"  # Specification path
    spec_name = "language-tool"  # Specification name
    cache_graph = True #set to false when you are actively changing the graph, however caching is useful for fast development when testing API side things
    generator = setup_request_generation(api_url, spec_path, spec_name, cached_graph=cache_graph)
    generator.perform_all_requests()