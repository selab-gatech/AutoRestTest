from dataclasses import dataclass
from typing import Dict, Set, Any, List, Optional

from requests import Response

from src.handle_response import ResponseHandler

from .generate_graph import OperationGraph, OperationNode, OperationEdge
from .specification_parser import OperationProperties, ParameterProperties, SchemaProperties
import requests
import pickle
import os 

from .value_generator import NaiveValueGenerator, identify_generator

@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: Dict[str, Any] # dict of parameter name to value
    request_body: Dict[str, Any]
    operation_id: str
    operation_properties: OperationProperties

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

class NaiveRequestGenerator:
    def __init__(self, operation_graph: OperationGraph, api_url: str):
        self.operation_graph: OperationGraph = operation_graph
        self.api_url = api_url  
        self.status_codes: Dict[int: StatusCode] = {} # dictionary to track status code occurrences
        self.requests_generated = 0  # Initialize the request count
        self.successful_query_data = [] # List to store successful query data
        self.response_handler = ResponseHandler()

    def generate_parameter_values(self, parameters: Dict[str, ParameterProperties], request_body: Dict[str, SchemaProperties]):
        value_generator = NaiveValueGenerator(parameters=parameters, request_body=request_body)
        return value_generator.generate_parameters() if parameters else None, value_generator.generate_request_body() if request_body else None

    def process_response(self, response: requests.Response, request_data: RequestData, operation_node: OperationNode):
        """
        Process the response from the API.
        """
        if response is None:
            return

        self.requests_generated += 1

        # print(response.text)
        request_and_response = RequestResponse(
            request=request_data,
            response=response,
            response_text=response.text
        )

        if response.status_code not in self.status_codes:
            self.status_codes[response.status_code] = StatusCode(
                status_code=response.status_code,
                count=1,
                requests_and_responses=[request_and_response],
            )
        else:
            self.status_codes[response.status_code].count += 1
            self.status_codes[response.status_code].requests_and_responses.append(request_and_response)

        if response.status_code // 100 == 2:
            self.successful_query_data.append(request_data)
        else:  # For non-2xx responses
            self.response_handler.handle_error(response, operation_node, request_data, self)
        
        print(f"Request {request_data.operation_id} completed with response text {response.text} and status code {response.status_code}")

    def create_and_send_request(self, operation_node: OperationNode):
        """
        Create a RequestData object from an OperationNode and send the request.
        """
        operation_properties = operation_node.operation_properties
        request_data = self.process_operation(operation_properties)
        return self.send_operation_request(request_data)

    def send_operation_request(self, request_data: RequestData) -> Optional[Response]:
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
        select_request_body = list(request_body.values())[0] if request_body else None
        # select the first value in the request body dictionary (key is MIME type)

        print("=====================================")
        print("Attempting to send request to endpoint: ", endpoint_path)
        print("ATTEMPTING WITH PARAMS: ", parameters)
        print("ATTEMPTING WITH REQUEST BODY: ", select_request_body)

        try:
            # Choose the appropriate method from the 'requests' library based on the HTTP method
            select_method = getattr(requests, http_method)
            full_url = f"{self.api_url}{endpoint_path}"

            if http_method in {"put", "post", "patch"}:
                # For PUT, POST, and PATCH requests, include the request body
                response = select_method(full_url, params=parameters, json=select_request_body)
            else:
                response = select_method(full_url, params=parameters)

            print(f"Request to {full_url} completed with status code {response.status_code}")
            #print(f"Response text: {response.text}")
            return response
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
        
    def process_operation(self, operation_properties: OperationProperties) -> RequestData:
        '''
        Process the operation properties to prepare the request data.
        '''
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method.lower()

        # Generate values for parameters and request body
        parameters, request_body = self.generate_parameter_values(
            operation_properties.parameters, operation_properties.request_body
        )

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
            operation_id=operation_properties.operation_id, 
            operation_properties=operation_properties
        )

    def depth_traversal(self, curr_node: OperationNode, visited: Set):
        '''
        Generate low-level requests (with no dependencies and hence high depth) first
        '''
        local_visited_set = set()
        visited.add(curr_node.operation_id)
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.depth_traversal(edge.destination, visited)
                local_visited_set.add(edge.destination.operation_id)
        request_data = self.process_operation(curr_node.operation_properties)
        #handle response
        response = self.send_operation_request(request_data)
        print("GOT RESPONSE ", response)
        if response is not None:
            self.process_response(response, request_data, curr_node)
            # self.attempt_retry(response, request_data)

    def generate_requests(self):
        '''
        Generate naive requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.depth_traversal(operation_node, visited)

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
    request_generator = NaiveRequestGenerator(operation_graph, api_url)
    return request_generator

if __name__ == "__main__":
    api_url = "http://0.0.0.0:9003"  # API URL for genome-nexus
    spec_path = "specs/original/oas/language-tool.yaml"  # Specification path
    spec_name = "language-tool"  # Specification name
    cache_graph = True #set to false when you are actively changing the graph, however caching is useful for fast development when testing API side things
    generator = setup_request_generation(api_url, spec_path, spec_name, cached_graph=cache_graph)
    generator.generate_requests()