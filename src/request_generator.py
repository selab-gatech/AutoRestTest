from dataclasses import dataclass
from typing import Dict, AnyStr, Set, Any

from .generate_graph import OperationGraph, OperationNode, OperationEdge
from .specification_parser import OperationProperties, ParameterProperties, SchemaProperties
import requests

from .value_generator import NaiveValueGenerator, identify_generator


@dataclass
class RequestData:
    endpoint_path: AnyStr
    http_method: AnyStr
    parameter_values: Dict[AnyStr, Any] # dict of parameter name to value
    request_body: AnyStr
    content_type: AnyStr
    operation_id: AnyStr
    operation_properties: OperationProperties

class NaiveRequestGenerator:
    def __init__(self, operation_graph: OperationGraph):
        self.operation_graph: OperationGraph = operation_graph
        self.api_url = api_url  # Set the base URL of your API here

    def generate_parameter_values(self, parameters: Dict[AnyStr, ParameterProperties], request_body: Dict[AnyStr, SchemaProperties]):
        value_generator = NaiveValueGenerator(parameters=parameters, request_body=request_body)
        return value_generator.generate_parameters() if parameters else None, value_generator.generate_request_body() if request_body else None

    def send_operation_request(self, request_data: RequestData):
        '''
        Generate naive requests based on the default values and types
        '''
        # endpoint_path: AnyStr = operation_properties.endpoint_path
        # for parameter_name, parameter_properties in operation_properties.parameters.items():
        #     if parameter_properties.in_value == "path":
        #         get_path_value = identify_generator(parameter_properties.schema.type)
        #         endpoint_path.replace("{"+parameter_name+"}", get_path_value)

        # parameters, request_body = self.generate_parameter_values(operation_properties.parameters, operation_properties.request_body)

        '''
        Send the request to the API using the request data.
        '''
        endpoint_path = request_data.endpoint_path
        http_method = request_data.http_method
        parameters = request_data.parameters
        request_body = request_data.request_body
        content_type = request_data.content_type

        try:
            # Choose the appropriate method from the 'requests' library based on the HTTP method
            select_method = getattr(requests, http_method)
            full_url = f"{self.api_url}{endpoint_path}"

            # Prepare and send the request
            if http_method in {"put", "post", "patch"}:
                # For PUT, POST, and PATCH requests, include the request body
                response = select_method(full_url, params=parameters, json=request_body)
            else:
                # For GET and DELETE requests, send only the parameters
                response = select_method(full_url, params=parameters)

            # Handle the response as needed
            print(f"Request to {http_method.upper()} {endpoint_path} completed with status code {response.status_code}")
            return response

        except requests.exceptions.RequestException as err:
            print(f"Request failed due to error: {err}")
            print(f"Endpoint Path: {endpoint_path}")
            print(f"Params: {parameters}")
            return None
        except Exception as err:
            print(f"Request failed due to error: {err}")
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

        # Determine the content type
        content_type = "application/json"  

        # Create RequestData object
        return RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=parameters,
            request_body=request_body,
            content_type=content_type,
            operation_id=operation_properties.operation_id
        )
    
    
    
    def depth_traversal(self, curr_node: OperationNode, visited: Set):
        '''
        Generate low-level requests (with no dependencies and hence high depth) first
        '''
        visited.add(curr_node.operation_id)
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.depth_traversal(edge.destination, visited)
        request_data = self.process_operation(curr_node.operation_properties)
        self.send_operation_request(request_data)


    def generate_requests(self):
        '''
        Generate naive requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.depth_traversal(operation_node, visited)

