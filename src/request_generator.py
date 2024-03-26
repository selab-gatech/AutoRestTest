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

    def generate_parameter_values(self, parameters: Dict[AnyStr, ParameterProperties], request_body: Dict[AnyStr, SchemaProperties]):
        value_generator = NaiveValueGenerator(parameters=parameters, request_body=request_body)
        return value_generator.generate_parameters() if parameters else None, value_generator.generate_request_body() if request_body else None

    def send_operation_request(self, operation_properties: OperationProperties):
        '''
        Generate naive requests based on the default values and types
        '''
        endpoint_path: AnyStr = operation_properties.endpoint_path
        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                get_path_value = identify_generator(parameter_properties.schema.type)
                endpoint_path.replace("{"+parameter_name+"}", get_path_value)

        parameters, request_body = self.generate_parameter_values(operation_properties.parameters, operation_properties.request_body)

    def depth_traversal(self, curr_node: OperationNode, visited: Set):
        '''
        Generate low-level requests (with no dependencies and hence high depth) first
        '''
        visited.add(curr_node.operation_id)
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.depth_traversal(edge.destination, visited)
        self.send_operation_request(curr_node.operation_properties)

    def generate_requests(self):
        '''
        Generate naive requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.depth_traversal(operation_node, visited)

