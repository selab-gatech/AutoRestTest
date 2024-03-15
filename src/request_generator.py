from .generate_graph import OperationGraph, OperationNode, OperationEdge
from .specification_parser import OperationProperties
import requests

class RequestGenerator:
    def __init__(self, operation_graph: OperationGraph):
        self.operation_graph: OperationGraph = operation_graph

    def naive_generate_request(self, operation_properties: OperationProperties):
        '''
        Generate naive requests based on the default values and types
        '''
        pass

    def naive_depth_traversal(self, curr_node: OperationNode, visited: set):
        '''
        Generate low-level requests (with no dependencies and hence high depth) first
        '''
        visited.add(curr_node.operation_id)
        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.naive_depth_traversal(edge.destination, visited)
        self.naive_generate_request(curr_node.operation_properties)

    def naive_generate_requests(self):
        '''
        Generate naive requests based on the operation graph
        '''
        visited = set()
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in visited:
                self.naive_depth_traversal(operation_node, visited)

