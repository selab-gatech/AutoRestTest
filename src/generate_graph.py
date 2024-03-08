from typing import List, Dict

from .specification_parser import OperationProperties, SpecificationParser
from .similarity_comparator import OperationDependencyComparator

class OperationNode:
    def __init__(self, operation_properties: OperationProperties):
        self.operation_id = operation_properties.operation_id
        self.operation_properties: OperationProperties = operation_properties

class OperationEdge:
    def __init__(self, source: OperationNode, destination: OperationNode):
        self.source = source
        self.destination = destination
        self.parameters = {} # have parameters as the weights

class OperationGraph:
    def __init__(self, spec_path, spec_name=None):
        self.spec_path = spec_path
        self.spec_name = spec_name
        self.operation_nodes: Dict[str, OperationNode] = {}
        self.operation_edges: List[OperationEdge] = []

    def add_operation_node(self, operation_properties: OperationProperties):
        self.operation_nodes[operation_properties.operation_id] = OperationNode(operation_properties)

    def add_operation_edge(self, operation_id: str, dependent_operation_id: str, parameters: Dict[str, str]):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        edge = OperationEdge(source=source_node, destination=destination_node)
        edge.parameters = parameters
        self.operation_edges.append(edge)

    def determine_dependencies(self, operations):
        operation_dependency_comparator = OperationDependencyComparator()
        visited = set()
        for operation_id, operation_properties in operations.items():
            for dependent_operation_id, dependent_operation_properties in operations.items():
                if not operation_properties.parameters or not dependent_operation_properties.responses:
                    continue
                similarity_1to2 = operation_dependency_comparator.compare(operation_properties, dependent_operation_properties)
                if len(similarity_1to2) > 0:
                    self.add_operation_edge(operation_id, dependent_operation_id, similarity_1to2)
                #if len(similarity_2to1) > 0:
                #    self.add_operation_edge(dependent_operation_id, operation_id, similarity_2to1)

    def create_graph(self):
        spec_parser = SpecificationParser(self.spec_path, self.spec_name)
        operations: Dict[str, OperationProperties] = spec_parser.parse_specification()
        for operation_id, operation_properties in operations.items():
            self.add_operation_node(operation_properties)
        self.determine_dependencies(operations)

if __name__ == "__main__":
    #operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print(f"Operation: {operation_id}")
    for operation_edge in operation_graph.operation_edges:
        print(f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.parameters}")